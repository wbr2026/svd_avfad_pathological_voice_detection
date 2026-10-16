#!/usr/bin/env python3
"""
TTS Synthetic Data Training Script v10 - TAPT / PTAPT (Whisper-small)
  *** orig_aug_only + speaker-disjoint TTS + Stage-2 1.2x balanced scale ***
======================================================================
Copy of ``..._bal11_tts_scale11.py``. Does NOT modify 1.1x / 1.25x / 1.5x
scripts, manifests, rankings, bal11, TTS pools, or existing results.

1.2x (num_augmented=1), nested prefix of 1.5x ranking:
    B = 2P
    Q = int(math.floor(0.20 * B + 0.5))   # half-up, not bankers round
    healthy TTS  = 2(P-H) + Q
    patho TTS    = Q
    Stage-2 entries: H = P = 2P + Q

1.0x âŠ?1.1x âŠ?1.2x âŠ?1.25x âŠ?1.5x  (prefix of existing ranking).

Audit-only (no training):
  cd "/home/beierwang/testvscode/AI4Voice copy/src"
  python -u train_with_tts_v10_tapt_ptapt_whisper_layersweep_orig_aug_disjoint_synth_bal11_tts_scale12.py \\
    --data_dir ./svd_cleaned_paper_tts \\
    --scale15_source_dir experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale15/feat8/tapt_then_selected_layers_L0-L8 \\
    --scale125_source_dir experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale125/feat8/tapt_then_selected_layers_L0-L8 \\
    --scale11_source_dir experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale11/feat8/tapt_then_selected_layers_L0-L8 \\
    --bal11_baseline_dir experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11/feat8/tapt_then_selected_layers_L0-L8 \\
    --task detection --recording_type sentence \\
    --method tapt_then_selected_layers \\
    --model_name openai/whisper-small \\
    --feature_layer 8 --strategy_layers "0,1,2,3,4,5,6,7,8" \\
    --num_augmented 1 --seed 42 --reseed_each_fold --n_folds 10 \\
    --exclude_test_speaker_synthetic --class_weight none \\
    --audit_tts_scale_only \\
    --output_dir experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale12/feat8/tapt_then_selected_layers_L0-L8
"""

from __future__ import annotations

import os
import math
import json
import csv
import types
import random
import argparse
import logging
import warnings
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    WhisperModel,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report

# Reuse v9 helpers verbatim (data IO, CV split, strategy, patch, safety, metrics)
from whisper_train_utils import (
    set_deterministic,
    VoiceDisorderDatasetTTS,
    build_sentence_augmenter,
    build_vowel_augmenter,
    load_metadata_with_tts,
    create_cv_folds_with_tts,
    parse_strategy_layers,
    apply_strategy,
    patch_forward_to_use_feature_layer,
    safety_check_and_log,
    compute_metrics_detection,
    compute_metrics_classification,
    evaluate_speaker_level,
    _is_whisper,
    HAS_AUDIOMENTATIONS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dynamic safety_check_and_log override
# The v9 original has expected_frozen hard-coded as (7,8,9,10,11), which
# breaks for any strategy_layers that include layer 7 or deeper.
# This replacement computes expected_frozen from the actual model's encoder
# size minus the strategy_layers that are being fine-tuned.
# ============================================================================

def safety_check_and_log(final_model, strategy, strategy_layers,
                         expected_frozen=None):
    """Raise RuntimeError if any param under frozen encoder layers is trainable.
    expected_frozen is computed dynamically: all encoder layer indices NOT in
    strategy_layers. Also asserts projector + classifier have trainable params.
    """
    underlying = final_model
    if hasattr(final_model, "base_model") and hasattr(final_model.base_model, "model"):
        underlying = final_model.base_model.model
    enc = underlying.encoder

    n_enc = len(enc.layers)
    if expected_frozen is None:
        strategy_set = set(strategy_layers)
        expected_frozen = [i for i in range(n_enc) if i not in strategy_set]

    bad = []
    for i in expected_frozen:
        if i >= n_enc:
            continue
        for n, p in enc.layers[i].named_parameters():
            if p.requires_grad:
                bad.append(f"encoder.layers.{i}.{n}")
    if bad:
        msg = (
            f"[strategy={strategy}] SAFETY CHECK FAILED: "
            f"{len(bad)} trainable param tensors found in frozen layers "
            f"{list(expected_frozen)}. Should be ZERO. Sample:\n  "
            + "\n  ".join(bad[:30])
            + (f"\n  ... and {len(bad) - 30} more" if len(bad) > 30 else "")
        )
        raise RuntimeError(msg)

    for hname in ("projector", "classifier"):
        mod = getattr(underlying, hname, None)
        if mod is None:
            raise RuntimeError(
                f"[strategy={strategy}] Whisper head module '{hname}' not found"
            )
        if not any(p.requires_grad for p in mod.parameters()):
            raise RuntimeError(
                f"[strategy={strategy}] Head module '{hname}' has no "
                "trainable params (must always be trainable)"
            )

    total = sum(p.numel() for p in final_model.parameters())
    trainable = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    ratio = (trainable / total) if total > 0 else 0.0
    trainable_names = [n for n, p in final_model.named_parameters() if p.requires_grad]

    logger.info("=" * 64)
    logger.info(f"[strategy] name={strategy}")
    logger.info(f"[strategy] strategy_layers={list(strategy_layers)}")
    logger.info(f"[strategy] total params:     {total/1e6:.3f}M  ({total})")
    logger.info(f"[strategy] trainable params: {trainable/1e6:.4f}M  ({trainable})")
    logger.info(f"[strategy] trainable ratio:  {ratio*100:.4f}%")
    logger.info(f"[strategy] #trainable tensors: {len(trainable_names)}")
    preview = 40
    logger.info(f"[strategy] trainable parameter names (first {preview}):")
    for n in trainable_names[:preview]:
        logger.info(f"    + {n}")
    if len(trainable_names) > preview:
        logger.info(f"    ... and {len(trainable_names) - preview} more")
    logger.info(f"[strategy] SAFETY CHECK OK: encoder.layers.{list(expected_frozen)} "
                f"are fully frozen; projector + classifier trainable.")
    logger.info("=" * 64)

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(ratio),
        "num_trainable_tensors": int(len(trainable_names)),
    }


# ============================================================================
# Constants / methods
# ============================================================================

VALID_METHODS = (
    "tapt_then_selected_layers",
    "ptapt_then_selected_layers",
)

STAGE2_STRATEGY = "selected_layers"


# ============================================================================
# Stage-2 dataset: online aug on REAL originals only (TTS = clean only)
# ============================================================================

class VoiceDisorderDatasetTTSOrigAugOnly(Dataset):
    """Paper-faithful data++: augment only non-synthetic train samples."""

    def __init__(
        self,
        file_paths,
        labels,
        feature_extractor,
        recording_types,
        is_synthetic,
        num_augmented=1,
        max_length_sec=5.0,
        sampling_rate=16000,
    ):
        if len(is_synthetic) != len(file_paths):
            raise ValueError(
                f"is_synthetic length {len(is_synthetic)} != "
                f"file_paths length {len(file_paths)}"
            )

        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.recording_types = recording_types
        self.num_augmented = num_augmented
        self.max_length = int(max_length_sec * sampling_rate)
        self.sampling_rate = sampling_rate

        self.n_files = len(file_paths)
        self._entries = []
        n_real_clean = n_real_aug = n_synth_clean = 0
        for i, synth in enumerate(is_synthetic):
            self._entries.append((i, False))
            if synth:
                n_synth_clean += 1
            else:
                n_real_clean += 1
                for _ in range(num_augmented):
                    self._entries.append((i, True))
                    n_real_aug += 1

        self._is_whisper_fe = type(feature_extractor).__name__ == \
            "WhisperFeatureExtractor"
        self.sentence_aug = build_sentence_augmenter()
        self.vowel_aug = build_vowel_augmenter()

        logger.info(
            f"  Dataset (orig_aug_only): {self.n_files} files -> "
            f"{n_real_clean} real clean + {n_real_aug} real aug + "
            f"{n_synth_clean} synth clean = {len(self)} total"
        )

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        import torchaudio

        file_idx, apply_aug = self._entries[idx]
        audio_path = self.file_paths[file_idx]
        label = self.labels[file_idx]
        rec_type = self.recording_types[file_idx]

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        if apply_aug and HAS_AUDIOMENTATIONS:
            audio_np = waveform.numpy().astype(np.float32)
            try:
                if rec_type == 'sentence':
                    if self.sentence_aug is not None:
                        audio_np = self.sentence_aug(
                            samples=audio_np, sample_rate=self.sampling_rate
                        )
                else:
                    if self.vowel_aug is not None:
                        audio_np = self.vowel_aug(
                            samples=audio_np, sample_rate=self.sampling_rate
                        )
                waveform = torch.tensor(audio_np, dtype=torch.float32)
            except Exception as e:
                logger.debug(f"Aug failed {audio_path}: {e}")

        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.shape[0] < self.max_length:
            waveform = torch.cat([
                waveform,
                torch.zeros(self.max_length - waveform.shape[0]),
            ])

        if self._is_whisper_fe:
            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
        else:
            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=False,
            )

        if 'input_values' in inputs:
            return {
                'input_values': inputs['input_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long),
            }
        if 'input_features' in inputs:
            return {
                'input_features': inputs['input_features'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long),
            }
        raise KeyError(
            f"Feature extractor returned unknown keys: {list(inputs.keys())}."
        )


# ============================================================================
# Adaptive-pretraining dataset (no labels, no augmentation, single copy)
# ============================================================================

class WhisperAdaptiveAudioDataset(Dataset):
    """Yields raw ``input_features`` tensors for masked-feature reconstruction.

    No augmentation, no labels, no synthetic-flag handling beyond what the
    caller already filtered. Identical waveform-loading logic to
    ``VoiceDisorderDatasetTTS`` (truncate/pad to ``max_length_sec`` then run
    the WhisperFeatureExtractor, which internally pads mel to 3000 frames).
    """

    def __init__(
        self,
        file_paths,
        feature_extractor,
        max_length_sec: float = 5.0,
        sampling_rate: int = 16000,
    ):
        if type(feature_extractor).__name__ != "WhisperFeatureExtractor":
            raise ValueError(
                "WhisperAdaptiveAudioDataset requires WhisperFeatureExtractor"
                f" (got {type(feature_extractor).__name__})"
            )
        self.file_paths = list(file_paths)
        self.feature_extractor = feature_extractor
        self.max_length = int(max_length_sec * sampling_rate)
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        import torchaudio

        audio_path = self.file_paths[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        if waveform.shape[0] > self.max_length:
            waveform = waveform[: self.max_length]
        elif waveform.shape[0] < self.max_length:
            waveform = torch.cat(
                [waveform, torch.zeros(self.max_length - waveform.shape[0])]
            )

        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        return {"input_features": inputs["input_features"].squeeze(0)}


# ============================================================================
# Masked Acoustic Feature Reconstruction model
# ============================================================================

class WhisperReconstructionModel(nn.Module):
    """Whisper encoder + reconstruction head for masked acoustic feature
    pretraining. The reconstruction is read from encoder hidden_states at
    ``feature_layer`` (default 6), matching the v9 classification head's
    feature_layer plumbing.

    Stage 1 trainable params:
        encoder.layers.0..feature_layer + reconstruction_head
    All other params (conv1, conv2, embed_positions, encoder.layers.{>fl},
    encoder.layer_norm) are frozen.
    """

    def __init__(self, model_name: str, feature_layer: int, num_mel_bins: int):
        super().__init__()
        whisper_full = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper_full.encoder
        # Free decoder memory - we never use it.
        del whisper_full

        d_model = self.encoder.config.d_model
        n_layers = len(self.encoder.layers)
        if not (0 <= feature_layer < n_layers):
            raise ValueError(
                f"feature_layer={feature_layer} out of range [0,{n_layers - 1}]"
            )
        self.feature_layer = feature_layer
        self.num_mel_bins = num_mel_bins
        self.n_layers = n_layers
        self.d_model = d_model

        self.reconstruction_head = nn.Linear(d_model, num_mel_bins)
        nn.init.normal_(self.reconstruction_head.weight, std=0.02)
        nn.init.zeros_(self.reconstruction_head.bias)

    def configure_stage1_grads(self):
        """Freeze everything, then unfreeze encoder.layers.0..feature_layer
        and reconstruction_head. Returns (trainable_param_count, total)."""
        for p in self.parameters():
            p.requires_grad = False
        for i in range(self.feature_layer + 1):
            for p in self.encoder.layers[i].parameters():
                p.requires_grad = True
        for p in self.reconstruction_head.parameters():
            p.requires_grad = True

        # Hard safety: encoder.layers.{feature_layer+1..end} must stay frozen.
        bad = []
        for i in range(self.feature_layer + 1, self.n_layers):
            for n, p in self.encoder.layers[i].named_parameters():
                if p.requires_grad:
                    bad.append(f"encoder.layers.{i}.{n}")
        if bad:
            raise RuntimeError(
                "[adaptive-stage1] frozen-layer leak in "
                f"encoder.layers.{self.feature_layer + 1}..{self.n_layers - 1}: "
                f"{bad[:5]}"
            )
        # And: conv1, conv2, embed_positions, layer_norm should be frozen.
        for mod_name in ("conv1", "conv2", "embed_positions", "layer_norm"):
            mod = getattr(self.encoder, mod_name, None)
            if mod is None:
                continue
            for n, p in mod.named_parameters():
                if p.requires_grad:
                    raise RuntimeError(
                        f"[adaptive-stage1] encoder.{mod_name}.{n} should be frozen"
                    )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Return reconstruction predictions of shape [B, T_enc, num_mel_bins]."""
        enc_out = self.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] = post-conv pre-layer-0 ;
        # hidden_states[i + 1] = output of encoder.layers[i]
        hidden = enc_out.hidden_states[self.feature_layer + 1]
        # Apply encoder.layer_norm to mirror the v9 classification head when
        # feature_layer < n_layers - 1 (LN is frozen here, just normalization).
        if self.feature_layer != (self.n_layers - 1):
            hidden = self.encoder.layer_norm(hidden)
        return self.reconstruction_head(hidden)


# ============================================================================
# Time masking (wav2vec2-style block masks at ENCODER resolution)
# ============================================================================

def compute_encoder_time_mask(
    batch_size: int,
    n_enc_frames: int,
    mask_time_prob: float,
    mask_time_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask of shape [B, n_enc_frames] (True = masked).

    Strategy: expected number of masked frames per sample ~ mask_time_prob * T.
    Number of mask spans per sample = max(1, round(mask_time_prob * T /
    mask_time_length)). Span starts are sampled uniformly in [0, T-L];
    spans may overlap, so realized masked count can be slightly less.
    """
    if mask_time_length <= 0 or mask_time_length > n_enc_frames:
        raise ValueError(
            f"mask_time_length={mask_time_length} invalid for T={n_enc_frames}"
        )
    if not (0.0 < mask_time_prob <= 1.0):
        raise ValueError(f"mask_time_prob={mask_time_prob} out of (0,1]")

    n_spans = max(1, int(round(mask_time_prob * n_enc_frames / mask_time_length)))
    mask = torch.zeros(batch_size, n_enc_frames, dtype=torch.bool, device=device)
    high = n_enc_frames - mask_time_length + 1
    if high <= 0:
        return mask
    starts = torch.randint(0, high, (batch_size, n_spans), device=device)
    # Vectorized scatter would be nicer; this is fine for B<=32.
    for b in range(batch_size):
        for s in starts[b].tolist():
            mask[b, s:s + mask_time_length] = True
    return mask


# ============================================================================
# Stage 1: adaptive pretraining loop
# ============================================================================

def _classify_label_role(label2id: dict):
    """Return (pathological_id, healthy_id_or_None). Heuristic match on
    label strings; raises if no pathological-like label is present.
    """
    keys_lower = {k.lower(): v for k, v in label2id.items()}
    patho_id = None
    healthy_id = None
    for k, v in keys_lower.items():
        if k.startswith("pathol") or k == "abnormal" or k == "patient":
            patho_id = v
        if k.startswith("health") or k == "normal" or k == "control":
            healthy_id = v
    if patho_id is None:
        raise ValueError(
            f"Cannot identify the pathological label among {list(label2id)}; "
            "PTAPT needs a label whose name starts with 'pathol' / "
            "'abnormal' / 'patient'."
        )
    return patho_id, healthy_id


def filter_adaptive_pool(
    method: str,
    train_files,
    train_labels,
    train_speaker_ids,
    train_is_synthetic,
    label2id,
    include_synthetic: bool = False,
):
    """Build the (files, speaker_ids) pool to use for adaptive pretraining.

    DEFAULT behavior (``include_synthetic=False``):
        Adaptive pretraining uses ONLY *real* train-split samples. This is
        required to honor the strict speaker-leak rule: TTS-synthetic
        samples in this dataset are conditioned on REAL speakers drawn
        from the entire corpus, and the v9 CV split intentionally puts
        ALL synthetic samples into every fold's train pool. So a
        synthetic sample's nominal ``speaker_id`` will frequently coincide
        with a test-fold speaker, which would (correctly) trip the
        speaker-overlap leak check. Excluding synthetic from stage 1 is
        therefore the only research-defensible choice â€?synthetic is
        designed as a stage-2 *augmentation*, not real domain audio.

    OPT-IN (``include_synthetic=True``):
        Includes synthetic samples too. The downstream leak check WILL
        raise on this dataset because of the speaker-id collision above;
        this flag exists only for completeness and for datasets where
        synthetic speakers are disjoint from test speakers.

    Returns a dict with the pool files, speaker_ids, scope string, and
    counts (n_total / n_original / n_synthetic / n_skipped_synthetic).
    """
    if method == "tapt_then_selected_layers":
        keep_mask = [True] * len(train_files)
        base_scope = "train_real_all_labels"
        if include_synthetic:
            base_scope = "train_all_real_and_synthetic"
    elif method == "ptapt_then_selected_layers":
        patho_id, _ = _classify_label_role(label2id)
        keep_mask = [lab == patho_id for lab in train_labels]
        base_scope = f"train_real_pathological_only(label_id={patho_id})"
        if include_synthetic:
            base_scope = (
                f"train_pathological_only_real_and_synthetic(label_id={patho_id})"
            )
    else:
        raise ValueError(f"Unknown method={method}")

    n_skipped_synth = 0
    keep_idx = []
    for i, k in enumerate(keep_mask):
        if not k:
            continue
        if not include_synthetic and train_is_synthetic[i]:
            n_skipped_synth += 1
            continue
        keep_idx.append(i)

    pool_files = [train_files[i] for i in keep_idx]
    pool_speakers = [train_speaker_ids[i] for i in keep_idx]
    n_synth = sum(1 for i in keep_idx if train_is_synthetic[i])
    n_orig = len(keep_idx) - n_synth
    return {
        "files": pool_files,
        "speaker_ids": pool_speakers,
        "scope": base_scope,
        "n_total": len(keep_idx),
        "n_original": n_orig,
        "n_synthetic": n_synth,
        "n_skipped_synthetic": n_skipped_synth,
        "include_synthetic": bool(include_synthetic),
    }


def assert_no_speaker_leak(
    fold_idx: int,
    adaptive_speaker_ids,
    train_speaker_ids,
    test_speaker_ids,
    method: str,
    pool_info: dict,
):
    """Print full audit + raise RuntimeError if adaptive âˆ?test is non-empty."""
    adaptive_set = set(map(str, adaptive_speaker_ids))
    train_set = set(map(str, train_speaker_ids))
    test_set = set(map(str, test_speaker_ids))
    overlap = adaptive_set & test_set

    logger.info("=" * 64)
    logger.info(f"[leak-check] fold_idx                          = {fold_idx}")
    logger.info(f"[leak-check] method                            = {method}")
    logger.info(f"[leak-check] adaptive_pretrain_num_samples     = {pool_info['n_total']}")
    logger.info(f"[leak-check]   .of which n_original            = {pool_info['n_original']}")
    logger.info(f"[leak-check]   .of which n_synthetic           = {pool_info['n_synthetic']}")
    logger.info(f"[leak-check]   .synthetic_skipped (leak-guard) = "
                f"{pool_info.get('n_skipped_synthetic', 0)}")
    logger.info(f"[leak-check] adaptive_pretrain_data_scope      = {pool_info['scope']}")
    logger.info(f"[leak-check] include_synthetic flag            = "
                f"{pool_info.get('include_synthetic', False)}")
    logger.info(f"[leak-check] #train speakers                   = {len(train_set)}")
    logger.info(f"[leak-check] #test  speakers                   = {len(test_set)}")
    logger.info(f"[leak-check] #adaptive speakers                = {len(adaptive_set)}")
    logger.info(f"[leak-check] #(adaptive âˆ?test) overlap        = {len(overlap)}")
    logger.info(f"[leak-check] #(adaptive âŠ?train) violations    = "
                f"{len(adaptive_set - train_set)}")
    logger.info("=" * 64)

    if overlap:
        raise RuntimeError(
            f"[fold {fold_idx} | method={method}] DATA LEAK: "
            f"{len(overlap)} speaker_id(s) appear in BOTH the adaptive "
            f"pretraining pool and the test set. Examples: "
            f"{sorted(overlap)[:10]}"
        )
    illegal = adaptive_set - train_set
    if illegal:
        raise RuntimeError(
            f"[fold {fold_idx} | method={method}] SCOPE VIOLATION: "
            f"{len(illegal)} adaptive-pool speakers are NOT in the train "
            f"split. Examples: {sorted(illegal)[:10]}"
        )


def filter_stage2_train_disjoint_synthetic(
    fold_idx: int,
    train_files,
    train_labels,
    train_rec_types,
    train_speaker_ids,
    train_is_synthetic,
    test_speaker_ids,
):
    """Drop synthetic train samples whose speaker_id is in the test set."""
    test_set = set(map(str, test_speaker_ids))
    out_files, out_labels, out_rec_types, out_speaker_ids, out_is_synth = (
        [], [], [], [], [],
    )
    n_synth_total = n_synth_kept = n_synth_excluded = 0
    n_real = 0

    for fpath, lab, rt, sid, synth in zip(
        train_files,
        train_labels,
        train_rec_types,
        train_speaker_ids,
        train_is_synthetic,
    ):
        if synth:
            n_synth_total += 1
            if str(sid) in test_set:
                n_synth_excluded += 1
                continue
            n_synth_kept += 1
        else:
            n_real += 1

        out_files.append(fpath)
        out_labels.append(lab)
        out_rec_types.append(rt)
        out_speaker_ids.append(sid)
        out_is_synth.append(synth)

    logger.info("=" * 64)
    logger.info(f"[stage2-synth][fold {fold_idx}] exclude_test_speaker_synthetic=True")
    logger.info(f"[stage2-synth][fold {fold_idx}] #test speakers              = "
                f"{len(test_set)}")
    logger.info(f"[stage2-synth][fold {fold_idx}] synthetic before filter     = "
                f"{n_synth_total}")
    logger.info(f"[stage2-synth][fold {fold_idx}] synthetic excluded (test spk)= "
                f"{n_synth_excluded}")
    logger.info(f"[stage2-synth][fold {fold_idx}] synthetic kept (train spk)  = "
                f"{n_synth_kept}")
    logger.info(f"[stage2-synth][fold {fold_idx}] real recordings (unchanged) = "
                f"{n_real}")
    logger.info(f"[stage2-synth][fold {fold_idx}] stage2 train pool after     = "
                f"{len(out_files)} (orig={n_real}, synth={n_synth_kept})")
    logger.info("=" * 64)

    return {
        "train_files": out_files,
        "train_labels": out_labels,
        "train_rec_types": out_rec_types,
        "train_speaker_ids": out_speaker_ids,
        "train_is_synthetic": out_is_synth,
        "n_synth_total": n_synth_total,
        "n_synth_excluded_test_speakers": n_synth_excluded,
        "n_synth_kept": n_synth_kept,
        "n_real": n_real,
    }


def _speaker_sort_key(sid):
    s = str(sid)
    return (0, int(s)) if s.isdigit() else (1, s)


def sample_stage2_synth_speaker_balanced(
    fold_idx: int,
    train_files,
    train_labels,
    train_rec_types,
    train_speaker_ids,
    train_is_synthetic,
    label2id,
    num_augmented: int,
    fold_seed: int,
    fold_dir: Path,
):
    """Keep all real rows; speaker-balanced subsample of TTS to 1:1 entries.

    With orig_aug_only, each real file becomes ``1 + num_augmented`` Stage-2
    entries and each TTS file stays one clean entry. Equalizing entry counts
    requires::

        n_tts_target = (1 + num_augmented) * (nP_real - nH_real)
    """
    if "healthy" not in label2id:
        raise ValueError(
            "balance_synth_to_ones requires a 'healthy' label in label2id, "
            f"got {label2id}"
        )
    healthy_id = label2id["healthy"]

    real_idx = [i for i, s in enumerate(train_is_synthetic) if not s]
    synth_idx = [i for i, s in enumerate(train_is_synthetic) if s]

    nP = sum(1 for i in real_idx if train_labels[i] != healthy_id)
    nH = sum(1 for i in real_idx if train_labels[i] == healthy_id)
    n_real = len(real_idx)
    n_synth_available = len(synth_idx)
    multiplier = 1 + int(num_augmented)
    n_tts_target = multiplier * (nP - nH)

    logger.info("=" * 64)
    logger.info(f"[stage2-bal11][fold {fold_idx}] speaker-balanced TTS quota")
    logger.info(f"[stage2-bal11][fold {fold_idx}] nP_real={nP} nH_real={nH} "
                f"num_augmented={num_augmented}")
    logger.info(f"[stage2-bal11][fold {fold_idx}] n_tts_target="
                f"{multiplier}*(nP-nH)={n_tts_target}")
    logger.info(f"[stage2-bal11][fold {fold_idx}] synth available "
                f"(after train-spk filter)={n_synth_available}")

    if n_tts_target < 0:
        raise RuntimeError(
            f"[stage2-bal11][fold {fold_idx}] nP_real ({nP}) < nH_real ({nH}); "
            "healthy-only TTS cannot 1:1-balance pathological:healthy entries."
        )
    if n_tts_target == 0:
        logger.info(f"[stage2-bal11][fold {fold_idx}] target=0; dropping all TTS")
        keep_synth = []
        copies_hist = {}
        n_spk_with_tts = 0
    else:
        if n_synth_available < n_tts_target:
            raise RuntimeError(
                f"[stage2-bal11][fold {fold_idx}] not enough train-speaker TTS: "
                f"need {n_tts_target}, have {n_synth_available}. Use the 4x "
                "pool (svd_cleaned_paper_tts), not _1x."
            )

        rng = np.random.RandomState(int(fold_seed))
        by_spk = defaultdict(list)
        n_nonhealthy_synth = 0
        for i in synth_idx:
            if train_labels[i] != healthy_id:
                n_nonhealthy_synth += 1
                continue
            by_spk[str(train_speaker_ids[i])].append(i)
        if n_nonhealthy_synth:
            logger.warning(
                f"[stage2-bal11][fold {fold_idx}] ignored "
                f"{n_nonhealthy_synth} non-healthy synthetic row(s)"
            )
        speakers = sorted(by_spk.keys(), key=_speaker_sort_key)
        n_spk_with_tts = len(speakers)
        if n_spk_with_tts == 0:
            raise RuntimeError(
                f"[stage2-bal11][fold {fold_idx}] no healthy TTS speakers "
                "after filter"
            )
        for sid in speakers:
            rng.shuffle(by_spk[sid])

        caps = {sid: len(by_spk[sid]) for sid in speakers}
        quota = {sid: 0 for sid in speakers}
        remaining = n_tts_target
        # Even water-fill: cycle shuffled speakers, +1 if under cap.
        order = list(speakers)
        rng.shuffle(order)
        guard = 0
        max_iters = n_tts_target + len(order) + 8
        while remaining > 0:
            progressed = False
            next_order = []
            for sid in order:
                if remaining <= 0:
                    next_order.append(sid)
                    continue
                if quota[sid] < caps[sid]:
                    quota[sid] += 1
                    remaining -= 1
                    progressed = True
                    if quota[sid] < caps[sid]:
                        next_order.append(sid)
                # depleted speakers are dropped from the next cycle
            order = next_order
            guard += 1
            if remaining > 0 and not progressed:
                raise RuntimeError(
                    f"[stage2-bal11][fold {fold_idx}] water-fill stalled with "
                    f"{remaining} TTS still needed"
                )
            if guard > max_iters:
                raise RuntimeError(
                    f"[stage2-bal11][fold {fold_idx}] water-fill exceeded "
                    f"iteration cap ({max_iters})"
                )

        keep_synth = []
        for sid in speakers:
            keep_synth.extend(by_spk[sid][:quota[sid]])
        copies_hist = dict(Counter(quota.values()))
        if len(keep_synth) != n_tts_target:
            raise RuntimeError(
                f"[stage2-bal11][fold {fold_idx}] selected "
                f"{len(keep_synth)} != target {n_tts_target}"
            )

    keep_idx = real_idx + keep_synth
    out_files = [train_files[i] for i in keep_idx]
    out_labels = [train_labels[i] for i in keep_idx]
    out_rec_types = [train_rec_types[i] for i in keep_idx]
    out_speaker_ids = [train_speaker_ids[i] for i in keep_idx]
    out_is_synth = [train_is_synthetic[i] for i in keep_idx]

    n_tts_selected = len(keep_synth)
    entries_p = nP * multiplier
    entries_h = nH * multiplier + n_tts_selected
    logger.info(
        f"[stage2-bal11][fold {fold_idx}] selected TTS={n_tts_selected} "
        f"from {n_spk_with_tts} speakers; copies/spk hist={copies_hist}"
    )
    logger.info(
        f"[stage2-bal11][fold {fold_idx}] Stage-2 entries after orig_aug: "
        f"P={entries_p} H={entries_h} "
        f"({'balanced' if entries_p == entries_h else 'NOT BALANCED'})"
    )
    logger.info("=" * 64)

    if entries_p != entries_h:
        raise RuntimeError(
            f"[stage2-bal11][fold {fold_idx}] entry counts not 1:1: "
            f"P={entries_p} H={entries_h}"
        )

    manifest = {
        "fold_idx": fold_idx,
        "fold_seed": int(fold_seed),
        "nP_real": nP,
        "nH_real": nH,
        "num_augmented": int(num_augmented),
        "n_tts_target": n_tts_target,
        "n_synth_available": n_synth_available,
        "n_tts_selected": n_tts_selected,
        "n_speakers_with_tts": n_spk_with_tts,
        "copies_per_speaker_hist": copies_hist,
        "entries_pathological": entries_p,
        "entries_healthy": entries_h,
        "selected_synth": [
            {
                "speaker_id": str(train_speaker_ids[i]),
                "label": int(train_labels[i]),
                "output_filename": Path(train_files[i]).name,
                "path": train_files[i],
            }
            for i in keep_synth
        ],
    }
    manifest_path = Path(fold_dir) / "stage2_tts_sample.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"[stage2-bal11][fold {fold_idx}] wrote {manifest_path}")

    return {
        "train_files": out_files,
        "train_labels": out_labels,
        "train_rec_types": out_rec_types,
        "train_speaker_ids": out_speaker_ids,
        "train_is_synthetic": out_is_synth,
        "nP_real": nP,
        "nH_real": nH,
        "n_tts_target": n_tts_target,
        "n_synth_available": n_synth_available,
        "n_tts_selected": n_tts_selected,
        "n_speakers_with_tts": n_spk_with_tts,
        "copies_per_speaker_hist": copies_hist,
        "entries_pathological": entries_p,
        "entries_healthy": entries_h,
        "n_real": n_real,
    }


# ============================================================================
# 1.5x nested TTS sampling (healthy 6x + patho 4x). Does not touch bal11 1.0.
# ============================================================================

SCALE15_HEALTHY_EXTRA_RNG_OFFSET = 15001
SCALE15_PATHO_RNG_OFFSET = 15002

_FORBIDDEN_WRITE_PATHS = (
    "svd_cleaned_paper_tts",
    "svd_cleaned_paper_aligned",
    "svd_cleaned_paper_tts_healthy_sentence_6x",
    "svd_cleaned_paper_tts_patho_sentence_4x",
    "svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11",
)


def extra_quota_half_up(nP: int) -> tuple:
    """Q = half-up(0.20 * B), B=2P. Not Python bankers round()."""
    B = 2 * int(nP)
    Q = int(math.floor(0.20 * B + 0.5))
    return B, Q


def extra_quota_half_up_scale11(nP: int) -> tuple:
    """Read-only 1.1x Q, used only to assert 1.1 âŠ?1.2."""
    B = 2 * int(nP)
    Q = int(math.floor(0.10 * B + 0.5))
    return B, Q


def _assert_new_output_dir(output_dir: Path) -> None:
    resolved = Path(output_dir).resolve()
    text = str(resolved)
    forbidden_sub = (
        "tts_scale15",
        "tts_scale125",
        "tts_scale175",
        "tts_scale18",
        "tts_scale20",
        "tts_scale11",
        "analysis_scale11_spkdrop",
    )
    for token in forbidden_sub:
        if token in text:
            raise ValueError(
                f"refusing to write into existing experiment dir ({token}): {resolved}"
            )
    # Allow only this new 1.2x dir. "tts_scale12" is a prefix of "tts_scale125",
    # so require the 1.2 token without 125.
    if "tts_scale12" in text and "tts_scale125" not in text:
        return
    for token in _FORBIDDEN_WRITE_PATHS:
        if text.rstrip("/").endswith(token) or f"/{token}/" in text + "/":
            if token == "svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11" and "tts_scale" not in text:
                raise ValueError(
                    f"refusing to write into existing bal11 experiment dir: {resolved}"
                )
            if token.startswith("svd_cleaned"):
                raise ValueError(
                    f"refusing to write into existing data pool: {resolved}"
                )
    raise ValueError(
        f"refusing to write outside the new 1.2x experiment dir: {resolved}"
    )


def load_synth_pool(pool_dir: Path, expected_label: str) -> list:
    """Read synthetic_metadata.csv; do not modify the pool."""
    pool_dir = Path(pool_dir)
    meta = pool_dir / "synthetic_metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"missing {meta}")
    records = []
    with open(meta, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("recording_type", "sentence") != "sentence":
                continue
            if row.get("label") != expected_label:
                continue
            fname = row["output_filename"]
            path = pool_dir / "synthetic" / "sentences" / fname
            if not path.exists() or path.stat().st_size <= 0:
                continue
            records.append({
                "speaker_id": str(row["speaker_id"]),
                "label": row["label"],
                "output_filename": fname,
                "path": str(path.resolve()),
                "extra_idx": int(row.get("extra_idx") or 0),
                "speaker_wav_source": row.get("speaker_wav_source", ""),
                "pool_dir": str(pool_dir.resolve()),
            })
    return records


def speaker_balanced_nested_rank(records: list, rng: np.random.RandomState) -> list:
    """Deterministic speaker-balanced ordering of TTS records.

    Each speaker is one water-fill unit (more refs/wavs does not raise the
    per-cycle probability). Within a speaker, ``rng.shuffle`` sets a fixed
    order over reference / base / extra files. The returned list is a total
    ranking: prefix k is the nested k-sample for 1.5x / 2.0x.
    """
    by_spk = defaultdict(list)
    for rec in records:
        by_spk[str(rec["speaker_id"])].append(rec)
    speakers = sorted(by_spk.keys(), key=_speaker_sort_key)
    if not speakers:
        return []
    for sid in speakers:
        rng.shuffle(by_spk[sid])
    order = list(speakers)
    rng.shuffle(order)
    pointers = {sid: 0 for sid in speakers}
    ranking = []
    guard = 0
    max_iters = sum(len(v) for v in by_spk.values()) + len(speakers) + 8
    while order:
        next_order = []
        progressed = False
        for sid in order:
            i = pointers[sid]
            if i < len(by_spk[sid]):
                ranking.append(by_spk[sid][i])
                pointers[sid] = i + 1
                progressed = True
                if pointers[sid] < len(by_spk[sid]):
                    next_order.append(sid)
        order = next_order
        guard += 1
        if order and not progressed:
            raise RuntimeError("nested rank water-fill stalled")
        if guard > max_iters:
            raise RuntimeError(f"nested rank exceeded iteration cap ({max_iters})")
    return ranking


def _quota_hist(selected: list) -> dict:
    counts = Counter(str(r["speaker_id"]) for r in selected)
    return dict(Counter(counts.values()))


def _load_bal11_baseline_manifest(baseline_dir: Path, fold_idx: int) -> dict:
    path = Path(baseline_dir) / f"fold_{fold_idx}" / "stage2_tts_sample.json"
    if not path.exists():
        raise FileNotFoundError(f"missing bal11 baseline manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _recompute_bal11_baseline_filenames(
    fold_idx: int,
    train_files,
    train_labels,
    train_rec_types,
    train_speaker_ids,
    train_is_synthetic,
    test_speaker_ids,
    label2id,
    num_augmented: int,
    fold_seed: int,
    scratch_dir: Path,
) -> list:
    """Replay the original bal11 sampler on the 4x healthy pool (read-only)."""
    filtered = filter_stage2_train_disjoint_synthetic(
        fold_idx=fold_idx,
        train_files=train_files,
        train_labels=train_labels,
        train_rec_types=train_rec_types,
        train_speaker_ids=train_speaker_ids,
        train_is_synthetic=train_is_synthetic,
        test_speaker_ids=test_speaker_ids,
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    sampled = sample_stage2_synth_speaker_balanced(
        fold_idx=fold_idx,
        train_files=filtered["train_files"],
        train_labels=filtered["train_labels"],
        train_rec_types=filtered["train_rec_types"],
        train_speaker_ids=filtered["train_speaker_ids"],
        train_is_synthetic=filtered["train_is_synthetic"],
        label2id=label2id,
        num_augmented=num_augmented,
        fold_seed=fold_seed,
        fold_dir=scratch_dir,
    )
    with open(scratch_dir / "stage2_tts_sample.json", "r", encoding="utf-8") as f:
        recomputed = json.load(f)
    names = [x["output_filename"] for x in recomputed["selected_synth"]]
    if len(names) != sampled["n_tts_selected"]:
        raise RuntimeError(
            f"[fold {fold_idx}] recompute count mismatch "
            f"{len(names)} vs {sampled['n_tts_selected']}"
        )
    return names, sampled, filtered


def _resolve_baseline_to_6x(baseline_items: list, healthy_6x_by_name: dict) -> list:
    resolved = []
    missing = []
    for item in baseline_items:
        fname = item["output_filename"]
        rec = healthy_6x_by_name.get(fname)
        if rec is None:
            missing.append(fname)
            continue
        resolved.append({
            **rec,
            "role": "baseline_healthy",
            "label": "healthy",
        })
    if missing:
        raise RuntimeError(
            f"{len(missing)} bal11 baseline filenames missing from healthy 6x "
            f"pool, e.g. {missing[:5]}"
        )
    return resolved


def sample_stage2_synth_scale15_fold(
    fold_idx: int,
    nH: int,
    nP: int,
    train_speaker_ids,
    test_speaker_ids,
    num_augmented: int,
    fold_seed: int,
    tts_scale: float,
    healthy_6x_records: list,
    patho_4x_records: list,
    baseline_manifest: dict,
    recomputed_baseline_filenames: list,
    fold_dir: Path,
) -> dict:
    """Lock 1.0 healthy baseline; add P extra healthy + P patho TTS."""
    raise RuntimeError(
        "1.25x script must not call sample_stage2_synth_scale15_fold; "
        "use locked 1.5x ranking prefixes."
    )
    if abs(float(tts_scale) - 1.5) > 1e-9:
        raise ValueError(f"this sampler is 1.5x only, got tts_scale={tts_scale}")
    if int(num_augmented) != 1:
        raise ValueError(f"num_augmented must be 1, got {num_augmented}")

    multiplier = 1 + int(num_augmented)
    n_tts_h_baseline = multiplier * (nP - nH)
    n_extra = nP  # (1.5-1)*B, B=2P
    n_tts_h_total = n_tts_h_baseline + n_extra
    n_tts_p = n_extra
    entries_h = nH * multiplier + n_tts_h_total
    entries_p = nP * multiplier + n_tts_p
    if entries_h != 3 * nP or entries_p != 3 * nP or entries_h != entries_p:
        raise RuntimeError(
            f"[fold {fold_idx}] 1.5x entry formula failed: "
            f"H={nH} P={nP} entries_h={entries_h} entries_p={entries_p}"
        )

    train_set = set(map(str, train_speaker_ids))
    test_set = set(map(str, test_speaker_ids))
    if train_set & test_set:
        raise RuntimeError(f"[fold {fold_idx}] train/test speaker overlap")

    gold_names = [x["output_filename"] for x in baseline_manifest["selected_synth"]]
    if baseline_manifest.get("nH_real") != nH or baseline_manifest.get("nP_real") != nP:
        raise RuntimeError(
            f"[fold {fold_idx}] H/P mismatch vs bal11 manifest: "
            f"now H={nH} P={nP} vs "
            f"H={baseline_manifest.get('nH_real')} P={baseline_manifest.get('nP_real')}"
        )
    if list(gold_names) != list(recomputed_baseline_filenames):
        gold_set = set(gold_names)
        rec_set = set(recomputed_baseline_filenames)
        raise RuntimeError(
            f"[fold {fold_idx}] baseline filenames != bal11 stage2_tts_sample.json "
            f"(order_or_set mismatch; only_in_gold={len(gold_set-rec_set)} "
            f"only_in_recompute={len(rec_set-gold_set)})"
        )
    if len(gold_names) != n_tts_h_baseline or len(set(gold_names)) != len(gold_names):
        raise RuntimeError(
            f"[fold {fold_idx}] baseline count/dup error: "
            f"{len(gold_names)} unique={len(set(gold_names))} target={n_tts_h_baseline}"
        )

    healthy_6x_by_name = {r["output_filename"]: r for r in healthy_6x_records}
    baseline_selected = _resolve_baseline_to_6x(
        baseline_manifest["selected_synth"], healthy_6x_by_name,
    )
    baseline_name_set = set(gold_names)

    h_avail = [
        r for r in healthy_6x_records
        if r["speaker_id"] in train_set and r["speaker_id"] not in test_set
    ]
    p_avail = [
        r for r in patho_4x_records
        if r["speaker_id"] in train_set and r["speaker_id"] not in test_set
    ]
    h_extra_cand = [r for r in h_avail if r["output_filename"] not in baseline_name_set]
    if len(h_avail) < n_tts_h_total:
        raise RuntimeError(
            f"[fold {fold_idx}] healthy 6x pool short: avail={len(h_avail)} "
            f"need total {n_tts_h_total}"
        )
    if len(h_extra_cand) < n_extra:
        raise RuntimeError(
            f"[fold {fold_idx}] extra healthy pool short: cand={len(h_extra_cand)} "
            f"need {n_extra}"
        )
    if len(p_avail) < n_tts_p:
        raise RuntimeError(
            f"[fold {fold_idx}] patho 4x pool short: avail={len(p_avail)} "
            f"need {n_tts_p}"
        )

    rng_h = np.random.RandomState(int(fold_seed) + SCALE15_HEALTHY_EXTRA_RNG_OFFSET)
    rng_p = np.random.RandomState(int(fold_seed) + SCALE15_PATHO_RNG_OFFSET)
    extra_h_rank = speaker_balanced_nested_rank(h_extra_cand, rng_h)
    patho_rank = speaker_balanced_nested_rank(p_avail, rng_p)
    extra_h_sel = [{**r, "role": "extra_healthy_1.5"} for r in extra_h_rank[:n_extra]]
    patho_sel = [{**r, "role": "pathological_1.5"} for r in patho_rank[:n_tts_p]]

    extra_h_names = [r["output_filename"] for r in extra_h_sel]
    patho_names = [r["output_filename"] for r in patho_sel]
    all_sel = baseline_selected + extra_h_sel + patho_sel
    all_names = [r["output_filename"] for r in all_sel]
    all_paths = [r["path"] for r in all_sel]
    all_sids = [str(r["speaker_id"]) for r in all_sel]

    issues = []
    if set(extra_h_names) & baseline_name_set:
        issues.append("baseline/extra healthy filename overlap")
    if len(all_names) != len(set(all_names)):
        issues.append("duplicate TTS filename")
    if len(all_paths) != len(set(all_paths)):
        issues.append("duplicate TTS path")
    if any(sid in test_set for sid in all_sids):
        issues.append("test-speaker TTS in selection")
    if any(sid not in train_set for sid in all_sids):
        issues.append("TTS speaker not in current train speakers")
    if len(extra_h_sel) != n_extra:
        issues.append(f"extra healthy {len(extra_h_sel)} != P={n_extra}")
    if len(patho_sel) != n_tts_p:
        issues.append(f"patho TTS {len(patho_sel)} != P={n_tts_p}")
    if entries_h != entries_p:
        issues.append("final entries not equal")
    if issues:
        raise RuntimeError(f"[fold {fold_idx}] audit failed: {issues}")

    n_test_tts = sum(1 for sid in all_sids if sid in test_set)
    overlap_be = len(set(extra_h_names) & baseline_name_set)

    report = {
        "fold_idx": fold_idx,
        "fold_seed": int(fold_seed),
        "tts_scale": 1.5,
        "nH_real": nH,
        "nP_real": nP,
        "num_augmented": int(num_augmented),
        "n_tts_healthy_baseline": n_tts_h_baseline,
        "baseline_filenames_match_bal11": True,
        "n_tts_healthy_extra": n_extra,
        "n_tts_pathological": n_tts_p,
        "n_tts_healthy_total": n_tts_h_total,
        "entries_healthy": entries_h,
        "entries_pathological": entries_p,
        "entries_equal": entries_h == entries_p,
        "n_healthy_tts_pool_train": len(h_avail),
        "n_patho_tts_pool_train": len(p_avail),
        "n_healthy_extra_candidates": len(h_extra_cand),
        "all_tts_speakers_in_train": True,
        "n_test_speaker_tts": n_test_tts,
        "n_duplicate_filename": 0,
        "n_duplicate_path": 0,
        "n_baseline_extra_healthy_overlap": overlap_be,
        "healthy_baseline_quota_hist": dict(
            baseline_manifest.get("copies_per_speaker_hist") or {}
        ),
        "healthy_extra_quota_hist": _quota_hist(extra_h_sel),
        "healthy_total_quota_hist": _quota_hist(baseline_selected + extra_h_sel),
        "pathological_quota_hist": _quota_hist(patho_sel),
        "n_healthy_extra_speakers": len({r["speaker_id"] for r in extra_h_sel}),
        "n_patho_tts_speakers": len({r["speaker_id"] for r in patho_sel}),
        "rng_offsets": {
            "healthy_extra": SCALE15_HEALTHY_EXTRA_RNG_OFFSET,
            "pathological": SCALE15_PATHO_RNG_OFFSET,
        },
        "selected_synth": [
            {
                "speaker_id": r["speaker_id"],
                "label": r["label"],
                "output_filename": r["output_filename"],
                "path": r["path"],
                "extra_idx": r.get("extra_idx"),
                "speaker_wav_source": r.get("speaker_wav_source"),
                "role": r.get("role"),
            }
            for r in all_sel
        ],
    }
    ranking_doc = {
        "fold_idx": fold_idx,
        "fold_seed": int(fold_seed),
        "note": "Nested ranking prefixes: 1.5x uses [:P]; 2.0x must keep this "
                "order and use [:2P]. Baseline healthy is locked separately.",
        "n_extra_1_5": n_extra,
        "n_extra_2_0": 2 * nP,
        "extra_healthy_ranking": [
            {
                "speaker_id": r["speaker_id"],
                "output_filename": r["output_filename"],
                "path": r["path"],
                "extra_idx": r.get("extra_idx"),
            }
            for r in extra_h_rank
        ],
        "pathological_ranking": [
            {
                "speaker_id": r["speaker_id"],
                "output_filename": r["output_filename"],
                "path": r["path"],
                "extra_idx": r.get("extra_idx"),
                "speaker_wav_source": r.get("speaker_wav_source"),
            }
            for r in patho_rank
        ],
    }

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    man_path = fold_dir / "stage2_tts_sample_scale15.json"
    rank_path = fold_dir / "tts_nested_ranking.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(rank_path, "w", encoding="utf-8") as f:
        json.dump(ranking_doc, f, indent=2)
    logger.info(
        f"[scale15][fold {fold_idx}] H={nH} P={nP}  "
        f"baseH={n_tts_h_baseline} extraH={n_extra} patho={n_tts_p}  "
        f"entries {entries_h}:{entries_p}  wrote {man_path.name}"
    )
    return report


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_scale12_fold_manifest(
    fold_idx: int,
    nH: int,
    nP: int,
    train_speaker_ids,
    test_speaker_ids,
    num_augmented: int,
    scale15_fold_dir: Path,
    scale125_fold_dir: Path,
    scale11_fold_dir: Path,
    bal11_baseline_dir: Path,
    out_fold_dir: Path,
) -> dict:
    """Lock bal11 baseline; take 1.5x ranking prefixes of length Q. No shuffle."""
    if int(num_augmented) != 1:
        raise ValueError("1.2x requires num_augmented=1")
    B, Q = extra_quota_half_up(nP)
    _B11, Q11 = extra_quota_half_up_scale11(nP)
    n_tts_h_baseline = 2 * (nP - nH)
    n_tts_h_total = n_tts_h_baseline + Q
    entries = 2 * nP + Q
    if entries != nH * 2 + n_tts_h_total:
        raise RuntimeError(f"[fold {fold_idx}] 1.2x entry formula failed")
    if Q11 > Q:
        raise RuntimeError(f"[fold {fold_idx}] 1.1x Q={Q11} > 1.2x Q={Q}")

    train_set = set(map(str, train_speaker_ids))
    test_set = set(map(str, test_speaker_ids))
    gold = _load_bal11_baseline_manifest(bal11_baseline_dir, fold_idx)
    gold_names = [x["output_filename"] for x in gold["selected_synth"]]
    if gold.get("nH_real") != nH or gold.get("nP_real") != nP:
        raise RuntimeError(
            f"[fold {fold_idx}] H/P vs bal11 manifest: {nH}/{nP} vs "
            f"{gold.get('nH_real')}/{gold.get('nP_real')}"
        )
    if len(gold_names) != n_tts_h_baseline or len(set(gold_names)) != len(gold_names):
        raise RuntimeError(f"[fold {fold_idx}] bal11 baseline count/dup error")

    man15_path = Path(scale15_fold_dir) / "stage2_tts_sample_scale15.json"
    rank_path = Path(scale15_fold_dir) / "tts_nested_ranking.json"
    man15 = _load_json(man15_path)
    ranking = _load_json(rank_path)
    extra15 = [x for x in man15["selected_synth"] if x.get("role") == "extra_healthy_1.5"]
    patho15 = [x for x in man15["selected_synth"] if x.get("role") == "pathological_1.5"]
    base15 = [x for x in man15["selected_synth"] if x.get("role") == "baseline_healthy"]
    extra15_names = [x["output_filename"] for x in extra15]
    patho15_names = [x["output_filename"] for x in patho15]
    base15_names = [x["output_filename"] for x in base15]
    rank_h_names = [x["output_filename"] for x in ranking["extra_healthy_ranking"]]
    rank_p_names = [x["output_filename"] for x in ranking["pathological_ranking"]]

    issues = []
    if list(base15_names) != list(gold_names):
        issues.append("1.5x baseline != bal11 baseline")
    if extra15_names != rank_h_names[:nP]:
        issues.append("1.5x extra healthy != ranking[:P]")
    if patho15_names != rank_p_names[:nP]:
        issues.append("1.5x patho != ranking[:P]")
    if Q > len(rank_h_names) or Q > len(rank_p_names):
        issues.append(f"ranking shorter than Q={Q}")
    if issues:
        raise RuntimeError(f"[fold {fold_idx}] nested prefix precheck failed: {issues}")

    extra_h = [
        {**x, "label": "healthy", "role": "extra_healthy_1.2"}
        for x in ranking["extra_healthy_ranking"][:Q]
    ]
    patho = [
        {**x, "label": "pathological", "role": "pathological_1.2"}
        for x in ranking["pathological_ranking"][:Q]
    ]
    extra_h_names = [x["output_filename"] for x in extra_h]
    patho_names = [x["output_filename"] for x in patho]
    if extra_h_names != extra15_names[:Q] or extra_h_names != rank_h_names[:Q]:
        raise RuntimeError(f"[fold {fold_idx}] extra healthy is not 1.5x ranking[:Q]")
    if patho_names != patho15_names[:Q] or patho_names != rank_p_names[:Q]:
        raise RuntimeError(f"[fold {fold_idx}] patho is not 1.5x ranking[:Q]")

    man125 = _load_json(Path(scale125_fold_dir) / "stage2_tts_sample_scale125.json")
    extra125_names = [
        x["output_filename"] for x in man125["selected_synth"]
        if x.get("role") == "extra_healthy_1.25"
    ]
    patho125_names = [
        x["output_filename"] for x in man125["selected_synth"]
        if x.get("role") == "pathological_1.25"
    ]
    if extra_h_names != extra125_names[:Q]:
        raise RuntimeError(
            f"[fold {fold_idx}] extra healthy is not a strict prefix of 1.25x extra"
        )
    if patho_names != patho125_names[:Q]:
        raise RuntimeError(
            f"[fold {fold_idx}] patho is not a strict prefix of 1.25x patho"
        )

    man11 = _load_json(Path(scale11_fold_dir) / "stage2_tts_sample_scale11.json")
    extra11_names = [
        x["output_filename"] for x in man11["selected_synth"]
        if x.get("role") == "extra_healthy_1.1"
    ]
    patho11_names = [
        x["output_filename"] for x in man11["selected_synth"]
        if x.get("role") == "pathological_1.1"
    ]
    if extra11_names != extra_h_names[:Q11]:
        raise RuntimeError(
            f"[fold {fold_idx}] 1.1x extra healthy is not a strict prefix of 1.2x extra"
        )
    if patho11_names != patho_names[:Q11]:
        raise RuntimeError(
            f"[fold {fold_idx}] 1.1x patho is not a strict prefix of 1.2x patho"
        )

    baseline_selected = []
    for item in gold["selected_synth"]:
        rec15 = next((x for x in base15 if x["output_filename"] == item["output_filename"]), None)
        if rec15 is None:
            raise RuntimeError(
                f"[fold {fold_idx}] bal11 baseline missing in 1.5x: "
                f"{item['output_filename']}"
            )
        baseline_selected.append({
            "speaker_id": str(rec15["speaker_id"]),
            "label": "healthy",
            "output_filename": rec15["output_filename"],
            "path": rec15["path"],
            "extra_idx": rec15.get("extra_idx"),
            "speaker_wav_source": rec15.get("speaker_wav_source"),
            "role": "baseline_healthy",
        })

    selected = baseline_selected + extra_h + patho
    all_names = [r["output_filename"] for r in selected]
    all_paths = [r["path"] for r in selected]
    all_sids = [str(r["speaker_id"]) for r in selected]
    overlap_be = len(set(gold_names) & set(extra_h_names))
    n_test = sum(1 for sid in all_sids if sid in test_set)
    n_not_train = sum(1 for sid in all_sids if sid not in train_set)
    post = []
    if overlap_be:
        post.append(f"baseline/extra overlap={overlap_be}")
    if len(all_names) != len(set(all_names)):
        post.append("duplicate filename")
    if len(all_paths) != len(set(all_paths)):
        post.append("duplicate path")
    if n_test:
        post.append(f"test-speaker TTS={n_test}")
    if n_not_train:
        post.append(f"TTS speaker not in train={n_not_train}")
    if len(extra_h) != Q or len(patho) != Q:
        post.append("Q count mismatch")
    if post:
        raise RuntimeError(f"[fold {fold_idx}] 1.2x safety failed: {post}")

    report = {
        "fold_idx": fold_idx,
        "tts_scale": 1.2,
        "nH_real": nH,
        "nP_real": nP,
        "B": B,
        "Q": Q,
        "num_augmented": 1,
        "n_tts_healthy_baseline": n_tts_h_baseline,
        "baseline_filenames_match_bal11": True,
        "n_tts_healthy_extra": Q,
        "n_tts_pathological": Q,
        "n_tts_healthy_total": n_tts_h_total,
        "entries_healthy": entries,
        "entries_pathological": entries,
        "entries_equal": True,
        "extra_healthy_is_15_prefix": True,
        "patho_is_15_prefix": True,
        "extra_healthy_is_125_prefix": True,
        "patho_is_125_prefix": True,
        "extra_healthy_is_11_prefix": True,
        "patho_is_11_prefix": True,
        "Q11": Q11,
        "all_tts_speakers_in_train": True,
        "n_test_speaker_tts": 0,
        "n_duplicate_filename": 0,
        "n_duplicate_path": 0,
        "n_baseline_extra_healthy_overlap": 0,
        "used_old_bal11_sampler": False,
        "used_nested_reshuffle": False,
        "healthy_extra_quota_hist": _quota_hist(extra_h),
        "pathological_quota_hist": _quota_hist(patho),
        "n_healthy_extra_speakers": len({r["speaker_id"] for r in extra_h}),
        "n_patho_tts_speakers": len({r["speaker_id"] for r in patho}),
        "scale15_ranking_path": str(rank_path.resolve()),
        "selected_synth": [
            {
                "speaker_id": r["speaker_id"],
                "label": r["label"],
                "output_filename": r["output_filename"],
                "path": r["path"],
                "extra_idx": r.get("extra_idx"),
                "speaker_wav_source": r.get("speaker_wav_source"),
                "role": r.get("role"),
            }
            for r in selected
        ],
    }
    out_fold_dir = Path(out_fold_dir)
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_fold_dir / "stage2_tts_sample_scale12.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(
        f"[scale12][fold {fold_idx}] H={nH} P={nP} B={B} Q={Q} Q11={Q11} "
        f"baseH={n_tts_h_baseline} extraH={Q} patho={Q} "
        f"entries {entries}:{entries}  wrote {man_path.name}"
    )
    return report


def run_tts_scale12_audit(args) -> dict:
    """10-fold 1.2x prefix audit + Stage-2 construction dry-run. No training."""
    output_dir = Path(args.output_dir)
    _assert_new_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    scale15_root = Path(args.scale15_source_dir)
    scale125_root = Path(args.scale125_source_dir)
    scale11_root = Path(args.scale11_source_dir)
    baseline_dir = Path(args.bal11_baseline_dir)
    if not scale15_root.exists():
        raise FileNotFoundError(scale15_root)
    if not scale125_root.exists():
        raise FileNotFoundError(scale125_root)
    if not scale11_root.exists():
        raise FileNotFoundError(scale11_root)

    logger.info("=" * 72)
    logger.info("TTS SCALE 1.2 AUDIT + STAGE-2 DRY-RUN â€?no Whisper / S1 / train")
    logger.info("  data_dir             = %s", args.data_dir)
    logger.info("  scale15_source_dir   = %s (read-only rankings)", scale15_root)
    logger.info("  scale125_source_dir  = %s (read-only 1.25 prefix)", scale125_root)
    logger.info("  scale11_source_dir   = %s (read-only 1.1 prefix)", scale11_root)
    logger.info("  bal11_baseline_dir   = %s", baseline_dir)
    logger.info("  output_dir           = %s", output_dir)
    logger.info("=" * 72)

    (file_paths, labels, speaker_ids, rec_types,
     is_synthetic, label2id, id2label) = load_metadata_with_tts(
        args.data_dir, task=args.task, recording_type=args.recording_type,
        max_synth_ratio=None,
    )
    folds = create_cv_folds_with_tts(
        speaker_ids, labels, is_synthetic,
        n_folds=args.n_folds, seed=args.seed,
    )
    healthy_id = label2id["healthy"]
    fold_reports = []
    for fold_idx in range(len(folds)):
        train_idx, test_idx = folds[fold_idx]
        train_files = [file_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_rec_types = [rec_types[i] for i in train_idx]
        train_speaker_ids = [speaker_ids[i] for i in train_idx]
        train_is_synth = [is_synthetic[i] for i in train_idx]
        test_speaker_ids = [speaker_ids[i] for i in test_idx]
        nH = sum(1 for s, lab in zip(train_is_synth, train_labels)
                 if (not s) and lab == healthy_id)
        nP = sum(1 for s, lab in zip(train_is_synth, train_labels)
                 if (not s) and lab != healthy_id)
        real_train_sids = [
            sid for sid, synth in zip(train_speaker_ids, train_is_synth) if not synth
        ]
        fold_dir = output_dir / f"fold_{fold_idx}"
        report = build_scale12_fold_manifest(
            fold_idx=fold_idx,
            nH=nH,
            nP=nP,
            train_speaker_ids=real_train_sids,
            test_speaker_ids=test_speaker_ids,
            num_augmented=args.num_augmented,
            scale15_fold_dir=scale15_root / f"fold_{fold_idx}",
            scale125_fold_dir=scale125_root / f"fold_{fold_idx}",
            scale11_fold_dir=scale11_root / f"fold_{fold_idx}",
            bal11_baseline_dir=baseline_dir,
            out_fold_dir=fold_dir,
        )
        constructed = build_stage2_from_scale12_manifest(
            fold_idx=fold_idx,
            train_files=train_files,
            train_labels=train_labels,
            train_rec_types=train_rec_types,
            train_speaker_ids=train_speaker_ids,
            train_is_synthetic=train_is_synth,
            test_speaker_ids=test_speaker_ids,
            label2id=label2id,
            num_augmented=args.num_augmented,
            manifest_path=fold_dir / "stage2_tts_sample_scale12.json",
            bal11_baseline_dir=baseline_dir,
        )
        report["stage2_dryrun"] = constructed["report"]
        if constructed["report"]["entries_healthy"] != report["entries_healthy"]:
            raise RuntimeError(f"[fold {fold_idx}] dry-run entries != audit formula")
        fold_reports.append(report)
        logger.info(
            f"[audit][fold {fold_idx}] H={nH} P={nP} Q={report['Q']} "
            f"Htts={report['n_tts_healthy_total']} Ptts={report['n_tts_pathological']} "
            f"entries {report['entries_healthy']}/{report['entries_pathological']} "
            f"testTTS=0 overlap=0 prefix=OK"
        )

    summary = {
        "script": "train_with_tts_v10_tapt_ptapt_whisper_layersweep_orig_aug_disjoint_synth_bal11_tts_scale12.py",
        "tts_scale": 1.2,
        "audit_only": True,
        "n_folds": len(fold_reports),
        "all_folds_ok": True,
        "used_old_bal11_sampler": False,
        "folds": [
            {k: v for k, v in r.items() if k != "selected_synth"}
            for r in fold_reports
        ],
    }
    out_path = output_dir / "tts_scale12_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("=" * 72)
    logger.info("AUDIT DONE â€?10 folds OK, wrote %s", out_path)
    logger.info("No training started.")
    logger.info("=" * 72)
    return summary


def _expand_orig_aug_entry_counts(labels, is_synthetic, num_augmented, healthy_id):
    """Mirror VoiceDisorderDatasetTTSOrigAugOnly expansion (no audio / no FE)."""
    n_real_h = n_real_p = 0
    n_aug_h = n_aug_p = 0
    n_synth_h = n_synth_p = 0
    for lab, synth in zip(labels, is_synthetic):
        is_h = int(lab) == int(healthy_id)
        if synth:
            if is_h:
                n_synth_h += 1
            else:
                n_synth_p += 1
        else:
            if is_h:
                n_real_h += 1
                n_aug_h += int(num_augmented)
            else:
                n_real_p += 1
                n_aug_p += int(num_augmented)
    return {
        "n_real_healthy": n_real_h,
        "n_real_pathological": n_real_p,
        "n_real_aug_healthy": n_aug_h,
        "n_real_aug_pathological": n_aug_p,
        "n_synth_healthy": n_synth_h,
        "n_synth_pathological": n_synth_p,
        "entries_healthy": n_real_h + n_aug_h + n_synth_h,
        "entries_pathological": n_real_p + n_aug_p + n_synth_p,
    }


def build_stage2_from_scale12_manifest(
    fold_idx: int,
    train_files,
    train_labels,
    train_rec_types,
    train_speaker_ids,
    train_is_synthetic,
    test_speaker_ids,
    label2id,
    num_augmented: int,
    manifest_path: Path,
    bal11_baseline_dir: Path,
):
    """Stage-2 lists = current-fold real train rows + locked 1.2x manifest TTS.

    Never calls ``sample_stage2_synth_speaker_balanced``. Does not read
    ``svd_cleaned_paper_tts`` synthetic rows into Stage 2.
    """
    if "healthy" not in label2id or "pathological" not in label2id:
        raise ValueError(f"expected healthy/pathological in label2id, got {label2id}")
    healthy_id = label2id["healthy"]
    patho_id = label2id["pathological"]
    label_to_id = {"healthy": healthy_id, "pathological": patho_id}

    real_files, real_labels, real_rec, real_sids, real_synth = [], [], [], [], []
    n_dropped_data_dir_synth = 0
    for fpath, lab, rt, sid, synth in zip(
        train_files, train_labels, train_rec_types,
        train_speaker_ids, train_is_synthetic,
    ):
        if synth:
            n_dropped_data_dir_synth += 1
            continue
        real_files.append(fpath)
        real_labels.append(lab)
        real_rec.append(rt)
        real_sids.append(sid)
        real_synth.append(False)

    nH = sum(1 for lab in real_labels if lab == healthy_id)
    nP = sum(1 for lab in real_labels if lab != healthy_id)
    train_set = set(map(str, real_sids))
    test_set = set(map(str, test_speaker_ids))

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"[fold {fold_idx}] missing 1.2x manifest {manifest_path}. "
            "Run --audit_tts_scale_only first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    selected = list(man.get("selected_synth") or [])
    if not selected:
        raise RuntimeError(f"[fold {fold_idx}] empty selected_synth in {manifest_path}")

    if man.get("nH_real") != nH or man.get("nP_real") != nP:
        raise RuntimeError(
            f"[fold {fold_idx}] real H/P {nH}/{nP} != manifest "
            f"{man.get('nH_real')}/{man.get('nP_real')}"
        )

    gold = _load_bal11_baseline_manifest(bal11_baseline_dir, fold_idx)
    gold_names = [x["output_filename"] for x in gold["selected_synth"]]
    base_items = [x for x in selected if x.get("role") == "baseline_healthy"]
    extra_items = [x for x in selected if x.get("role") == "extra_healthy_1.2"]
    patho_items = [x for x in selected if x.get("role") == "pathological_1.2"]
    unknown_roles = [
        x.get("role") for x in selected
        if x.get("role") not in (
            "baseline_healthy", "extra_healthy_1.2", "pathological_1.2",
        )
    ]
    if unknown_roles:
        raise RuntimeError(f"[fold {fold_idx}] unknown TTS roles: {set(unknown_roles)}")

    base_names = [x["output_filename"] for x in base_items]
    extra_names = [x["output_filename"] for x in extra_items]
    patho_names = [x["output_filename"] for x in patho_items]
    if list(base_names) != list(gold_names):
        raise RuntimeError(
            f"[fold {fold_idx}] manifest baseline healthy filenames != "
            "bal11 stage2_tts_sample.json"
        )

    n_tts_h_baseline = 2 * (nP - nH)
    _B, Q = extra_quota_half_up(nP)
    n_tts_h_extra = Q
    n_tts_p = Q
    n_tts_h_total = n_tts_h_baseline + Q
    n_entries = 2 * nP + Q
    if len(selected) != man.get("n_tts_healthy_total", -1) + man.get("n_tts_pathological", -1):
        raise RuntimeError(
            f"[fold {fold_idx}] selected TTS {len(selected)} != "
            f"audit healthy_total+patho "
            f"{man.get('n_tts_healthy_total')}+{man.get('n_tts_pathological')}"
        )
    if len(base_names) != n_tts_h_baseline or len(extra_names) != n_tts_h_extra:
        raise RuntimeError(
            f"[fold {fold_idx}] baseline/extra counts "
            f"{len(base_names)}/{len(extra_names)} != {n_tts_h_baseline}/{n_tts_h_extra}"
        )
    if len(patho_names) != n_tts_p:
        raise RuntimeError(
            f"[fold {fold_idx}] patho TTS {len(patho_names)} != P={n_tts_p}"
        )
    if len(base_names) + len(extra_names) != n_tts_h_total:
        raise RuntimeError(f"[fold {fold_idx}] healthy TTS total mismatch")

    synth_files, synth_labels, synth_rec, synth_sids, synth_flags = (
        [], [], [], [], [],
    )
    missing = []
    for rec in selected:
        fpath = Path(rec["path"])
        if not fpath.exists() or fpath.stat().st_size <= 0:
            missing.append(rec["output_filename"])
            continue
        lab_name = rec.get("label")
        if lab_name not in label_to_id:
            raise RuntimeError(
                f"[fold {fold_idx}] TTS label {lab_name!r} not in label2id"
            )
        synth_files.append(str(fpath))
        synth_labels.append(label_to_id[lab_name])
        synth_rec.append("sentence")
        synth_sids.append(str(rec["speaker_id"]))
        synth_flags.append(True)
    if missing:
        raise RuntimeError(
            f"[fold {fold_idx}] {len(missing)} manifest TTS wavs missing, "
            f"e.g. {missing[:5]}"
        )

    out_files = real_files + synth_files
    out_labels = real_labels + synth_labels
    out_rec = real_rec + synth_rec
    out_sids = real_sids + synth_sids
    out_is_synth = real_synth + synth_flags

    fed_names = [Path(p).name for p in synth_files]
    man_names = [x["output_filename"] for x in selected]
    overlap_be = len(set(base_names) & set(extra_names))
    n_dup_name = len(fed_names) - len(set(fed_names))
    n_dup_path = len(synth_files) - len(set(synth_files))
    n_test_overlap = sum(1 for sid in synth_sids if sid in test_set)
    n_not_train = sum(1 for sid in synth_sids if sid not in train_set)
    counts = _expand_orig_aug_entry_counts(
        out_labels, out_is_synth, num_augmented, healthy_id,
    )

    issues = []
    if list(fed_names) != list(man_names):
        issues.append("dataset TTS filenames != manifest selected_synth")
    if overlap_be:
        issues.append(f"baseline/extra healthy overlap={overlap_be}")
    if n_dup_name:
        issues.append(f"duplicate filename={n_dup_name}")
    if n_dup_path:
        issues.append(f"duplicate path={n_dup_path}")
    if n_test_overlap:
        issues.append(f"test-speaker TTS={n_test_overlap}")
    if n_not_train:
        issues.append(f"TTS speaker not in train={n_not_train}")
    if counts["n_synth_healthy"] != n_tts_h_total:
        issues.append(
            f"healthy TTS {counts['n_synth_healthy']} != 2(P-H)+Q={n_tts_h_total}"
        )
    if counts["n_synth_pathological"] != n_tts_p:
        issues.append(
            f"patho TTS {counts['n_synth_pathological']} != Q={n_tts_p}"
        )
    if counts["entries_healthy"] != n_entries or counts["entries_pathological"] != n_entries:
        issues.append(
            f"expanded entries H/P={counts['entries_healthy']}/"
            f"{counts['entries_pathological']} != 2P+Q={n_entries}"
        )
    if counts["n_real_healthy"] != nH or counts["n_real_pathological"] != nP:
        issues.append("real file counts drifted")
    if counts["n_real_aug_healthy"] != nH * int(num_augmented):
        issues.append("real healthy aug count mismatch")
    if counts["n_real_aug_pathological"] != nP * int(num_augmented):
        issues.append("real patho aug count mismatch")
    if issues:
        raise RuntimeError(f"[fold {fold_idx}] Stage-2 assert failed: {issues}")

    report = {
        "fold_idx": fold_idx,
        "nH_real": nH,
        "nP_real": nP,
        "B": _B,
        "Q": Q,
        "n_dropped_data_dir_synth": n_dropped_data_dir_synth,
        "n_real_healthy": counts["n_real_healthy"],
        "n_real_pathological": counts["n_real_pathological"],
        "n_real_aug_healthy": counts["n_real_aug_healthy"],
        "n_real_aug_pathological": counts["n_real_aug_pathological"],
        "n_tts_healthy_baseline": len(base_names),
        "n_tts_healthy_extra": len(extra_names),
        "n_tts_pathological": len(patho_names),
        "n_tts_healthy_total": n_tts_h_total,
        "entries_healthy": counts["entries_healthy"],
        "entries_pathological": counts["entries_pathological"],
        "entries_equal": counts["entries_healthy"] == counts["entries_pathological"],
        "baseline_filenames_match_bal11": True,
        "manifest_vs_dataset_tts_identical": True,
        "n_test_speaker_tts": n_test_overlap,
        "n_duplicate_filename": n_dup_name,
        "n_duplicate_path": n_dup_path,
        "n_baseline_extra_healthy_overlap": overlap_be,
        "n_selected_synth_manifest": len(selected),
        "n_synth_fed_to_dataset": len(synth_files),
        "used_old_bal11_sampler": False,
        "manifest_path": str(manifest_path.resolve()),
    }
    logger.info("=" * 72)
    logger.info(f"[scale12-stage2][fold {fold_idx}] real H/P={nH}/{nP} "
                f"(dropped data_dir synth={n_dropped_data_dir_synth})")
    logger.info(f"[scale12-stage2][fold {fold_idx}] TTS baselineH={len(base_names)} "
                f"extraH={len(extra_names)} patho={len(patho_names)} Q={Q}")
    logger.info(f"[scale12-stage2][fold {fold_idx}] expanded entries "
                f"H={counts['entries_healthy']} P={counts['entries_pathological']}")
    logger.info("=" * 72)
    return {
        "train_files": out_files,
        "train_labels": out_labels,
        "train_rec_types": out_rec,
        "train_speaker_ids": out_sids,
        "train_is_synthetic": out_is_synth,
        "report": report,
    }


def run_stage2_data_dryrun(args) -> dict:
    """Construct Stage-2 lists for one fold. No Whisper, no Stage 1, no training."""
    output_dir = Path(args.output_dir)
    _assert_new_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_idx = 0 if args.only_fold is None else int(args.only_fold)
    logger.info("=" * 72)
    logger.info("STAGE-2 DATA DRY-RUN â€?no Whisper / Stage-1 / training")
    logger.info("  fold                 = %s", fold_idx)
    logger.info("  data_dir (real only for S2) = %s", args.data_dir)
    logger.info("  output_dir           = %s", output_dir)
    logger.info("=" * 72)

    (file_paths, labels, speaker_ids, rec_types,
     is_synthetic, label2id, id2label) = load_metadata_with_tts(
        args.data_dir, task=args.task, recording_type=args.recording_type,
        max_synth_ratio=None,
    )
    folds = create_cv_folds_with_tts(
        speaker_ids, labels, is_synthetic,
        n_folds=args.n_folds, seed=args.seed,
    )
    if fold_idx < 0 or fold_idx >= len(folds):
        raise ValueError(f"fold {fold_idx} out of range")

    train_idx, test_idx = folds[fold_idx]
    train_files = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_rec_types = [rec_types[i] for i in train_idx]
    train_speaker_ids = [speaker_ids[i] for i in train_idx]
    train_is_synth = [is_synthetic[i] for i in train_idx]
    test_speaker_ids = [speaker_ids[i] for i in test_idx]

    manifest_root = Path(getattr(args, "scale12_manifest_dir", None) or args.output_dir)
    constructed = build_stage2_from_scale12_manifest(
        fold_idx=fold_idx,
        train_files=train_files,
        train_labels=train_labels,
        train_rec_types=train_rec_types,
        train_speaker_ids=train_speaker_ids,
        train_is_synthetic=train_is_synth,
        test_speaker_ids=test_speaker_ids,
        label2id=label2id,
        num_augmented=args.num_augmented,
        manifest_path=manifest_root / f"fold_{fold_idx}" / "stage2_tts_sample_scale12.json",
        bal11_baseline_dir=Path(args.bal11_baseline_dir),
    )
    report = constructed["report"]
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    out_path = fold_dir / "stage2_data_dryrun.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("[dry-run] real healthy              = %d", report["n_real_healthy"])
    logger.info("[dry-run] real pathological         = %d", report["n_real_pathological"])
    logger.info("[dry-run] real aug healthy          = %d", report["n_real_aug_healthy"])
    logger.info("[dry-run] real aug pathological     = %d", report["n_real_aug_pathological"])
    logger.info("[dry-run] baseline healthy TTS      = %d", report["n_tts_healthy_baseline"])
    logger.info("[dry-run] extra healthy TTS         = %d", report["n_tts_healthy_extra"])
    logger.info("[dry-run] pathological TTS          = %d", report["n_tts_pathological"])
    logger.info("[dry-run] final healthy entries     = %d", report["entries_healthy"])
    logger.info("[dry-run] final pathological entries= %d", report["entries_pathological"])
    logger.info("[dry-run] test-speaker overlap      = %d", report["n_test_speaker_tts"])
    logger.info("[dry-run] duplicate filename        = %d", report["n_duplicate_filename"])
    logger.info("[dry-run] duplicate path            = %d", report["n_duplicate_path"])
    logger.info("[dry-run] manifest vs dataset TTS   = %s",
                "100% identical" if report["manifest_vs_dataset_tts_identical"] else "MISMATCH")
    logger.info("[dry-run] wrote %s", out_path)
    logger.info("No training started.")
    return report


def run_adaptive_pretraining(
    fold_idx: int,
    method: str,
    pool_info: dict,
    model_name: str,
    feature_extractor,
    feature_layer: int,
    fold_dir: Path,
    epochs: int,
    lr: float,
    mask_time_prob: float,
    mask_time_length: int,
    batch_size: int,
    max_length_sec: float,
    fold_seed: int,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> tuple[Path, dict]:
    """Run masked acoustic feature reconstruction for `epochs` epochs.
    Saves the adapted encoder state_dict to ``fold_dir/adaptive_encoder.pt``
    and a JSON log to ``fold_dir/adaptive_pretrain_log.json``.

    Returns (adaptive_ckpt_path, stats_dict).
    """
    set_deterministic(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adaptive_ckpt = fold_dir / "adaptive_encoder.pt"
    log_path = fold_dir / "adaptive_pretrain_log.json"

    # Allow resuming: if a completed adaptive checkpoint already exists, reuse.
    if adaptive_ckpt.exists() and log_path.exists():
        try:
            with open(log_path) as f:
                cached_stats = json.load(f)
            logger.info(
                f"[adaptive][fold {fold_idx}] cached checkpoint found "
                f"at {adaptive_ckpt}, skipping stage 1."
            )
            return adaptive_ckpt, cached_stats
        except Exception:
            logger.warning(
                f"[adaptive][fold {fold_idx}] cached log unreadable, "
                "re-running stage 1."
            )

    dataset = WhisperAdaptiveAudioDataset(
        file_paths=pool_info["files"],
        feature_extractor=feature_extractor,
        max_length_sec=max_length_sec,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = WhisperReconstructionModel(
        model_name=model_name,
        feature_layer=feature_layer,
        num_mel_bins=feature_extractor.feature_size,
    )
    trainable_n, total_n = model.configure_stage1_grads()
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("=" * 64)
    logger.info(f"[adaptive][fold {fold_idx}] method={method}  "
                f"scope={pool_info['scope']}")
    logger.info(f"[adaptive][fold {fold_idx}] #samples={len(dataset)}  "
                f"batch_size={batch_size}  epochs={epochs}  lr={lr}")
    logger.info(f"[adaptive][fold {fold_idx}] mask_time_prob={mask_time_prob}  "
                f"mask_time_length={mask_time_length}")
    logger.info(f"[adaptive][fold {fold_idx}] trainable params: "
                f"{trainable_n/1e6:.4f}M / total {total_n/1e6:.3f}M "
                f"({100*trainable_n/max(1,total_n):.4f}%)")
    logger.info(f"[adaptive][fold {fold_idx}] reconstruction head: "
                f"Linear({model.d_model} -> {model.num_mel_bins})")
    logger.info(f"[adaptive][fold {fold_idx}] feature_layer for reconstruction = "
                f"{feature_layer} (hidden_states[{feature_layer + 1}])")
    logger.info("=" * 64)

    if len(dataset) == 0:
        raise RuntimeError(
            f"[adaptive][fold {fold_idx}] empty pool after filtering "
            f"(scope={pool_info['scope']})"
        )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay
    )
    total_steps = max(1, len(loader) * epochs)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    epoch_losses = []
    epoch_masked_ratios = []
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0
        sum_masked = 0
        sum_total_mask_pos = 0
        for batch in loader:
            input_features = batch["input_features"].to(device, non_blocking=True)
            B, _, T_in = input_features.shape
            if T_in % 2 != 0:
                # Should never happen for Whisper (T_in=3000); guard anyway.
                pad = 2 - (T_in % 2)
                input_features = F.pad(input_features, (0, pad))
                T_in = input_features.shape[-1]
            T_enc = T_in // 2

            encoder_mask = compute_encoder_time_mask(
                B, T_enc, mask_time_prob, mask_time_length, device,
            )
            input_mask = encoder_mask.repeat_interleave(2, dim=1)  # [B, T_in]

            target = F.avg_pool1d(input_features, kernel_size=2, stride=2)
            target = target.transpose(1, 2)  # [B, T_enc, num_mel_bins]

            masked_features = input_features.masked_fill(
                input_mask.unsqueeze(1), 0.0
            )

            pred = model(masked_features)  # [B, T_enc, num_mel_bins]
            sq_err = (pred - target) ** 2                         # [B, T_enc, M]
            per_frame_mse = sq_err.mean(dim=-1)                   # [B, T_enc]
            mask_f = encoder_mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            loss = (per_frame_mse * mask_f).sum() / denom

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item() * B
            total_examples += B
            sum_masked += int(mask_f.sum().item())
            sum_total_mask_pos += int(mask_f.numel())

        avg_loss = total_loss / max(1, total_examples)
        masked_ratio = sum_masked / max(1, sum_total_mask_pos)
        epoch_losses.append(avg_loss)
        epoch_masked_ratios.append(masked_ratio)
        logger.info(
            f"[adaptive][fold {fold_idx}] epoch {epoch + 1}/{epochs}  "
            f"avg_loss={avg_loss:.5f}  masked_ratio={masked_ratio:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    # Persist encoder state dict (drop reconstruction head).
    encoder_state = {
        k: v.detach().cpu() for k, v in model.encoder.state_dict().items()
    }
    torch.save(encoder_state, adaptive_ckpt)
    logger.info(
        f"[adaptive][fold {fold_idx}] saved adapted encoder to {adaptive_ckpt} "
        f"({len(encoder_state)} tensors)"
    )

    stats = {
        "method": method,
        "scope": pool_info["scope"],
        "num_samples": pool_info["n_total"],
        "num_original": pool_info["n_original"],
        "num_synthetic": pool_info["n_synthetic"],
        "epochs_run": len(epoch_losses),
        "epoch_losses": epoch_losses,
        "epoch_masked_ratios": epoch_masked_ratios,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "mean_loss": (
            float(np.mean(epoch_losses)) if epoch_losses else None
        ),
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "mask_time_prob": mask_time_prob,
        "mask_time_length": mask_time_length,
        "batch_size": batch_size,
        "global_steps": global_step,
        "feature_layer": feature_layer,
        "num_mel_bins": int(model.num_mel_bins),
        "d_model": int(model.d_model),
        "trainable_params": int(trainable_n),
        "total_params": int(total_n),
    }
    with open(log_path, "w") as f:
        json.dump(stats, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adaptive_ckpt, stats


# ============================================================================
# Stage 2: classification fine-tuning (close mirror of v9 train_single_fold)
# ============================================================================

def train_classification_fold(
    fold_idx,
    train_files, train_labels, train_rec_types, train_is_synthetic,
    test_files, test_labels, test_rec_types,
    test_speaker_ids,
    model_name, feature_extractor,
    num_labels, id2label, label2id,
    task, output_dir,
    num_epochs, batch_size, lr, warmup_ratio,
    max_length_sec, gradient_accumulation_steps,
    num_augmented,
    seed,
    reseed_each_fold: bool,
    metric_for_best_model: str,
    strategy_layers,
    feature_layer: int,
    adaptive_ckpt_path: Path,
    class_weight_mode: str = "none",
):
    """Stage 2: load fresh WhisperForAudioClassification, load adaptive
    encoder weights, then run v9 selected_layers fine-tuning."""

    if reseed_each_fold:
        fold_seed = seed + fold_idx * 1000
        set_deterministic(fold_seed)
    else:
        fold_seed = seed

    fold_dir = Path(output_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    n_synth_train = sum(1 for f in train_files if 'synthetic' in str(f))
    n_orig_train = len(train_files) - n_synth_train
    logger.info(f"\n{'=' * 60}")
    logger.info(f"[stage2][fold {fold_idx}] Train={len(train_files)} "
                f"(orig={n_orig_train}, synth={n_synth_train}), "
                f"Test={len(test_files)}")
    logger.info(f"{'=' * 60}")

    train_dataset = VoiceDisorderDatasetTTSOrigAugOnly(
        train_files, train_labels, feature_extractor,
        recording_types=train_rec_types,
        is_synthetic=train_is_synthetic,
        num_augmented=num_augmented,
        max_length_sec=max_length_sec,
    )
    test_dataset = VoiceDisorderDatasetTTS(
        test_files, test_labels, feature_extractor,
        recording_types=test_rec_types,
        num_augmented=0,
        max_length_sec=max_length_sec,
    )

    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    if not _is_whisper(model):
        raise ValueError(
            "v10 requires a Whisper model "
            f"(got class={model.__class__.__name__}). "
            "Use --model_name openai/whisper-small."
        )

    # ----- Load adaptive encoder weights -----
    if not Path(adaptive_ckpt_path).exists():
        raise FileNotFoundError(
            f"[stage2][fold {fold_idx}] adaptive checkpoint missing: "
            f"{adaptive_ckpt_path}"
        )
    encoder_state = torch.load(adaptive_ckpt_path, map_location="cpu")
    missing, unexpected = model.encoder.load_state_dict(
        encoder_state, strict=True,
    )
    logger.info(
        f"[stage2][fold {fold_idx}] loaded adaptive encoder from "
        f"{adaptive_ckpt_path} (missing={len(missing)}, "
        f"unexpected={len(unexpected)})"
    )
    if missing or unexpected:
        logger.warning(
            f"[stage2][fold {fold_idx}] state_dict mismatch detail: "
            f"missing={list(missing)[:5]} ; unexpected={list(unexpected)[:5]}"
        )

    # ----- Apply v9 selected_layers strategy -----
    final_model, strategy_info = apply_strategy(
        model,
        strategy=STAGE2_STRATEGY,
        strategy_layers=strategy_layers,
    )
    feature_layer_book = patch_forward_to_use_feature_layer(
        final_model, feature_layer
    )
    audit = safety_check_and_log(
        final_model, STAGE2_STRATEGY, strategy_layers,
    )

    # ----- Class weights -----
    # After 1:1 Stage-2 entry balancing, uniform weights match the sampling
    # target. real_invfreq is the old real-only inverse-frequency recipe.
    if class_weight_mode == "none":
        class_weights = torch.ones(num_labels, dtype=torch.float32)
        logger.info(
            f"[stage2][fold {fold_idx}] class weights (uniform / none): "
            f"{class_weights.tolist()}"
        )
    elif class_weight_mode == "real_invfreq":
        orig_labels = [
            l for f, l in zip(train_files, train_labels)
            if 'synthetic' not in str(f)
        ]
        weight_source = orig_labels if orig_labels else train_labels
        label_counts = Counter(weight_source)
        total = sum(label_counts.values())
        raw_weights = [
            total / (num_labels * label_counts.get(i, 1))
            for i in range(num_labels)
        ]
        class_weights = torch.tensor(
            [w ** 1.0 for w in raw_weights], dtype=torch.float32,
        )
        class_weights = class_weights / class_weights.mean()
        logger.info(
            f"[stage2][fold {fold_idx}] class weights "
            f"(real_invfreq from {len(weight_source)} original samples): "
            f"{class_weights.tolist()}"
        )
    else:
        raise ValueError(
            f"Unknown class_weight_mode={class_weight_mode!r}; "
            "use 'none' or 'real_invfreq'."
        )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_t = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            device = logits.device

            if model.training and torch.rand(1).item() < 0.3:
                lam = torch.distributions.Beta(0.4, 0.4).sample().to(device)
                bs = logits.size(0)
                index = torch.randperm(bs, device=device)
                loss_fn = nn.CrossEntropyLoss(
                    weight=class_weights.to(device), reduction='none')
                loss = lam * loss_fn(logits, labels_t) + \
                    (1 - lam) * loss_fn(logits, labels_t[index])
                loss = loss.mean()
            else:
                loss_fn = nn.CrossEntropyLoss(
                    weight=class_weights.to(device))
                loss = loss_fn(logits, labels_t)
            return (loss, outputs) if return_outputs else loss

    steps_per_epoch = max(
        len(train_dataset) // (batch_size * gradient_accumulation_steps), 1
    )
    eval_steps = steps_per_epoch
    logger.info(f"[stage2][fold {fold_idx}] steps_per_epoch={steps_per_epoch} "
                f"eval every {eval_steps} steps")

    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        label_smoothing_factor=0.05,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_steps=max(steps_per_epoch // 5, 1),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        seed=fold_seed,
        data_seed=fold_seed,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )

    compute_fn = compute_metrics_detection if task == 'detection' \
        else compute_metrics_classification

    trainer = WeightedTrainer(
        model=final_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_fn,
    )

    trainer.train()
    results = trainer.evaluate()
    logger.info(f"[stage2][fold {fold_idx}] recording-level results: {results}")

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    if task == 'detection' and test_speaker_ids is not None:
        spk_results = evaluate_speaker_level(
            predictions.predictions, test_labels, test_speaker_ids,
            num_labels, id2label,
        )
        results.update(spk_results)
        logger.info(
            f"[stage2][fold {fold_idx}] speaker-level (thr=0.5): "
            f"Acc={spk_results['spk_accuracy']:.3f}, "
            f"F1={spk_results['spk_f1_macro']:.3f}, "
            f"AUC={spk_results['spk_auc']:.3f}"
        )
        logger.info(
            f"[stage2][fold {fold_idx}] speaker-level (thr="
            f"{spk_results['spk_best_threshold']:.2f}): "
            f"Acc={spk_results['spk_accuracy_opt']:.3f}, "
            f"F1={spk_results['spk_f1_macro_opt']:.3f}"
        )

    results['strategy'] = STAGE2_STRATEGY
    results['strategy_layers'] = list(strategy_layers)
    results['feature_layer'] = feature_layer
    results['feature_layer_used'] = feature_layer_book.get('feature_layer_used')
    results['feature_layer_post_ln'] = feature_layer_book.get(
        'post_layernorm_applied', False
    )
    results['backbone_type'] = 'whisper'
    results['strategy_info'] = strategy_info
    results['param_audit'] = audit

    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        test_labels, preds,
        target_names=[id2label[i] for i in range(num_labels)],
        output_dict=True,
    )
    with open(fold_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    del model, final_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiment(args):
    output_dir = Path(args.output_dir)
    _assert_new_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "audit_tts_scale_only", False):
        return run_tts_scale12_audit(args)
    if getattr(args, "dry_run_stage2_data", False):
        return run_stage2_data_dryrun(args)

    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    (file_paths, labels, speaker_ids, rec_types,
     is_synthetic, label2id, id2label) = load_metadata_with_tts(
        args.data_dir, task=args.task, recording_type=args.recording_type,
        max_synth_ratio=args.max_synth_ratio,
    )

    num_labels = len(label2id)
    logger.info(f"Task: {args.task} | Labels: {num_labels} | "
                f"Model: {args.model_name}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    folds = create_cv_folds_with_tts(
        speaker_ids, labels, is_synthetic,
        n_folds=args.n_folds, seed=args.seed,
    )

    if sum(x is not None for x in (args.only_fold, args.max_folds, args.fold_list)) > 1:
        raise ValueError(
            "Use only one of --only_fold, --max_folds, and --fold_list"
        )

    if args.fold_list is not None:
        raw = [s.strip() for s in str(args.fold_list).split(",") if s.strip()]
        if not raw:
            raise ValueError("--fold_list is empty")
        fold_ids = []
        for tok in raw:
            i = int(tok)
            if i < 0 or i >= len(folds):
                raise ValueError(
                    f"--fold_list contains out-of-range fold {i} "
                    f"(valid [0, {len(folds) - 1}])"
                )
            fold_ids.append(i)
        seen = set()
        fold_ids_u = []
        for i in fold_ids:
            if i not in seen:
                seen.add(i)
                fold_ids_u.append(i)
        folds_indexed = [
            (i, folds[i][0], folds[i][1]) for i in fold_ids_u
        ]
        logger.info(
            f"--fold_list={fold_ids_u} set: built {len(folds)}-fold "
            f"split, running {len(fold_ids_u)} folds"
        )
    elif args.only_fold is not None:
        if args.only_fold < 0 or args.only_fold >= len(folds):
            raise ValueError(
                f"--only_fold={args.only_fold} out of range "
                f"[0, {len(folds) - 1}]"
            )
        folds_indexed = [
            (args.only_fold, folds[args.only_fold][0], folds[args.only_fold][1])
        ]
        logger.info(
            f"--only_fold={args.only_fold} set: built {len(folds)}-fold "
            f"split, running fold_{args.only_fold} only"
        )
    elif args.max_folds is not None:
        if args.max_folds <= 0:
            raise ValueError("--max_folds must be positive")
        if args.max_folds > len(folds):
            raise ValueError(
                f"--max_folds ({args.max_folds}) cannot exceed "
                f"n_folds ({args.n_folds})"
            )
        folds_indexed = [
            (i, folds[i][0], folds[i][1]) for i in range(args.max_folds)
        ]
        logger.info(
            f"--max_folds={args.max_folds} set: built {len(folds)}-fold "
            f"split, running fold_0 .. fold_{args.max_folds - 1}"
        )
    else:
        folds_indexed = [
            (i, folds[i][0], folds[i][1]) for i in range(len(folds))
        ]

    strategy_layers = parse_strategy_layers(args.strategy_layers, n_layers=64)

    # Stash for final summary
    method = args.method
    if method not in VALID_METHODS:
        raise ValueError(
            f"--method must be one of {VALID_METHODS}, got '{method}'"
        )

    all_results = []
    adaptive_summaries = []
    backbone_type_observed = None

    for fold_idx, train_idx, test_idx in folds_indexed:
        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        cached_path = fold_dir / 'results.json'
        if cached_path.exists():
            with open(cached_path) as f:
                cached = json.load(f)
            logger.info(
                f"[skip] fold_{fold_idx} already finished, "
                f"reusing {cached_path}"
            )
            if backbone_type_observed is None:
                backbone_type_observed = cached.get('backbone_type', 'whisper')
            all_results.append(cached)

            adaptive_log = fold_dir / 'adaptive_pretrain_log.json'
            if adaptive_log.exists():
                try:
                    with open(adaptive_log) as f:
                        adaptive_summaries.append({
                            "fold_idx": fold_idx, **json.load(f),
                        })
                except Exception:
                    pass
            continue

        train_files = [file_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_rec_types = [rec_types[i] for i in train_idx]
        train_speaker_ids = [speaker_ids[i] for i in train_idx]
        train_is_synth = [is_synthetic[i] for i in train_idx]

        test_files = [file_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        test_rec_types = [rec_types[i] for i in test_idx]
        test_speaker_ids = [speaker_ids[i] for i in test_idx]

        # ===== Stage 1: adaptive pretraining =====
        pool_info = filter_adaptive_pool(
            method=method,
            train_files=train_files,
            train_labels=train_labels,
            train_speaker_ids=train_speaker_ids,
            train_is_synthetic=train_is_synth,
            label2id=label2id,
            include_synthetic=args.include_synthetic_in_adaptive,
        )
        assert_no_speaker_leak(
            fold_idx=fold_idx,
            adaptive_speaker_ids=pool_info["speaker_ids"],
            train_speaker_ids=train_speaker_ids,
            test_speaker_ids=test_speaker_ids,
            method=method,
            pool_info=pool_info,
        )

        if args.reseed_each_fold:
            fold_seed = args.seed + fold_idx * 1000
        else:
            fold_seed = args.seed

        adaptive_ckpt, adaptive_stats = run_adaptive_pretraining(
            fold_idx=fold_idx,
            method=method,
            pool_info=pool_info,
            model_name=args.model_name,
            feature_extractor=feature_extractor,
            feature_layer=args.feature_layer,
            fold_dir=fold_dir,
            epochs=args.adaptive_pretrain_epochs,
            lr=args.adaptive_pretrain_lr,
            mask_time_prob=args.mask_time_prob,
            mask_time_length=args.mask_time_length,
            batch_size=args.adaptive_batch_size,
            max_length_sec=args.max_length_sec,
            fold_seed=fold_seed,
        )
        adaptive_summaries.append({"fold_idx": fold_idx, **adaptive_stats})

        # ===== Stage 2: locked 1.2x manifest (real train + ranking[:Q]) =====
        manifest_root = Path(getattr(args, "scale12_manifest_dir", None) or args.output_dir)
        man_path = manifest_root / f"fold_{fold_idx}" / "stage2_tts_sample_scale12.json"
        if not man_path.exists():
            healthy_id = label2id["healthy"]
            nH = sum(1 for s, lab in zip(train_is_synth, train_labels)
                     if (not s) and lab == healthy_id)
            nP = sum(1 for s, lab in zip(train_is_synth, train_labels)
                     if (not s) and lab != healthy_id)
            real_train_sids = [
                sid for sid, synth in zip(train_speaker_ids, train_is_synth) if not synth
            ]
            build_scale12_fold_manifest(
                fold_idx=fold_idx,
                nH=nH,
                nP=nP,
                train_speaker_ids=real_train_sids,
                test_speaker_ids=test_speaker_ids,
                num_augmented=args.num_augmented,
                scale15_fold_dir=Path(args.scale15_source_dir) / f"fold_{fold_idx}",
                scale125_fold_dir=Path(args.scale125_source_dir) / f"fold_{fold_idx}",
                scale11_fold_dir=Path(args.scale11_source_dir) / f"fold_{fold_idx}",
                bal11_baseline_dir=Path(args.bal11_baseline_dir),
                out_fold_dir=manifest_root / f"fold_{fold_idx}",
            )
        constructed = build_stage2_from_scale12_manifest(
            fold_idx=fold_idx,
            train_files=train_files,
            train_labels=train_labels,
            train_rec_types=train_rec_types,
            train_speaker_ids=train_speaker_ids,
            train_is_synthetic=train_is_synth,
            test_speaker_ids=test_speaker_ids,
            label2id=label2id,
            num_augmented=args.num_augmented,
            manifest_path=man_path,
            bal11_baseline_dir=Path(args.bal11_baseline_dir),
        )
        stage2_train_files = constructed["train_files"]
        stage2_train_labels = constructed["train_labels"]
        stage2_train_rec_types = constructed["train_rec_types"]
        stage2_train_speaker_ids = constructed["train_speaker_ids"]
        stage2_train_is_synth = constructed["train_is_synthetic"]
        stage2_synth_info = {
            "exclude_test_speaker_synthetic": True,
            "balance_synth_to_ones": False,
            "tts_scale": 1.2,
            "used_old_bal11_sampler": False,
            **constructed["report"],
        }

        # ===== Stage 2: classification fine-tuning =====
        results = train_classification_fold(
            fold_idx=fold_idx,
            train_files=stage2_train_files,
            train_labels=stage2_train_labels,
            train_rec_types=stage2_train_rec_types,
            train_is_synthetic=stage2_train_is_synth,
            test_files=test_files,
            test_labels=test_labels,
            test_rec_types=test_rec_types,
            test_speaker_ids=test_speaker_ids,
            model_name=args.model_name,
            feature_extractor=feature_extractor,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            task=args.task,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            max_length_sec=args.max_length_sec,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_augmented=args.num_augmented,
            seed=args.seed,
            reseed_each_fold=args.reseed_each_fold,
            metric_for_best_model=args.metric_for_best_model,
            strategy_layers=strategy_layers,
            feature_layer=args.feature_layer,
            adaptive_ckpt_path=adaptive_ckpt,
            class_weight_mode=args.class_weight,
        )

        # Annotate fold results with stage-1 info (for downstream summaries).
        results['method'] = method
        results['adaptive_pretrain_data_scope'] = pool_info["scope"]
        results['adaptive_pretrain_num_samples'] = pool_info["n_total"]
        results['adaptive_pretrain_final_loss'] = (
            adaptive_stats.get("final_loss")
        )
        results['adaptive_pretrain_mean_loss'] = (
            adaptive_stats.get("mean_loss")
        )
        results.update(stage2_synth_info)

        # Re-persist enriched per-fold results.json with stage-1 fields
        with open(fold_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        if backbone_type_observed is None:
            backbone_type_observed = results.get('backbone_type', 'whisper')
        all_results.append(results)

    # ===== Final summary =====
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (mean +/- std across folds)")
    logger.info("=" * 60)

    metric_keys = ['eval_accuracy', 'eval_f1_macro']
    if args.task == 'detection':
        metric_keys.append('eval_auc')

    summary = {}
    for key in metric_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            mean, std = float(np.mean(values)), float(np.std(values))
            short = key.replace('eval_', '')
            summary[short] = {'mean': mean, 'std': std}
            logger.info(f"  {short}: {mean:.3f} +/- {std:.3f}")

    spk_keys = ['spk_accuracy', 'spk_f1_macro', 'spk_auc']
    if all_results and any(k in all_results[0] for k in spk_keys):
        logger.info("\n  --- Speaker-level metrics (threshold=0.5) ---")
        for key in spk_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                mean, std = float(np.mean(values)), float(np.std(values))
                summary[key] = {'mean': mean, 'std': std}
                logger.info(f"  {key}: {mean:.3f} +/- {std:.3f}")

        opt_keys = ['spk_accuracy_opt', 'spk_f1_macro_opt']
        logger.info("\n  --- Speaker-level metrics (optimal threshold per fold) ---")
        for key in opt_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                mean, std = float(np.mean(values)), float(np.std(values))
                summary[key] = {'mean': mean, 'std': std}
                logger.info(f"  {key}: {mean:.3f} +/- {std:.3f}")

        thresholds = [r.get('spk_best_threshold', 0.5) for r in all_results]
        if thresholds:
            logger.info(
                f"  avg optimal threshold: {np.mean(thresholds):.3f} "
                f"(range: {min(thresholds):.2f} - {max(thresholds):.2f})"
            )

    # adaptive_pretrain_data_scope is identical across folds for a given
    # method, just use the first non-empty value.
    scope_for_summary = next(
        (s.get("scope") for s in adaptive_summaries if s.get("scope")),
        None,
    )

    final_summary = {
        'task': args.task,
        'model': args.model_name,
        'backbone_type': backbone_type_observed,
        'recording_type': args.recording_type,
        'n_folds': args.n_folds,
        'max_folds': args.max_folds,
        'only_fold': args.only_fold,
        'fold_list': args.fold_list,
        'num_folds_run': len(all_results),
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'num_augmented': args.num_augmented,
        'aug_originals_only': True,
        'max_synth_ratio': args.max_synth_ratio,

        'method': method,
        'strategy': STAGE2_STRATEGY,
        'strategy_layers': strategy_layers,
        'feature_layer': args.feature_layer,
        'feature_layer_post_ln': any(
            r.get('feature_layer_post_ln', False) for r in all_results
        ),
        'adaptive_pretrain_epochs': args.adaptive_pretrain_epochs,
        'adaptive_pretrain_lr': args.adaptive_pretrain_lr,
        'adaptive_pretrain_data_scope': scope_for_summary,
        'mask_time_prob': args.mask_time_prob,
        'mask_time_length': args.mask_time_length,
        'adaptive_batch_size': args.adaptive_batch_size,
        'exclude_test_speaker_synthetic': args.exclude_test_speaker_synthetic,
        'tts_scale': 1.2,
        'stage2_tts_sampling': 'locked_scale12_prefix_of_scale15_ranking',
        'used_old_bal11_sampler': False,
        'stage2_class_ratio': '1:1',
        'stage2_manifest_dir': str(
            Path(getattr(args, "scale12_manifest_dir", None) or args.output_dir).resolve()
        ),
        'balance_synth_to_ones': False,
        'class_weight': args.class_weight,

        'results': summary,
        'fold_results': all_results,
        'adaptive_summaries': adaptive_summaries,
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    logger.info(f"\nResults saved to {output_dir / 'final_results.json'}")
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TTS Training v10 - TAPT / PTAPT two-stage on Whisper-small.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # v9-equivalents
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--task',           type=str, default='detection',
                        choices=['detection', 'classification'])
    parser.add_argument('--recording_type', type=str, default='sentence',
                        choices=['sentence', 'vowel', 'all'])
    parser.add_argument('--output_dir',     type=str, required=True)
    parser.add_argument('--model_name',     type=str,
                        default='openai/whisper-small')
    parser.add_argument('--num_epochs',     type=int, default=50)
    parser.add_argument('--batch_size',     type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--lr',             type=float, default=1e-5)
    parser.add_argument('--warmup_ratio',   type=float, default=0.12)
    parser.add_argument('--max_length_sec', type=float, default=5.0)
    parser.add_argument('--num_augmented',  type=int, default=1)
    parser.add_argument('--max_synth_ratio', type=float, default=None)
    parser.add_argument('--n_folds',        type=int, default=10)
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--metric_for_best_model', type=str, default='f1_macro',
                        choices=['f1_macro', 'auc'])
    parser.add_argument('--reseed_each_fold', action='store_true')
    parser.add_argument('--max_folds', type=int, default=None)
    parser.add_argument(
        '--only_fold', type=int, default=None,
        help='Run a single fold K only (mutually exclusive with --max_folds '
             '/ --fold_list).',
    )
    parser.add_argument(
        '--fold_list', type=str, default=None,
        help='Comma-separated fold indices to run, e.g. "0,1" for multi-GPU '
             'sharding (mutually exclusive with --only_fold / --max_folds).',
    )

    parser.add_argument(
        '--strategy_layers', type=str, default='0,1,2,3,4,5,6',
        help='Comma-separated encoder layer ids that stage 2 fine-tunes. '
             'Layers outside this set are FROZEN (safety-checked).',
    )
    parser.add_argument(
        '--feature_layer', type=int, default=6,
        help='Encoder layer whose hidden_states feed both the stage-1 '
             'reconstruction head and the stage-2 classifier head.',
    )

    # v10-specific
    parser.add_argument(
        '--method', type=str, required=True,
        choices=list(VALID_METHODS),
        help='Two-stage method: TAPT or PTAPT followed by selected_layers FT.',
    )
    parser.add_argument(
        '--adaptive_pretrain_epochs', type=int, default=5,
        help='Number of epochs for the stage-1 masked-feature reconstruction.',
    )
    parser.add_argument(
        '--adaptive_pretrain_lr', type=float, default=1e-5,
        help='Learning rate for stage-1 adaptive pretraining (AdamW).',
    )
    parser.add_argument(
        '--mask_time_prob', type=float, default=0.15,
        help='Fraction of *encoder* time frames to mask per sample (1500 '
             'frames/sample for whisper-small @ 30s mel pad).',
    )
    parser.add_argument(
        '--mask_time_length', type=int, default=10,
        help='Length (in *encoder* timesteps) of each masked block. '
             'Each encoder timestep covers 2 mel frames (stride-2 conv).',
    )
    parser.add_argument(
        '--adaptive_batch_size', type=int, default=8,
        help='Batch size for stage-1 adaptive pretraining (default matches '
             'stage-2 per-device train batch size).',
    )
    parser.add_argument(
        '--include_synthetic_in_adaptive', action='store_true',
        help='OPT-IN: include TTS-synthetic train samples in the adaptive '
             'pretraining pool. DEFAULT is OFF because in this dataset the '
             'synthetic samples are conditioned on real speakers drawn from '
             'the whole corpus, so their speaker_ids overlap test speakers '
             'and will (correctly) trigger the speaker-leak guard. Enable '
             'only if you have verified your synthetic split is speaker-'
             'disjoint from test.',
    )
    parser.add_argument(
        '--exclude_test_speaker_synthetic', action='store_true',
        help='Stage 2 only: remove TTS-synthetic train samples whose '
             'speaker_id is in the current fold test set (speaker-disjoint '
             'synthetic augmentation ablation). Real train recordings are '
             'unchanged. Stage 1 is unaffected.',
    )
    parser.add_argument(
        '--balance_synth_to_ones',
        dest='balance_synth_to_ones',
        action='store_true',
        help='After the train-speaker TTS filter, speaker-balanced subsample '
             'of healthy TTS so Stage-2 training entries are 1:1 '
             '(default ON in this script).',
    )
    parser.add_argument(
        '--no_balance_synth_to_ones',
        dest='balance_synth_to_ones',
        action='store_false',
        help='Disable 1:1 TTS quota sampling (keeps all post-filter TTS).',
    )
    parser.set_defaults(balance_synth_to_ones=True)
    parser.add_argument(
        '--class_weight',
        type=str,
        default='none',
        choices=['none', 'real_invfreq'],
        help="Stage-2 CE class weights. 'none' = uniform (default, matches "
             "1:1 entry balancing). 'real_invfreq' = old real-only "
             "inverse-frequency weights.",
    )
    parser.add_argument(
        '--tts_scale', type=float, default=1.2,
        help='Stage-2 TTS scale. This script is 1.2 only (prefix of 1.5x/1.25x/1.1x).',
    )
    parser.add_argument(
        '--healthy_tts_dir', type=str,
        default='./svd_cleaned_paper_tts_healthy_sentence_6x',
        help='Read-only healthy 6x TTS pool.',
    )
    parser.add_argument(
        '--patho_tts_dir', type=str,
        default='./svd_cleaned_paper_tts_patho_sentence_4x',
        help='Read-only pathological 4x TTS pool.',
    )
    parser.add_argument(
        '--bal11_baseline_dir', type=str,
        default=('experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11/'
                 'feat8/tapt_then_selected_layers_L0-L8'),
        help='Existing bal11 feat8 run with per-fold stage2_tts_sample.json '
             '(read-only).',
    )
    parser.add_argument(
        '--audit_tts_scale_only', action='store_true',
        help='Run 10-fold TTS sampling audit and write manifests; do not train.',
    )
    parser.add_argument(
        '--dry_run_stage2_data', action='store_true',
        help='Construct Stage-2 file lists from the locked 1.2x manifest '
             '(default fold 0). No Whisper load, no Stage 1, no training.',
    )
    parser.add_argument(
        '--scale15_source_dir', type=str,
        default=('experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale15/'
                 'feat8/tapt_then_selected_layers_L0-L8'),
        help='Read-only 1.5x audit dir with tts_nested_ranking.json.',
    )
    parser.add_argument(
        '--scale125_source_dir', type=str,
        default=('experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale125/'
                 'feat8/tapt_then_selected_layers_L0-L8'),
        help='Read-only 1.25x dir used to assert 1.2x âŠ?1.25x.',
    )
    parser.add_argument(
        '--scale11_source_dir', type=str,
        default=('experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_tts_scale11/'
                 'feat8/tapt_then_selected_layers_L0-L8'),
        help='Read-only 1.1x dir used to assert 1.1x âŠ?1.2x.',
    )
    parser.add_argument(
        '--scale12_manifest_dir', type=str, default=None,
        help='Directory containing fold_*/stage2_tts_sample_scale12.json. '
             'Default: --output_dir.',
    )

    args = parser.parse_args()

    if args.num_augmented != 1:
        parser.error(
            "This 1.2x experiment is defined only for --num_augmented 1."
        )
    if args.balance_synth_to_ones and not args.exclude_test_speaker_synthetic:
        parser.error(
            "--balance_synth_to_ones requires --exclude_test_speaker_synthetic "
            "so the TTS quota is drawn only from train speakers."
        )
    if args.balance_synth_to_ones and args.max_synth_ratio is not None:
        logger.warning(
            "--max_synth_ratio downsamples TTS globally BEFORE folds and can "
            "starve the per-fold 1:1 quota; leave it unset for this experiment."
        )
    if abs(float(args.tts_scale) - 1.2) > 1e-9:
        parser.error("this script currently supports --tts_scale 1.2 only")
    if args.audit_tts_scale_only and args.dry_run_stage2_data:
        parser.error("use only one of --audit_tts_scale_only and --dry_run_stage2_data")

    set_deterministic(args.seed)

    multiplier = 1 + args.num_augmented
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("\n--- Config ---")
    logger.info(f"  Method:              {args.method}")
    logger.info(f"  Task:                {args.task}")
    logger.info(f"  Rec type:            {args.recording_type}")
    logger.info(f"  Model:               {args.model_name}")
    logger.info(f"  Stage2 epochs:       {args.num_epochs}")
    logger.info(f"  Effective BS:        {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Stage2 LR:           {args.lr}")
    logger.info(f"  Warmup:              {args.warmup_ratio}")
    logger.info(f"  Aug copies:          {args.num_augmented} ({multiplier}x real-only)")
    logger.info(f"  Aug originals only:  True (TTS = clean only)")
    logger.info(f"  Synth ratio:         {args.max_synth_ratio or 'unlimited'}")
    logger.info(f"  Auglib:              {'audiomentations' if HAS_AUDIOMENTATIONS else 'DISABLED'}")
    logger.info(f"  Best metric:         {args.metric_for_best_model}")
    logger.info(f"  Reseed fold:         {args.reseed_each_fold}")
    logger.info(f"  Stage2 strategy:     {STAGE2_STRATEGY} (FIXED for v10)")
    logger.info(f"  Strategy layers:     {args.strategy_layers}")
    logger.info(f"  Feature layer:       {args.feature_layer}")
    logger.info(f"  n_folds:             {args.n_folds}")
    logger.info(f"  max_folds:           {args.max_folds}")
    logger.info(f"  only_fold:           {args.only_fold}")
    logger.info(f"  fold_list:           {args.fold_list}")
    logger.info(f"  --- Stage-1 (adaptive) ---")
    logger.info(f"  Adaptive epochs:     {args.adaptive_pretrain_epochs}")
    logger.info(f"  Adaptive LR:         {args.adaptive_pretrain_lr}")
    logger.info(f"  Mask time prob:      {args.mask_time_prob}")
    logger.info(f"  Mask time length:    {args.mask_time_length}  (enc frames)")
    logger.info(f"  Adaptive batch size: {args.adaptive_batch_size}")
    logger.info(f"  Include synthetic in adaptive: "
                f"{args.include_synthetic_in_adaptive} "
                f"(default OFF; synthetic speakers overlap test in this dataset)")
    logger.info(f"  Exclude test-spk synth (S2): "
                f"{args.exclude_test_speaker_synthetic}")
    logger.info(f"  Balance synth to 1:1 entries: "
                f"{args.balance_synth_to_ones}")
    logger.info(f"  Class weight:        {args.class_weight}")

    run_experiment(args)


if __name__ == '__main__':
    main()
