#!/usr/bin/env python3

from __future__ import annotations

import os
import math
import json
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
from train_with_tts_v9_strategy_ablation_whisper import (
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

    seq_head = getattr(underlying, "seq_clf_head", None)
    if seq_head is not None and any(True for _ in seq_head.parameters()):
        if not any(p.requires_grad for p in seq_head.parameters()):
            raise RuntimeError(
                f"[strategy={strategy}] seq_clf_head has no trainable params"
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
# Classification-head ablation (sequence models on projected encoder frames)
# ============================================================================
# Shared interface: input [B, T, d_model] after Whisper projector (256-d),
# output [B, T, d_model]. Temporal mean + Linear(d_model → num_labels) is
# applied in the patched forward, matching mean_linear pooling.

VALID_CLASSIFIER_HEADS = ("mean_linear", "lstm", "transformer", "mamba", "mamba3")


class LSTMSeqHead(nn.Module):
    """1-layer unidirectional LSTM; same hidden size as projector dim."""

    def __init__(self, d_model: int, hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.out_proj = (
            nn.Identity() if hidden_size == d_model
            else nn.Linear(hidden_size, d_model)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.out_proj(out)


class TransformerSeqHead(nn.Module):
    """1-layer Transformer encoder on the projected frame sequence."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        t = x.size(1)
        if t > self.pos.size(1):
            raise ValueError(
                f"sequence length {t} exceeds pos-embed max_len {self.pos.size(1)}"
            )
        return self.encoder(x + self.pos[:, :t])


class MambaLiteBlock(nn.Module):
    """Single Mamba / S6 block in pure PyTorch (no mamba-ssm CUDA kernel).

    Same layout as Gu & Dao Mamba: gated in-proj, depthwise conv, selective
    scan, out-proj. Used when ``mamba_ssm`` is not installed.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seqlen]
        x_inner = F.silu(x_conv.transpose(1, 2))

        x_dbl = self.x_proj(x_inner)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x_inner, dt, A, B, C, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)

    @staticmethod
    def _selective_scan(u, delta, A, B, C, D):
        """u, delta: [B, T, D]; A: [D, N]; B, C: [B, T, N]; D: [D]."""
        bsz, seqlen, d_inner = u.shape
        n_state = A.size(-1)
        delta_a = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_inner, n_state))
        delta_b_u = (
            delta.unsqueeze(-1)
            * B.unsqueeze(2)
            * u.unsqueeze(-1)
        )
        h = u.new_zeros(bsz, d_inner, n_state)
        ys = []
        for t in range(seqlen):
            h = delta_a[:, t] * h + delta_b_u[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + u * D.view(1, 1, -1)


def _build_seq_clf_head(kind: str, d_model: int):
    """Return (module, implementation_tag). kind is lstm/transformer/mamba/mamba3."""
    if kind == "lstm":
        return LSTMSeqHead(d_model=d_model, hidden_size=d_model, num_layers=1), "lstm_1l_uni"
    if kind == "transformer":
        return TransformerSeqHead(
            d_model=d_model, nhead=4, dim_feedforward=512, num_layers=1,
        ), "transformer_1l"
    if kind == "mamba":
        try:
            from mamba_ssm import Mamba as MambaSSM
        except Exception as exc:
            raise RuntimeError(
                "--classifier_head mamba requires the official mamba-ssm "
                "package (from mamba_ssm import Mamba). Install mamba-ssm "
                "before the formal run; fallback to MambaLiteBlock is disabled."
            ) from exc
        return MambaSSM(d_model=d_model, d_state=16, d_conv=4, expand=2), "mamba_ssm"
    if kind == "mamba3":
        try:
            from mamba_ssm import Mamba3 as Mamba3SSM
        except Exception as exc:
            raise RuntimeError(
                "--classifier_head mamba3 requires a recent official "
                "mamba-ssm that exports Mamba3 "
                "(from mamba_ssm import Mamba3). Use the ai4voice_mamba3 "
                "env; do not fall back to Mamba-1 or MambaLiteBlock."
            ) from exc
        # Projector dim is 256: expand=2 -> d_inner=512, 512 % headdim(64) == 0.
        # SISO + paper defaults so the row is "official Mamba-3 block", not MIMO.
        return Mamba3SSM(
            d_model=d_model,
            d_state=128,
            headdim=64,
            expand=2,
            is_mimo=False,
            chunk_size=64,
            is_outproj_norm=False,
        ), "mamba3_ssm_siso"
    raise ValueError(f"unknown classifier head {kind!r}")


def patch_forward_with_classifier_head(model, feature_layer_idx, classifier_head: str):
    """Patch Whisper forward. mean_linear == original v9 patch (no extra module).
    Other heads: projector → seq_clf_head → mean → classifier.
    """
    if classifier_head == "mean_linear":
        book = patch_forward_to_use_feature_layer(model, feature_layer_idx)
        book["classifier_head"] = "mean_linear"
        book["seq_head_impl"] = "mean_linear"
        book["seq_head_params"] = 0
        return book

    bookkeep = {
        "feature_layer_used": None,
        "post_layernorm_applied": False,
        "classifier_head": classifier_head,
        "seq_head_impl": None,
        "seq_head_params": 0,
    }
    if feature_layer_idx is None or feature_layer_idx < 0:
        logger.info("[feature-layer] feature_layer<0 -> no forward patch.")
        return bookkeep

    underlying = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        underlying = model.base_model.model
    if not _is_whisper(underlying):
        raise ValueError("patch_forward expects a Whisper model")

    n_layers = len(underlying.encoder.layers)
    if feature_layer_idx >= n_layers:
        raise ValueError(
            f"--feature_layer={feature_layer_idx} out of range "
            f"[0, {n_layers - 1}]"
        )

    d_model = int(underlying.projector.out_features)
    seq_head, impl_tag = _build_seq_clf_head(classifier_head, d_model)
    underlying.seq_clf_head = seq_head
    for p in underlying.seq_clf_head.parameters():
        p.requires_grad = True
    n_head = sum(p.numel() for p in seq_head.parameters())
    bookkeep["seq_head_impl"] = impl_tag
    bookkeep["seq_head_params"] = int(n_head)
    if classifier_head == "mamba" and impl_tag != "mamba_ssm":
        raise RuntimeError(
            "--classifier_head mamba requires seq_head_impl='mamba_ssm', "
            f"got {impl_tag!r}. Official mamba-ssm must be used."
        )
    if classifier_head == "mamba3" and impl_tag != "mamba3_ssm_siso":
        raise RuntimeError(
            "--classifier_head mamba3 requires seq_head_impl='mamba3_ssm_siso', "
            f"got {impl_tag!r}. Official mamba-ssm Mamba3 must be used."
        )

    underlying.config.output_hidden_states = True
    hidden_states_index = feature_layer_idx + 1
    is_last = (feature_layer_idx == n_layers - 1)
    apply_extra_ln = not is_last
    bookkeep["post_layernorm_applied"] = apply_extra_ln
    bookkeep["feature_layer_used"] = feature_layer_idx

    from transformers.modeling_outputs import SequenceClassifierOutput

    def forward(
        self,
        input_features=None,
        head_mask=None,
        encoder_outputs=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = encoder_outputs.hidden_states[hidden_states_index]
        if apply_extra_ln:
            hidden_states = self.encoder.layer_norm(hidden_states)
        hidden_states = self.projector(hidden_states)
        hidden_states = self.seq_clf_head(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )

    underlying.forward = types.MethodType(forward, underlying)
    logger.info(
        f"[classifier-head] {classifier_head} ({impl_tag}): "
        f"{n_head} params on projected dim {d_model}; "
        f"encoder layer {feature_layer_idx} "
        f"(hidden_states[{hidden_states_index}]); extra LN={apply_extra_ln}"
    )
    return bookkeep


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
        therefore the only research-defensible choice — synthetic is
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
    """Print full audit + raise RuntimeError if adaptive ∩ test is non-empty."""
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
    logger.info(f"[leak-check] #(adaptive ∩ test) overlap        = {len(overlap)}")
    logger.info(f"[leak-check] #(adaptive ⊆ train) violations    = "
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
    classifier_head: str = "mean_linear",
):
    """Stage 2: load fresh WhisperForAudioClassification, load adaptive
    encoder weights, then run v9 selected_layers fine-tuning with the
    requested classification head."""

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
    feature_layer_book = patch_forward_with_classifier_head(
        final_model, feature_layer, classifier_head
    )
    if classifier_head == "mamba" and feature_layer_book.get("seq_head_impl") != "mamba_ssm":
        raise RuntimeError(
            "--classifier_head mamba requires seq_head_impl='mamba_ssm', "
            f"got {feature_layer_book.get('seq_head_impl')!r}. "
            "Official mamba-ssm must be used."
        )
    if classifier_head == "mamba3" and feature_layer_book.get("seq_head_impl") != "mamba3_ssm_siso":
        raise RuntimeError(
            "--classifier_head mamba3 requires seq_head_impl='mamba3_ssm_siso', "
            f"got {feature_layer_book.get('seq_head_impl')!r}. "
            "Official mamba-ssm Mamba3 must be used."
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
    results['classifier_head'] = classifier_head
    results['seq_head_impl'] = feature_layer_book.get('seq_head_impl')
    results['seq_head_params'] = feature_layer_book.get('seq_head_params', 0)
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
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if args.classifier_head not in VALID_CLASSIFIER_HEADS:
        raise ValueError(
            f"--classifier_head must be one of {VALID_CLASSIFIER_HEADS}, "
            f"got '{args.classifier_head}'"
        )

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

        # ===== Stage 2: optionally remove test-speaker TTS from train pool =====
        stage2_train_files = train_files
        stage2_train_labels = train_labels
        stage2_train_rec_types = train_rec_types
        stage2_train_speaker_ids = train_speaker_ids
        stage2_train_is_synth = train_is_synth
        stage2_synth_info = {
            "exclude_test_speaker_synthetic": False,
            "n_synth_excluded_test_speakers": 0,
            "n_synth_kept": sum(1 for s in train_is_synth if s),
            "n_synth_total_before_filter": sum(1 for s in train_is_synth if s),
            "balance_synth_to_ones": False,
        }
        if args.exclude_test_speaker_synthetic:
            filtered = filter_stage2_train_disjoint_synthetic(
                fold_idx=fold_idx,
                train_files=train_files,
                train_labels=train_labels,
                train_rec_types=train_rec_types,
                train_speaker_ids=train_speaker_ids,
                train_is_synthetic=train_is_synth,
                test_speaker_ids=test_speaker_ids,
            )
            stage2_train_files = filtered["train_files"]
            stage2_train_labels = filtered["train_labels"]
            stage2_train_rec_types = filtered["train_rec_types"]
            stage2_train_speaker_ids = filtered["train_speaker_ids"]
            stage2_train_is_synth = filtered["train_is_synthetic"]
            stage2_synth_info = {
                "exclude_test_speaker_synthetic": True,
                "n_synth_excluded_test_speakers": filtered[
                    "n_synth_excluded_test_speakers"
                ],
                "n_synth_kept": filtered["n_synth_kept"],
                "n_synth_total_before_filter": filtered["n_synth_total"],
                "balance_synth_to_ones": False,
            }

        if args.balance_synth_to_ones:
            sampled = sample_stage2_synth_speaker_balanced(
                fold_idx=fold_idx,
                train_files=stage2_train_files,
                train_labels=stage2_train_labels,
                train_rec_types=stage2_train_rec_types,
                train_speaker_ids=stage2_train_speaker_ids,
                train_is_synthetic=stage2_train_is_synth,
                label2id=label2id,
                num_augmented=args.num_augmented,
                fold_seed=fold_seed,
                fold_dir=fold_dir,
            )
            stage2_train_files = sampled["train_files"]
            stage2_train_labels = sampled["train_labels"]
            stage2_train_rec_types = sampled["train_rec_types"]
            stage2_train_speaker_ids = sampled["train_speaker_ids"]
            stage2_train_is_synth = sampled["train_is_synthetic"]
            stage2_synth_info.update({
                "balance_synth_to_ones": True,
                "nP_real": sampled["nP_real"],
                "nH_real": sampled["nH_real"],
                "n_tts_target": sampled["n_tts_target"],
                "n_synth_available": sampled["n_synth_available"],
                "n_tts_selected": sampled["n_tts_selected"],
                "n_speakers_with_tts": sampled["n_speakers_with_tts"],
                "copies_per_speaker_hist": sampled["copies_per_speaker_hist"],
                "entries_pathological": sampled["entries_pathological"],
                "entries_healthy": sampled["entries_healthy"],
                "n_synth_kept": sampled["n_tts_selected"],
            })

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
            classifier_head=args.classifier_head,
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
        'classifier_head': args.classifier_head,
        'seq_head_impl': next(
            (r.get('seq_head_impl') for r in all_results if r.get('seq_head_impl')),
            args.classifier_head,
        ),
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
        'balance_synth_to_ones': args.balance_synth_to_ones,
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
        description='TTS Training v10 - bal11 + classifier-head ablation '
                    '(copy of bal11; new output_dir only).',
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
    parser.add_argument('--num_augmented',  type=int, default=3)
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
    parser.add_argument(
        '--classifier_head', type=str, default='mean_linear',
        choices=list(VALID_CLASSIFIER_HEADS),
        help='Stage-2 classification head on projected encoder frames. '
             'mean_linear is the current projector→mean→Linear path. '
             'lstm / transformer / mamba / mamba3 replace the mean with a '
             '1-layer sequence model, then still mean-pool and Linear(256→2). '
             'mamba3 is official Mamba-3 SISO (d_state=128, headdim=64).',
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

    args = parser.parse_args()

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
    logger.info(f"  Classifier head:     {args.classifier_head}")
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
