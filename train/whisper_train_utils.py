#!/usr/bin/env python3
"""
TTS Synthetic Data Training Script v9 - Strategy Ablation (Whisper-small)
=========================================================================
Direct extension of ``train_with_tts_v8_layer_feature_whisper.py``.

Compared to v8, the ONLY new research variable is *which fine-tuning
strategy* is applied to a FIXED encoder-layer range (default 0..6).
Layer range, feature_layer=6 plumbing, data pipeline, CV split,
augmentation, class weights, mixup, evaluation, and final_results.json
schema are all preserved.

Supported --strategy values (all act ONLY on encoder.layers.{strategy_layers},
which defaults to "0,1,2,3,4,5,6"):

  1. selected_layers      - full fine-tune on the listed layers (v8 baseline)
  2. lora                 - PEFT LoRA on Q/K/V/O of listed layers
  3. adapter              - Houlsby-style bottleneck adapter inserted after
                            each listed layer's output (zero-init up-proj)
  4. bitfit               - only "*.bias" params of listed layers trainable
  5. layernorm_tuning     - only "*layer_norm*" params of listed layers
  6. lora_bitfit          - LoRA + bias unfreeze of listed layers
  7. lora_layernorm       - LoRA + LayerNorm unfreeze of listed layers
  8. adapter_layernorm    - Adapter + LayerNorm unfreeze of listed layers

In every strategy, ``projector`` + ``classifier`` + ``encoder.layer_norm``
are always trainable (head-side, tiny param budget; needed so the
randomly-initialized head can converge).

Safety check (called after strategy + feature_layer patch):
- Any param under encoder.layers.{7..11} with requires_grad=True
  -> RuntimeError (covers everything except LoRA/Adapter newly-added
  params, which we never inject above the strategy_layers range).
- projector and classifier MUST have trainable params.

Example:
  cd "/home/beierwang/testvscode/AI4Voice copy/src"
  CUDA_VISIBLE_DEVICES=1 nohup /home/beierwang/anaconda3/envs/ai4voice/bin/python -u \\
    whisper_train_utils.py \\
    --data_dir ./svd_cleaned --task detection --recording_type sentence \\
    --model_name openai/whisper-small \\
    --output_dir ./experiments/whisper_feat6/strategy_ablation_L0-L6/lora_L0-L6 \\
    --max_synth_ratio 1.5 --seed 42 --metric_for_best_model f1_macro --reseed_each_fold \\
    --n_folds 10 --max_folds 3 \\
    --feature_layer 6 \\
    --strategy lora --strategy_layers "0,1,2,3,4,5,6" \\
    >> ./logs/whisper_feat6_strategy_ablation_L0-L6/lora_L0-L6.log 2>&1 &
"""

import os
import csv
import json
import types
import random
import argparse
import logging
import warnings
from pathlib import Path
from collections import Counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("[WARNING] audiomentations not installed: pip install audiomentations")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Determinism helpers (unchanged from v8)
# ============================================================================

def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Augmentation (unchanged from v8)
# ============================================================================

def build_sentence_augmenter():
    if not HAS_AUDIOMENTATIONS:
        return None
    return Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    ])


def build_vowel_augmenter():
    if not HAS_AUDIOMENTATIONS:
        return None
    return Compose([
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
    ])


# ============================================================================
# Dataset (Whisper input_features; same logic as v8)
# ============================================================================

class VoiceDisorderDatasetTTS(Dataset):
    """Original + TTS synthetic samples. Emits ``input_features`` for Whisper."""

    def __init__(
        self,
        file_paths,
        labels,
        feature_extractor,
        recording_types,
        num_augmented=1,
        max_length_sec=5.0,
        sampling_rate=16000,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.recording_types = recording_types
        self.num_augmented = num_augmented
        self.max_length = int(max_length_sec * sampling_rate)
        self.sampling_rate = sampling_rate

        self.n_original = len(file_paths)
        self._is_whisper_fe = type(feature_extractor).__name__ == \
            "WhisperFeatureExtractor"
        if self._is_whisper_fe:
            logger.info(
                "  Detected WhisperFeatureExtractor: feature extractor will "
                "internally pad mel to 3000 frames (30 s); waveform itself is "
                f"still truncated/padded to {max_length_sec:.2f} s."
            )

        self.sentence_aug = build_sentence_augmenter()
        self.vowel_aug = build_vowel_augmenter()

        logger.info(
            f"  Dataset: {self.n_original} samples + "
            f"{self.n_original * num_augmented} augmented copies = "
            f"{len(self)} total"
        )

    def __len__(self):
        return self.n_original * (1 + self.num_augmented)

    def __getitem__(self, idx):
        import torchaudio

        original_idx = idx % self.n_original
        is_augmented = idx >= self.n_original

        audio_path = self.file_paths[original_idx]
        label = self.labels[original_idx]
        rec_type = self.recording_types[original_idx]

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

        if is_augmented and HAS_AUDIOMENTATIONS:
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
            waveform = torch.cat([waveform,
                                  torch.zeros(self.max_length - waveform.shape[0])])

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
# Data loading + CV folds (unchanged from v8)
# ============================================================================

def load_metadata_with_tts(data_dir, task='detection', recording_type='sentence',
                           max_synth_ratio=None):
    data_path = Path(data_dir)

    if task == 'detection':
        combined_file = data_path / 'combined_detection_metadata.csv'
        fallback_file = data_path / 'detection_metadata.csv'
    else:
        combined_file = data_path / 'combined_classification_metadata.csv'
        fallback_file = data_path / 'classification_metadata.csv'

    meta_file = combined_file if combined_file.exists() else fallback_file
    logger.info(f"Reading metadata from: {meta_file.name}")

    rows = []
    with open(meta_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if recording_type == 'sentence':
        rows = [r for r in rows if r['recording_type'] == 'sentence']
    elif recording_type == 'vowel':
        rows = [r for r in rows if r['recording_type'].startswith('vowel')]

    if not rows:
        raise ValueError(f"No recordings found for type='{recording_type}'")

    if max_synth_ratio is not None:
        orig_rows = [r for r in rows if r.get('source_dir', 'original') != 'synthetic']
        synth_rows = [r for r in rows if r.get('source_dir', 'original') == 'synthetic']
        max_synth = int(len(orig_rows) * max_synth_ratio)
        if len(synth_rows) > max_synth:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(synth_rows), max_synth, replace=False)
            synth_rows = [synth_rows[i] for i in sorted(idx)]
            logger.info(f"Downsampled synthetic: {len(synth_rows)} "
                        f"(max_ratio={max_synth_ratio}, orig={len(orig_rows)})")
        rows = orig_rows + synth_rows

    label_key = 'label' if task == 'detection' else 'pathology'
    unique_labels = sorted(set(r[label_key] for r in rows))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    file_paths, labels, speaker_ids, rec_types, is_synthetic = [], [], [], [], []

    for r in rows:
        fname = r['output_filename']
        rtype = r['recording_type']
        source = r.get('source_dir', 'original')

        if source == 'synthetic':
            if rtype == 'sentence':
                fpath = data_path / 'synthetic' / 'sentences' / fname
            else:
                fpath = data_path / 'synthetic' / 'vowels' / fname
            synth = True
        else:
            if rtype == 'sentence':
                fpath = data_path / 'sentences' / fname
            else:
                fpath = data_path / 'vowels' / fname
            synth = False

        if not fpath.exists():
            logger.debug(f"File not found, skipping: {fpath}")
            continue

        file_paths.append(str(fpath))
        labels.append(label2id[r[label_key]])
        speaker_ids.append(r['speaker_id'])
        rec_types.append(rtype)
        is_synthetic.append(synth)

    n_orig = sum(1 for s in is_synthetic if not s)
    n_synth = sum(1 for s in is_synthetic if s)
    logger.info(f"Loaded {len(file_paths)} samples | original={n_orig} | synthetic={n_synth}")
    logger.info(f"Label distribution: {Counter(labels)}")
    logger.info(f"Labels: {label2id}")

    if len(label2id) != 2 and task == 'detection':
        logger.warning(f"Detection task expects 2 labels, found {len(label2id)}: {label2id}")

    return file_paths, labels, speaker_ids, rec_types, is_synthetic, label2id, id2label


def create_cv_folds_with_tts(speaker_ids, labels, is_synthetic,
                              n_folds=10, seed=42):
    orig_indices = [i for i, s in enumerate(is_synthetic) if not s]
    synth_indices = [i for i, s in enumerate(is_synthetic) if s]

    logger.info(f"CV split: {len(orig_indices)} original + "
                f"{len(synth_indices)} synthetic (always in train)")

    spk_labels = {}
    spk_indices = {}
    for i in orig_indices:
        sid = speaker_ids[i]
        if sid not in spk_labels:
            spk_labels[sid] = labels[i]
            spk_indices[sid] = []
        spk_indices[sid].append(i)

    unique_spks = sorted(spk_labels.keys())
    spk_label_array = [spk_labels[s] for s in unique_spks]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_spk_idx, test_spk_idx in skf.split(unique_spks, spk_label_array):
        train_spks = {unique_spks[i] for i in train_spk_idx}
        test_spks = {unique_spks[i] for i in test_spk_idx}
        train_orig = [j for sid in train_spks for j in spk_indices[sid]]
        test_orig = [j for sid in test_spks for j in spk_indices[sid]]
        train_all = sorted(train_orig + synth_indices)
        folds.append((train_all, sorted(test_orig)))

    t, te = folds[0]
    t_synth = sum(1 for i in t if is_synthetic[i])
    t_orig = len(t) - t_synth
    logger.info(f"Fold 0 example: Train={len(t)} (orig={t_orig}, synth={t_synth}), "
                f"Test={len(te)} (orig only)")

    return folds


# ============================================================================
# Strategy core (NEW in v9)
# ============================================================================

VALID_STRATEGIES = (
    "selected_layers",
    "lora",
    "adapter",
    "bitfit",
    "layernorm_tuning",
    "lora_bitfit",
    "lora_layernorm",
    "adapter_layernorm",
)

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")


class WhisperAdapterBlock(nn.Module):
    """Houlsby-style bottleneck adapter: Linear -> GELU -> Linear (zero-init up)."""

    def __init__(self, d_model, adapter_dim, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(adapter_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x))))


def parse_strategy_layers(spec: str, n_layers: int):
    if spec is None or str(spec).strip() == "":
        raise ValueError("--strategy_layers must be a non-empty comma list")
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    ids = []
    for p in parts:
        try:
            i = int(p)
        except ValueError as e:
            raise ValueError(f"--strategy_layers contains non-integer '{p}'") from e
        if i < 0 or i >= n_layers:
            raise ValueError(
                f"--strategy_layers id {i} out of range [0, {n_layers - 1}]"
            )
        ids.append(i)
    ids = sorted(set(ids))
    if not ids:
        raise ValueError("--strategy_layers parsed empty")
    return ids


def _is_whisper(model):
    enc = getattr(model, "encoder", None)
    return enc is not None and hasattr(enc, "conv1") and hasattr(enc, "conv2") \
        and hasattr(enc, "embed_positions")


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _open_module(mod):
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = True


def _open_layer_full(layer):
    for p in layer.parameters():
        p.requires_grad = True


def _open_bias_in_layer(layer):
    """Open parameters whose leaf name == 'bias' (e.g. q_proj.bias, fc1.bias,
    self_attn_layer_norm.bias, final_layer_norm.bias)."""
    for n, p in layer.named_parameters():
        if n.split(".")[-1] == "bias":
            p.requires_grad = True


def _open_layernorm_in_layer(layer):
    """Open all params belonging to any submodule of type nn.LayerNorm
    OR whose name contains 'layer_norm' / 'layernorm'."""
    # Direct module-type detection (most reliable)
    for sub_name, sub_mod in layer.named_modules():
        if isinstance(sub_mod, nn.LayerNorm):
            for p in sub_mod.parameters(recurse=False):
                p.requires_grad = True
    # Fallback: name-based (covers any oddly-named LN clones)
    for n, p in layer.named_parameters():
        low = n.lower()
        if "layer_norm" in low or "layernorm" in low:
            p.requires_grad = True


def _inject_adapters(model, strategy_layers, adapter_dim, dropout=0.0):
    """Insert bottleneck adapters after each listed encoder.layer's forward
    output via forward hook. Returns the ModuleList of adapters (already
    registered on the model so they move with .cuda())."""
    enc_layers = model.encoder.layers
    d_model = model.config.d_model
    adapters = nn.ModuleList([
        WhisperAdapterBlock(d_model, adapter_dim, dropout=dropout)
        for _ in strategy_layers
    ])
    for p in adapters.parameters():
        p.requires_grad = True
    if hasattr(model, "strategy_adapters"):
        delattr(model, "strategy_adapters")
    model.add_module("strategy_adapters", adapters)

    def make_hook(adapter):
        def hook(module, args, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + adapter(h)
                return (h,) + output[1:]
            return output + adapter(output)
        return hook

    for k, i in enumerate(strategy_layers):
        enc_layers[i].register_forward_hook(make_hook(adapters[k]))
    return adapters


def apply_strategy(
    model,
    strategy: str,
    strategy_layers,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    adapter_dim: int = 64,
    adapter_dropout: float = 0.0,
):
    """Configure the model per strategy. Returns:
        (final_model_to_train, info_dict)

    final_model_to_train may be a PeftModel wrapping the original Whisper
    model. Caller should pass final_model_to_train into Trainer. The
    underlying Whisper model is available at:
        final_model_to_train.base_model.model   (when PEFT-wrapped)
    or just `final_model_to_train` (otherwise).
    """
    if not _is_whisper(model):
        raise ValueError(
            "v9_strategy_ablation script supports ONLY Whisper backbones "
            f"(WhisperForAudioClassification). Got class={model.__class__.__name__}"
        )
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown --strategy '{strategy}'. Valid: {list(VALID_STRATEGIES)}"
        )

    enc_layers = model.encoder.layers
    n_enc = len(enc_layers)
    for i in strategy_layers:
        if not (0 <= i < n_enc):
            raise ValueError(
                f"strategy_layer id {i} out of range [0, {n_enc - 1}]"
            )

    _freeze_all(model)
    # Always-trainable head-side modules
    _open_module(getattr(model, "projector", None))
    _open_module(getattr(model, "classifier", None))
    _open_module(model.encoder.layer_norm)

    info = {
        "strategy": strategy,
        "strategy_layers": list(strategy_layers),
        "added_modules": [],
        "lora": None,
        "adapter": None,
    }

    use_lora = strategy in ("lora", "lora_bitfit", "lora_layernorm")
    use_adapter = strategy in ("adapter", "adapter_layernorm")
    add_bitfit = strategy in ("bitfit", "lora_bitfit")
    add_ln = strategy in ("layernorm_tuning", "lora_layernorm", "adapter_layernorm")
    use_selected = strategy == "selected_layers"

    if use_selected:
        for i in strategy_layers:
            _open_layer_full(enc_layers[i])

    if add_bitfit:
        for i in strategy_layers:
            _open_bias_in_layer(enc_layers[i])

    if add_ln:
        for i in strategy_layers:
            _open_layernorm_in_layer(enc_layers[i])

    if use_adapter:
        adapters = _inject_adapters(
            model, strategy_layers, adapter_dim, dropout=adapter_dropout,
        )
        info["adapter"] = {
            "adapter_dim": adapter_dim,
            "adapter_dropout": adapter_dropout,
            "n_adapters": len(adapters),
        }
        info["added_modules"].append("strategy_adapters")

    final_model = model

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                f"Strategy '{strategy}' requires the 'peft' package. "
                "Install with:\n    pip install peft"
            ) from e

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
            layers_to_transform=list(strategy_layers),
            # WhisperForAudioClassification layout: encoder.layers.{i}.self_attn.q_proj
            # peft uses re.search(rf"{layers_pattern}\.(\d+)\.") on each
            # candidate full module name to match. Using "layers" works.
            layers_pattern="layers",
        )
        info["lora"] = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
            "layers_to_transform": list(strategy_layers),
        }

        final_model = get_peft_model(model, lora_config)

        # get_peft_model freezes EVERYTHING in the base model then opens LoRA
        # adapters. Re-open head-side modules and any strategy-specific
        # extras (bitfit / ln) so they remain trainable.
        underlying = final_model.base_model.model
        _open_module(getattr(underlying, "projector", None))
        _open_module(getattr(underlying, "classifier", None))
        _open_module(underlying.encoder.layer_norm)
        if add_bitfit:
            for i in strategy_layers:
                _open_bias_in_layer(underlying.encoder.layers[i])
        if add_ln:
            for i in strategy_layers:
                _open_layernorm_in_layer(underlying.encoder.layers[i])

    return final_model, info


# ============================================================================
# Feature-layer forward patch (verbatim Whisper branch from v8)
# ============================================================================

def patch_forward_to_use_feature_layer(model, feature_layer_idx):
    """Whisper: head reads encoder.hidden_states[feature_layer_idx + 1].
    For an intermediate layer (i < n_layers-1) we apply encoder.layer_norm
    once so the head sees the same kind of post-LN representation as the
    default last-layer path. For i == n_layers-1, hidden_states[-1] is
    already LN'd so we skip the extra LN."""
    bookkeep = {"feature_layer_used": None, "post_layernorm_applied": False}
    if feature_layer_idx is None or feature_layer_idx < 0:
        logger.info("[feature-layer] feature_layer<0 -> no forward patch.")
        return bookkeep

    # Find underlying whisper model (in case of peft wrap)
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

    underlying.config.output_hidden_states = True
    hidden_states_index = feature_layer_idx + 1
    is_last = (feature_layer_idx == n_layers - 1)
    apply_extra_ln = not is_last
    bookkeep["post_layernorm_applied"] = apply_extra_ln

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
    bookkeep["feature_layer_used"] = feature_layer_idx
    logger.info(
        f"[feature-layer] whisper: head reads encoder layer {feature_layer_idx} "
        f"(hidden_states[{hidden_states_index}]) out of {n_layers}; "
        f"extra encoder.layer_norm = {apply_extra_ln}"
    )
    return bookkeep


# ============================================================================
# Safety check
# ============================================================================

def safety_check_and_log(final_model, strategy, strategy_layers,
                         expected_frozen=(7, 8, 9, 10, 11)):
    """Raise RuntimeError if any param under encoder.layers.{expected_frozen}
    is trainable. Also assert projector + classifier have trainable params.
    Then log a full param audit."""
    underlying = final_model
    if hasattr(final_model, "base_model") and hasattr(final_model.base_model, "model"):
        underlying = final_model.base_model.model
    enc = underlying.encoder

    bad = []
    for i in expected_frozen:
        if i >= len(enc.layers):
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

    # Param audit
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
# Metrics (unchanged from v8)
# ============================================================================

def compute_metrics_detection(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    try:
        auc = roc_auc_score(labels, probs[:, 1]) if probs.shape[1] == 2 else \
              roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    return {'accuracy': acc, 'f1_macro': f1, 'auc': auc}


def evaluate_speaker_level(predictions, test_labels, test_speaker_ids,
                           num_labels, id2label):
    from collections import defaultdict

    probs = torch.softmax(torch.tensor(predictions), dim=-1).numpy()

    spk_probs = defaultdict(list)
    spk_labels = {}
    for i, (sid, label) in enumerate(zip(test_speaker_ids, test_labels)):
        spk_probs[sid].append(probs[i])
        spk_labels[sid] = label

    spk_final_probs = []
    spk_final_labels = []
    for sid in sorted(spk_probs.keys()):
        avg_prob = np.mean(spk_probs[sid], axis=0)
        spk_final_probs.append(avg_prob)
        spk_final_labels.append(spk_labels[sid])

    spk_final_probs = np.array(spk_final_probs)
    spk_final_labels = np.array(spk_final_labels)

    spk_preds_default = np.argmax(spk_final_probs, axis=-1)
    acc_default = accuracy_score(spk_final_labels, spk_preds_default)
    f1_default = f1_score(spk_final_labels, spk_preds_default, average='macro')

    try:
        if spk_final_probs.shape[1] == 2:
            auc = roc_auc_score(spk_final_labels, spk_final_probs[:, 1])
        else:
            auc = roc_auc_score(spk_final_labels, spk_final_probs,
                                multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    best_threshold = 0.5
    best_f1 = f1_default
    best_acc = acc_default
    if spk_final_probs.shape[1] == 2:
        pos_probs = spk_final_probs[:, 1]
        for t in np.arange(0.25, 0.75, 0.01):
            preds_t = (pos_probs >= t).astype(int)
            f1_t = f1_score(spk_final_labels, preds_t, average='macro')
            if f1_t > best_f1:
                best_f1 = f1_t
                best_acc = accuracy_score(spk_final_labels, preds_t)
                best_threshold = float(t)

    return {
        'spk_accuracy': float(acc_default),
        'spk_f1_macro': float(f1_default),
        'spk_auc': float(auc),
        'spk_accuracy_opt': float(best_acc),
        'spk_f1_macro_opt': float(best_f1),
        'spk_best_threshold': float(best_threshold),
        'n_speakers': len(spk_final_labels),
    }


def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}


# ============================================================================
# Single fold training
# ============================================================================

def train_single_fold(
    fold_idx,
    train_files, train_labels, train_rec_types,
    test_files,  test_labels,  test_rec_types,
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
    strategy: str,
    strategy_layers,
    feature_layer: int,
    lora_r, lora_alpha, lora_dropout,
    adapter_dim, adapter_dropout,
):
    if reseed_each_fold:
        fold_seed = seed + fold_idx * 1000
        set_deterministic(fold_seed)
    else:
        fold_seed = seed

    fold_dir = Path(output_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    n_synth_train = sum(1 for f in train_files if 'synthetic' in f)
    n_orig_train = len(train_files) - n_synth_train
    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_idx}: Train={len(train_files)} "
                f"(orig={n_orig_train}, synth={n_synth_train}), "
                f"Test={len(test_files)}")
    logger.info(f"{'='*60}")

    train_dataset = VoiceDisorderDatasetTTS(
        train_files, train_labels, feature_extractor,
        recording_types=train_rec_types,
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
            "v9_strategy_ablation requires a Whisper model "
            f"(got class={model.__class__.__name__}). Use --model_name openai/whisper-small."
        )

    final_model, strategy_info = apply_strategy(
        model,
        strategy=strategy,
        strategy_layers=strategy_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
    )

    feature_layer_book = patch_forward_to_use_feature_layer(
        final_model, feature_layer
    )

    audit = safety_check_and_log(final_model, strategy, strategy_layers)

    orig_labels = [l for f, l in zip(train_files, train_labels)
                   if 'synthetic' not in str(f)]
    weight_source = orig_labels if orig_labels else train_labels
    label_counts = Counter(weight_source)
    total = sum(label_counts.values())

    raw_weights = [total / (num_labels * label_counts.get(i, 1))
                   for i in range(num_labels)]
    class_weights = torch.tensor(
        [w ** 1.0 for w in raw_weights],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.mean()
    logger.info(f"Class weights (from {len(weight_source)} original samples): "
                f"{class_weights.tolist()}")
    logger.info(f"  Raw inverse-freq: {[f'{w:.3f}' for w in raw_weights]}")

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
                loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
                loss = loss_fn(logits, labels_t)

            return (loss, outputs) if return_outputs else loss

    steps_per_epoch = max(
        len(train_dataset) // (batch_size * gradient_accumulation_steps), 1
    )
    eval_steps = steps_per_epoch
    logger.info(f"Steps per epoch: {steps_per_epoch}, eval every {eval_steps} steps")

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
        # PEFT (LoRA) wraps the model so its forward signature is
        # (*args, **kwargs); HF Trainer's auto-detection of label fields then
        # fails, eval skips compute_metrics, and metric_for_best_model raises
        # KeyError('eval_f1_macro'). Setting label_names explicitly fixes it
        # and is harmless for the non-PEFT strategies.
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
    logger.info(f"Fold {fold_idx} recording-level results: {results}")

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    if task == 'detection' and test_speaker_ids is not None:
        spk_results = evaluate_speaker_level(
            predictions.predictions, test_labels, test_speaker_ids,
            num_labels, id2label,
        )
        results.update(spk_results)
        logger.info(f"Fold {fold_idx} speaker-level (threshold=0.5): "
                    f"Acc={spk_results['spk_accuracy']:.3f}, "
                    f"F1={spk_results['spk_f1_macro']:.3f}, "
                    f"AUC={spk_results['spk_auc']:.3f}")
        logger.info(f"Fold {fold_idx} speaker-level (threshold={spk_results['spk_best_threshold']:.2f}): "
                    f"Acc={spk_results['spk_accuracy_opt']:.3f}, "
                    f"F1={spk_results['spk_f1_macro_opt']:.3f}")

    results['strategy'] = strategy
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
    logger.info(f"Task: {args.task} | Labels: {num_labels} | Model: {args.model_name}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    folds = create_cv_folds_with_tts(
        speaker_ids, labels, is_synthetic,
        n_folds=args.n_folds, seed=args.seed
    )

    if args.max_folds is not None:
        if args.max_folds <= 0:
            raise ValueError("--max_folds must be positive")
        if args.max_folds > len(folds):
            raise ValueError(
                f"--max_folds ({args.max_folds}) cannot exceed "
                f"n_folds ({args.n_folds})"
            )
        folds_to_run = folds[:args.max_folds]
        logger.info(
            f"--max_folds={args.max_folds} set: still built {len(folds)}-fold "
            f"split, but only running fold_0 .. fold_{args.max_folds - 1}"
        )
    else:
        folds_to_run = folds

    # Whisper-small has 12 encoder layers; use a safe ceiling for parsing.
    strategy_layers = parse_strategy_layers(args.strategy_layers, n_layers=64)

    all_results = []
    backbone_type_observed = None
    for fold_idx, (train_idx, test_idx) in enumerate(folds_to_run):
        cached_path = output_dir / f'fold_{fold_idx}' / 'results.json'
        if cached_path.exists():
            with open(cached_path) as f:
                cached = json.load(f)
            logger.info(
                f"[skip] fold_{fold_idx} already finished, reusing {cached_path}"
            )
            if backbone_type_observed is None:
                backbone_type_observed = cached.get('backbone_type', 'whisper')
            all_results.append(cached)
            continue
        results = train_single_fold(
            fold_idx=fold_idx,
            train_files=[file_paths[i] for i in train_idx],
            train_labels=[labels[i] for i in train_idx],
            train_rec_types=[rec_types[i] for i in train_idx],
            test_files=[file_paths[i] for i in test_idx],
            test_labels=[labels[i] for i in test_idx],
            test_rec_types=[rec_types[i] for i in test_idx],
            test_speaker_ids=[speaker_ids[i] for i in test_idx],
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
            strategy=args.strategy,
            strategy_layers=strategy_layers,
            feature_layer=args.feature_layer,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            adapter_dim=args.adapter_dim,
            adapter_dropout=args.adapter_dropout,
        )
        if backbone_type_observed is None:
            backbone_type_observed = results.get('backbone_type', 'whisper')
        all_results.append(results)

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
            mean, std = np.mean(values), np.std(values)
            short_key = key.replace('eval_', '')
            summary[short_key] = {'mean': float(mean), 'std': float(std)}
            logger.info(f"  {short_key}: {mean:.3f} +/- {std:.3f}")

    spk_keys = ['spk_accuracy', 'spk_f1_macro', 'spk_auc']
    has_spk = any(k in all_results[0] for k in spk_keys)
    if has_spk:
        logger.info("\n  --- Speaker-level metrics (threshold=0.5) ---")
        for key in spk_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                mean, std = np.mean(values), np.std(values)
                summary[key] = {'mean': float(mean), 'std': float(std)}
                logger.info(f"  {key}: {mean:.3f} +/- {std:.3f}")

        opt_keys = ['spk_accuracy_opt', 'spk_f1_macro_opt']
        logger.info("\n  --- Speaker-level metrics (optimal threshold per fold) ---")
        for key in opt_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                mean, std = np.mean(values), np.std(values)
                summary[key] = {'mean': float(mean), 'std': float(std)}
                logger.info(f"  {key}: {mean:.3f} +/- {std:.3f}")

        thresholds = [r.get('spk_best_threshold', 0.5) for r in all_results]
        logger.info(f"  avg optimal threshold: {np.mean(thresholds):.3f} "
                    f"(range: {min(thresholds):.2f} - {max(thresholds):.2f})")

    final_summary = {
        'task': args.task,
        'model': args.model_name,
        'backbone_type': backbone_type_observed,
        'recording_type': args.recording_type,
        'n_folds': args.n_folds,
        'max_folds': args.max_folds,
        'num_folds_run': len(all_results),
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'num_augmented': args.num_augmented,
        'max_synth_ratio': args.max_synth_ratio,
        'strategy': args.strategy,
        'strategy_layers': strategy_layers,
        'feature_layer': args.feature_layer,
        'feature_layer_post_ln': any(
            r.get('feature_layer_post_ln', False) for r in all_results
        ),
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'adapter_dim': args.adapter_dim,
        'adapter_dropout': args.adapter_dropout,
        'results': summary,
        'fold_results': all_results,
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
        description='TTS Training v9 - Strategy ablation on Whisper-small '
                    '(fixed layer range, feature_layer plumbing).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        '--strategy', type=str, required=True,
        choices=list(VALID_STRATEGIES),
        help='Which fine-tuning strategy to apply on encoder.layers.{strategy_layers}.',
    )
    parser.add_argument(
        '--strategy_layers', type=str, default='0,1,2,3,4,5,6',
        help='Comma-separated encoder layer ids the strategy acts on. '
             'All layers outside this set are FROZEN (and safety-checked).',
    )
    parser.add_argument(
        '--feature_layer', type=int, default=6,
        help='Which encoder layer\'s output feeds the classifier head. '
             'Default 6.',
    )

    parser.add_argument('--lora_r',         type=int, default=8)
    parser.add_argument('--lora_alpha',     type=int, default=16)
    parser.add_argument('--lora_dropout',   type=float, default=0.05)
    parser.add_argument('--adapter_dim',    type=int, default=64)
    parser.add_argument('--adapter_dropout', type=float, default=0.0)

    args = parser.parse_args()

    set_deterministic(args.seed)

    multiplier = 1 + args.num_augmented
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("\n--- Config ---")
    logger.info(f"  Task:            {args.task}")
    logger.info(f"  Rec type:        {args.recording_type}")
    logger.info(f"  Model:           {args.model_name}")
    logger.info(f"  Epochs:          {args.num_epochs}")
    logger.info(f"  Effective BS:    {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  LR:              {args.lr}")
    logger.info(f"  Warmup:          {args.warmup_ratio}")
    logger.info(f"  Scheduler:       cosine")
    logger.info(f"  Label smooth:    0.05")
    logger.info(f"  Aug copies:      {args.num_augmented} ({multiplier}x data)")
    logger.info(f"  Synth ratio:     {args.max_synth_ratio or 'unlimited'}")
    logger.info(f"  Auglib:          {'audiomentations' if HAS_AUDIOMENTATIONS else 'DISABLED'}")
    logger.info(f"  Best metric:     {args.metric_for_best_model}")
    logger.info(f"  Reseed fold:     {args.reseed_each_fold}")
    logger.info(f"  Strategy:        {args.strategy}")
    logger.info(f"  Strategy layers: {args.strategy_layers}")
    logger.info(f"  Feature layer:   {args.feature_layer}")
    if args.strategy in ('lora', 'lora_bitfit', 'lora_layernorm'):
        logger.info(f"  LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    if args.strategy in ('adapter', 'adapter_layernorm'):
        logger.info(f"  Adapter dim/dropout:  {args.adapter_dim}/{args.adapter_dropout}")
    logger.info(f"  n_folds:         {args.n_folds}")
    logger.info(f"  max_folds:       {args.max_folds}")

    run_experiment(args)


if __name__ == '__main__':
    main()
