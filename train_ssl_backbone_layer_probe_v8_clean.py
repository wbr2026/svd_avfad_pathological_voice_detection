#!/usr/bin/env python3

import os
import csv
import json
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
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parent.parent

DATASET_PRESETS = {
    'svd': {
        'data_dir': _SCRIPT_DIR / 'svd_cleaned_paper_aligned',
        'max_length_sec': 5.0,
    },
    'avfad': {
        'data_dir': _WORKSPACE_ROOT / 'avfad_paper_aligned',
        'max_length_sec': 24.0,
    },
}


SUPPORTED_BACKBONES = ("hubert", "wavlm", "wav2vec2", "whisper")
WAVEFORM_BACKBONES = ("hubert", "wavlm", "wav2vec2")  # consume input_values
SPECTROGRAM_BACKBONES = ("whisper",)                  # consume input_features


# ============================================================================
# Determinism helpers (identical to v8 reseed f1best)
# ============================================================================

def set_deterministic(seed: int):
    """Seed every RNG we touch and force deterministic cuDNN/cuBLAS.

    Same caveats as the v8 baseline: we deliberately skip
    torch.use_deterministic_algorithms(True) because it crashes HuBERT's
    SpecAugment _mask_hidden_states. cuDNN determinism + fixed RNG seeds +
    fixed data_seed keep run-to-run variation within ~1%.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Augmentation (identical to v8 baseline)
# ============================================================================

def build_sentence_augmenter():
    """Stronger augmentation for sentences."""
    if not HAS_AUDIOMENTATIONS:
        return None
    return Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    ])

def build_vowel_augmenter():
    """Weaker augmentation for vowels."""
    if not HAS_AUDIOMENTATIONS:
        return None
    return Compose([
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
    ])


# ============================================================================
# Dataset (identical to v8 baseline; backbone-agnostic input fields)
# ============================================================================

class VoiceDisorderDatasetTTS(Dataset):
    """Original (clean) + augmented copies + TTS synthetic samples.

    Index mapping (n_original=N, num_augmented=K):
      idx 0..N-1                  -> original sample i, NO augmentation
      idx N..(K+1)*N-1            -> augmented copy of original sample i
      Total __len__ = N * (1 + K)

    Test set must use num_augmented=0 (originals only, clean).

    The dataset is backbone-agnostic with respect to which input field the
    feature extractor produces. We forward whatever the feature extractor
    returns:
      - HuBERT / WavLM / wav2vec2 feature extractors -> 'input_values'
      - Whisper feature extractor                    -> 'input_features'
      - Either may also return 'attention_mask'      -> 'attention_mask'
    Everything else (waveform load, mono mixdown, resample, augmentation,
    crop/pad to max_length) is identical to the v8 baseline.
    """

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

        # Backbone-specific padding policy (kept implicit via FE defaults):
        #   Wav2Vec2FeatureExtractor (used by HuBERT / WavLM / wav2vec2):
        #     default padding=False -> input_values is left at the waveform
        #     length we already cropped/padded above (e.g. 5s = 80000).
        #   WhisperFeatureExtractor:
        #     default padding="max_length" -> waveform is internally padded
        #     to n_samples=480000 (30s) before mel computation, producing the
        #     3000-frame mel that the Whisper encoder hard-requires.
        # We deliberately do NOT pass padding=False here, because that would
        # override Whisper's default and produce a 500-frame mel, which the
        # Whisper encoder rejects ("expects ... 3000, but found 500").
        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        item = {'labels': torch.tensor(label, dtype=torch.long)}
        if 'input_values' in inputs:
            item['input_values'] = inputs['input_values'].squeeze(0)
        if 'input_features' in inputs:
            item['input_features'] = inputs['input_features'].squeeze(0)
        if 'attention_mask' in inputs:
            item['attention_mask'] = inputs['attention_mask'].squeeze(0)
        return item


# ============================================================================
# Data loading (identical to v8 baseline)
# ============================================================================

def load_metadata_with_tts(data_dir, task='detection', recording_type='sentence',
                           max_synth_ratio=None, original_only=True):
    data_path = Path(data_dir)

    if task == 'detection':
        combined_file = data_path / 'combined_detection_metadata.csv'
        fallback_file = data_path / 'detection_metadata.csv'
    else:
        combined_file = data_path / 'combined_classification_metadata.csv'
        fallback_file = data_path / 'classification_metadata.csv'

    if original_only:
        meta_file = fallback_file
        if not meta_file.exists():
            raise FileNotFoundError(
                f"--original_only requires {meta_file.name} under {data_path}"
            )
        logger.info(
            f"Reading metadata from: {meta_file.name} (--original_only; "
            f"ignoring combined_* even if present)"
        )
    else:
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

    if original_only:
        before = len(rows)
        rows = [r for r in rows if r.get('source_dir', 'original') != 'synthetic']
        logger.info(
            f"Original-only filter: kept {len(rows)}/{before} rows "
            f"(dropped synthetic/TTS)"
        )
        if not rows:
            raise ValueError(
                f"No original recordings left after --original_only filter "
                f"for type='{recording_type}'"
            )
    elif max_synth_ratio is not None:
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


# ============================================================================
# CV folds — synthetic samples always in TRAIN (identical to v8 baseline)
# ============================================================================

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
# Backbone helpers (load + transformer-layer count)
# ============================================================================

def get_num_transformer_layers(backbone_type: str, model_name: str) -> int:
    """Read the backbone's transformer-layer count from its config.

    HuBERT / WavLM / wav2vec2 expose ``num_hidden_layers`` (the number of
    transformer encoder layers). Whisper exposes ``encoder_layers``. We
    only count *transformer* layers; the pre-transformer feature
    projection / conv embed layer is NOT counted.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    if backbone_type in WAVEFORM_BACKBONES:
        n = getattr(cfg, "num_hidden_layers", None)
    elif backbone_type == "whisper":
        # WhisperConfig does not have num_hidden_layers; use encoder_layers.
        n = getattr(cfg, "encoder_layers", None)
        if n is None:
            n = getattr(cfg, "num_hidden_layers", None)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")
    if n is None:
        raise ValueError(
            f"Could not read transformer layer count from config for "
            f"backbone_type={backbone_type}, model_name={model_name}"
        )
    return int(n)


def assert_twelve_layer_backbone(backbone_type: str, model_name: str):
    """Hard-fail when the backbone is not 12 transformer layers.

    The whole point of this script is a strict 12-layer probe across the
    four supported backbone families. If the user picks a non-12-layer
    checkpoint (e.g. ``whisper-base`` has 6 encoder layers) we abort
    immediately so the user can switch to a 12-layer checkpoint.
    """
    n = get_num_transformer_layers(backbone_type, model_name)
    if n != 12:
        raise ValueError(
            f"backbone_type={backbone_type} model_name={model_name} exposes "
            f"{n} transformer layers, but this layer-probe script is "
            f"designed for a strict 12-layer comparison against "
            f"HuBERT-base / WavLM-base / wav2vec2-base. "
            f"Please pick a 12-layer checkpoint. For Whisper, use "
            f"openai/whisper-small (12 encoder layers) or an equivalent "
            f"local 12-layer checkpoint."
        )


def load_backbone(backbone_type: str, model_name: str):
    """Load only the *encoder* part of the requested SSL backbone.

    For HuBERT / WavLM / wav2vec2, ``AutoModel.from_pretrained`` already
    returns the encoder-only model (HubertModel / WavLMModel /
    Wav2Vec2Model). For Whisper, ``AutoModel`` would return the full
    encoder-decoder; we explicitly take ``WhisperModel(...).encoder`` and
    let the decoder be garbage collected.
    """
    if backbone_type in WAVEFORM_BACKBONES:
        backbone = AutoModel.from_pretrained(model_name)
    elif backbone_type == "whisper":
        # Local import keeps this import optional for non-Whisper runs.
        from transformers import WhisperModel
        whisper_full = WhisperModel.from_pretrained(model_name)
        backbone = whisper_full.encoder
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")
    return backbone


def get_backbone_hidden_size(backbone_type: str, backbone) -> int:
    """Return the channel dimension at the transformer-layer output."""
    cfg = backbone.config
    if backbone_type in WAVEFORM_BACKBONES:
        h = getattr(cfg, "hidden_size", None)
    elif backbone_type == "whisper":
        # Whisper encoder outputs at d_model.
        h = getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size", None)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")
    if h is None:
        raise ValueError(
            f"Could not read hidden_size from backbone config "
            f"(backbone_type={backbone_type})"
        )
    return int(h)


# ============================================================================
# Frozen SSL-backbone layer-probe model
# ============================================================================

class FrozenSSLBackboneLayerProbeClassifier(nn.Module):
    """Probe a single transformer layer of a frozen SSL backbone.

    The backbone (HuBERT / WavLM / wav2vec2 / Whisper-encoder) is loaded
    via ``load_backbone`` and ALL of its parameters are frozen
    (``requires_grad=False``, ``eval`` mode). We request
    ``output_hidden_states=True`` and pick exactly one transformer layer:

        selected = outputs.hidden_states[ssl_layer + 1]

    because ``hidden_states[0]`` is the pre-transformer (feature
    projection / conv embed) representation, and ``hidden_states[i]``
    for ``i in 1..L`` corresponds to transformer layer ``i``. So
    ``--ssl_layer 0`` reads the output of transformer layer 1.

    Forward dispatch:
      - ``backbone_type in {hubert, wavlm, wav2vec2}`` -> ``input_values``
      - ``backbone_type == 'whisper'``                 -> ``input_features``

    Pooling: mean over the time dimension (kept fixed across backbones and
    layers so that only the layer index varies).

    Head: dropout + ``Linear(hidden_size, num_labels)`` — identical
    structure across all backbones / layers.

    Output: ``SequenceClassifierOutput(loss=None, logits=...)`` so HF
    Trainer.compute_loss / predict / evaluate all work unchanged.
    """

    def __init__(
        self,
        backbone_type: str,
        model_name: str,
        num_labels: int,
        ssl_layer: int,
        hidden_dropout: float = 0.1,
        id2label=None,
        label2id=None,
    ):
        super().__init__()
        if backbone_type not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone_type must be one of {SUPPORTED_BACKBONES}, "
                f"got {backbone_type!r}"
            )
        if not (0 <= ssl_layer <= 11):
            raise ValueError(
                f"ssl_layer must be in [0, 11] (got {ssl_layer}). "
                f"This indexes transformer layers 0..11; the code reads "
                f"hidden_states[ssl_layer + 1]."
            )

        self.backbone_type = backbone_type
        self.model_name = model_name
        self.num_labels = num_labels
        self.ssl_layer = ssl_layer
        self.hidden_state_index = ssl_layer + 1  # we read this directly

        self.backbone = load_backbone(backbone_type, model_name)

        # Strict freeze: every backbone parameter is a non-trainable
        # buffer of weights for inference. eval() also disables
        # dropout / SpecAugment-style variability inside the backbone.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.hidden_size = get_backbone_hidden_size(backbone_type, self.backbone)

        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # HF expects a config-ish attribute on custom models; this also
        # makes id2label / label2id available downstream if needed.
        self.config = self.backbone.config
        if id2label is not None:
            self.config.id2label = id2label
        if label2id is not None:
            self.config.label2id = label2id
        self.config.num_labels = num_labels

    def train(self, mode: bool = True):
        """Override so the frozen backbone stays in eval mode even under .train().

        This guarantees no dropout / SpecAugment randomness inside the
        backbone, which is what we want for a clean layer-probe
        comparison.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        input_values=None,
        input_features=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        with torch.no_grad():
            if self.backbone_type in WAVEFORM_BACKBONES:
                if input_values is None:
                    raise ValueError(
                        f"backbone_type={self.backbone_type} expects "
                        f"'input_values', but none was provided."
                    )
                outputs = self.backbone(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            elif self.backbone_type == "whisper":
                if input_features is None:
                    raise ValueError(
                        "backbone_type=whisper expects 'input_features', "
                        "but none was provided. Make sure you used a "
                        "Whisper feature extractor."
                    )
                # Whisper encoder does not consume a frame-level
                # attention_mask in the same way wav2vec2 does, so we
                # only forward input_features.
                outputs = self.backbone(
                    input_features=input_features,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                raise ValueError(
                    f"Unsupported backbone_type: {self.backbone_type}"
                )

        hidden_states = outputs.hidden_states
        if not (0 <= self.hidden_state_index < len(hidden_states)):
            raise IndexError(
                f"hidden_state_index={self.hidden_state_index} is out of "
                f"range [0, {len(hidden_states)-1}]. The backbone exposed "
                f"{len(hidden_states)} hidden states (= 1 + transformer "
                f"layers). Did you load a 12-layer checkpoint?"
            )
        selected = hidden_states[self.hidden_state_index]

        # Mean pooling over time (kept fixed for all layers / backbones).
        # Only valid for waveform backbones whose attention_mask matches
        # the time dim of the transformer output. For Whisper we
        # (correctly) fall through to the unmasked mean.
        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.shape[1] == selected.shape[1]
        ):
            mask = attention_mask.unsqueeze(-1).to(selected.dtype)
            pooled = (selected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = selected.mean(dim=1)

        # The classifier is trainable (cast to backbone dtype if needed).
        pooled = pooled.to(self.classifier.weight.dtype)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None  # WeightedTrainer.compute_loss handles loss computation
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def log_param_audit(model: FrozenSSLBackboneLayerProbeClassifier):
    """Verify the freeze and report trainable parameter location.

    This is the layer-probe contract: the backbone must be 100% frozen
    and the only trainable parameters must live under the classifier
    head (dropout has no params; ``self.classifier`` is the only trainable
    submodule). Both invariants are hard-asserted.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )

    head_total = sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith("backbone.")
    )
    head_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith("backbone.") and p.requires_grad
    )

    logger.info(
        f"  Params: total={total/1e6:.3f}M  trainable={trainable/1e6:.3f}M"
    )
    logger.info(
        f"   - {model.backbone_type} backbone: "
        f"backbone_total={backbone_total/1e6:.3f}M, "
        f"backbone_trainable={backbone_trainable} (must be 0)"
    )
    logger.info(
        f"   - Classifier head: head_total={head_total/1e6:.3f}M, "
        f"head_trainable={head_trainable/1e6:.3f}M"
    )

    if backbone_trainable != 0:
        bad = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
        raise RuntimeError(
            f"{model.backbone_type} backbone is NOT fully frozen; "
            f"{len(bad)} params still require grad: {bad[:5]}..."
        )
    if trainable != head_trainable:
        raise RuntimeError(
            "Trainable params leak outside the classifier head."
        )


# ============================================================================
# Metrics (identical to v8 baseline)
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
# Single fold training (layer-probe variant)
# ============================================================================

def train_single_fold(
    fold_idx,
    train_files, train_labels, train_rec_types,
    test_files,  test_labels,  test_rec_types,
    test_speaker_ids,
    backbone_type, model_name, feature_extractor,
    num_labels, id2label, label2id,
    task, output_dir,
    num_epochs, batch_size, lr, warmup_ratio,
    max_length_sec, gradient_accumulation_steps,
    num_augmented,
    seed,
    reseed_each_fold: bool,
    metric_for_best_model: str,
    ssl_layer: int,
    label_smoothing: float,
    mixup_prob: float,
    use_class_weights: bool,
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

    # Frozen layer-probe model (only the classifier head is trained).
    model = FrozenSSLBackboneLayerProbeClassifier(
        backbone_type=backbone_type,
        model_name=model_name,
        num_labels=num_labels,
        ssl_layer=ssl_layer,
        id2label=id2label,
        label2id=label2id,
    )
    log_param_audit(model)

    if use_class_weights:
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
    else:
        class_weights = None
        logger.info("Class weights: DISABLED (probe-friendly default; "
                    "pass --use_class_weights to re-enable)")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_t = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            device = logits.device

            weight = class_weights.to(device) if class_weights is not None else None

            do_mixup = (
                mixup_prob > 0
                and model.training
                and torch.rand(1).item() < mixup_prob
            )

            if do_mixup:
                lam = torch.distributions.Beta(0.4, 0.4).sample().to(device)
                bs = logits.size(0)
                index = torch.randperm(bs, device=device)
                loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='none')
                loss = lam * loss_fn(logits, labels_t) + \
                       (1 - lam) * loss_fn(logits, labels_t[index])
                loss = loss.mean()
            else:
                loss_fn = nn.CrossEntropyLoss(weight=weight)
                loss = loss_fn(logits, labels_t)

            return (loss, outputs) if return_outputs else loss

    steps_per_epoch = max(
        len(train_dataset) // (batch_size * gradient_accumulation_steps), 1
    )
    eval_steps = steps_per_epoch
    logger.info(f"Steps per epoch: {steps_per_epoch}, "
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
        label_smoothing_factor=label_smoothing,
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
    )

    compute_fn = compute_metrics_detection if task == 'detection' \
                 else compute_metrics_classification

    trainer = WeightedTrainer(
        model=model,
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

    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        test_labels, preds,
        target_names=[id2label[i] for i in range(num_labels)],
        output_dict=True,
    )
    with open(fold_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    del model, trainer
    torch.cuda.empty_cache()

    return results


# ============================================================================
# Per-layer experiment runner
# ============================================================================

def _aggregate_summary(all_results, args, ssl_layer):
    """Build the per-layer mean/std summary used in final_results.json."""
    metric_keys = ['eval_accuracy', 'eval_f1_macro']
    if args.task == 'detection':
        metric_keys.append('eval_auc')

    summary = {}
    for key in metric_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            short_key = key.replace('eval_', '')
            summary[short_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }

    spk_keys = ['spk_accuracy', 'spk_f1_macro', 'spk_auc',
                'spk_accuracy_opt', 'spk_f1_macro_opt']
    for key in spk_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }

    final_summary = {
        'task': args.task,
        'backbone_type': args.backbone_type,
        'model': args.model_name,
        'model_name': args.model_name,
        'recording_type': args.recording_type,
        'n_folds': args.n_folds,
        'max_folds': args.max_folds,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'num_augmented': args.num_augmented,
        'max_synth_ratio': args.max_synth_ratio,
        'ssl_layer': ssl_layer,
        # Backwards-compat alias for old tooling that read 'hubert_layer'.
        'hubert_layer': ssl_layer,
        'hidden_state_index': ssl_layer + 1,
        'backbone_frozen': True,
        # Backwards-compat alias for old tooling that read 'hubert_frozen'.
        'hubert_frozen': True,
        'pooling': 'mean',
        'metric_for_best_model': args.metric_for_best_model,
        'reseed_each_fold': args.reseed_each_fold,
        'seed': args.seed,
        'results': summary,
        'fold_results': all_results,
    }
    return final_summary


def run_layer_experiment(args, ssl_layer, file_paths, labels, speaker_ids,
                         rec_types, is_synthetic, label2id, id2label,
                         feature_extractor, folds):
    """Run all (capped) folds for a single transformer layer of the SSL backbone."""
    layer_dir = Path(args.output_dir) / f'layer_{ssl_layer}'
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Re-pin the seed at the start of each layer so layer N's RNG is
    # independent of layer N-1's RNG.
    set_deterministic(args.seed)

    logger.info("\n" + "=" * 60)
    logger.info(f"Layer Probe: {args.backbone_type} transformer layer {ssl_layer}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Using hidden_states[{ssl_layer + 1}]")
    logger.info(f"Backbone frozen: yes")
    logger.info(f"Pooling: mean")
    logger.info(f"Trainable parameters: classifier only")
    logger.info("=" * 60)

    num_labels = len(label2id)

    # Save per-layer config for reproducibility / debugging.
    with open(layer_dir / 'experiment_config.json', 'w') as f:
        cfg = vars(args).copy()
        cfg['ssl_layer'] = ssl_layer
        cfg['hidden_state_index'] = ssl_layer + 1
        json.dump(cfg, f, indent=2)

    n_folds_to_run = len(folds)
    if args.max_folds is not None:
        n_folds_to_run = min(args.max_folds, len(folds))
        logger.info(
            f"max_folds={args.max_folds}: running fold_0..fold_{n_folds_to_run-1} "
            f"out of {len(folds)} total folds (full split kept for alignment)."
        )

    all_results = []
    for fold_idx in range(n_folds_to_run):
        # Reuse a previously-computed fold result if requested. Because the
        # fold split is built deterministically from (n_folds, seed) and
        # the per-fold RNG is fold_seed = seed + fold_idx*1000 when
        # --reseed_each_fold is on, fold_idx's training is reproducible
        # across runs as long as data + model + hyper-parameters are
        # unchanged. So we can safely load layer_X/fold_Y/results.json
        # from a previous screening run instead of retraining.
        existing_results_json = layer_dir / f'fold_{fold_idx}' / 'results.json'
        if args.skip_existing_folds and existing_results_json.exists():
            with open(existing_results_json, 'r') as f:
                cached = json.load(f)
            logger.info(
                f"\n[skip_existing_folds] Layer {ssl_layer} fold "
                f"{fold_idx}: reusing {existing_results_json}"
            )
            all_results.append(cached)
            continue

        train_idx, test_idx = folds[fold_idx]
        results = train_single_fold(
            fold_idx=fold_idx,
            train_files=[file_paths[i] for i in train_idx],
            train_labels=[labels[i] for i in train_idx],
            train_rec_types=[rec_types[i] for i in train_idx],
            test_files=[file_paths[i] for i in test_idx],
            test_labels=[labels[i] for i in test_idx],
            test_rec_types=[rec_types[i] for i in test_idx],
            test_speaker_ids=[speaker_ids[i] for i in test_idx],
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            feature_extractor=feature_extractor,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            task=args.task,
            output_dir=str(layer_dir),
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
            ssl_layer=ssl_layer,
            label_smoothing=args.label_smoothing,
            mixup_prob=args.mixup_prob,
            use_class_weights=args.use_class_weights,
        )
        all_results.append(results)

    logger.info("\n" + "=" * 60)
    logger.info(f"LAYER {ssl_layer} FINAL RESULTS "
                f"(mean +/- std across {len(all_results)} folds)")
    logger.info("=" * 60)

    final_summary = _aggregate_summary(all_results, args, ssl_layer)
    summary = final_summary['results']

    rec_keys = ['accuracy', 'f1_macro']
    if args.task == 'detection':
        rec_keys.append('auc')
    for key in rec_keys:
        if key in summary:
            logger.info(f"  {key}: {summary[key]['mean']:.3f} +/- "
                        f"{summary[key]['std']:.3f}")

    if any(k in all_results[0] for k in ['spk_accuracy', 'spk_f1_macro', 'spk_auc']):
        logger.info("\n  --- Speaker-level metrics (threshold=0.5) ---")
        for key in ['spk_accuracy', 'spk_f1_macro', 'spk_auc']:
            if key in summary:
                logger.info(f"  {key}: {summary[key]['mean']:.3f} +/- "
                            f"{summary[key]['std']:.3f}")

        logger.info("\n  --- Speaker-level metrics (optimal threshold per fold) ---")
        for key in ['spk_accuracy_opt', 'spk_f1_macro_opt']:
            if key in summary:
                logger.info(f"  {key}: {summary[key]['mean']:.3f} +/- "
                            f"{summary[key]['std']:.3f}")

        thresholds = [r.get('spk_best_threshold', 0.5) for r in all_results]
        logger.info(f"  avg optimal threshold: {np.mean(thresholds):.3f} "
                    f"(range: {min(thresholds):.2f} - {max(thresholds):.2f})")

    with open(layer_dir / 'final_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    logger.info(f"\nLayer {ssl_layer} results saved to "
                f"{layer_dir / 'final_results.json'}")

    return final_summary


# ============================================================================
# Cross-layer summary
# ============================================================================

SUMMARY_COLUMNS = [
    'backbone_type',
    'model_name',
    'ssl_layer',
    # 'layer' is kept (== ssl_layer) for backwards-compatible plotting.
    'layer',
    'hidden_state_index',
    'recording_accuracy_mean',
    'recording_accuracy_std',
    'recording_f1_macro_mean',
    'recording_f1_macro_std',
    'recording_auc_mean',
    'recording_auc_std',
    'spk_accuracy_mean',
    'spk_accuracy_std',
    'spk_f1_macro_mean',
    'spk_f1_macro_std',
    'spk_auc_mean',
    'spk_auc_std',
    'spk_accuracy_opt_mean',
    'spk_accuracy_opt_std',
    'spk_f1_macro_opt_mean',
    'spk_f1_macro_opt_std',
]


def _row_from_layer_summary(layer_summary):
    s = layer_summary['results']

    def get(key, stat):
        return s.get(key, {}).get(stat)

    return {
        'backbone_type': layer_summary.get('backbone_type'),
        'model_name': layer_summary.get('model_name'),
        'ssl_layer': layer_summary['ssl_layer'],
        'layer': layer_summary['ssl_layer'],
        'hidden_state_index': layer_summary['hidden_state_index'],
        'recording_accuracy_mean': get('accuracy', 'mean'),
        'recording_accuracy_std': get('accuracy', 'std'),
        'recording_f1_macro_mean': get('f1_macro', 'mean'),
        'recording_f1_macro_std': get('f1_macro', 'std'),
        'recording_auc_mean': get('auc', 'mean'),
        'recording_auc_std': get('auc', 'std'),
        'spk_accuracy_mean': get('spk_accuracy', 'mean'),
        'spk_accuracy_std': get('spk_accuracy', 'std'),
        'spk_f1_macro_mean': get('spk_f1_macro', 'mean'),
        'spk_f1_macro_std': get('spk_f1_macro', 'std'),
        'spk_auc_mean': get('spk_auc', 'mean'),
        'spk_auc_std': get('spk_auc', 'std'),
        'spk_accuracy_opt_mean': get('spk_accuracy_opt', 'mean'),
        'spk_accuracy_opt_std': get('spk_accuracy_opt', 'std'),
        'spk_f1_macro_opt_mean': get('spk_f1_macro_opt', 'mean'),
        'spk_f1_macro_opt_std': get('spk_f1_macro_opt', 'std'),
    }


def write_layer_probe_summary(output_dir, layer_summaries):
    """Write layer_probe_summary.csv and .json across all probed layers."""
    output_dir = Path(output_dir)
    rows = [_row_from_layer_summary(s) for s in layer_summaries]
    rows.sort(key=lambda r: r['ssl_layer'])

    csv_path = output_dir / 'layer_probe_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = output_dir / 'layer_probe_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    logger.info(f"\nLayer probe summary written to:")
    logger.info(f"  {csv_path}")
    logger.info(f"  {json_path}")

    def _argbest(metric_key):
        scored = [(r['ssl_layer'], r.get(metric_key)) for r in rows
                  if r.get(metric_key) is not None]
        if not scored:
            return None
        return max(scored, key=lambda kv: kv[1])

    best_auc = _argbest('spk_auc_mean')
    best_f1 = _argbest('spk_f1_macro_mean')
    best_acc = _argbest('spk_accuracy_mean')
    best_f1_opt = _argbest('spk_f1_macro_opt_mean')

    logger.info("\n" + "=" * 60)
    logger.info("LAYER PROBE: BEST LAYERS")
    logger.info("=" * 60)
    if best_auc:
        logger.info(f"Best by speaker-level AUC: layer {best_auc[0]}  "
                    f"(spk_auc={best_auc[1]:.4f})")
    if best_f1:
        logger.info(f"Best by speaker-level F1:  layer {best_f1[0]}  "
                    f"(spk_f1_macro={best_f1[1]:.4f})")
    if best_acc:
        logger.info(f"Best by speaker-level ACC: layer {best_acc[0]}  "
                    f"(spk_accuracy={best_acc[1]:.4f})")
    if best_f1_opt:
        logger.info(f"Best by speaker-level F1 (optimal threshold): "
                    f"layer {best_f1_opt[0]}  "
                    f"(spk_f1_macro_opt={best_f1_opt[1]:.4f})")
    logger.info("=" * 60)


# ============================================================================
# Top-level experiment runner
# ============================================================================

def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Resolve which layers we will probe up front so we can decide whether
    # the strict 12-layer requirement applies (run_all_layers, or any
    # explicit layer in [0,11] inherently assumes 12 layers).
    if args.layers is not None:
        layers_to_run = [int(x) for x in args.layers.split(',') if x.strip()]
    elif args.run_all_layers:
        layers_to_run = list(range(12))
    else:
        layers_to_run = [args.ssl_layer]

    # Strict 12-layer guard (always required by this script's design).
    assert_twelve_layer_backbone(args.backbone_type, args.model_name)

    (file_paths, labels, speaker_ids, rec_types,
     is_synthetic, label2id, id2label) = load_metadata_with_tts(
        args.data_dir, task=args.task, recording_type=args.recording_type,
        max_synth_ratio=args.max_synth_ratio,
        original_only=args.original_only,
    )

    num_labels = len(label2id)
    logger.info(f"Task: {args.task} | Labels: {num_labels} | "
                f"Backbone: {args.backbone_type} | Model: {args.model_name}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    # Always build the FULL n_folds split — --max_folds only caps how many
    # of those folds we actually run, so the split is identical to the
    # other v8 ablations and we can compare fold-by-fold.
    folds = create_cv_folds_with_tts(
        speaker_ids, labels, is_synthetic,
        n_folds=args.n_folds, seed=args.seed,
    )

    layer_summaries = []
    for ssl_layer in layers_to_run:
        layer_summary = run_layer_experiment(
            args=args,
            ssl_layer=ssl_layer,
            file_paths=file_paths,
            labels=labels,
            speaker_ids=speaker_ids,
            rec_types=rec_types,
            is_synthetic=is_synthetic,
            label2id=label2id,
            id2label=id2label,
            feature_extractor=feature_extractor,
            folds=folds,
        )
        layer_summaries.append(layer_summary)

    # Always write the cross-layer summary when more than one layer ran.
    # This covers --run_all_layers as well as --layers "5,7,9".
    if len(layer_summaries) > 1:
        write_layer_probe_summary(output_dir, layer_summaries)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Frozen SSL Backbone 12-Layer Probe — clean paper-aligned '
                    '(HuBERT / WavLM / wav2vec2 / Whisper; SVD or AVFAD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['svd', 'avfad'],
        help='Paper-aligned preset: sets default data_dir and max_length_sec. '
             'Override with explicit --data_dir / --max_length_sec if needed.',
    )
    parser.add_argument('--data_dir',       type=str, default=None)
    parser.add_argument('--task',           type=str, default='detection',
                        choices=['detection', 'classification'])
    parser.add_argument('--recording_type', type=str, default='sentence',
                        choices=['sentence', 'vowel', 'all'])
    parser.add_argument('--output_dir',     type=str,
                        default='./experiments/ssl_backbone_layer_probe_v8_clean')
    parser.add_argument(
        '--backbone_type',
        type=str,
        required=True,
        choices=list(SUPPORTED_BACKBONES),
        help='Which SSL backbone family to probe. Determines the input '
             'field (input_values for hubert/wavlm/wav2vec2, '
             'input_features for whisper) and how the encoder is '
             'extracted (Whisper uses .encoder, dropping the decoder).',
    )
    parser.add_argument('--model_name',     type=str, required=True,
                        help='HF hub id or local path to the backbone '
                             'checkpoint. Must be a 12-layer checkpoint '
                             '(e.g. hubert-base-ls960, wavlm-base-plus, '
                             'wav2vec2-base, whisper-small).')
    parser.add_argument('--num_epochs',     type=int, default=50)
    parser.add_argument('--batch_size',     type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate for the trainable surface. The default 1e-3 is '
             'tuned for a frozen-backbone linear probe (only ~1.5K head '
             'params, randomly initialized). If you are switching this '
             'script to compare against the v8 finetune baseline configuration, '
             'pass --lr 1e-5.',
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.05,
        help='Cosine warmup ratio. Default 0.05 is appropriate for a probe '
             'head; the v8 finetune baseline uses 0.12.',
    )
    parser.add_argument('--max_length_sec', type=float, default=None)
    parser.add_argument('--num_augmented',  type=int, default=0)
    parser.add_argument('--max_synth_ratio', type=float, default=None)
    parser.add_argument(
        '--original_only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use detection_metadata.csv only and drop synthetic/TTS rows '
             '(default: True). Pass --no-original_only to allow combined_* '
             'and optional --max_synth_ratio downsampling.',
    )
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        help='label_smoothing_factor passed to TrainingArguments. Default 0 '
             'for the probe (was hard-coded to 0.05 to match the v8 finetune '
             'baseline; that level of smoothing flattens the logits of an '
             'underfit linear head and was responsible for fold-to-fold '
             'majority/minority collapse). Pass 0.05 to reproduce baseline.',
    )
    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=0.0,
        help='Probability of applying mixup in the WeightedTrainer. Default 0 '
             'for the probe (was hard-coded to 0.3 in the v8 finetune '
             'baseline). On a frozen feature space mixup blurs linear '
             'separability and is detrimental to a small linear head. Pass '
             '0.3 to reproduce baseline.',
    )
    parser.add_argument(
        '--use_class_weights',
        action='store_true',
        help='If set, apply full inverse-frequency class weighting computed '
             'from ORIGINAL training samples only (matches the v8 finetune '
             'baseline). Default OFF for the probe: heavy class weights on '
             'top of an underfit linear head distort decision boundaries '
             'fold-to-fold and inflate the variance of speaker-level '
             'metrics.',
    )
    parser.add_argument('--n_folds',        type=int, default=10,
                        help='Build a stratified K-fold split with this K. '
                             'The split is built in full and is independent '
                             'of --max_folds.')
    parser.add_argument('--max_folds',      type=int, default=None,
                        help='If set, only run the first N folds out of the '
                             '--n_folds split. Default None = run all folds. '
                             'Used to align e.g. "first 5 of the 10-fold '
                             'split" with other ablations without changing '
                             'the underlying split.')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument(
        '--metric_for_best_model',
        type=str,
        default='f1_macro',
        choices=['f1_macro', 'auc'],
    )
    parser.add_argument(
        '--reseed_each_fold',
        action='store_true',
        help='If set, re-seed RNGs at the start of each fold '
             '(seed + fold_idx*1000) for fold-independent determinism.',
    )

    # Layer-probe specific knobs
    parser.add_argument(
        '--ssl_layer',
        type=int,
        default=None,
        help='Which transformer layer of the SSL backbone to probe. '
             'Range [0, 11]. The code reads '
             'outputs.hidden_states[ssl_layer + 1] because '
             'hidden_states[0] is the pre-transformer (feature '
             'projection / conv embed) output. Ignored if '
             '--run_all_layers is set or --layers is provided.',
    )
    parser.add_argument(
        '--hubert_layer',
        type=int,
        default=None,
        help='[DEPRECATED] Alias for --ssl_layer, kept for backwards '
             'compatibility with command lines from '
             'train_hubert_layer_probe_v8_reseed_f1best.py. '
             'Prefer --ssl_layer.',
    )
    parser.add_argument(
        '--run_all_layers',
        action='store_true',
        help='Loop --ssl_layer over 0..11. Each layer writes its own '
             'final_results.json under output_dir/layer_X/, and the '
             'top-level output_dir/layer_probe_summary.{csv,json} is '
             'written at the end. Requires the backbone to expose '
             'exactly 12 transformer layers.',
    )
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated list of transformer layers to probe '
             '(e.g. "5,7,9"). Each must be in [0, 11]. Overrides both '
             '--ssl_layer and --run_all_layers. Useful for two-stage '
             'workflows: stage 1 = --run_all_layers --max_folds 3, '
             'stage 2 = --layers <top-3> (no --max_folds).',
    )
    parser.add_argument(
        '--skip_existing_folds',
        action='store_true',
        help='If set, skip folds whose layer_X/fold_Y/results.json '
             'already exists and reload that JSON for the layer summary. '
             'Use this to extend a screening run (e.g. 3 folds) into a '
             'full run (10 folds) without re-training the first 3 folds, '
             'or to resume after a crash. Requires that data / split / '
             'seed / hyper-parameters are unchanged across runs.',
    )

    args = parser.parse_args()

    if args.data_dir is None:
        if args.dataset is None:
            parser.error('Provide --dataset {svd,avfad} or --data_dir')
        args.data_dir = str(DATASET_PRESETS[args.dataset]['data_dir'])

    if args.max_length_sec is None:
        if args.dataset is not None:
            args.max_length_sec = DATASET_PRESETS[args.dataset]['max_length_sec']
        else:
            args.max_length_sec = 5.0

    if args.original_only and args.max_synth_ratio is not None:
        parser.error(
            '--max_synth_ratio cannot be used with --original_only (default). '
            'Pass --no-original_only if you intentionally want TTS in train.'
        )

    # Resolve --hubert_layer (deprecated) -> --ssl_layer.
    if args.hubert_layer is not None:
        if args.ssl_layer is not None and args.ssl_layer != args.hubert_layer:
            parser.error(
                f"Both --ssl_layer ({args.ssl_layer}) and --hubert_layer "
                f"({args.hubert_layer}) were provided with conflicting "
                f"values. --hubert_layer is deprecated; please pass only "
                f"--ssl_layer."
            )
        logger.warning(
            "[DEPRECATED] --hubert_layer is deprecated; please use "
            "--ssl_layer instead. Treating --hubert_layer=%d as "
            "--ssl_layer=%d.",
            args.hubert_layer, args.hubert_layer,
        )
        if args.ssl_layer is None:
            args.ssl_layer = args.hubert_layer

    if args.ssl_layer is None:
        args.ssl_layer = 0

    if args.layers is not None:
        try:
            parsed_layers = [int(x) for x in args.layers.split(',') if x.strip()]
        except ValueError:
            parser.error(
                f"--layers must be a comma-separated list of integers, "
                f"got {args.layers!r}"
            )
        if not parsed_layers:
            parser.error("--layers parsed to an empty list")
        for lyr in parsed_layers:
            if not (0 <= lyr <= 11):
                parser.error(
                    f"--layers entries must be in [0, 11], got {lyr}"
                )
    elif not args.run_all_layers:
        if not (0 <= args.ssl_layer <= 11):
            parser.error(
                f"--ssl_layer must be in [0, 11], got {args.ssl_layer}"
            )
    if args.max_folds is not None and args.max_folds <= 0:
        parser.error(f"--max_folds must be positive, got {args.max_folds}")

    set_deterministic(args.seed)

    multiplier = 1 + args.num_augmented
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("\n--- Config ---")
    logger.info(f"  Backbone:     {args.backbone_type}")
    logger.info(f"  Model:        {args.model_name}")
    logger.info(f"  Task:         {args.task}")
    logger.info(f"  Dataset:      {args.dataset or '(custom data_dir)'}")
    logger.info(f"  Data dir:     {args.data_dir}")
    logger.info(f"  Rec type:     {args.recording_type}")
    logger.info(f"  Max length:   {args.max_length_sec}s")
    logger.info(f"  Epochs:       {args.num_epochs}")
    logger.info(f"  Effective BS: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  LR:           {args.lr}")
    logger.info(f"  Warmup:       {args.warmup_ratio}")
    logger.info(f"  Scheduler:    cosine")
    logger.info(f"  Label smooth: {args.label_smoothing}")
    logger.info(f"  Mixup prob:   {args.mixup_prob}")
    logger.info(f"  Class weight: {'ON (inverse-freq)' if args.use_class_weights else 'OFF'}")
    logger.info(f"  Aug copies:   {args.num_augmented} ({multiplier}x data)")
    logger.info(f"  Original only:{args.original_only}")
    logger.info(f"  Synth ratio:  {args.max_synth_ratio or 'unlimited'}")
    logger.info(f"  Backbone:     FROZEN (eval mode, no_grad in forward)")
    logger.info(f"  Pooling:      mean (time)")
    logger.info(f"  Auglib:       {'audiomentations' if HAS_AUDIOMENTATIONS else 'DISABLED'}")
    logger.info(f"  Best metric:  {args.metric_for_best_model}")
    logger.info(f"  Reseed fold:  {args.reseed_each_fold}")
    logger.info(f"  n_folds:      {args.n_folds}")
    logger.info(f"  max_folds:    {args.max_folds if args.max_folds is not None else 'all'}")

    # Underfit guard: probe-mode + finetune-style lr is the failure mode that
    # produced the original near-random screening run. Warn loudly so the
    # user can abort before burning GPU time.
    if args.lr <= 1e-4:
        logger.warning(
            "[UNDERFIT WARNING] You are running a frozen-backbone linear "
            f"probe with lr={args.lr:.0e}. This learning rate is tuned for "
            "full-backbone finetune (~94M params); on a randomly-initialized "
            "Linear head it typically leaves the head near initialization "
            "and produces fold-to-fold majority/minority collapse. "
            "Consider --lr 1e-3 unless you are deliberately reproducing the "
            "baseline regularizer ablation."
        )
    if args.layers is not None:
        logger.info(f"  Mode:         layers={args.layers} "
                    f"(parsed -> {parsed_layers})")
    elif args.run_all_layers:
        logger.info(f"  Mode:         ALL 12 layers (0..11)")
    else:
        logger.info(f"  Mode:         single layer "
                    f"(ssl_layer={args.ssl_layer}, "
                    f"hidden_states[{args.ssl_layer + 1}])")
    logger.info(f"  Skip existing folds: {args.skip_existing_folds}")

    run_experiment(args)


if __name__ == '__main__':
    main()
