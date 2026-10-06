#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("tts_synthesis_paper")

# ---------------------------------------------------------------------------
# Paper text (verbatim from public data_generation.py)
# ---------------------------------------------------------------------------

SVD_SENTENCE_TEXT_DE = "Guten Morgen, wie geht es Ihnen?"
SVD_VOWEL_TEXT_DE = "aaaaaaaaaaaaaaaaaaaaaaa"

AVFAD_CAPE_V_SENTENCES_PT = [
    "A Marta e o avô vivem naquele casarão rosa velho.",
    "Sofia saiu cedo da sala.",
    "A asa do avião andava avariada.",
    "Agora é hora de acabar.",
    "A minha mãe mandou-me embora.",
    "O Tiago comeu quatro peras.",
]
AVFAD_SENTENCE_TEXT_PT = " ".join(AVFAD_CAPE_V_SENTENCES_PT)

PAPER_DATASET_CONFIG = {
    "svd": {
        "label_filter": "healthy",
        "language": "de",
        "supports_vowels": True,
    },
    "avfad": {
        "label_filter": "pathological",
        "language": "pt",
        "supports_vowels": False,
    },
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _setup_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False


def _safe_name(s: str) -> str:
    s = str(s).strip().replace("'", "_")
    s = _SAFE_NAME_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unknown"


def _check_reference_audio(audio_path: str, min_duration: float = 1.0) -> Tuple[bool, str]:
    try:
        info = sf.info(audio_path)
        if info.duration < min_duration:
            return False, f"too short ({info.duration:.1f}s)"
        return True, f"OK ({info.duration:.1f}s, sr={info.samplerate})"
    except Exception as e:
        return False, str(e)


def _paper_text(dataset: str, recording_type: str) -> str:
    if dataset == "svd":
        if recording_type == "sentence":
            return SVD_SENTENCE_TEXT_DE
        return SVD_VOWEL_TEXT_DE
    return AVFAD_SENTENCE_TEXT_PT


def _audio_subdir(recording_type: str) -> str:
    return "sentences" if recording_type == "sentence" else "vowels"


def _synth_out_dir(output_dir: Path, recording_type: str) -> Path:
    return output_dir / "synthetic" / _audio_subdir(recording_type)


def _synth_out_fname(dataset: str, row: dict) -> str:
    if dataset == "svd":
        return f"tts_{row['output_filename']}"
    safe_patho = _safe_name(row["pathology"])
    rep = row.get("rep", "")
    return f"tts_{row['speaker_id']}_{safe_patho}_sentence_rep{rep}.wav"


def _extra_fname(base_fname: str, extra_idx: int) -> str:
    """Insert an ``_extra{n}`` tag before the extension of a base synth name."""
    stem = base_fname[:-4] if base_fname.endswith(".wav") else base_fname
    return f"{stem}_extra{extra_idx}.wav"


def _build_synth_work(dataset: str, rows: List[dict], target_synth_ratio: float,
                      seed: int) -> List[dict]:
    """Expand eligible base rows into a deterministic synthesis work list.

    Each base row yields exactly one base synth (extra_idx=0). If
    ``target_synth_ratio > 1.0``, additional ``_extra{n}`` synths are appended
    until ``int(n_group * ratio)`` is reached, grouped per recording_type.

    Nested-subset guarantee: extras are emitted in a single fixed global order
    (round-robin over a seed-shuffled row order), so the extra set for a smaller
    ratio is always a prefix-subset of a larger ratio's. This lets a single
    high-ratio generation be sliced into smaller ratios deterministically.
    """
    by_type: Dict[str, List[dict]] = {}
    for r in rows:
        by_type.setdefault(r["recording_type"], []).append(r)

    work: List[dict] = []
    for rtype in sorted(by_type):
        group = by_type[rtype]
        n_group = len(group)
        # Base synths (extra_idx = 0), original order preserved.
        for r in group:
            base_fname = _synth_out_fname(dataset, r)
            work.append({
                "row": r,
                "out_fname": base_fname,
                "recording_type": rtype,
                "extra_idx": 0,
                "global_order": -1,
            })
        if target_synth_ratio <= 1.0 or n_group == 0:
            continue
        target_total = int(n_group * target_synth_ratio)
        n_extra = max(0, target_total - n_group)
        if n_extra == 0:
            continue
        rng = np.random.RandomState(seed)
        shuffled = list(rng.permutation(n_group))
        for j in range(n_extra):
            r = group[shuffled[j % n_group]]
            extra_idx = (j // n_group) + 1
            base_fname = _synth_out_fname(dataset, r)
            work.append({
                "row": r,
                "out_fname": _extra_fname(base_fname, extra_idx),
                "recording_type": rtype,
                "extra_idx": extra_idx,
                "global_order": j,
            })
    return work


def _resolve_real_path(data_dir: Path, row: dict) -> Path:
    sub = _audio_subdir(row["recording_type"])
    return data_dir / sub / row["output_filename"]


def _load_svd_rows(data_dir: Path, label_filter: str, sentences_only: bool) -> List[dict]:
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")

    rows: List[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] != label_filter:
                continue
            rtype = row["recording_type"]
            if sentences_only and rtype != "sentence":
                continue
            if rtype != "sentence" and not rtype.startswith("vowel"):
                continue
            real_path = _resolve_real_path(data_dir, row)
            if not real_path.exists():
                logger.warning("missing wav, skip: %s", real_path)
                continue
            rows.append({
                "speaker_id": row["speaker_id"],
                "label": row["label"],
                "pathology": row.get("pathology", ""),
                "recording_type": rtype,
                "output_filename": row["output_filename"],
                "real_filename": row["output_filename"],
                "real_path": str(real_path),
                "rep": row.get("rep", ""),
            })
    rows.sort(key=lambda r: (r["speaker_id"], r["recording_type"], r["output_filename"]))
    return rows


def _load_avfad_rows(data_dir: Path, label_filter: str) -> List[dict]:
    det_path = data_dir / "detection_metadata.csv"
    if not det_path.exists():
        raise FileNotFoundError(f"missing {det_path}")

    spk_to_pathology: Dict[str, str] = {}
    clin_path = data_dir / "participants_clinical.csv"
    if clin_path.exists():
        with open(clin_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                spk_to_pathology[row["speaker_id"]] = (row.get("cmvd_word") or "").strip()

    cls_path = data_dir / "classification_metadata.csv"
    file_to_pathology: Dict[str, str] = {}
    if cls_path.exists():
        with open(cls_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                file_to_pathology[row["output_filename"]] = row.get("pathology", "")

    rows: List[dict] = []
    with open(det_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["recording_type"] != "sentence":
                continue
            if row["label"] != label_filter:
                continue
            fname = row["output_filename"]
            patho = file_to_pathology.get(fname, "") or spk_to_pathology.get(row["speaker_id"], "")
            if not patho:
                patho = "Normal" if row["label"] == "healthy" else "Unknown"
            real_path = data_dir / "sentences" / fname
            if not real_path.exists():
                logger.warning("missing wav, skip: %s", real_path)
                continue
            try:
                rep = int(row.get("rep") or 0)
            except (TypeError, ValueError):
                rep = 0
            rows.append({
                "speaker_id": row["speaker_id"],
                "label": row["label"],
                "pathology": patho,
                "recording_type": "sentence",
                "output_filename": fname,
                "real_filename": fname,
                "real_path": str(real_path),
                "rep": rep,
            })
    rows.sort(key=lambda r: (r["speaker_id"], r["rep"]))
    return rows


def _backup_existing_tts(data_dir: Path) -> Optional[Path]:
    synth_dir = data_dir / "synthetic"
    meta = data_dir / "synthetic_metadata.csv"
    if not synth_dir.exists() and not meta.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = data_dir / f"backup_tts_pre_paper_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    for name in (
        "synthetic",
        "synthetic_metadata.csv",
        "combined_detection_metadata.csv",
        "combined_classification_metadata.csv",
        "tts_generation_config.json",
        "tts_generation_paper_aligned_config.json",
        "tts_synthesis_paper_config.json",
    ):
        src = data_dir / name
        if src.exists():
            dst = backup_root / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            logger.info("backed up %s -> %s", src.name, backup_root)

    return backup_root


def _clear_synthetic_outputs(data_dir: Path) -> None:
    synth_root = data_dir / "synthetic"
    if synth_root.exists():
        shutil.rmtree(synth_root)
    for sub in ("sentences", "vowels"):
        (synth_root / sub).mkdir(parents=True, exist_ok=True)


def _prepare_isolated_output_dir(data_dir: Path, output_dir: Path) -> None:
    """Create output_dir with symlinks to read-only assets in data_dir (no copy/move)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    link_names = [
        "sentences",
        "vowels",
        "detection_metadata.csv",
        "classification_metadata.csv",
        "augmented_detection_metadata.csv",
        "augmented_classification_metadata.csv",
        "dataset_config.json",
        "metadata.csv",
        "participants_clinical.csv",
        "cv_folds.json",
    ]
    for name in link_names:
        src = data_dir / name
        dst = output_dir / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())
        logger.info("symlink %s -> %s", dst, src)


def _load_base_detection_rows(data_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for fname in ("detection_metadata.csv", "augmented_detection_metadata.csv"):
        path = data_dir / fname
        if not path.exists():
            continue
        source = "original" if fname.startswith("detection") else "augmented"
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source_dir"] = source
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no detection metadata under {data_dir}")
    return rows


def _load_base_classification_rows(data_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for fname in ("classification_metadata.csv", "augmented_classification_metadata.csv"):
        path = data_dir / fname
        if not path.exists():
            continue
        source = "original" if fname.startswith("classification") else "augmented"
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source_dir"] = source
                rows.append(row)
    return rows


def _write_combined_metadata(
    meta_dir: Path,
    output_dir: Path,
    synth_records: List[dict],
) -> None:
    det_fields = ["speaker_id", "label", "recording_type", "output_filename", "source_dir"]
    cls_fields = ["speaker_id", "pathology", "recording_type", "output_filename", "source_dir"]

    base_det = _load_base_detection_rows(meta_dir)
    tts_det = [{
        "speaker_id": r["speaker_id"],
        "label": r["label"],
        "recording_type": r["recording_type"],
        "output_filename": r["output_filename"],
        "source_dir": "synthetic",
    } for r in synth_records]
    combined_det = base_det + tts_det
    with open(output_dir / "combined_detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        for row in combined_det:
            w.writerow({k: row.get(k, "") for k in det_fields})
    logger.info(
        "combined_detection_metadata.csv: %d rows (%s)",
        len(combined_det),
        dict(Counter(r["source_dir"] for r in combined_det)),
    )

    base_cls = _load_base_classification_rows(meta_dir)
    if not base_cls:
        return

    top_k: List[str] = []
    cfg = meta_dir / "dataset_config.json"
    if cfg.exists():
        with open(cfg) as f:
            top_k = json.load(f).get("top_k_classes", [])

    tts_cls = []
    for r in synth_records:
        if not top_k or r["pathology"] in top_k:
            tts_cls.append({
                "speaker_id": r["speaker_id"],
                "pathology": r["pathology"],
                "recording_type": r["recording_type"],
                "output_filename": r["output_filename"],
                "source_dir": "synthetic",
            })
    combined_cls = base_cls + tts_cls
    with open(output_dir / "combined_classification_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cls_fields)
        w.writeheader()
        for row in combined_cls:
            w.writerow({k: row.get(k, "") for k in cls_fields})
    logger.info(
        "combined_classification_metadata.csv: %d rows (%s)",
        len(combined_cls),
        dict(Counter(r["source_dir"] for r in combined_cls)),
    )


def _write_synthetic_metadata(output_dir: Path, synth_records: List[dict]) -> None:
    if not synth_records:
        return
    fieldnames = list(synth_records[0].keys())
    with open(output_dir / "synthetic_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in synth_records:
            w.writerow(row)


def _save_wav(wav: np.ndarray, out_path: Path, target_sr: int, native_sr: int) -> None:
    wav = np.asarray(wav, dtype=np.float32)
    if native_sr != target_sr:
        if not HAS_LIBROSA:
            raise RuntimeError("librosa required for resampling (pip install librosa)")
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=target_sr, res_type="polyphase")
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16, target_sr, subtype="PCM_16")


def _make_synth_record(
    dataset: str,
    row: dict,
    out_fname: str,
    out_path: Path,
    text: str,
    extra_idx: int = 0,
    global_order: int = -1,
) -> dict:
    try:
        duration = round(float(sf.info(str(out_path)).duration), 3)
    except Exception:
        duration = 0.0
    rec = {
        "speaker_id": row["speaker_id"],
        "label": row["label"],
        "pathology": row["pathology"],
        "recording_type": row["recording_type"],
        "output_filename": out_fname,
        "processed_path": str(out_path.resolve()),
        "source_dir": "synthetic",
        "speaker_wav_source": row["real_filename"],
        "synth_text": text,
        "is_extra": int(extra_idx > 0),
        "extra_idx": extra_idx,
        "extra_global_order": global_order,
    }
    if dataset == "avfad":
        rec["rep"] = row["rep"]
        rec["duration_sec"] = duration
        rec["n_source_sentences"] = len(AVFAD_CAPE_V_SENTENCES_PT)
    return rec


def run_paper_tts(
    dataset: str,
    data_dir: Path,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    seed: int = 42,
    sentences_only: bool = False,
    target_sr: int = 16000,
    target_synth_ratio: float = 1.0,
    force: bool = False,
    backup: bool = True,
    dry_run: bool = False,
    max_rows: Optional[int] = None,
) -> int:
    cfg = PAPER_DATASET_CONFIG[dataset]
    data_dir = data_dir.resolve()
    output_dir = (output_dir or data_dir).resolve()
    isolated = output_dir != data_dir
    if isolated:
        backup = False

    log_path = output_dir / "tts_synthesis_paper.log"
    _setup_logging(log_path)

    logger.info("=" * 72)
    logger.info("PAPER-ALIGNED TTS — dataset=%s", dataset.upper())
    logger.info("  data_dir        = %s (read originals)", data_dir)
    logger.info("  output_dir      = %s (write synth + metadata)", output_dir)
    logger.info("  isolated_mode   = %s", isolated)
    logger.info("  label_filter    = %s", cfg["label_filter"])
    logger.info("  language        = %s", cfg["language"])
    logger.info("  sentences_only  = %s", sentences_only)
    logger.info("  target_ratio    = %s", target_synth_ratio)
    logger.info("  device          = %s", device)
    logger.info("  target_sr       = %d", target_sr)
    logger.info("  backup          = %s", backup)
    logger.info("  force           = %s", force)
    logger.info("  dry_run         = %s", dry_run)
    logger.info("=" * 72)

    if dataset == "svd":
        rows = _load_svd_rows(data_dir, cfg["label_filter"], sentences_only)
    else:
        if sentences_only:
            logger.info("sentences_only ignored for AVFAD (sentence-only dataset)")
        rows = _load_avfad_rows(data_dir, cfg["label_filter"])

    if max_rows is not None:
        rows = rows[:max_rows]
        logger.info("capped to max_rows=%d", max_rows)

    type_counts = Counter(r["recording_type"] for r in rows)
    label_counts = Counter(r["label"] for r in rows)
    logger.info("eligible rows: %d | labels: %s | types: %s", len(rows), dict(label_counts), dict(type_counts))

    if not rows:
        logger.error("no eligible rows")
        return 1

    work = _build_synth_work(dataset, rows, target_synth_ratio, seed)
    n_base = sum(1 for w in work if w["extra_idx"] == 0)
    n_extra = len(work) - n_base
    logger.info("synthesis work: %d total (%d base + %d extra) @ ratio=%.3f",
                len(work), n_base, n_extra, target_synth_ratio)

    if dry_run:
        logger.info("[DRY RUN] would synthesize %d wavs (%d base + %d extra); "
                    "no files written.", len(work), n_base, n_extra)
        return 0

    if isolated:
        _prepare_isolated_output_dir(data_dir, output_dir)
    elif backup:
        backup_path = _backup_existing_tts(data_dir)
        if backup_path:
            logger.info("previous TTS backed up to %s", backup_path)

    if isolated:
        if force:
            _clear_synthetic_outputs(output_dir)
        else:
            for sub in ("sentences", "vowels"):
                (output_dir / "synthetic" / sub).mkdir(parents=True, exist_ok=True)
    elif force or backup:
        _clear_synthetic_outputs(output_dir)
    else:
        for sub in ("sentences", "vowels"):
            (output_dir / "synthetic" / sub).mkdir(parents=True, exist_ok=True)

    import torch
    from TTS.api import TTS

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("loading XTTS-v2 ...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    try:
        native_sr = int(tts_model.synthesizer.output_sample_rate)
    except Exception:
        native_sr = 24000
    logger.info("XTTS loaded (native_sr=%d, target_sr=%d)", native_sr, target_sr)

    synth_records: List[dict] = []
    skipped = 0
    errors = 0
    t0 = time.time()

    iterator = tqdm(work, desc=f"TTS-{dataset}") if HAS_TQDM else work
    for item in iterator:
        row = item["row"]
        extra_idx = item["extra_idx"]
        global_order = item["global_order"]
        text = _paper_text(dataset, row["recording_type"])
        out_fname = item["out_fname"]
        out_path = _synth_out_dir(output_dir, row["recording_type"]) / out_fname

        ok, msg = _check_reference_audio(row["real_path"])
        if not ok:
            logger.warning("skip bad ref %s: %s", row["real_filename"], msg)
            skipped += 1
            continue

        if out_path.exists() and not force:
            synth_records.append(_make_synth_record(
                dataset, row, out_fname, out_path, text, extra_idx, global_order))
            continue

        # Deterministic per-item seed: base reproduces the original 1x run
        # (extra_idx=0 -> seed); each extra gets a distinct seed so the cloned
        # wav differs from the base and from other extras of the same speaker.
        item_seed = seed if extra_idx == 0 else seed + 100000 + global_order + 1
        torch.manual_seed(item_seed)
        np.random.seed(item_seed)

        try:
            wav = tts_model.tts(
                text=text,
                speaker_wav=row["real_path"],
                language=cfg["language"],
            )
            _save_wav(wav, out_path, target_sr, native_sr)
            synth_records.append(_make_synth_record(
                dataset, row, out_fname, out_path, text, extra_idx, global_order))
        except Exception as e:
            logger.warning("TTS failed %s -> %s: %s", row["real_filename"], out_fname, e)
            errors += 1

    elapsed = time.time() - t0
    _write_synthetic_metadata(output_dir, synth_records)
    _write_combined_metadata(data_dir, output_dir, synth_records)

    config = {
        "script": "tts_synthesis_paper_extend.py",
        "extends_previous_ratio": 3.0,
        "dataset": dataset.upper(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "isolated_mode": isolated,
        "paper_reference": "Koudounas et al., Interspeech 2024",
        "paper_code_reference": "koudounasalkis/AI4Voice/src/data_generation.py",
        "label_filter": cfg["label_filter"],
        "language": cfg["language"],
        "method": ("1 eligible real recording -> 1 base synthetic wav; "
                   "if target_synth_ratio > 1.0, seed-shuffled round-robin "
                   "_extra{n} wavs are added (nested across ratios) until "
                   "int(n_group * ratio) is reached; speaker_wav = real wav"),
        "sentences_only": sentences_only,
        "target_synth_ratio": target_synth_ratio,
        "n_eligible_rows": len(rows),
        "n_base_planned": n_base,
        "n_extra_planned": n_extra,
        "n_synthetic_written": len(synth_records),
        "n_synthetic_base": sum(1 for r in synth_records if not r.get("is_extra")),
        "n_synthetic_extra": sum(1 for r in synth_records if r.get("is_extra")),
        "n_skipped_bad_reference": skipped,
        "n_errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "device": device,
        "target_sr": target_sr,
        "xtts_native_sr": native_sr,
        "seed": seed,
        "label_counts_synthetic": dict(Counter(r["label"] for r in synth_records)),
        "type_counts_synthetic": dict(Counter(r["recording_type"] for r in synth_records)),
    }
    with open(output_dir / "tts_synthesis_paper_extend_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("=" * 72)
    logger.info("DONE — synth=%d skipped=%d errors=%d elapsed=%.1fs",
                len(synth_records), skipped, errors, elapsed)
    logger.info("  synthetic_metadata.csv")
    logger.info("  combined_detection_metadata.csv")
    logger.info("  tts_synthesis_paper_extend_config.json")
    logger.info("=" * 72)
    return 0 if errors == 0 else 4


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend existing SVD/AVFAD TTS pool to a higher synth ratio "
                    "(copy of tts_synthesis_paper_all.py; skips existing wavs)",
    )
    parser.add_argument("--dataset", required=True, choices=["svd", "avfad"])
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Source cleaned data (original wavs + base metadata); read-only in isolated mode")
    parser.add_argument("--output_dir", default=None, type=Path,
                        help="Write synthetic wavs + combined metadata here. "
                             "If set and != data_dir, originals are symlinked (no copy/move/overwrite of data_dir)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument(
        "--sentences_only",
        action="store_true",
        help="SVD only: skip healthy vowel synthesis (sentence experiments)",
    )
    parser.add_argument(
        "--target_synth_ratio", type=float, default=4.0,
        help="Target synthetic/eligible-real ratio per recording_type. "
             "Default 4.0 for extending an existing 3.0x pool. "
             ">1.0 keeps all base synths and adds _extra{n} wavs (nested "
             "across ratios) until int(n_group * ratio) is reached. "
             "Existing wavs are reused (resume-friendly) unless --force.",
    )
    parser.add_argument("--no_backup", action="store_true", help="do not backup existing synthetic/")
    parser.add_argument("--force", action="store_true", help="overwrite existing output wavs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None, help="smoke-test cap")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data_dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    return run_paper_tts(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        sentences_only=args.sentences_only,
        target_sr=args.target_sr,
        target_synth_ratio=args.target_synth_ratio,
        force=args.force,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
