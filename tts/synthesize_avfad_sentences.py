#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import librosa

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("avfad_tts_paper")


# ============================================================================
# AVFAD CAPE-V six Portuguese sentences (verbatim from public
# ``koudounasalkis/AI4Voice/src/data_generation.py`` -> AVFAD branch).
# ============================================================================

AVFAD_CAPE_V_SENTENCES_PT = [
    "A Marta e o avô vivem naquele casarão rosa velho.",
    "Sofia saiu cedo da sala.",
    "A asa do avião andava avariada.",
    "Agora é hora de acabar.",
    "A minha mãe mandou-me embora.",
    "O Tiago comeu quatro peras.",
]

# The paper's published code joins the six sentences with the original
# Python multi-line-string indentation whitespace; equivalent to a
# single space separator after stripping.  We use single spaces.
PAPER_TEXT_PT = " ".join(AVFAD_CAPE_V_SENTENCES_PT)


# ============================================================================
# Helpers
# ============================================================================

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


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_name(s: str) -> str:
    s = str(s).strip().replace("'", "_")
    s = _SAFE_NAME_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unknown"


def _check_reference_audio(audio_path: str, min_duration: float = 1.0,
                           max_duration: float = 30.0):
    try:
        info = sf.info(audio_path)
        d = info.duration
        if d < min_duration:
            return False, f"too short ({d:.1f}s < {min_duration}s)"
        return True, f"OK ({d:.1f}s, sr={info.samplerate})"
    except Exception as e:
        return False, str(e)


def _resolve_pathology(label: str, pathology_field: str) -> str:
    """Mimic ``tts_synthesis_avfad.py``'s rule: if the metadata pathology
    field is empty, fall back to ``Normal`` for healthy and ``Unknown``
    for pathological speakers."""
    p = (pathology_field or "").strip()
    if p:
        return p
    return "Normal" if label == "healthy" else "Unknown"


# ============================================================================
# Manifest
# ============================================================================

def _load_real_manifest(data_dir: Path):
    """Read ``detection_metadata.csv`` + ``participants_clinical.csv`` from
    ``data_dir`` and return a list of dicts:
        [{"speaker_id", "label", "pathology", "rep", "real_filename",
          "real_path", "real_duration_sec"}, ...]
    Sentence rows only (no vowels).
    """
    det_path = data_dir / "detection_metadata.csv"
    if not det_path.exists():
        raise FileNotFoundError(f"missing {det_path}")

    spk_to_pathology: Dict[str, str] = {}
    clin_path = data_dir / "participants_clinical.csv"
    if clin_path.exists():
        with open(clin_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cmvd_word = (row.get("cmvd_word") or "").strip()
                spk_to_pathology[row["speaker_id"]] = cmvd_word

    rows: List[dict] = []
    with open(det_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["recording_type"] != "sentence":
                continue
            spk = r["speaker_id"]
            label = r["label"]
            patho_field = r.get("pathology", "") or spk_to_pathology.get(spk, "")
            patho = _resolve_pathology(label, patho_field)
            real_path = data_dir / "sentences" / r["output_filename"]
            try:
                rep = int(r["rep"])
            except (KeyError, ValueError, TypeError):
                rep = 0
            try:
                dur = float(r.get("duration_sec") or 0.0)
            except ValueError:
                dur = 0.0
            rows.append({
                "speaker_id": spk,
                "label": label,
                "pathology": patho,
                "rep": rep,
                "real_filename": r["output_filename"],
                "real_path": str(real_path),
                "real_duration_sec": dur,
            })
    return rows


# ============================================================================
# Combined metadata writer (drop-in compatible with the previous
# ``tts_synthesis_avfad.py`` layout, so downstream training code can
# read this dir without changes)
# ============================================================================

def _build_combined_metadata(data_dir: Path, tts_records: List[dict]) -> None:
    det_fields = ["speaker_id", "label", "recording_type",
                  "output_filename", "source_dir"]
    cls_fields = ["speaker_id", "pathology", "recording_type",
                  "output_filename", "source_dir"]

    base_det = data_dir / "detection_metadata.csv"
    base_det_rows: List[dict] = []
    with open(base_det, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row.setdefault("source_dir", "original")
            base_det_rows.append(row)

    tts_det_rows = [{
        "speaker_id": r["speaker_id"], "label": r["label"],
        "recording_type": r["recording_type"],
        "output_filename": r["output_filename"], "source_dir": "synthetic",
    } for r in tts_records]
    combined_det = base_det_rows + tts_det_rows
    out_det = data_dir / "combined_detection_metadata.csv"
    with open(out_det, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        for r in combined_det:
            w.writerow({k: r.get(k, "") for k in det_fields})
    logger.info("  combined_detection_metadata.csv: %d rows (real %d + synth %d)",
                len(combined_det), len(base_det_rows), len(tts_det_rows))

    config_path = data_dir / "dataset_config.json"
    top_k_classes: List[str] = []
    if config_path.exists():
        with open(config_path) as f:
            top_k_classes = json.load(f).get("top_k_classes", [])

    base_cls = data_dir / "classification_metadata.csv"
    if not base_cls.exists():
        return
    base_cls_rows: List[dict] = []
    with open(base_cls, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row.setdefault("source_dir", "original")
            base_cls_rows.append(row)

    tts_cls_rows: List[dict] = []
    for r in tts_records:
        if r["pathology"] in top_k_classes:
            tts_cls_rows.append({
                "speaker_id": r["speaker_id"], "pathology": r["pathology"],
                "recording_type": r["recording_type"],
                "output_filename": r["output_filename"], "source_dir": "synthetic",
            })
    combined_cls = base_cls_rows + tts_cls_rows
    out_cls = data_dir / "combined_classification_metadata.csv"
    with open(out_cls, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cls_fields)
        w.writeheader()
        for r in combined_cls:
            w.writerow({k: r.get(k, "") for k in cls_fields})
    logger.info("  combined_classification_metadata.csv: %d rows (real %d + synth %d)",
                len(combined_cls), len(base_cls_rows), len(tts_cls_rows))


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="paper-aligned cleaning output dir (must contain "
                         "detection_metadata.csv and sentences/*.wav)")
    ap.add_argument("--language", default="pt", choices=["pt"],
                    help="paper hard-codes Portuguese for AVFAD; only 'pt' is supported")
    ap.add_argument("--only_pathological", action="store_true",
                    help="If set, mimic the AVFAD branch of the public "
                         "``data_generation.py`` and only synthesize pathological "
                         "speakers. Default: synthesize both H and P.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_sr", type=int, default=16000,
                    help="output sample rate. XTTS-v2 generates at 24000 Hz; "
                         "we downsample to 16000 by default to match the "
                         "real sentence wavs in --data_dir/sentences/, "
                         "which are 16 kHz. Pass --target_sr 24000 to keep "
                         "the native XTTS sample rate.")
    ap.add_argument("--max_speakers", type=int, default=None,
                    help="optional speaker cap for smoke testing")
    ap.add_argument("--max_reps", type=int, default=None,
                    help="optional rep cap (e.g. 1) for smoke testing")
    ap.add_argument("--target_synth_ratio", type=float, default=1.0,
                    help="target synthetic/original sentence ratio. 1.0 = one base "
                         "synth per real rep (paper default). 1.5 = keep all base "
                         "synths and add extras with _extra1 suffix until the ratio "
                         "is reached (matches training --max_synth_ratio 1.5).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing synthetic wavs (default: skip if exists)")
    ap.add_argument("--dry_run", action="store_true",
                    help="don't load TTS, don't write wavs; just report what would happen")
    args = ap.parse_args()

    data_dir: Path = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        return 2

    out_audio_dir = data_dir / "synthetic" / "sentences"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "tts_generation_paper_aligned.log"
    _setup_logging(log_path)

    logger.info("=" * 72)
    logger.info("AVFAD PAPER-ALIGNED TTS SYNTHESIS (sentences only)")
    logger.info("  DATA_DIR    = %s", data_dir)
    logger.info("  OUT_AUDIO   = %s", out_audio_dir)
    logger.info("  LANGUAGE    = %s", args.language)
    logger.info("  CLASSES     = %s", "pathological-only (paper code)"
                if args.only_pathological else "both H and P (paper text)")
    logger.info("  DEVICE      = %s", args.device)
    logger.info("  TARGET_SR   = %d", args.target_sr)
    logger.info("  SEED        = %d", args.seed)
    logger.info("  MAX_SPK     = %s", args.max_speakers)
    logger.info("  MAX_REPS    = %s", args.max_reps)
    logger.info("  SYNTH_RATIO = %s", args.target_synth_ratio)
    logger.info("  FORCE       = %s", args.force)
    logger.info("  DRY_RUN     = %s", args.dry_run)
    logger.info("  TEXT (6 sentences):")
    for i, s in enumerate(AVFAD_CAPE_V_SENTENCES_PT):
        logger.info("    [%d] %s", i, s)
    logger.info("=" * 72)

    rows = _load_real_manifest(data_dir)
    logger.info("loaded %d real sentence rows", len(rows))

    spk_label = {(r["speaker_id"], r["label"]) for r in rows}
    spk_count = len({s for s, _ in spk_label})
    label_counts = Counter(l for _, l in spk_label)
    logger.info("real speakers: %d (H=%d P=%d)", spk_count,
                label_counts.get("healthy", 0),
                label_counts.get("pathological", 0))

    if args.only_pathological:
        rows = [r for r in rows if r["label"] == "pathological"]
        logger.info("filter only_pathological -> %d rows", len(rows))

    if args.max_reps is not None:
        rows = [r for r in rows if r["rep"] <= args.max_reps]
        logger.info("filter max_reps=%d -> %d rows", args.max_reps, len(rows))

    if args.max_speakers is not None:
        keep_spk = sorted({r["speaker_id"] for r in rows})[:args.max_speakers]
        keep_spk_set = set(keep_spk)
        rows = [r for r in rows if r["speaker_id"] in keep_spk_set]
        logger.info("filter max_speakers=%d -> %d rows", args.max_speakers, len(rows))

    rows.sort(key=lambda r: (r["speaker_id"], r["rep"]))

    def _base_out_fname(r: dict) -> str:
        safe_patho = _safe_name(r["pathology"])
        return f"tts_{r['speaker_id']}_{safe_patho}_sentence_rep{r['rep']}.wav"

    def _extra_out_fname(r: dict) -> str:
        safe_patho = _safe_name(r["pathology"])
        return f"tts_{r['speaker_id']}_{safe_patho}_sentence_rep{r['rep']}_extra1.wav"

    def _make_synth_record(r: dict, out_fname: str, out_path: str) -> dict:
        try:
            d = sf.info(out_path).duration
        except Exception:
            d = 0.0
        return {
            "speaker_id": r["speaker_id"], "label": r["label"],
            "pathology": r["pathology"], "recording_type": "sentence",
            "output_filename": out_fname,
            "processed_path": out_path, "source_dir": "synthetic",
            "rep": r["rep"], "duration_sec": round(float(d), 3),
            "speaker_wav_source": r["real_filename"],
            "synth_text": PAPER_TEXT_PT,
            "n_source_sentences": len(AVFAD_CAPE_V_SENTENCES_PT),
        }

    skipped_existing: List[str] = []
    skipped_bad_ref: List[str] = []
    pending_rows: List[dict] = []
    eligible_for_extra: List[dict] = []
    for r in rows:
        out_fname = _base_out_fname(r)
        out_path = out_audio_dir / out_fname
        ok, msg = _check_reference_audio(r["real_path"])
        if not ok:
            skipped_bad_ref.append(f"{r['speaker_id']} rep{r['rep']}: {msg}")
            r["status"] = "skip_bad_ref"
            continue
        eligible_for_extra.append(r)
        if out_path.exists() and not args.force:
            skipped_existing.append(out_fname)
            r["output_filename"] = out_fname
            r["output_path"] = str(out_path)
            r["status"] = "skip_existing"
            continue
        r["output_filename"] = out_fname
        r["output_path"] = str(out_path)
        r["status"] = "pending"
        pending_rows.append(r)

    skipped_existing_extra: List[str] = []
    pending_extra_rows: List[dict] = []
    extra_rows_tracked: List[dict] = []
    n_base_total = 0
    if args.target_synth_ratio > 1.0:
        n_base_total = sum(
            1 for r in eligible_for_extra
            if (out_audio_dir / _base_out_fname(r)).exists() or r.get("status") == "pending"
        )
        target_total = int(len(rows) * args.target_synth_ratio)
        n_extra_target = max(0, target_total - n_base_total)
        if n_extra_target > 0:
            rng = np.random.RandomState(args.seed)
            pick_n = min(n_extra_target, len(eligible_for_extra))
            pick_idx = sorted(rng.choice(len(eligible_for_extra), pick_n, replace=False))
            for idx in pick_idx:
                r = dict(eligible_for_extra[idx])
                out_fname = _extra_out_fname(r)
                out_path = out_audio_dir / out_fname
                if out_path.exists() and not args.force:
                    skipped_existing_extra.append(out_fname)
                    r["output_filename"] = out_fname
                    r["output_path"] = str(out_path)
                    r["status"] = "skip_existing_extra"
                    extra_rows_tracked.append(r)
                    continue
                r["output_filename"] = out_fname
                r["output_path"] = str(out_path)
                r["status"] = "pending_extra"
                pending_extra_rows.append(r)
                extra_rows_tracked.append(r)
        logger.info(
            "extra plan: target_ratio=%.2f real=%d base_total=%d target_synth=%d "
            "extra_target=%d pending_extra=%d skip_existing_extra=%d",
            args.target_synth_ratio, len(rows), n_base_total, target_total,
            n_extra_target, len(pending_extra_rows), len(skipped_existing_extra),
        )
    else:
        target_total = n_base_total = sum(
            1 for r in eligible_for_extra
            if (out_audio_dir / _base_out_fname(r)).exists() or r.get("status") == "pending"
        )

    logger.info("base pending=%d, skip_existing=%d, skip_bad_ref=%d",
                len(pending_rows), len(skipped_existing), len(skipped_bad_ref))

    if args.dry_run:
        logger.info("[DRY RUN] would synthesize base=%d extra=%d wavs (text length=%d chars). Stop.",
                    len(pending_rows), len(pending_extra_rows), len(PAPER_TEXT_PT))
        return 0

    tts_model = None
    xtts_native_sr: Optional[int] = None
    if pending_rows or pending_extra_rows:
        logger.info("loading XTTS-v2 (~30s on first call) ...")
        try:
            import torch
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            from TTS.api import TTS
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(args.device)
            logger.info("XTTS-v2 loaded on %s", args.device)
            try:
                xtts_native_sr = int(tts_model.synthesizer.output_sample_rate)
            except Exception:
                xtts_native_sr = 24000
            logger.info("XTTS native sr=%d, target sr=%d (resample on save: %s)",
                        xtts_native_sr, args.target_sr,
                        xtts_native_sr != args.target_sr)
        except Exception as e:
            logger.error("failed to load XTTS-v2: %s", e)
            logger.error("hint: pip install TTS torch in the ai4voice env")
            return 3

    synth_records: List[dict] = []
    for r in rows:
        if r.get("status") == "skip_existing":
            synth_records.append(_make_synth_record(r, r["output_filename"], r["output_path"]))

    errors = 0
    extra_errors = 0

    def _run_tts_batch(batch: List[dict], desc: str) -> int:
        batch_errors = 0
        iterator = tqdm(batch, desc=desc) if (HAS_TQDM and batch) else batch
        for r in iterator:
            try:
                wav = tts_model.tts(
                    text=PAPER_TEXT_PT,
                    speaker_wav=r["real_path"],
                    language=args.language,
                )
                wav = np.asarray(wav, dtype=np.float32)
                if xtts_native_sr is not None and args.target_sr != xtts_native_sr:
                    wav = librosa.resample(
                        wav,
                        orig_sr=xtts_native_sr,
                        target_sr=args.target_sr,
                        res_type="polyphase",
                    )
                wav = np.clip(wav, -1.0, 1.0)
                pcm16 = (wav * 32767.0).astype(np.int16)
                sf.write(r["output_path"], pcm16, args.target_sr, subtype="PCM_16")
                synth_records.append(
                    _make_synth_record(r, r["output_filename"], r["output_path"])
                )
            except Exception as e:
                logger.warning("  TTS failed for %s rep%d (%s): %s",
                               r["speaker_id"], r["rep"], r["output_filename"], e)
                batch_errors += 1
        return batch_errors

    t0 = time.time()
    base_errors = 0
    if pending_rows:
        base_errors = _run_tts_batch(pending_rows, "TTS-base")
        errors += base_errors
        n_new_base = len(pending_rows) - base_errors
        logger.info("base synthesized %d new wavs (%d errors)",
                    n_new_base, base_errors)

    for r in extra_rows_tracked:
        if r.get("status") == "skip_existing_extra":
            synth_records.append(
                _make_synth_record(r, r["output_filename"], r["output_path"])
            )

    if pending_extra_rows:
        extra_errors = _run_tts_batch(pending_extra_rows, "TTS-extra")
        errors += extra_errors
        n_new_extra = len(pending_extra_rows) - extra_errors
        logger.info("extra synthesized %d new wavs (%d errors)",
                    n_new_extra, extra_errors)

    elapsed = time.time() - t0
    n_new = (len(pending_rows) - base_errors) + (len(pending_extra_rows) - extra_errors)
    if pending_rows or pending_extra_rows:
        logger.info("total synthesized %d new wavs (%d errors), %.1f min, "
                    "%.2f s/wav", n_new, errors, elapsed / 60.0,
                    elapsed / max(1, n_new))

    synth_records.sort(key=lambda r: (r["speaker_id"], r["rep"], r["output_filename"]))
    syn_fields = [
        "speaker_id", "label", "pathology", "recording_type",
        "output_filename", "processed_path", "source_dir",
        "rep", "duration_sec", "speaker_wav_source",
        "n_source_sentences", "synth_text",
    ]
    syn_meta_path = data_dir / "synthetic_metadata.csv"
    with open(syn_meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=syn_fields)
        w.writeheader()
        for rec in synth_records:
            w.writerow({k: rec.get(k, "") for k in syn_fields})
    logger.info("wrote synthetic_metadata.csv (%d rows)", len(synth_records))

    logger.info("building combined metadata ...")
    _build_combined_metadata(data_dir, synth_records)

    durs = [r["duration_sec"] for r in synth_records if r.get("duration_sec")]
    config = {
        "dataset": "AVFAD",
        "mode": "paper_aligned_sentence_only",
        "language": args.language,
        "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "paper_reference": "Koudounas et al., Interspeech 2024",
        "paper_code_reference": "koudounasalkis/AI4Voice/src/data_generation.py (AVFAD branch)",
        "target_synth_ratio": args.target_synth_ratio,
        "method": (
            "1 real sentence rep wav -> 1 base synthetic sentence rep wav; "
            "if target_synth_ratio > 1.0, randomly selected real reps also get "
            "one extra synthetic wav with suffix _extra1 until the target ratio "
            "is reached. speaker_wav = the real rep wav itself (zero-shot voice "
            "cloning). text = the 6 AVFAD CAPE-V Portuguese sentences joined "
            "into one utterance."
        ),
        "text_used": AVFAD_CAPE_V_SENTENCES_PT,
        "n_source_sentences": len(AVFAD_CAPE_V_SENTENCES_PT),
        "classes_synthesized": "pathological_only" if args.only_pathological else "both",
        "n_real_rows_total": len(rows),
        "n_synthetic_rows_total": len(synth_records),
        "n_new_this_run": n_new,
        "n_new_base_this_run": max(0, len(pending_rows) - base_errors),
        "n_new_extra_this_run": max(0, len(pending_extra_rows) - extra_errors),
        "n_skipped_existing": len(skipped_existing),
        "n_skipped_existing_extra": len(skipped_existing_extra),
        "n_skipped_bad_reference": len(skipped_bad_ref),
        "n_errors": errors,
        "elapsed_seconds_this_run": round(elapsed, 1),
        "duration_sec_stats": ({
            "mean": float(np.mean(durs)),
            "median": float(np.median(durs)),
            "min": float(np.min(durs)),
            "max": float(np.max(durs)),
        } if durs else {}),
        "differences_vs_previous_avfad_tts": {
            "old_method": "9 distinct generic Portuguese sentences per speaker, "
                          "synthesized as 9 short wavs, then concatenated and "
                          "split into 3 reps -> matches a 'sentence rep' "
                          "structure but uses different text than AVFAD",
            "new_method": "1:1 mapping; for each real rep wav, synthesize ONCE "
                          "with the 6 AVFAD CAPE-V sentences as text; speaker_wav "
                          "is that rep wav itself",
            "preserved": "synthetic_metadata.csv schema, combined_*_metadata.csv "
                         "schema, file naming pattern tts_{spk}_{pathology}_sentence_rep{r}.wav",
        },
        "device": args.device,
        "target_sample_rate": args.target_sr,
        "xtts_native_sample_rate": xtts_native_sr,
        "seed": args.seed,
    }
    cfg_path = data_dir / "tts_generation_paper_aligned_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", cfg_path)

    logger.info("=" * 72)
    logger.info("DONE -> %s", data_dir)
    logger.info("  synthetic wavs total: %d (new this run: %d, errors: %d)",
                len(synth_records), n_new, errors)
    if durs:
        logger.info("  duration mean=%.2fs median=%.2fs",
                    config["duration_sec_stats"]["mean"],
                    config["duration_sec_stats"]["median"])
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
