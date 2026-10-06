#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

import librosa
from sklearn.model_selection import StratifiedKFold


SENTENCE_INDICES = ("004", "005", "006", "007", "008", "009")
VOWEL_INDICES = ("001", "002", "003")
ALL_REQUIRED_INDICES = VOWEL_INDICES + SENTENCE_INDICES


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("avfad_paper_aligned")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _load_xlsx_labels(xlsx_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Return a DataFrame indexed by File ID with label / cmvd columns."""
    df = pd.read_excel(xlsx_path, sheet_name="AVFAD")
    df = df.rename(columns=str.strip)
    label_col_num = "CMVD-I Dimension 1 (numeric system)"
    label_col_word = "CMVD-I Dimension 1 (word system)"
    if label_col_num not in df.columns:
        raise ValueError(f"missing column {label_col_num!r} in {xlsx_path}")
    df = df[["File ID", label_col_num, label_col_word]].copy()
    df = df.rename(
        columns={
            "File ID": "speaker_id",
            label_col_num: "cmvd_numeric",
            label_col_word: "cmvd_word",
        }
    )
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df = df.dropna(subset=["speaker_id"])
    df = df[df["speaker_id"] != ""]
    df["cmvd_numeric"] = pd.to_numeric(df["cmvd_numeric"], errors="coerce")
    df = df.dropna(subset=["cmvd_numeric"])
    df["cmvd_numeric"] = df["cmvd_numeric"].astype(int)
    df["label"] = np.where(df["cmvd_numeric"] == 0, "healthy", "pathological")
    df["cmvd_word"] = df["cmvd_word"].fillna("Normal").astype(str).str.strip()
    n_h = int((df["label"] == "healthy").sum())
    n_p = int((df["label"] == "pathological").sum())
    logger.info("loaded %d speakers from xlsx (H=%d P=%d)", len(df), n_h, n_p)
    return df.set_index("speaker_id")


def _index_zips(raw_dir: Path, audio_zips: List[str], logger: logging.Logger
                ) -> Dict[str, Tuple[zipfile.ZipFile, Dict[str, zipfile.ZipInfo]]]:
    """Open every zip and build {speaker_id: (zfile, {idx: ZipInfo})}."""
    spk_to_zip: Dict[str, Tuple[zipfile.ZipFile, Dict[str, zipfile.ZipInfo]]] = {}
    for zip_name in audio_zips:
        zpath = raw_dir / zip_name
        logger.info("indexing %s ...", zip_name)
        zfile = zipfile.ZipFile(zpath, "r")
        for info in zfile.infolist():
            name = info.filename
            if not name.lower().endswith(".wav"):
                continue
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            stem = Path(name).stem
            if len(stem) < 6:
                continue
            spk = stem[:3]
            idx = stem[3:6]
            if idx not in ALL_REQUIRED_INDICES:
                continue
            entry = spk_to_zip.setdefault(spk, (zfile, {}))
            entry[1][idx] = info
    return spk_to_zip


def _load_wav_from_zip(zfile: zipfile.ZipFile, info: zipfile.ZipInfo,
                       target_sr: int) -> np.ndarray:
    """Read a wav out of a zip member, return mono float32 at target_sr."""
    with zfile.open(info, "r") as fp:
        data = fp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # ``polyphase`` uses scipy (no extra deps), and AVFAD wavs are
        # 48 kHz -> 16 kHz which is exactly /3, so it is fast and clean.
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr,
                                 res_type="polyphase")
    return audio.astype(np.float32, copy=False)


def _remove_silence(audio: np.ndarray, method: str, top_db: float) -> np.ndarray:
    """Remove silence per-sentence.

    method:
        - ``"none"``: return audio unchanged
        - ``"trim"``: ``librosa.effects.trim`` — strips head/tail silence only
        - ``"split"``: ``librosa.effects.split`` — also drops internal silences
          between voiced segments (recommended for matching paper T(s))
    """
    if audio.size == 0 or method == "none":
        return audio
    if method == "trim":
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed if trimmed.size > 0 else audio
    if method == "split":
        intervals = librosa.effects.split(audio, top_db=top_db)
        if intervals.shape[0] == 0:
            return audio
        return np.concatenate([audio[s:e] for s, e in intervals])
    raise ValueError(f"unknown vad_method {method!r}")


def _split_into_reps(audio: np.ndarray, num_reps: int) -> List[np.ndarray]:
    """Split audio into ``num_reps`` equal-length pieces.

    The tail piece absorbs any remainder so total length is preserved.
    """
    n = audio.shape[0]
    if n == 0 or num_reps <= 0:
        return [audio]
    step = n // num_reps
    pieces: List[np.ndarray] = []
    for r in range(num_reps):
        start = r * step
        end = n if r == num_reps - 1 else (r + 1) * step
        pieces.append(audio[start:end])
    return pieces


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = np.clip(audio, -1.0, 1.0)
    pcm16 = (clip * 32767.0).astype(np.int16)
    sf.write(str(path), pcm16, sr, subtype="PCM_16")
    return pcm16.shape[0] / sr


def _stratified_cv_folds(spk_label_pairs: List[Tuple[str, str]], n_folds: int,
                         seed: int) -> List[Dict[str, List[str]]]:
    speakers = np.array([s for s, _ in spk_label_pairs])
    y = np.array([0 if l == "healthy" else 1 for _, l in spk_label_pairs])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: List[Dict[str, List[str]]] = []
    for k, (tr, te) in enumerate(skf.split(speakers, y)):
        folds.append({
            "fold": k,
            "train_speakers": sorted(speakers[tr].tolist()),
            "test_speakers": sorted(speakers[te].tolist()),
        })
    return folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path,
                    help="dir containing AVFAD_01_00_00_*.zip")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="NEW output dir, must NOT overlap with avfad_paper_cleaned/")
    ap.add_argument("--xlsx", required=True, type=Path,
                    help="AVFAD_01_00_00.xlsx (use the one already in avfad_paper_cleaned/_aux/)")
    ap.add_argument("--audio_zips", nargs="+", default=[
        "AVFAD_01_00_00_2_A_to_C.zip",
        "AVFAD_01_00_00_3_D_to_L.zip",
        "AVFAD_01_00_00_4_M.zip",
        "AVFAD_01_00_00_5_N_to_Z.zip",
    ])
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--vad_method", type=str, default="split",
                    choices=["none", "trim", "split"],
                    help="per-sentence silence removal before concatenation. "
                         "'split' (default) drops internal pauses too and is "
                         "the recommended setting for matching the paper's "
                         "T(s)=15.86s. 'trim' only strips head/tail silence. "
                         "'none' disables silence removal.")
    ap.add_argument("--top_db", type=float, default=20.0,
                    help="top_db threshold (dB below max ref) for the "
                         "selected --vad_method. 20 is a good starting point "
                         "with split; 30 with trim.")
    ap.add_argument("--num_reps", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=6,
                    help="number of most-frequent pathology macro-classes for classification CSV")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_vowels_for_filter", action="store_true", default=True,
                    help="Require all 3 vowel wavs to also exist for speaker to be kept (default ON, matches paper 663-spk count target)")
    ap.add_argument("--no_require_vowels_for_filter", dest="require_vowels_for_filter",
                    action="store_false",
                    help="If set, only require the 6 sentence wavs to exist (=> closer to previous 707 count)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only run speaker filtering + report counts. No audio written.")
    args = ap.parse_args()

    out_dir: Path = args.output_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        existing = sorted(p.name for p in out_dir.iterdir())
        if not args.dry_run:
            print(f"ERROR: {out_dir} already exists and is not empty:", file=sys.stderr)
            for n in existing[:20]:
                print(f"  {n}", file=sys.stderr)
            print("Refusing to write. Pick a fresh --output_dir.", file=sys.stderr)
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_aux").mkdir(parents=True, exist_ok=True)
    (out_dir / "sentences").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "cleaning_paper_aligned.log"
    logger = _setup_logging(log_path)

    logger.info("=" * 72)
    logger.info("AVFAD PAPER-ALIGNED CLEANING (sentences only)")
    logger.info("  RAW_DIR     = %s", args.raw_dir)
    logger.info("  OUT_DIR     = %s", out_dir)
    logger.info("  XLSX        = %s", args.xlsx)
    logger.info("  AUDIO_ZIPS  = %s", args.audio_zips)
    logger.info("  TARGET_SR   = %d", args.target_sr)
    logger.info("  VAD_METHOD  = %s (top_db=%.1f)", args.vad_method, args.top_db)
    logger.info("  NUM_REPS    = %d", args.num_reps)
    logger.info("  SPK FILTER  = %s", "require XXX001..XXX009 all present"
                if args.require_vowels_for_filter else "require XXX004..XXX009 only")
    logger.info("  DRY RUN     = %s", args.dry_run)
    logger.info("=" * 72)

    aux_xlsx = out_dir / "_aux" / args.xlsx.name
    if not aux_xlsx.exists():
        try:
            shutil.copy2(args.xlsx, aux_xlsx)
        except Exception as e:
            logger.warning("could not copy xlsx into _aux/: %s", e)

    labels = _load_xlsx_labels(args.xlsx, logger)

    spk_to_zip = _index_zips(args.raw_dir, args.audio_zips, logger)
    logger.info("zips cover %d candidate speakers", len(spk_to_zip))

    required = ALL_REQUIRED_INDICES if args.require_vowels_for_filter else SENTENCE_INDICES

    kept: List[str] = []
    dropped_no_label: List[str] = []
    dropped_missing: List[Tuple[str, List[str]]] = []
    for spk, (_zf, idx_map) in spk_to_zip.items():
        if spk not in labels.index:
            dropped_no_label.append(spk)
            continue
        missing = [i for i in required if i not in idx_map or idx_map[i].file_size <= 44]
        if missing:
            dropped_missing.append((spk, missing))
            continue
        kept.append(spk)
    kept.sort()

    n_h = sum(labels.loc[s, "label"] == "healthy" for s in kept)
    n_p = len(kept) - n_h
    logger.info("speakers kept after strict filter: %d (H=%d, P=%d)", len(kept), n_h, n_p)
    logger.info("  dropped no_label: %d, dropped missing-files: %d",
                len(dropped_no_label), len(dropped_missing))
    if dropped_missing[:5]:
        for spk, miss in dropped_missing[:5]:
            logger.info("    e.g. %s missing %s", spk, ",".join(miss))

    if args.dry_run:
        logger.info("DRY RUN — no audio written. Stop.")
        return 0

    detection_rows: List[Dict] = []
    classification_pathology_counter: Counter = Counter()
    speaker_pathology: Dict[str, str] = {}
    failures: List[Dict] = []
    short_warnings: List[Dict] = []
    duration_records: List[float] = []

    t0 = time.time()
    for i, spk in enumerate(kept, 1):
        zfile, idx_map = spk_to_zip[spk]
        row = labels.loc[spk]
        label = row["label"]
        cmvd_numeric = int(row["cmvd_numeric"])
        cmvd_word = str(row["cmvd_word"])
        speaker_pathology[spk] = cmvd_word
        if label == "pathological":
            classification_pathology_counter[cmvd_word] += 1

        try:
            trimmed_pieces = []
            for idx in SENTENCE_INDICES:
                audio = _load_wav_from_zip(zfile, idx_map[idx], args.target_sr)
                audio = _remove_silence(audio, args.vad_method, args.top_db)
                if audio.size > 0:
                    trimmed_pieces.append(audio)
            if not trimmed_pieces:
                raise RuntimeError("all 6 sentences became empty after VAD")
            concat = np.concatenate(trimmed_pieces, axis=0)
            reps = _split_into_reps(concat, args.num_reps)
        except Exception as e:
            logger.warning("  [skip] %s: %s", spk, e)
            failures.append({"speaker_id": spk, "reason": str(e)})
            continue

        per_rep_dur = []
        for r, piece in enumerate(reps, 1):
            out_name = f"{spk}_{label}_sentence_rep{r}.wav"
            out_path = out_dir / "sentences" / out_name
            dur = _save_wav(out_path, piece, args.target_sr)
            per_rep_dur.append(dur)
            duration_records.append(dur)
            detection_rows.append({
                "speaker_id": spk,
                "label": label,
                "recording_type": "sentence",
                "output_filename": out_name,
                "source_dir": "original",
                "rep": r,
                "duration_sec": round(dur, 3),
            })
            if dur < 2.0:
                short_warnings.append({
                    "speaker_id": spk, "rep": r, "duration_sec": dur,
                })

        if i % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d (elapsed %.1fs)", i, len(kept), elapsed)

    if not detection_rows:
        logger.error("no rows produced — aborting before writing CSV / folds")
        return 3

    det_df = pd.DataFrame(detection_rows)
    det_df.to_csv(out_dir / "detection_metadata.csv", index=False)
    logger.info("  wrote detection_metadata.csv (%d rows)", len(det_df))

    det_df.drop(columns=["duration_sec"]).to_csv(
        out_dir / "combined_detection_metadata.csv", index=False
    )

    top_k_classes = [c for c, _ in classification_pathology_counter.most_common(args.top_k)]
    cls_rows: List[Dict] = []
    spk_in_topk = {
        spk for spk, path in speaker_pathology.items() if path in top_k_classes
    }
    for r in detection_rows:
        if r["speaker_id"] in spk_in_topk and r["label"] == "pathological":
            cls_rows.append({
                "speaker_id": r["speaker_id"],
                "pathology": speaker_pathology[r["speaker_id"]],
                "recording_type": r["recording_type"],
                "output_filename": r["output_filename"],
                "source_dir": r["source_dir"],
                "rep": r["rep"],
            })
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(out_dir / "classification_metadata.csv", index=False)
    cls_df.to_csv(out_dir / "combined_classification_metadata.csv", index=False)
    logger.info("  wrote classification_metadata.csv (%d rows, top-%d)",
                len(cls_df), args.top_k)

    clinical_rows = []
    for spk in kept:
        if spk not in {r["speaker_id"] for r in detection_rows}:
            continue
        row = labels.loc[spk]
        clinical_rows.append({
            "speaker_id": spk,
            "label": row["label"],
            "cmvd_numeric": int(row["cmvd_numeric"]),
            "cmvd_word": row["cmvd_word"],
        })
    pd.DataFrame(clinical_rows).to_csv(out_dir / "participants_clinical.csv", index=False)

    spk_label_pairs = sorted({(r["speaker_id"], r["label"]) for r in detection_rows})
    folds = _stratified_cv_folds(spk_label_pairs, args.n_folds, args.seed)
    with open(out_dir / "cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2, ensure_ascii=False)
    for f in folds:
        logger.info("  fold_%d: train=%d test=%d", f["fold"],
                    len(f["train_speakers"]), len(f["test_speakers"]))

    durs = np.array(duration_records, dtype=np.float64)
    dur_stats = {
        "mean": float(durs.mean()),
        "median": float(np.median(durs)),
        "min": float(durs.min()),
        "max": float(durs.max()),
        "p25": float(np.percentile(durs, 25)),
        "p75": float(np.percentile(durs, 75)),
    }

    dataset_config = {
        "dataset": "AVFAD",
        "total_speakers": len({r["speaker_id"] for r in detection_rows}),
        "healthy_speakers": int(sum(1 for r in clinical_rows if r["label"] == "healthy")),
        "pathological_speakers": int(sum(1 for r in clinical_rows if r["label"] == "pathological")),
        "recording_type_counts": {"sentence": len(detection_rows)},
        "total_recordings": len(detection_rows),
        "top_k_classes": top_k_classes,
        "n_folds": args.n_folds,
        "sampling_rate": args.target_sr,
        "label_mapping": {"healthy": 0, "pathological": 1},
        "skipped_recording_types": [
            "vowel_a", "vowel_e", "vowel_o", "passage", "spontaneous_speech",
        ],
    }
    with open(out_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)

    summary = {
        "mode": "paper_aligned_sentence_only",
        "input_dir": str(args.raw_dir),
        "output_dir": str(out_dir),
        "audio_zips_used": args.audio_zips,
        "xlsx_path": str(args.xlsx),
        "label_source": {
            "label_source_column": "CMVD-I Dimension 1 (numeric system)",
            "label_rule": "value == 0 -> healthy ; value != 0 -> pathological "
                          "(consistent with the AVFAD xlsx Classifications sheet, "
                          "where CMVD-I Dim 1 == 0 has 363 speakers labelled Normal)",
        },
        "paper_reference": "Koudounas et al., Interspeech 2024",
        "paper_quote": "we concatenate the six sentences together, having three repetitions for each obtained audio",
        "sentence_handling": {
            "indices_used": list(SENTENCE_INDICES),
            "method": "per-sentence VAD={vm} (librosa.effects.{vm}, top_db={td}); "
                      "concatenate XXX004..XXX009; split into {n} equal-time sub-recordings".format(
                          vm=args.vad_method, td=args.top_db, n=args.num_reps),
            "num_reps_per_speaker": args.num_reps,
            "vad_method": args.vad_method,
            "vad_top_db": args.top_db,
        },
        "speaker_filter": {
            "rule": "speaker kept iff has valid xlsx label AND ALL of XXX{ix} exist with file_size > 44 bytes".format(
                ix="/".join(required)),
            "n_speakers_indexed_in_zips": len(spk_to_zip),
            "n_speakers_dropped_no_label": len(dropped_no_label),
            "n_speakers_dropped_missing_files": len(dropped_missing),
            "examples_dropped_missing_files": [
                {"speaker_id": s, "missing": m} for s, m in dropped_missing[:20]
            ],
        },
        "target_sampling_rate": args.target_sr,
        "target_channels": 1,
        "target_format": "WAV PCM_16",
        "n_speakers_kept": len({r["speaker_id"] for r in detection_rows}),
        "n_healthy_speakers": dataset_config["healthy_speakers"],
        "n_pathological_speakers": dataset_config["pathological_speakers"],
        "n_output_sentence_files": len(detection_rows),
        "n_failures_in_audio_stage": len(failures),
        "failures": failures,
        "short_warnings": short_warnings,
        "actual_duration_sec_stats": dur_stats,
        "top_k_pathologies": top_k_classes,
        "n_folds": args.n_folds,
        "differences_from_avfad_paper_cleaned": {
            "filter": "previous run required only XXX004..XXX009 to be loadable; "
                      "this run additionally requires XXX001..XXX003 to exist "
                      "with file_size>44. Empirically, both rules give 708 "
                      "speakers (only PLS is dropped, because PLS007.wav is a "
                      "44-byte empty file). The paper's 663-speaker count "
                      "cannot be reproduced from the public repo because the "
                      "metadata.csv that defines it is not published.",
            "silence_removal": "previous run did NOT remove silence before "
                               "concatenation; this run uses librosa.effects.{vm} "
                               "(top_db={td}) per-sentence before concatenation"
                               .format(vm=args.vad_method, td=args.top_db),
            "synthetic_branch": "this script does NOT produce TTS / synthetic data. "
                                "If needed, point the existing TTS scripts at this new "
                                "directory after this run completes.",
            "preserved_from_previous_run": "label rule, 16 kHz mono PCM_16, "
                                           "stratified 10-fold CV (seed=42), top-K "
                                           "pathology classification CSV layout.",
        },
        "paper_target_numbers": {
            "speakers_with_sentences": 663,
            "n_sentence_files": 1989,
            "mean_duration_sec": 15.86,
        },
        "seed": args.seed,
    }
    with open(out_dir / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 72)
    logger.info("CLEANING COMPLETE -> %s", out_dir)
    logger.info("  speakers kept: %d (H=%d, P=%d)",
                dataset_config["total_speakers"],
                dataset_config["healthy_speakers"],
                dataset_config["pathological_speakers"])
    logger.info("  output sentence files: %d", len(detection_rows))
    logger.info("  duration mean=%.2fs median=%.2fs (paper target 15.86s)",
                dur_stats["mean"], dur_stats["median"])
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
