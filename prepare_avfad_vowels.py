#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

import librosa
from sklearn.model_selection import StratifiedKFold


SENTENCE_INDICES = ("004", "005", "006", "007", "008", "009")
VOWEL_INDICES = ("001", "002", "003")
ALL_REQUIRED_INDICES = VOWEL_INDICES + SENTENCE_INDICES

# Raw AVFAD index -> recording_type (matches avfad_data_cleaning.py / paper MoE)
VOWEL_INDEX_TO_TYPE = {
    "001": "vowel_i",
    "002": "vowel_a",
    "003": "vowel_u",
}

DEFAULT_OUTPUT_DIR = Path("/home/beierwang/testvscode/avfad_paper_aligned_sv")


def _setup_logging(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("avfad_paper_aligned_sv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _load_xlsx_labels(xlsx_path: Path, logger: logging.Logger) -> pd.DataFrame:
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
    with zfile.open(info, "r") as fp:
        data = fp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr,
                                 res_type="polyphase")
    return audio.astype(np.float32, copy=False)


def _remove_silence(audio: np.ndarray, method: str, top_db: float) -> np.ndarray:
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


def _duration_stats(durs: List[float]) -> Dict[str, float]:
    if not durs:
        return {}
    arr = np.array(durs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _process_sentences(
    spk: str,
    label: str,
    zfile: zipfile.ZipFile,
    idx_map: Dict[str, zipfile.ZipInfo],
    out_dir: Path,
    target_sr: int,
    vad_method: str,
    top_db: float,
    num_reps: int,
) -> Tuple[List[Dict], List[float], List[Dict]]:
    """Return (detection_rows, durations, short_warnings)."""
    trimmed_pieces = []
    for idx in SENTENCE_INDICES:
        audio = _load_wav_from_zip(zfile, idx_map[idx], target_sr)
        audio = _remove_silence(audio, vad_method, top_db)
        if audio.size > 0:
            trimmed_pieces.append(audio)
    if not trimmed_pieces:
        raise RuntimeError("all 6 sentences became empty after VAD")
    concat = np.concatenate(trimmed_pieces, axis=0)
    reps = _split_into_reps(concat, num_reps)

    rows: List[Dict] = []
    durs: List[float] = []
    short_warnings: List[Dict] = []
    for r, piece in enumerate(reps, 1):
        out_name = f"{spk}_{label}_sentence_rep{r}.wav"
        dur = _save_wav(out_dir / "sentences" / out_name, piece, target_sr)
        durs.append(dur)
        rows.append({
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
                "speaker_id": spk, "recording_type": "sentence",
                "rep": r, "duration_sec": dur,
            })
    return rows, durs, short_warnings


def _process_vowels(
    spk: str,
    label: str,
    zfile: zipfile.ZipFile,
    idx_map: Dict[str, zipfile.ZipInfo],
    out_dir: Path,
    target_sr: int,
    vad_method: str,
    top_db: float,
) -> Tuple[List[Dict], List[float], List[Dict]]:
    """Return (detection_rows, durations, short_warnings)."""
    rows: List[Dict] = []
    durs: List[float] = []
    short_warnings: List[Dict] = []
    for idx in VOWEL_INDICES:
        rec_type = VOWEL_INDEX_TO_TYPE[idx]
        audio = _load_wav_from_zip(zfile, idx_map[idx], target_sr)
        audio = _remove_silence(audio, vad_method, top_db)
        if audio.size == 0:
            raise RuntimeError(f"{rec_type} became empty after VAD")
        out_name = f"{spk}_{label}_{rec_type}.wav"
        dur = _save_wav(out_dir / "vowels" / out_name, audio, target_sr)
        durs.append(dur)
        rows.append({
            "speaker_id": spk,
            "label": label,
            "recording_type": rec_type,
            "output_filename": out_name,
            "source_dir": "original",
            "rep": "",
            "duration_sec": round(dur, 3),
        })
        if dur < 1.0:
            short_warnings.append({
                "speaker_id": spk, "recording_type": rec_type,
                "duration_sec": dur,
            })
    return rows, durs, short_warnings


def main() -> int:
    ap = argparse.ArgumentParser(
        description="AVFAD paper-aligned cleaning: sentences + vowels -> NEW output dir",
    )
    ap.add_argument("--raw_dir", required=True, type=Path,
                    help="dir containing AVFAD_01_00_00_*.zip")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"NEW output dir (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--xlsx", required=True, type=Path,
                    help="AVFAD_01_00_00.xlsx path")
    ap.add_argument("--audio_zips", nargs="+", default=[
        "AVFAD_01_00_00_2_A_to_C.zip",
        "AVFAD_01_00_00_3_D_to_L.zip",
        "AVFAD_01_00_00_4_M.zip",
        "AVFAD_01_00_00_5_N_to_Z.zip",
    ])
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--sentence_vad_method", type=str, default="split",
                    choices=["none", "trim", "split"],
                    help="VAD for sentences before 6-way concat (default: split)")
    ap.add_argument("--sentence_top_db", type=float, default=20.0,
                    help="top_db for sentence VAD (default: 20 with split)")
    ap.add_argument("--vowel_vad_method", type=str, default="trim",
                    choices=["none", "trim", "split"],
                    help="VAD for vowels; trim recommended (default: trim)")
    ap.add_argument("--vowel_top_db", type=float, default=30.0,
                    help="top_db for vowel VAD (default: 30 with trim)")
    ap.add_argument("--num_reps", type=int, default=3,
                    help="sentence repetitions after concat (default: 3)")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_sentences", action="store_true",
                    help="Only write vowels/ (speaker filter unchanged)")
    ap.add_argument("--skip_vowels", action="store_true",
                    help="Only write sentences/ (same as original script)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Speaker filter + counts only; no audio written")
    args = ap.parse_args()

    if args.skip_sentences and args.skip_vowels:
        print("ERROR: cannot set both --skip_sentences and --skip_vowels", file=sys.stderr)
        return 2

    out_dir: Path = args.output_dir.resolve()
    if not args.dry_run:
        if out_dir.exists() and any(out_dir.iterdir()):
            existing = sorted(p.name for p in out_dir.iterdir())
            print(f"ERROR: {out_dir} already exists and is not empty:", file=sys.stderr)
            for n in existing[:20]:
                print(f"  {n}", file=sys.stderr)
            print("Refusing to write. Pick a fresh --output_dir.", file=sys.stderr)
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_aux").mkdir(parents=True, exist_ok=True)
        if not args.skip_sentences:
            (out_dir / "sentences").mkdir(parents=True, exist_ok=True)
        if not args.skip_vowels:
            (out_dir / "vowels").mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "cleaning_paper_aligned_sv.log"
    else:
        log_path = None
    logger = _setup_logging(log_path)

    logger.info("=" * 72)
    logger.info("AVFAD PAPER-ALIGNED CLEANING (sentences + vowels)")
    logger.info("  RAW_DIR              = %s", args.raw_dir)
    logger.info("  OUT_DIR              = %s", out_dir)
    logger.info("  XLSX                 = %s", args.xlsx)
    logger.info("  TARGET_SR            = %d", args.target_sr)
    logger.info("  SENTENCE VAD         = %s (top_db=%.1f)",
                args.sentence_vad_method, args.sentence_top_db)
    logger.info("  VOWEL VAD            = %s (top_db=%.1f)",
                args.vowel_vad_method, args.vowel_top_db)
    logger.info("  NUM_SENTENCE_REPS    = %d", args.num_reps)
    logger.info("  SKIP_SENTENCES       = %s", args.skip_sentences)
    logger.info("  SKIP_VOWELS          = %s", args.skip_vowels)
    logger.info("  DRY RUN              = %s", args.dry_run)
    logger.info("=" * 72)

    aux_xlsx = out_dir / "_aux" / args.xlsx.name
    if not aux_xlsx.exists() and not args.dry_run:
        try:
            shutil.copy2(args.xlsx, aux_xlsx)
        except Exception as e:
            logger.warning("could not copy xlsx into _aux/: %s", e)

    labels = _load_xlsx_labels(args.xlsx, logger)
    spk_to_zip = _index_zips(args.raw_dir, args.audio_zips, logger)
    logger.info("zips cover %d candidate speakers", len(spk_to_zip))

    kept: List[str] = []
    dropped_no_label: List[str] = []
    dropped_missing: List[Tuple[str, List[str]]] = []
    for spk, (_zf, idx_map) in spk_to_zip.items():
        if spk not in labels.index:
            dropped_no_label.append(spk)
            continue
        missing = [
            i for i in ALL_REQUIRED_INDICES
            if i not in idx_map or idx_map[i].file_size <= 44
        ]
        if missing:
            dropped_missing.append((spk, missing))
            continue
        kept.append(spk)
    kept.sort()

    n_h = sum(labels.loc[s, "label"] == "healthy" for s in kept)
    n_p = len(kept) - n_h
    n_sent_expected = 0 if args.skip_sentences else len(kept) * args.num_reps
    n_vowel_expected = 0 if args.skip_vowels else len(kept) * len(VOWEL_INDICES)
    logger.info("speakers kept: %d (H=%d P=%d)", len(kept), n_h, n_p)
    logger.info("  expected sentence files: %d", n_sent_expected)
    logger.info("  expected vowel files:    %d", n_vowel_expected)
    logger.info("  dropped no_label: %d, dropped missing: %d",
                len(dropped_no_label), len(dropped_missing))

    if args.dry_run:
        logger.info("DRY RUN — no audio written.")
        return 0

    detection_rows: List[Dict] = []
    classification_pathology_counter: Counter = Counter()
    speaker_pathology: Dict[str, str] = {}
    failures: List[Dict] = []
    short_warnings: List[Dict] = []
    sentence_durs: List[float] = []
    vowel_durs: List[float] = []

    t0 = time.time()
    for i, spk in enumerate(kept, 1):
        zfile, idx_map = spk_to_zip[spk]
        row = labels.loc[spk]
        label = row["label"]
        speaker_pathology[spk] = str(row["cmvd_word"])
        if label == "pathological":
            classification_pathology_counter[speaker_pathology[spk]] += 1

        try:
            if not args.skip_sentences:
                s_rows, s_durs, s_short = _process_sentences(
                    spk, label, zfile, idx_map, out_dir,
                    args.target_sr, args.sentence_vad_method,
                    args.sentence_top_db, args.num_reps,
                )
                detection_rows.extend(s_rows)
                sentence_durs.extend(s_durs)
                short_warnings.extend(s_short)

            if not args.skip_vowels:
                v_rows, v_durs, v_short = _process_vowels(
                    spk, label, zfile, idx_map, out_dir,
                    args.target_sr, args.vowel_vad_method, args.vowel_top_db,
                )
                detection_rows.extend(v_rows)
                vowel_durs.extend(v_durs)
                short_warnings.extend(v_short)
        except Exception as e:
            logger.warning("  [skip] %s: %s", spk, e)
            failures.append({"speaker_id": spk, "reason": str(e)})
            continue

        if i % 50 == 0:
            logger.info("  progress %d/%d (%.1fs)", i, len(kept), time.time() - t0)

    if not detection_rows:
        logger.error("no rows produced — aborting")
        return 3

    det_df = pd.DataFrame(detection_rows)
    det_df.to_csv(out_dir / "detection_metadata.csv", index=False)
    det_df.drop(columns=["duration_sec"]).to_csv(
        out_dir / "combined_detection_metadata.csv", index=False
    )
    logger.info("  wrote detection_metadata.csv (%d rows)", len(det_df))

    top_k_classes = [c for c, _ in classification_pathology_counter.most_common(args.top_k)]
    spk_in_topk = {
        spk for spk, path in speaker_pathology.items() if path in top_k_classes
    }
    cls_rows = [
        {
            "speaker_id": r["speaker_id"],
            "pathology": speaker_pathology[r["speaker_id"]],
            "recording_type": r["recording_type"],
            "output_filename": r["output_filename"],
            "source_dir": r["source_dir"],
            "rep": r["rep"],
        }
        for r in detection_rows
        if r["speaker_id"] in spk_in_topk and r["label"] == "pathological"
    ]
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(out_dir / "classification_metadata.csv", index=False)
    cls_df.to_csv(out_dir / "combined_classification_metadata.csv", index=False)

    clinical_rows = [
        {
            "speaker_id": spk,
            "label": labels.loc[spk, "label"],
            "cmvd_numeric": int(labels.loc[spk, "cmvd_numeric"]),
            "cmvd_word": labels.loc[spk, "cmvd_word"],
        }
        for spk in kept
        if spk in {r["speaker_id"] for r in detection_rows}
    ]
    pd.DataFrame(clinical_rows).to_csv(out_dir / "participants_clinical.csv", index=False)

    spk_label_pairs = sorted({(r["speaker_id"], r["label"]) for r in detection_rows})
    folds = _stratified_cv_folds(spk_label_pairs, args.n_folds, args.seed)
    with open(out_dir / "cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2, ensure_ascii=False)

    rec_type_counts = dict(Counter(r["recording_type"] for r in detection_rows))
    dataset_config = {
        "dataset": "AVFAD",
        "total_speakers": len({r["speaker_id"] for r in detection_rows}),
        "healthy_speakers": int(sum(1 for r in clinical_rows if r["label"] == "healthy")),
        "pathological_speakers": int(sum(1 for r in clinical_rows if r["label"] == "pathological")),
        "recording_type_counts": rec_type_counts,
        "total_recordings": len(detection_rows),
        "top_k_classes": top_k_classes,
        "n_folds": args.n_folds,
        "sampling_rate": args.target_sr,
        "label_mapping": {"healthy": 0, "pathological": 1},
        "skipped_recording_types": ["passage", "spontaneous_speech"],
    }
    with open(out_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)

    summary = {
        "mode": "paper_aligned_sentences_and_vowels",
        "input_dir": str(args.raw_dir),
        "output_dir": str(out_dir),
        "paper_reference": "Koudounas et al., Interspeech 2024",
        "sentence_handling": {
            "skipped": args.skip_sentences,
            "indices": list(SENTENCE_INDICES),
            "vad_method": args.sentence_vad_method,
            "top_db": args.sentence_top_db,
            "num_reps": args.num_reps,
        },
        "vowel_handling": {
            "skipped": args.skip_vowels,
            "indices": list(VOWEL_INDICES),
            "index_to_type": VOWEL_INDEX_TO_TYPE,
            "vad_method": args.vowel_vad_method,
            "top_db": args.vowel_top_db,
            "note": "Raw AVFAD vowel wavs already contain 3 repetitions; "
                    "we resample + light trim, no rep-split.",
        },
        "n_speakers_kept": len(kept),
        "n_output_files": len(detection_rows),
        "recording_type_counts": rec_type_counts,
        "sentence_duration_sec_stats": _duration_stats(sentence_durs),
        "vowel_duration_sec_stats": _duration_stats(vowel_durs),
        "n_failures": len(failures),
        "failures": failures,
        "short_warnings": short_warnings,
        "seed": args.seed,
        "does_not_modify": [
            "avfad_paper_aligned/",
            "avfad_paper_cleaned/",
            "avfad_cleaned/",
            "avfad_clean_paper_aligned.py",
        ],
    }
    with open(out_dir / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 72)
    logger.info("DONE -> %s", out_dir)
    logger.info("  speakers: %d | total files: %d", len(kept), len(detection_rows))
    logger.info("  recording types: %s", rec_type_counts)
    if sentence_durs:
        logger.info("  sentence dur median=%.2fs", float(np.median(sentence_durs)))
    if vowel_durs:
        logger.info("  vowel dur median=%.2fs", float(np.median(vowel_durs)))
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
