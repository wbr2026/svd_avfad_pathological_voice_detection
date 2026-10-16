#!/usr/bin/env python3
"""Package 2038 complete-9 Barche-9 concats for the vowel bal11 experiment.

NEW output directory only. Does NOT modify existing Barche-9 / whole_iautext
pools, training scripts, or experiment dirs.

Reuses already-concatenated real wavs (order is already
i_low â†?a_low â†?u_low â†?i_normal â†?a_normal â†?u_normal â†?i_high â†?a_high â†?u_high).
Rewrites metadata so each row is one speaker / one concat with
recording_type=vowel_9concat. Audio is symlinked, not re-encoded.

Usage:
  python build_vowel9_concat.py
  python build_vowel9_concat.py --src_dir ./svd_cleaned_paper_tts_vowel_iau_barche9_speaker_aligned \\
      --out_dir ./svd_cleaned_paper_tts_vowel_9concat_bal11
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

DET_FIELDS = ["speaker_id", "label", "recording_type", "output_filename", "source_dir"]
REC = "vowel_9concat"
EXPECTED_ORDER = [
    "i_low", "a_low", "u_low",
    "i_normal", "a_normal", "u_normal",
    "i_high", "a_high", "u_high",
]


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src.resolve())


def _load_orig_rows(src_dir: Path) -> list[dict]:
    meta = src_dir / "detection_metadata.csv"
    if not meta.exists():
        meta = src_dir / "combined_detection_metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"no detection metadata under {src_dir}")
    rows = []
    with open(meta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("source_dir", "original") == "synthetic":
                continue
            if not str(r.get("recording_type", "")).startswith("vowel"):
                continue
            if r.get("recording_type") not in ("vowel_iau_barche9", REC):
                continue
            rows.append(dict(r))
    return rows


def _duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def build(src_dir: Path, out_dir: Path) -> dict:
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_rows = _load_orig_rows(src_dir)
    src_vowels = src_dir / "vowels"
    if not src_vowels.exists():
        raise FileNotFoundError(f"missing vowels dir: {src_vowels}")

    by_spk: dict[str, list[dict]] = {}
    for r in src_rows:
        sid = str(r["speaker_id"]).strip()
        if not sid:
            continue
        by_spk.setdefault(sid, []).append(r)

    det_rows: list[dict] = []
    durations: list[float] = []
    skipped_dup = skipped_missing = skipped_conflict = 0
    out_vowels = out_dir / "vowels"
    out_vowels.mkdir(parents=True, exist_ok=True)

    for sid in sorted(by_spk.keys(), key=lambda x: int(x) if x.isdigit() else x):
        rows = by_spk[sid]
        labels = {r["label"] for r in rows}
        if len(labels) != 1:
            skipped_conflict += 1
            continue
        if len(rows) != 1:
            skipped_dup += 1
            continue
        r = rows[0]
        src_wav = src_vowels / r["output_filename"]
        if not src_wav.exists():
            skipped_missing += 1
            continue
        label = r["label"]
        out_name = r["output_filename"]
        _symlink(src_wav, out_vowels / out_name)
        dur = _duration_sec(src_wav)
        durations.append(dur)
        det_rows.append({
            "speaker_id": sid,
            "label": label,
            "recording_type": REC,
            "output_filename": out_name,
            "source_dir": "original",
        })

    with open(out_dir / "detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DET_FIELDS)
        w.writeheader()
        for r in det_rows:
            w.writerow(r)

    with open(out_dir / "combined_detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DET_FIELDS)
        w.writeheader()
        for r in det_rows:
            w.writerow(r)

    arr = np.asarray(durations, dtype=np.float64)
    n = int(arr.size)
    label_counts = dict(Counter(r["label"] for r in det_rows))
    duration_report = {
        "n": n,
        "min": float(arr.min()) if n else None,
        "mean": float(arr.mean()) if n else None,
        "median": float(np.median(arr)) if n else None,
        "p95": float(np.percentile(arr, 95)) if n else None,
        "max": float(arr.max()) if n else None,
        "n_gt_20s": int((arr > 20.0).sum()) if n else 0,
        "n_gt_25s": int((arr > 25.0).sum()) if n else 0,
        "n_gt_30s": int((arr > 30.0).sum()) if n else 0,
    }
    recommended = 20.0
    if n and duration_report["n_gt_30s"] > 0:
        recommended = None
    elif n and duration_report["n_gt_25s"] > 0:
        recommended = 30.0
    elif n and duration_report["n_gt_20s"] > 0:
        recommended = 25.0

    report = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "recording_type": REC,
        "concat_order": EXPECTED_ORDER,
        "reused_existing_barche9_wavs": True,
        "n_speakers": len(det_rows),
        "label_counts": label_counts,
        "n_unique_speakers": len({r["speaker_id"] for r in det_rows}),
        "rows_per_speaker": dict(Counter(Counter(r["speaker_id"] for r in det_rows).values())),
        "skipped_duplicate_rows": skipped_dup,
        "skipped_missing_wav": skipped_missing,
        "skipped_label_conflict": skipped_conflict,
        "missing_speaker_id": 0,
        "duration": duration_report,
        "recommended_max_length_sec": recommended,
    }
    with open(out_dir / "build_vowel9_concat_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(out_dir / "duration_report.json", "w", encoding="utf-8") as f:
        json.dump(duration_report, f, indent=2)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Package 2038 Barche-9 concats as vowel_9concat.")
    ap.add_argument(
        "--src_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_tts_vowel_iau_barche9_speaker_aligned"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_tts_vowel_9concat_bal11"),
    )
    args = ap.parse_args()
    report = build(args.src_dir, args.out_dir)
    print(json.dumps(report, indent=2))
    n = report["n_speakers"]
    lc = report["label_counts"]
    if n != 2038 or lc.get("healthy") != 687 or lc.get("pathological") != 1351:
        print(
            f"WARNING: expected 2038 (687/1351), got {n} ({lc})",
            flush=True,
        )
        return 3
    if report["rows_per_speaker"] != {"1": 2038} and report["rows_per_speaker"] != {1: 2038}:
        print(f"WARNING: not 1-row-per-speaker: {report['rows_per_speaker']}", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
