#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("avfad_barche3_whole_iautext")

REC = "vowel_iau_barche3"
LANGUAGE = "pt"

# Same 23-char vowel blocks as SVD whole_iautext, but i→a→u once (3 blocks).
# SVD 9concat uses this 3 times (low / mid / high). Barche-3 is one 24 s clip.
_I = "i" * 23
_A = "a" * 23
_U = "u" * 23
BARCHE3_IAUTEXT = f"{_I}. {_A}. {_U}."

DEFAULT_DATA_DIR = Path(
    "/home/beierwang/testvscode/"
    "avfad_paper_aligned_sv_paper_tts_vowel_barche3_stable_8s_ratio_pool"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/beierwang/testvscode/"
    "avfad_paper_aligned_sv_paper_tts_vowel_barche3_stable_whole_iautext"
)

FORBIDDEN_OUTPUT_DIRS = [
    Path("/home/beierwang/testvscode/avfad_paper_aligned"),
    Path("/home/beierwang/testvscode/avfad_paper_aligned_sv"),
    Path(
        "/home/beierwang/testvscode/"
        "avfad_paper_aligned_sv_paper_tts_vowel_barche3_stable_8s_ratio_pool"
    ),
    Path(
        "/home/beierwang/testvscode/"
        "avfad_paper_aligned_sv_paper_tts_vowel_ratio_pool"
    ),
    Path(
        "/home/beierwang/testvscode/"
        "avfad_paper_aligned_sv_paper_tts_vowel_barche3_stable_8s_ratio_pool/"
        "metadata_synth_100"
    ),
    Path(
        "/home/beierwang/testvscode/"
        "avfad_paper_aligned_sv_paper_tts_vowel_barche3_stable_8s_ratio_pool/"
        "metadata_synth_200"
    ),
    Path("/home/beierwang/testvscode/svd_cleaned_paper_tts_vowel_9concat_bal11"),
    Path(
        "/home/beierwang/testvscode/"
        "svd_cleaned_paper_tts_vowel_9concat_bal11_scale_candidates"
    ),
]

FORBIDDEN_PREFIXES = [
    Path("/home/beierwang/testvscode/AI4Voice copy/src/experiments"),
    Path("/home/beierwang/testvscode/AI4Voice copy/experiments"),
]


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_forbidden_output(out: Path) -> str | None:
    out_r = _resolve(out)
    for f in FORBIDDEN_OUTPUT_DIRS:
        if out_r == _resolve(f):
            return f"output_dir is a forbidden existing pool: {f}"
    for pref in FORBIDDEN_PREFIXES:
        pref_r = _resolve(pref)
        try:
            out_r.relative_to(pref_r)
            return f"output_dir sits under forbidden prefix: {pref}"
        except ValueError:
            pass
    return None


def _save_wav(path: Path, wav, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    sf.write(str(path), wav, sr)


def _pathology_from_filename(fname: str) -> str:
    # AAF_healthy_vowel_iau_barche3.wav → healthy
    # AAC_pathological_vowel_iau_barche3.wav → pathological
    stem = Path(fname).stem
    suffix = f"_{REC}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else "unknown"


def _load_real_concat_rows(data_dir: Path) -> list[dict]:
    meta = data_dir / "detection_metadata.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"missing {meta}")
    rows = []
    with meta.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("recording_type") != REC:
                continue
            if r.get("source_dir", "original") == "synthetic":
                continue
            rows.append(r)
    if not rows:
        raise RuntimeError(f"no original {REC} rows in {meta}")
    return rows


def _build_work_for_label(
    speaker_ids: list[str],
    label: str,
    rec: str,
    target_synth_ratio: float,
    seed: int,  # kept for call-site compatibility; extras are round-robin
) -> list[dict]:
    """1x per speaker, then extra copies independently for this label."""
    work: list[dict] = []
    for sid in speaker_ids:
        work.append(
            {
                "speaker_id": sid,
                "label": label,
                "recording_type": rec,
                "copy": "1x",
                "seed_offset": 0,
                "out_name": f"tts_{sid}_{label}_{rec}.wav",
            }
        )
    n_1x = len(work)
    n_extra_total = max(0, int(round((target_synth_ratio - 1.0) * n_1x)))
    if n_extra_total <= 0:
        return work
    extra_counts = {sid: 0 for sid in speaker_ids}
    for i in range(n_extra_total):
        extra_counts[speaker_ids[i % len(speaker_ids)]] += 1
    n = 1
    for sid in speaker_ids:
        for k in range(extra_counts[sid]):
            work.append(
                {
                    "speaker_id": sid,
                    "label": label,
                    "recording_type": rec,
                    "copy": f"extra{n}",
                    "seed_offset": n,
                    "out_name": f"tts_{sid}_{label}_{rec}_extra{n}.wav",
                }
            )
            n += 1
    return work


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AVFAD barche-3 whole-clip XTTS (script only; default dry-run).",
    )
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--language", default=LANGUAGE)
    p.add_argument("--xtts_model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--target_synth_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_ref_duration_sec", type=float, default=1.0)
    p.add_argument("--max_speakers", type=int, default=None)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print jobs and exit. No XTTS, no wavs, no output_dir writes.",
    )
    p.add_argument(
        "--write_job_list",
        action="store_true",
        help="With --dry_run, still write generation_jobs.json under output_dir "
        "(only if output_dir does not already exist).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    out_dir = _resolve(args.output_dir)

    if not data_dir.is_dir():
        logger.error("data_dir does not exist: %s", data_dir)
        return 2

    reason = _is_forbidden_output(out_dir)
    if reason:
        logger.error("REFUSING to write: %s", reason)
        return 2
    if out_dir == data_dir:
        logger.error("REFUSING: output_dir must not equal data_dir")
        return 2
    try:
        out_dir.relative_to(data_dir)
        logger.error("REFUSING: output_dir is inside data_dir (%s)", data_dir)
        return 2
    except ValueError:
        pass

    if out_dir.exists() and not args.dry_run:
        logger.error("REFUSING: output_dir already exists: %s", out_dir)
        return 2
    if out_dir.exists() and args.dry_run and args.write_job_list:
        logger.error("REFUSING: --write_job_list but output_dir exists: %s", out_dir)
        return 2

    real_rows = _load_real_concat_rows(data_dir)
    vowels_dir = data_dir / "vowels"

    by_label: dict[str, list[str]] = {"healthy": [], "pathological": []}
    row_by_sid: dict[str, dict] = {}
    missing_wav = 0
    short_wav = 0
    for r in real_rows:
        sid = r["speaker_id"]
        label = r["label"]
        fname = r["output_filename"]
        wav_path = vowels_dir / fname
        if not wav_path.is_file():
            missing_wav += 1
            logger.warning("missing real wav, skip: %s", wav_path)
            continue
        try:
            info = sf.info(str(wav_path))
            dur = float(info.frames) / float(info.samplerate)
        except Exception as exc:
            logger.warning("cannot read %s: %s", wav_path, exc)
            missing_wav += 1
            continue
        if dur < args.min_ref_duration_sec:
            short_wav += 1
            logger.warning("ref too short (%.2fs): %s", dur, wav_path)
            continue
        if sid in row_by_sid:
            logger.warning("duplicate speaker_id %s, keeping first", sid)
            continue
        row_by_sid[sid] = {
            **r,
            "ref_path": str(wav_path),
            "ref_duration_sec": dur,
            "pathology": r.get("pathology") or _pathology_from_filename(fname),
        }
        by_label.setdefault(label, []).append(sid)

    h_ids = sorted(by_label.get("healthy", []))
    p_ids = sorted(by_label.get("pathological", []))
    if args.max_speakers is not None:
        h_ids = h_ids[: args.max_speakers]
        p_ids = p_ids[: args.max_speakers]

    work = _build_work_for_label(
        h_ids, "healthy", REC, args.target_synth_ratio, args.seed
    ) + _build_work_for_label(
        p_ids, "pathological", REC, args.target_synth_ratio, args.seed + 1
    )

    logger.info("data_dir (read-only) = %s", data_dir)
    logger.info("output_dir            = %s", out_dir)
    logger.info("real concat rows      = %d (missing_wav=%d short=%d)",
                len(real_rows), missing_wav, short_wav)
    logger.info("eligible healthy      = %d", len(h_ids))
    logger.info("eligible pathological = %d", len(p_ids))
    logger.info("jobs                  = %d (ratio=%.3f)", len(work), args.target_synth_ratio)
    logger.info("language              = %s", args.language)
    logger.info("recording_type        = %s", REC)
    logger.info("speaker_wav           = whole concat under %s", vowels_dir)
    logger.info("text                  = %s", BARCHE3_IAUTEXT)

    jobs_preview = []
    for job in work:
        ref = row_by_sid[job["speaker_id"]]
        jobs_preview.append(
            {
                **job,
                "ref_path": ref["ref_path"],
                "ref_duration_sec": ref["ref_duration_sec"],
                "out_relpath": f"synthetic/vowels/{job['out_name']}",
            }
        )

    if args.dry_run:
        logger.info("DRY RUN — no XTTS, no wavs")
        for j in jobs_preview[:8]:
            logger.info("  %s <- %s", j["out_name"], j["ref_path"])
        if len(jobs_preview) > 8:
            logger.info("  ... %d more", len(jobs_preview) - 8)
        if args.write_job_list:
            out_dir.mkdir(parents=False, exist_ok=False)
            (out_dir / "generation_jobs.json").write_text(
                json.dumps(
                    {
                        "protocol": "avfad_barche3_concat_whole_iautext",
                        "data_dir": str(data_dir),
                        "output_dir": str(out_dir),
                        "language": args.language,
                        "text": BARCHE3_IAUTEXT,
                        "n_jobs": len(jobs_preview),
                        "jobs": jobs_preview,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("wrote %s", out_dir / "generation_jobs.json")
        return 0

    # Synthesis path — only reached when the user explicitly runs without --dry_run.
    from TTS.api import TTS  # lazy import so dry_run needs no GPU / TTS

    out_dir.mkdir(parents=False, exist_ok=False)
    syn_dir = out_dir / "synthetic" / "vowels"
    syn_dir.mkdir(parents=True, exist_ok=True)
    # Symlink only inside the new tree so later training can resolve real wavs.
    # Does not copy or modify source vowels/.
    vowels_link = out_dir / "vowels"
    if not vowels_link.exists():
        os.symlink(vowels_dir, vowels_link, target_is_directory=True)

    shutil.copy2(data_dir / "detection_metadata.csv", out_dir / "detection_metadata.csv")
    for optional in ("participants_clinical.csv", "dataset_config.json", "cv_folds.json"):
        src = data_dir / optional
        if src.is_file():
            shutil.copy2(src, out_dir / optional)

    (out_dir / "generation_jobs.json").write_text(
        json.dumps(
            {
                "protocol": "avfad_barche3_concat_whole_iautext",
                "copied_from": [
                    "tts_synthesis_vowel9_concat_whole_iautext.py",
                    "tts_synthesis_vowel9_concat_scale_candidates.py",
                ],
                "data_dir": str(data_dir),
                "output_dir": str(out_dir),
                "language": args.language,
                "xtts_model": args.xtts_model,
                "text": BARCHE3_IAUTEXT,
                "recording_type": REC,
                "speaker_wav_policy": "whole concatenated barche-3 wav, one XTTS call",
                "n_jobs": len(jobs_preview),
                "jobs": jobs_preview,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("loading XTTS %s on %s", args.xtts_model, args.device)
    tts = TTS(args.xtts_model).to(args.device)
    xtts_sr = int(tts.synthesizer.tts_config.audio.sample_rate)

    synth_rows: list[dict] = []
    failed: list[dict] = []
    for i, job in enumerate(jobs_preview, 1):
        out_wav = syn_dir / job["out_name"]
        logger.info("[%d/%d] %s", i, len(jobs_preview), job["out_name"])
        try:
            wav = tts.tts(
                text=BARCHE3_IAUTEXT,
                speaker_wav=job["ref_path"],
                language=args.language,
            )
            _save_wav(out_wav, wav, xtts_sr)
            info = sf.info(str(out_wav))
            dur = float(info.frames) / float(info.samplerate)
            ref = row_by_sid[job["speaker_id"]]
            synth_rows.append(
                {
                    "speaker_id": job["speaker_id"],
                    "label": job["label"],
                    "pathology": ref["pathology"],
                    "recording_type": REC,
                    "output_filename": job["out_name"],
                    "source_dir": "synthetic",
                    "rep": job["copy"],
                    "duration_sec": f"{dur:.1f}",
                    "processed_path": str(out_wav),
                    "source_recording": Path(job["ref_path"]).name,
                    "seed_offset": job["seed_offset"],
                }
            )
        except Exception as exc:
            logger.exception("FAILED %s: %s", job["out_name"], exc)
            failed.append({"job": job, "error": str(exc)})

    fieldnames = [
        "speaker_id",
        "label",
        "pathology",
        "recording_type",
        "output_filename",
        "source_dir",
        "rep",
        "duration_sec",
        "processed_path",
        "source_recording",
        "seed_offset",
    ]
    with (out_dir / "synthetic_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(synth_rows)

    combined_fields = [
        "speaker_id",
        "label",
        "recording_type",
        "output_filename",
        "source_dir",
        "rep",
        "duration_sec",
    ]
    combined = []
    for r in real_rows:
        combined.append({k: r.get(k, "") for k in combined_fields})
    for r in synth_rows:
        combined.append({k: r.get(k, "") for k in combined_fields})
    with (out_dir / "combined_detection_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=combined_fields)
        w.writeheader()
        w.writerows(combined)

    report = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_real": len(real_rows),
        "n_eligible_healthy": len(h_ids),
        "n_eligible_pathological": len(p_ids),
        "n_jobs": len(jobs_preview),
        "n_ok": len(synth_rows),
        "n_failed": len(failed),
        "failed": failed,
        "pid": os.getpid(),
        "argv": sys.argv,
    }
    (out_dir / "synthesis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("done ok=%d failed=%d -> %s", len(synth_rows), len(failed), out_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
