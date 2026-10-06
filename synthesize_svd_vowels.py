#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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
logger = logging.getLogger("tts_vowel9_concat")

SVD_VOWEL_TEXT_A = "aaaaaaaaaaaaaaaaaaaaaaa"
SVD_VOWEL_TEXT_I = "iiiiiiiiiiiiiiiiiiiiiii"
SVD_VOWEL_TEXT_U = "uuuuuuuuuuuuuuuuuuuuuuu"
ORDER = [(p, v) for p in ("l", "n", "h") for v in ("i", "a", "u")]
REC = "vowel_9concat"
DET_FIELDS = ["speaker_id", "label", "recording_type", "output_filename", "source_dir"]


def vowel9_iautext() -> str:
    """Same string as barche9_iautext(): i/a/u per pitch band, 3 bands."""
    parts = []
    for _p, v in ORDER:
        if v == "i":
            parts.append(SVD_VOWEL_TEXT_I)
        elif v == "u":
            parts.append(SVD_VOWEL_TEXT_U)
        else:
            parts.append(SVD_VOWEL_TEXT_A)
    return "".join(parts)


def _setup_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False


def _extra_fname(base: str, extra_idx: int) -> str:
    stem = base[:-4] if base.endswith(".wav") else base
    return f"{stem}_extra{extra_idx}.wav"


def _build_synth_work(speaker_ids: List[str], target_synth_ratio: float, seed: int) -> List[dict]:
    work: List[dict] = []
    n = len(speaker_ids)
    for sid in speaker_ids:
        base = f"tts_{sid}_healthy_{REC}.wav"
        work.append({"speaker_id": sid, "out_fname": base, "extra_idx": 0, "global_order": -1})
    if target_synth_ratio <= 1.0 or n == 0:
        return work
    target_total = int(n * target_synth_ratio)
    n_extra = max(0, target_total - n)
    if n_extra == 0:
        return work
    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(n))
    for j in range(n_extra):
        sid = speaker_ids[shuffled[j % n]]
        extra_idx = (j // n) + 1
        base = f"tts_{sid}_healthy_{REC}.wav"
        work.append({
            "speaker_id": sid,
            "out_fname": _extra_fname(base, extra_idx),
            "extra_idx": extra_idx,
            "global_order": j,
        })
    return work


def _save_wav(wav: np.ndarray, out_path: Path, target_sr: int, native_sr: int) -> None:
    wav = np.asarray(wav, dtype=np.float32)
    if native_sr != target_sr:
        if not HAS_LIBROSA:
            raise RuntimeError("librosa required for resampling")
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=target_sr, res_type="polyphase")
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16, target_sr, subtype="PCM_16")


def _load_healthy_real(data_dir: Path) -> Dict[str, dict]:
    meta = data_dir / "detection_metadata.csv"
    by_sid: Dict[str, dict] = {}
    with open(meta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("source_dir", "original") == "synthetic":
                continue
            if r.get("label") != "healthy":
                continue
            if r.get("recording_type") != REC:
                continue
            sid = str(r["speaker_id"])
            by_sid[sid] = dict(r)
    return by_sid


def _rewrite_combined(data_dir: Path, synth_records: List[dict]) -> None:
    orig = []
    with open(data_dir / "detection_metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            orig.append({k: r.get(k, "") for k in DET_FIELDS})
    tts_det = [{
        "speaker_id": r["speaker_id"],
        "label": r["label"],
        "recording_type": REC,
        "output_filename": r["output_filename"],
        "source_dir": "synthetic",
    } for r in synth_records]
    with open(data_dir / "combined_detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DET_FIELDS)
        w.writeheader()
        for row in orig + tts_det:
            w.writerow(row)


def run(
    data_dir: Path,
    *,
    device: str = "cuda",
    seed: int = 42,
    target_sr: int = 16000,
    target_synth_ratio: float = 3.0,
    min_ref_duration: float = 1.0,
    force: bool = False,
    dry_run: bool = False,
    shard_index: Optional[int] = None,
    n_shards: Optional[int] = None,
) -> int:
    data_dir = data_dir.resolve()
    log_path = data_dir / "tts_vowel9_concat.log"
    _setup_logging(log_path)

    synth_text = vowel9_iautext()
    healthy = _load_healthy_real(data_dir)
    speaker_ids = sorted(healthy.keys(), key=lambda x: int(x) if x.isdigit() else x)

    logger.info("=" * 72)
    logger.info("vowel_9concat whole-IAU TTS")
    logger.info("  data_dir           = %s", data_dir)
    logger.info("  n_healthy_real     = %d", len(speaker_ids))
    logger.info("  target_synth_ratio = %s", target_synth_ratio)
    logger.info("  synth_text_len     = %d", len(synth_text))
    logger.info("  FINAL TARGET TEXT  = %s", synth_text)
    logger.info("=" * 72)

    if not speaker_ids:
        logger.error("no healthy vowel_9concat rows in %s", data_dir)
        return 1

    vowels = data_dir / "vowels"
    eligible: List[str] = []
    miss_ref = 0
    for sid in speaker_ids:
        row = healthy[sid]
        ref = vowels / row["output_filename"]
        if not ref.exists():
            miss_ref += 1
            continue
        dur = float(sf.info(str(ref)).duration)
        if dur < min_ref_duration:
            miss_ref += 1
            continue
        eligible.append(sid)
    logger.info("eligible healthy refs: %d / %d (miss/short=%d)", len(eligible), len(speaker_ids), miss_ref)

    work = _build_synth_work(eligible, target_synth_ratio, seed)
    if shard_index is not None and n_shards is not None:
        work = [w for i, w in enumerate(work) if i % n_shards == shard_index]
        logger.info("shard %d/%d → %d jobs", shard_index, n_shards, len(work))
    logger.info("synthesis jobs: %d", len(work))

    if dry_run:
        logger.info("[DRY RUN] no wavs written")
        return 0

    out_syn = data_dir / "synthetic" / "vowels"
    out_syn.mkdir(parents=True, exist_ok=True)

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

    synth_records: List[dict] = []
    skipped = errors = 0
    t0 = time.time()
    iterator = tqdm(work, desc="TTS-vowel9") if HAS_TQDM else work
    for item in iterator:
        sid = item["speaker_id"]
        real = healthy[sid]
        out_fname = item["out_fname"]
        extra_idx = item["extra_idx"]
        global_order = item["global_order"]
        out_path = out_syn / out_fname
        ref_path = vowels / real["output_filename"]
        ref_dur = float(sf.info(str(ref_path)).duration)

        rec = {
            "speaker_id": real["speaker_id"],
            "label": real["label"],
            "pathology": "healthy",
            "recording_type": REC,
            "output_filename": out_fname,
            "processed_path": str(out_path.resolve()),
            "source_dir": "synthetic",
            "speaker_wav_source": real["output_filename"],
            "ref_fallback": "vowel_9concat_real",
            "ref_concat_duration_sec": round(ref_dur, 3),
            "synth_text": synth_text,
            "is_extra": int(extra_idx > 0),
            "extra_idx": extra_idx,
            "extra_global_order": global_order,
        }

        if out_path.exists() and not force:
            synth_records.append(rec)
            continue

        item_seed = seed if extra_idx == 0 else seed + 100000 + global_order + 1
        torch.manual_seed(item_seed)
        np.random.seed(item_seed)
        try:
            wav = tts_model.tts(
                text=synth_text,
                speaker_wav=str(ref_path),
                language="de",
            )
            _save_wav(wav, out_path, target_sr, native_sr)
            synth_records.append(rec)
        except Exception as e:
            logger.warning("TTS failed spk=%s -> %s: %s", sid, out_fname, e)
            errors += 1
            skipped += 1

    elapsed = time.time() - t0
    shard_tag = "" if shard_index is None else f".shard{shard_index}"
    syn_meta = data_dir / f"synthetic_metadata{shard_tag}.csv"
    if synth_records:
        fields = list(synth_records[0].keys())
        with open(syn_meta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in synth_records:
                w.writerow(row)

    if shard_index is None:
        _rewrite_combined(data_dir, synth_records)
        if syn_meta.name != "synthetic_metadata.csv":
            pass
        else:
            pass
    config = {
        "script": "tts_synthesis_vowel9_concat_whole_iautext.py",
        "data_dir": str(data_dir),
        "recording_type": REC,
        "synth_text": synth_text,
        "synth_text_len": len(synth_text),
        "target_synth_ratio": target_synth_ratio,
        "n_healthy_real": len(speaker_ids),
        "n_eligible": len(eligible),
        "n_jobs": len(work),
        "n_synthetic_written": len(synth_records),
        "n_errors": errors,
        "n_skipped": skipped,
        "elapsed_seconds": round(elapsed, 1),
        "seed": seed,
        "shard_index": shard_index,
        "n_shards": n_shards,
    }
    cfg_name = f"tts_vowel9_concat_config{shard_tag}.json"
    with open(data_dir / cfg_name, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(
        "DONE shard=%s synth=%d errors=%d elapsed=%.1fs",
        shard_index, len(synth_records), errors, elapsed,
    )
    return 0 if errors == 0 else 4


def merge_shards(data_dir: Path, n_shards: int) -> int:
    data_dir = data_dir.resolve()
    rows: List[dict] = []
    fields = None
    for i in range(n_shards):
        p = data_dir / f"synthetic_metadata.shard{i}.csv"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows.extend(list(reader))
    with open(data_dir / "synthetic_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _rewrite_combined(data_dir, rows)
    print(f"merged {len(rows)} synth rows -> {data_dir / 'synthetic_metadata.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="vowel_9concat whole-IAU TTS (new dir only)")
    ap.add_argument("--data_dir", type=Path, default=Path("./svd_cleaned_paper_tts_vowel_9concat_bal11"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--target_synth_ratio", type=float, default=3.0)
    ap.add_argument("--min_ref_duration", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--shard_index", type=int, default=None)
    ap.add_argument("--n_shards", type=int, default=None)
    ap.add_argument("--merge_shards", action="store_true")
    args = ap.parse_args()
    if args.merge_shards:
        if args.n_shards is None:
            print("--merge_shards requires --n_shards", file=sys.stderr)
            return 2
        return merge_shards(args.data_dir, args.n_shards)
    return run(
        args.data_dir,
        device=args.device,
        seed=args.seed,
        target_sr=args.target_sr,
        target_synth_ratio=args.target_synth_ratio,
        min_ref_duration=args.min_ref_duration,
        force=args.force,
        dry_run=args.dry_run,
        shard_index=args.shard_index,
        n_shards=args.n_shards,
    )


if __name__ == "__main__":
    raise SystemExit(main())
