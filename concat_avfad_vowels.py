#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from avfad_barche3_segment_utils import extract_segment  # noqa: E402

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

VOWEL_ORDER = ("vowel_i", "vowel_a", "vowel_u")
REC_TYPE = "vowel_iau_barche3"
PLAN = "B_stable_trim_onset_concat_i_a_u"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_name(s: str) -> str:
    s = str(s).strip().replace("'", "_")
    return _SAFE_NAME_RE.sub("_", s)


def load_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        if not HAS_LIBROSA:
            raise RuntimeError("librosa required for resampling")
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="polyphase")
        sr = target_sr
    return y.astype(np.float32, copy=False), sr


def concat_barche3_stable(
    paths: Dict[str, Path],
    segment_sec: float,
    target_sr: int,
    top_db: float,
    post_onset_sec: float,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for vtype in VOWEL_ORDER:
        if vtype not in paths:
            raise FileNotFoundError(f"missing {vtype} in {paths}")
        y, _ = load_mono(paths[vtype], target_sr)
        chunks.append(
            extract_segment(
                y,
                target_sr,
                segment_sec,
                mode="stable",
                top_db=top_db,
                post_onset_sec=post_onset_sec,
            )
        )
    return np.concatenate(chunks)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), clip, sr, subtype="PCM_16")
    return len(clip) / sr


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if not src.exists():
        return
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def build_original_barche3(
    aligned_sv_dir: Path,
    out_dir: Path,
    segment_sec: float,
    target_sr: int,
    top_db: float,
    post_onset_sec: float,
) -> Tuple[List[dict], dict]:
    vowels_in = aligned_sv_dir / "vowels"
    meta_in = aligned_sv_dir / "detection_metadata.csv"
    out_vowels = out_dir / "vowels"

    by_spk: Dict[str, Dict[str, dict]] = defaultdict(dict)
    with open(meta_in, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rtype = row["recording_type"]
            if rtype not in VOWEL_ORDER:
                continue
            by_spk[row["speaker_id"]][rtype] = row

    det_rows: List[dict] = []
    miss = 0
    durs: List[float] = []

    for spk in sorted(by_spk.keys()):
        mp = by_spk[spk]
        if not all(v in mp for v in VOWEL_ORDER):
            miss += 1
            continue
        paths = {v: vowels_in / mp[v]["output_filename"] for v in VOWEL_ORDER}
        if not all(p.exists() for p in paths.values()):
            miss += 1
            continue
        label = mp["vowel_i"]["label"]
        try:
            audio = concat_barche3_stable(
                paths, segment_sec, target_sr, top_db, post_onset_sec
            )
        except Exception:
            miss += 1
            continue
        out_name = f"{spk}_{label}_{REC_TYPE}.wav"
        dur = _write_wav(out_vowels / out_name, audio, target_sr)
        durs.append(dur)
        det_rows.append({
            "speaker_id": spk,
            "label": label,
            "recording_type": REC_TYPE,
            "output_filename": out_name,
            "source_dir": "original",
            "rep": "",
            "duration_sec": round(dur, 3),
        })

    stats = {
        "n_speakers_attempted": len(by_spk),
        "n_original": len(det_rows),
        "miss_original": miss,
        "duration_sec": {
            "min": float(min(durs)) if durs else 0.0,
            "max": float(max(durs)) if durs else 0.0,
            "mean": float(np.mean(durs)) if durs else 0.0,
        },
        "expected_duration_sec": 3 * segment_sec,
    }
    return det_rows, stats


def _synth_variant_key(row: dict) -> Tuple[str, str, int]:
    spk = row["speaker_id"]
    is_extra = int(row.get("is_extra", 0) or 0)
    extra_idx = int(row.get("extra_idx", 0) or 0)
    if is_extra:
        return spk, "extra", extra_idx
    return spk, "base", 0


def build_synthetic_barche3(
    vowel_tts_dir: Path,
    out_dir: Path,
    segment_sec: float,
    target_sr: int,
    top_db: float,
    post_onset_sec: float,
) -> Tuple[List[dict], dict]:
    synth_meta_path = vowel_tts_dir / "synthetic_metadata.csv"
    synth_src = vowel_tts_dir / "synthetic" / "vowels"
    out_syn = out_dir / "synthetic" / "vowels"

    by_key: Dict[Tuple[str, str, int], Dict[str, dict]] = defaultdict(dict)
    with open(synth_meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rtype = row["recording_type"]
            if rtype not in VOWEL_ORDER:
                continue
            vkey = _synth_variant_key(row)
            by_key[vkey][rtype] = row

    synth_rows: List[dict] = []
    miss = 0
    durs: List[float] = []
    synth_tag = "iau_barche3_stable_concat"

    for vkey in sorted(by_key.keys(), key=lambda k: (k[0], k[1], k[2])):
        mp = by_key[vkey]
        if not all(v in mp for v in VOWEL_ORDER):
            miss += 1
            continue
        paths = {}
        ok = True
        for v in VOWEL_ORDER:
            fname = mp[v]["output_filename"]
            p = synth_src / fname
            if not p.exists():
                ok = False
                break
            paths[v] = p
        if not ok:
            miss += 1
            continue

        spk, variant, extra_idx = vkey
        row0 = mp["vowel_i"]
        label = row0["label"]
        pathology = row0.get("pathology", "")
        safe_patho = _safe_name(pathology)

        try:
            audio = concat_barche3_stable(
                paths, segment_sec, target_sr, top_db, post_onset_sec
            )
        except Exception:
            miss += 1
            continue

        if variant == "base":
            out_name = f"tts_{spk}_{safe_patho}_{REC_TYPE}.wav"
            is_extra, extra_idx_out, extra_order = 0, 0, -1
        else:
            out_name = f"tts_{spk}_{safe_patho}_{REC_TYPE}_extra{extra_idx}.wav"
            is_extra, extra_idx_out = 1, extra_idx
            extra_order = extra_idx

        out_path = out_syn / out_name
        dur = _write_wav(out_path, audio, target_sr)
        durs.append(dur)

        synth_rows.append({
            "speaker_id": spk,
            "label": label,
            "pathology": pathology,
            "recording_type": REC_TYPE,
            "output_filename": out_name,
            "processed_path": str(out_path.resolve()),
            "source_dir": "synthetic",
            "duration_sec": round(dur, 3),
            "speaker_wav_source": row0.get("speaker_wav_source", ""),
            "synth_text": synth_tag,
            "is_extra": is_extra,
            "extra_idx": extra_idx_out,
            "extra_global_order": extra_order,
        })

    base_by_spk: Dict[str, Dict[str, Path]] = {}
    with open(synth_meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("is_extra", 0) or 0) != 0:
                continue
            rtype = row["recording_type"]
            if rtype not in VOWEL_ORDER:
                continue
            spk = row["speaker_id"]
            base_by_spk.setdefault(spk, {})
            p = synth_src / row["output_filename"]
            if p.exists():
                base_by_spk[spk][rtype] = p

    extra_rows_raw: Dict[Tuple[str, int], Dict[str, Path]] = defaultdict(dict)
    extra_meta: Dict[Tuple[str, int], dict] = {}
    with open(synth_meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("is_extra", 0) or 0) != 1:
                continue
            rtype = row["recording_type"]
            if rtype not in VOWEL_ORDER:
                continue
            spk = row["speaker_id"]
            eidx = int(row.get("extra_idx", 0) or 0)
            p = synth_src / row["output_filename"]
            if p.exists():
                extra_rows_raw[(spk, eidx)][rtype] = p
                extra_meta[(spk, eidx)] = row

    existing_extra_names = {
        r["output_filename"] for r in synth_rows if int(r.get("is_extra", 0)) == 1
    }
    filled = 0
    for (spk, eidx), vow_map in sorted(extra_rows_raw.items()):
        if spk not in base_by_spk:
            continue
        safe_patho = _safe_name(extra_meta[(spk, eidx)].get("pathology", ""))
        out_name = f"tts_{spk}_{safe_patho}_{REC_TYPE}_extra{eidx}.wav"
        if out_name in existing_extra_names:
            continue
        paths = dict(base_by_spk[spk])
        paths.update(vow_map)
        if not all(v in paths for v in VOWEL_ORDER):
            continue
        try:
            audio = concat_barche3_stable(
                paths, segment_sec, target_sr, top_db, post_onset_sec
            )
        except Exception:
            continue
        out_path = out_syn / out_name
        dur = _write_wav(out_path, audio, target_sr)
        durs.append(dur)
        row0 = extra_meta[(spk, eidx)]
        synth_rows.append({
            "speaker_id": spk,
            "label": row0["label"],
            "pathology": row0.get("pathology", ""),
            "recording_type": REC_TYPE,
            "output_filename": out_name,
            "processed_path": str(out_path.resolve()),
            "source_dir": "synthetic",
            "duration_sec": round(dur, 3),
            "speaker_wav_source": row0.get("speaker_wav_source", ""),
            "synth_text": "iau_barche3_stable_concat_fill_base",
            "is_extra": 1,
            "extra_idx": eidx,
            "extra_global_order": eidx,
        })
        filled += 1

    stats = {
        "n_synthetic": len(synth_rows),
        "miss_incomplete_triplets": miss,
        "n_filled_extra_with_base_fallback": filled,
        "duration_sec": {
            "min": float(min(durs)) if durs else 0.0,
            "max": float(max(durs)) if durs else 0.0,
            "mean": float(np.mean(durs)) if durs else 0.0,
        },
    }
    return synth_rows, stats


def write_metadata(
    out_dir: Path,
    det_rows: List[dict],
    synth_rows: List[dict],
    aligned_sv_dir: Path,
) -> None:
    det_fields = [
        "speaker_id", "label", "recording_type", "output_filename",
        "source_dir", "rep", "duration_sec",
    ]
    with open(out_dir / "detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        for row in det_rows:
            w.writerow({k: row.get(k, "") for k in det_fields})

    syn_fields = [
        "speaker_id", "label", "pathology", "recording_type", "output_filename",
        "processed_path", "source_dir", "duration_sec", "speaker_wav_source",
        "synth_text", "is_extra", "extra_idx", "extra_global_order",
    ]
    with open(out_dir / "synthetic_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=syn_fields)
        w.writeheader()
        for row in synth_rows:
            w.writerow({k: row.get(k, "") for k in syn_fields})

    combined_fields = ["speaker_id", "label", "recording_type", "output_filename", "source_dir"]
    combined = (
        [{k: r[k] for k in combined_fields} for r in det_rows]
        + [{k: r[k] for k in combined_fields} for r in synth_rows]
    )
    with open(out_dir / "combined_detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=combined_fields)
        w.writeheader()
        w.writerows(combined)

    for name in ("cv_folds.json", "participants_clinical.csv", "dataset_config.json"):
        _symlink_or_copy(aligned_sv_dir / name, out_dir / name)


def _pick_extra_indices(n_eligible: int, n_extra: int, seed: int) -> List[int]:
    if n_extra <= 0:
        return []
    rng = np.random.RandomState(seed)
    pick_n = min(n_extra, n_eligible)
    return sorted(rng.choice(n_eligible, pick_n, replace=False).tolist())


def build_ratio_view(
    out_dir: Path,
    synth_rows: List[dict],
    det_rows: List[dict],
    ratio: float,
    seed: int,
    segment_sec: float,
    top_db: float,
    post_onset_sec: float,
) -> dict:
    tag = str(int(round(ratio * 100))).zfill(3)
    view_dir = out_dir / f"metadata_synth_{tag}"
    view_dir.mkdir(parents=True, exist_ok=True)

    base_rows = [r for r in synth_rows if int(r.get("is_extra", 0) or 0) == 0]
    extra_rows = [r for r in synth_rows if int(r.get("is_extra", 0) or 0) == 1]
    base_rows.sort(key=lambda r: (r["speaker_id"], r["output_filename"]))
    extra_rows.sort(key=lambda r: (r["speaker_id"], int(r.get("extra_idx", 0))))

    n_real = len(base_rows)
    target_total = int(round(n_real * ratio))
    n_extra_target = max(0, target_total - len(base_rows))
    pick_idx = _pick_extra_indices(len(extra_rows), n_extra_target, seed)
    selected_extra = [extra_rows[i] for i in pick_idx]
    selected_synth = base_rows + selected_extra

    syn_fields = list(selected_synth[0].keys()) if selected_synth else []
    with open(view_dir / "synthetic_metadata.csv", "w", newline="", encoding="utf-8") as f:
        if selected_synth:
            w = csv.DictWriter(f, fieldnames=syn_fields)
            w.writeheader()
            w.writerows(selected_synth)

    combined_fields = ["speaker_id", "label", "recording_type", "output_filename", "source_dir"]
    combined = (
        [{k: r[k] for k in combined_fields} for r in det_rows]
        + [{k: r[k] for k in combined_fields} for r in selected_synth]
    )
    with open(view_dir / "combined_detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=combined_fields)
        w.writeheader()
        w.writerows(combined)

    _symlink_or_copy(out_dir / "detection_metadata.csv", view_dir / "detection_metadata.csv")
    for name in ("vowels", "synthetic", "cv_folds.json", "participants_clinical.csv", "dataset_config.json"):
        _symlink_or_copy(out_dir / name, view_dir / name)

    manifest = {
        "target_synth_ratio": ratio,
        "seed": seed,
        "modality": "vowel_iau_barche3",
        "segment_plan": PLAN,
        "segment_sec_per_vowel": segment_sec,
        "top_db": top_db,
        "post_onset_sec": post_onset_sec,
        "n_real_eligible": n_real,
        "n_base_included": len(base_rows),
        "n_extra_target": n_extra_target,
        "n_extra_included": len(selected_extra),
        "n_synthetic_total": len(selected_synth),
        "expected_synthetic_total": target_total,
        "ready_for_training": len(selected_synth) == target_total or ratio == 1.0,
        "training_data_dir": str(view_dir.resolve()),
        "training_note": (
            "Barche3 stable-segment ratio view. --recording_type vowel, --max_length_sec 24.0. "
            "No --max_synth_ratio."
        ),
    }
    with open(view_dir / "synth_ratio_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    if not HAS_LIBROSA:
        print("ERROR: librosa is required for Plan B stable segmentation.", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Build AVFAD Barche-3 Plan B (stable 8s × i/a/u).")
    ap.add_argument(
        "--aligned_sv_dir",
        type=Path,
        default=Path("/home/beierwang/testvscode/avfad_paper_aligned_sv"),
    )
    ap.add_argument(
        "--vowel_tts_dir",
        type=Path,
        default=Path(
            "/home/beierwang/testvscode/avfad_paper_aligned_sv_paper_tts_vowel_ratio_pool"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/home/beierwang/testvscode/"
            "avfad_paper_aligned_sv_paper_tts_vowel_barche3_stable_8s_ratio_pool"
        ),
    )
    ap.add_argument("--segment_sec", type=float, default=8.0)
    ap.add_argument("--top_db", type=float, default=30.0)
    ap.add_argument("--post_onset_sec", type=float, default=0.2)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--build_ratios",
        action="store_true",
        help="Write metadata_synth_100 and metadata_synth_200 ratio views.",
    )
    args = ap.parse_args()

    aligned = args.aligned_sv_dir.resolve()
    tts_root = args.vowel_tts_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    det_rows, orig_stats = build_original_barche3(
        aligned,
        out_dir,
        args.segment_sec,
        args.target_sr,
        args.top_db,
        args.post_onset_sec,
    )
    synth_rows, synth_stats = build_synthetic_barche3(
        tts_root,
        out_dir,
        args.segment_sec,
        args.target_sr,
        args.top_db,
        args.post_onset_sec,
    )
    write_metadata(out_dir, det_rows, synth_rows, aligned)

    report = {
        "plan": PLAN,
        "segment_sec_per_vowel": args.segment_sec,
        "top_db": args.top_db,
        "post_onset_sec": args.post_onset_sec,
        "target_sr": args.target_sr,
        "recording_type": REC_TYPE,
        "aligned_sv_dir": str(aligned),
        "vowel_tts_dir": str(tts_root),
        "out_dir": str(out_dir),
        "original": orig_stats,
        "synthetic": synth_stats,
        "does_not_modify": [
            str(aligned),
            str(tts_root),
            str(Path("/home/beierwang/testvscode/avfad_paper_aligned_sv_paper_tts_vowel_barche3_8s_ratio_pool")),
        ],
    }

    ratio_manifests = []
    if args.build_ratios:
        for ratio in (1.0, 2.0):
            ratio_manifests.append(
                build_ratio_view(
                    out_dir,
                    synth_rows,
                    det_rows,
                    ratio,
                    args.seed,
                    args.segment_sec,
                    args.top_db,
                    args.post_onset_sec,
                )
            )
        report["ratio_views"] = ratio_manifests

    report_path = out_dir / "build_barche3_stable_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
