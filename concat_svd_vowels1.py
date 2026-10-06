#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ORDER = [(p, v) for p in ("l", "n", "h") for v in ("i", "a", "u")]
PITCH_NAMES = {"l": "low", "n": "normal", "h": "high"}
REC = "vowel_iau_barche9"
RTYPES = [f"vowel_{v}_{PITCH_NAMES[p]}" for p, v in ORDER]


def read_nsp(path: Path):
    b = path.read_bytes()
    pos = 12
    chunks = {}
    while pos + 8 <= len(b):
        cid = b[pos : pos + 4].decode("latin1")
        clen = struct.unpack("<I", b[pos + 4 : pos + 8])[0]
        chunks[cid] = b[pos + 8 : pos + 8 + clen]
        pos += 8 + clen + (clen % 2)
    sr = struct.unpack("<I", chunks["HEDR"][20:24])[0]
    x = np.frombuffer(chunks["SDA_"], dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def find_vdir(raw_root: Path, sp: str):
    for d in raw_root.iterdir():
        if d.is_dir() and (d / sp / "vowels").exists():
            return d / sp / "vowels"
    return None


def load_wav(path: Path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def concat_chunks(chunks, sr0):
    return np.concatenate(chunks), sr0


def diag_name(label: str, spk: str, ref_name: str | None = None) -> str:
    if label == "healthy":
        return "healthy"
    if ref_name:
        m = re.match(rf"{spk}_(.+)_vowel_", ref_name)
        if m:
            return m.group(1)
    return "pathological"


def build_barche9(
    paper_dir: Path,
    multi_dir: Path,
    synth_root: Path,
    out_dir: Path,
    raw_root: Path | None = None,
) -> dict:
    out_vowels = out_dir / "vowels"
    out_syn = out_dir / "synthetic/vowels"
    out_vowels.mkdir(parents=True, exist_ok=True)
    out_syn.mkdir(parents=True, exist_ok=True)

    spk_info = {}
    for r in csv.DictReader(open(paper_dir / "detection_metadata.csv", encoding="utf-8")):
        if r["recording_type"] == "vowel_a_normal":
            spk_info[r["speaker_id"]] = {
                "label": r["label"],
                "ref_name": r["output_filename"],
            }

    det_rows = []
    miss_orig = 0
    for spk, info in sorted(spk_info.items(), key=lambda x: int(x[0])):
        label = info["label"]
        chunks, sr0 = [], None
        ok = True

        if label == "healthy":
            for p, v in ORDER:
                pname = PITCH_NAMES[p]
                f = multi_dir / "vowels" / f"{spk}_healthy_vowel_{v}_{pname}.wav"
                if not f.exists():
                    ok = False
                    break
                y, sr = load_wav(f)
                sr0 = sr0 or sr
                chunks.append(y)
        else:
            if raw_root is None:
                ok = False
            else:
                vdir = find_vdir(raw_root, spk)
                if vdir is None:
                    ok = False
                else:
                    for p, v in ORDER:
                        f = vdir / f"{spk}-{v}_{p}.nsp"
                        if not f.exists():
                            ok = False
                            break
                        y, sr = read_nsp(f)
                        sr0 = sr0 or sr
                        chunks.append(y)

        if not ok:
            miss_orig += 1
            continue

        y_cat, sr0 = concat_chunks(chunks, sr0)
        dname = diag_name(label, spk, info["ref_name"])
        out_name = f"{spk}_{dname}_{REC}.wav"
        sf.write(str(out_vowels / out_name), y_cat, sr0)
        det_rows.append({
            "speaker_id": spk,
            "label": label,
            "recording_type": REC,
            "output_filename": out_name,
            "source_dir": "original",
        })

    syn_src = synth_root / "synthetic/vowels"
    syn_meta_path = synth_root / "synthetic_metadata.csv"
    syn_meta_in = list(csv.DictReader(open(syn_meta_path, encoding="utf-8")))
    meta_by_key = {}
    for r in syn_meta_in:
        spk = r["speaker_id"]
        rtype = r["recording_type"]
        is_extra = int(r.get("is_extra", 0) or 0)
        extra_idx = int(r.get("extra_idx", 0) or 0)
        variant = ("base", 0) if not is_extra else ("extra", extra_idx)
        meta_by_key[(spk, variant, rtype)] = r

    by_variant = defaultdict(dict)
    for r in syn_meta_in:
        spk = r["speaker_id"]
        rtype = r["recording_type"]
        fname = r["output_filename"]
        is_extra = int(r.get("is_extra", 0) or 0)
        extra_idx = int(r.get("extra_idx", 0) or 0)
        variant = ("base", 0) if not is_extra else ("extra", extra_idx)
        p = syn_src / fname
        if p.exists():
            by_variant[(spk, variant)][rtype] = p

    syn_rows = []
    miss_syn = 0
    for (spk, variant), mp in sorted(by_variant.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        if not all(rt in mp for rt in RTYPES):
            miss_syn += 1
            continue
        chunks, sr0 = [], None
        for rt in RTYPES:
            y, sr = load_wav(mp[rt])
            sr0 = sr0 or sr
            chunks.append(y)
        y_cat, sr0 = concat_chunks(chunks, sr0)

        if variant[0] == "base":
            out_name = f"tts_{spk}_healthy_{REC}.wav"
            is_extra, extra_idx, extra_order = 0, 0, -1
        else:
            n = variant[1]
            out_name = f"tts_{spk}_healthy_{REC}_extra{n}.wav"
            is_extra, extra_idx = 1, n
            orders = []
            for rt in RTYPES:
                row = meta_by_key.get((spk, variant, rt))
                if row is not None:
                    orders.append(int(row.get("extra_global_order", -1) or -1))
            extra_order = min(orders) if orders else -1

        out_path = out_syn / out_name
        sf.write(str(out_path), y_cat, sr0)
        syn_rows.append({
            "speaker_id": spk,
            "label": "healthy",
            "pathology": "healthy",
            "recording_type": REC,
            "output_filename": out_name,
            "processed_path": str(out_path.resolve()),
            "source_dir": "synthetic",
            "speaker_wav_source": f"{spk}_healthy_vowel_a_normal.wav",
            "synth_text": "iau_barche9_concat",
            "is_extra": is_extra,
            "extra_idx": extra_idx,
            "extra_global_order": extra_order,
        })

    det_fields = ["speaker_id", "label", "recording_type", "output_filename", "source_dir"]
    with open(out_dir / "detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        w.writerows(det_rows)

    syn_fields = [
        "speaker_id", "label", "pathology", "recording_type", "output_filename",
        "processed_path", "source_dir", "speaker_wav_source", "synth_text",
        "is_extra", "extra_idx", "extra_global_order",
    ]
    with open(out_dir / "synthetic_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=syn_fields)
        w.writeheader()
        w.writerows(syn_rows)

    combined = det_rows + [{k: r[k] for k in det_fields} for r in syn_rows]
    with open(out_dir / "combined_detection_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        w.writerows(combined)

    for name in ["sentences", "cv_folds.json", "dataset_config.json", "classification_metadata.csv"]:
        src = paper_dir / name
        dst = out_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    report = {
        "paper_dir": str(paper_dir.resolve()),
        "multi_dir": str(multi_dir.resolve()),
        "synth_root": str(synth_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_original": len(det_rows),
        "n_synthetic": len(syn_rows),
        "miss_original": miss_orig,
        "miss_synthetic_variants": miss_syn,
        "n_combined": len(combined),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build barche9 IAU from speaker-aligned multipitch TTS.")
    p.add_argument(
        "--paper_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_aligned"),
        help="Real sentence/paper metadata (symlinks cv_folds etc.).",
    )
    p.add_argument(
        "--multi_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_multipitch_lnh"),
        help="Real multipitch vowels for original barche9.",
    )
    p.add_argument(
        "--synth_root",
        type=Path,
        default=Path("./svd_cleaned_paper_tts_vowel_multipitch_lnh_speaker_aligned"),
        help="Speaker-aligned multipitch TTS output.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_tts_vowel_iau_barche9_speaker_aligned"),
    )
    p.add_argument(
        "--raw_root",
        type=Path,
        default=Path("/home/beierwang/Downloads/16874898"),
        help="Raw SVD NSP root for pathological originals (optional).",
    )
    args = p.parse_args()

    raw = args.raw_root.resolve() if args.raw_root and args.raw_root.exists() else None
    report = build_barche9(
        args.paper_dir.resolve(),
        args.multi_dir.resolve(),
        args.synth_root.resolve(),
        args.out_dir.resolve(),
        raw_root=raw,
    )

    report_path = args.out_dir / "build_barche9_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"original IAU: {report['n_original']} (miss={report['miss_original']})")
    print(f"synthetic IAU: {report['n_synthetic']} (incomplete variant skipped={report['miss_synthetic_variants']})")
    print(f"combined rows: {report['n_combined']}")
    print(f"output -> {args.out_dir.resolve()}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
