#!/usr/bin/env python3
"""10-fold nested TTS-scale dry-run for vowel_9concat (no training).

Locks the existing bal11 healthy-balancing subsample (2(P-H)) from the
vowel_9concat ``aug_sentence`` ``stage2_tts_sample.json`` manifests.

Extra healthy and pathological TTS are ranked once per fold (seed protocol
matches sentence nested scales) and sliced as prefixes:

    1.1x extra ‚ä?1.2x extra ‚ä?1.3x extra

Does NOT modify:
  - svd_cleaned_paper_tts_vowel_9concat_bal11
  - existing vowel_9concat training scripts / result dirs
  - 10-fold split, num_augmented=1, speaker-level eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from whisper_train_utils import (
    create_cv_folds_with_tts,
    load_metadata_with_tts,
)
from train_with_tts_v10_tapt_ptapt_whisper_layersweep_orig_aug_disjoint_synth_bal11_vowel9concat import (
    _speaker_sort_key,
    filter_stage2_train_disjoint_synthetic,
    sample_stage2_synth_speaker_balanced,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vowel9_scale_audit")

REC = "vowel_9concat"
NUM_AUGMENTED = 1
SCALES = (1.1, 1.2, 1.3)
SCALE_FRACS = {1.1: 0.10, 1.2: 0.20, 1.3: 0.30}
HEALTHY_EXTRA_RNG_OFFSET = 15001
PATHO_RNG_OFFSET = 15002
FORBIDDEN_WRITE = (
    "svd_cleaned_paper_tts_vowel_9concat_bal11",
)


def extra_quota_half_up_frac(nP: int, frac: float) -> tuple[int, int]:
    """Q = round_half_up((scale-1)*B), B=2P. Not Python bankers round()."""
    B = 2 * int(nP)
    Q = int(math.floor(float(frac) * B + 0.5))
    return B, Q


def _assert_new_output_dir(output_dir: Path) -> Path:
    resolved = Path(output_dir).resolve()
    text = str(resolved)
    for token in FORBIDDEN_WRITE:
        if text.rstrip("/").endswith(token) or f"/{token}/" in text + "/":
            raise ValueError(f"refusing to write into existing pool: {resolved}")
    if "/aug_sentence/" in text + "/" or "/aug_vowelweak/" in text + "/":
        raise ValueError(f"refusing to write into existing vowel_9concat results: {resolved}")
    return resolved


def speaker_balanced_nested_rank(records: list, rng: np.random.RandomState) -> list:
    """Deterministic speaker-balanced ordering of TTS records (sentence protocol)."""
    by_spk = defaultdict(list)
    for rec in records:
        by_spk[str(rec["speaker_id"])].append(rec)
    speakers = sorted(by_spk.keys(), key=_speaker_sort_key)
    if not speakers:
        return []
    for sid in speakers:
        rng.shuffle(by_spk[sid])
    order = list(speakers)
    rng.shuffle(order)
    pointers = {sid: 0 for sid in speakers}
    ranking = []
    guard = 0
    max_iters = sum(len(v) for v in by_spk.values()) + len(speakers) + 8
    while order:
        next_order = []
        progressed = False
        for sid in order:
            i = pointers[sid]
            if i < len(by_spk[sid]):
                ranking.append(by_spk[sid][i])
                pointers[sid] = i + 1
                progressed = True
                if pointers[sid] < len(by_spk[sid]):
                    next_order.append(sid)
        order = next_order
        guard += 1
        if order and not progressed:
            raise RuntimeError("nested rank water-fill stalled")
        if guard > max_iters:
            raise RuntimeError(f"nested rank exceeded iteration cap ({max_iters})")
    return ranking


def load_synth_pool(pool_dir: Path, expected_label: str) -> list:
    pool_dir = Path(pool_dir)
    meta = pool_dir / "synthetic_metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"missing {meta}")
    records = []
    missing_wav = []
    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("recording_type") != REC:
                continue
            if row.get("label") != expected_label:
                continue
            fname = row["output_filename"]
            stored = row.get("processed_path") or ""
            cand = []
            if stored:
                cand.append(Path(stored))
            cand.append(pool_dir / "synthetic" / "vowels" / fname)
            path = next((p for p in cand if p.exists() and p.stat().st_size > 0), None)
            if path is None:
                missing_wav.append(fname)
                continue
            records.append({
                "speaker_id": str(row["speaker_id"]),
                "label": row["label"],
                "output_filename": fname,
                "path": str(path.resolve()),
                "extra_idx": int(row.get("extra_idx") or 0),
                "generation_index": int(row.get("generation_index") or row.get("extra_idx") or 0),
                "copy_index": int(row.get("copy_index") or (int(row.get("extra_idx") or 0) + 1)),
                "speaker_wav_source": row.get("speaker_wav_source", ""),
                "origin": row.get("origin", ""),
                "pool_dir": str(pool_dir.resolve()),
            })
    if missing_wav:
        raise FileNotFoundError(
            f"{len(missing_wav)} {expected_label} wavs missing under {pool_dir}, "
            f"e.g. {missing_wav[:5]}"
        )
    return records


def _load_bal11_manifest(baseline_dir: Path, fold_idx: int) -> dict:
    path = Path(baseline_dir) / f"fold_{fold_idx}" / "stage2_tts_sample.json"
    if not path.exists():
        raise FileNotFoundError(f"missing bal11 baseline manifest: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compact_rec(r: dict, role: str) -> dict:
    return {
        "speaker_id": str(r["speaker_id"]),
        "label": r.get("label"),
        "role": role,
        "output_filename": r["output_filename"],
        "path": r["path"],
        "extra_idx": r.get("extra_idx"),
        "generation_index": r.get("generation_index"),
        "copy_index": r.get("copy_index"),
        "origin": r.get("origin", ""),
        "speaker_wav_source": r.get("speaker_wav_source", ""),
        "pool_dir": r.get("pool_dir", ""),
    }


def audit_fold(
    fold_idx: int,
    n_folds: int,
    seed: int,
    reseed_each_fold: bool,
    file_paths,
    labels,
    speaker_ids,
    rec_types,
    is_synthetic,
    label2id,
    train_idx,
    test_idx,
    healthy_3x: list,
    healthy_extra3: list,
    patho_1x: list,
    baseline_dir: Path,
    out_fold_dir: Path,
) -> dict:
    if reseed_each_fold:
        fold_seed = int(seed) + int(fold_idx) * 1000
    else:
        fold_seed = int(seed)

    train_files = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_rec_types = [rec_types[i] for i in train_idx]
    train_speaker_ids = [str(speaker_ids[i]) for i in train_idx]
    train_is_synth = [is_synthetic[i] for i in train_idx]
    test_speaker_ids = [str(speaker_ids[i]) for i in test_idx]
    train_set = set(str(speaker_ids[i]) for i in train_idx if not is_synthetic[i])
    # train_idx from create_cv_folds includes synth rows; real train speakers:
    train_real_set = set(
        str(speaker_ids[i]) for i in train_idx if not is_synthetic[i]
    )
    test_set = set(test_speaker_ids)
    if train_real_set & test_set:
        raise RuntimeError(f"[fold {fold_idx}] train/test speaker overlap")

    healthy_id = label2id["healthy"]
    real_train_idx = [i for i in train_idx if not is_synthetic[i]]
    nH = sum(1 for i in real_train_idx if labels[i] == healthy_id)
    nP = sum(1 for i in real_train_idx if labels[i] != healthy_id)

    filtered = filter_stage2_train_disjoint_synthetic(
        fold_idx=fold_idx,
        train_files=train_files,
        train_labels=train_labels,
        train_rec_types=train_rec_types,
        train_speaker_ids=train_speaker_ids,
        train_is_synthetic=train_is_synth,
        test_speaker_ids=test_speaker_ids,
    )
    scratch = out_fold_dir / "_bal11_recompute"
    scratch.mkdir(parents=True, exist_ok=True)
    sampled = sample_stage2_synth_speaker_balanced(
        fold_idx=fold_idx,
        train_files=filtered["train_files"],
        train_labels=filtered["train_labels"],
        train_rec_types=filtered["train_rec_types"],
        train_speaker_ids=filtered["train_speaker_ids"],
        train_is_synthetic=filtered["train_is_synthetic"],
        label2id=label2id,
        num_augmented=NUM_AUGMENTED,
        fold_seed=fold_seed,
        fold_dir=scratch,
    )
    with open(scratch / "stage2_tts_sample.json", encoding="utf-8") as f:
        recomputed = json.load(f)
    gold = _load_bal11_manifest(baseline_dir, fold_idx)
    gold_names = [x["output_filename"] for x in gold["selected_synth"]]
    rec_names = [x["output_filename"] for x in recomputed["selected_synth"]]
    if gold.get("nH_real") != nH or gold.get("nP_real") != nP:
        raise RuntimeError(
            f"[fold {fold_idx}] H/P mismatch vs bal11: now H={nH} P={nP} vs "
            f"H={gold.get('nH_real')} P={gold.get('nP_real')}"
        )
    if gold_names != rec_names:
        raise RuntimeError(
            f"[fold {fold_idx}] recomputed healthy-balancing filenames != "
            f"aug_sentence stage2_tts_sample.json "
            f"(only_gold={len(set(gold_names)-set(rec_names))} "
            f"only_recompute={len(set(rec_names)-set(gold_names))})"
        )
    if sampled["nH_real"] != nH or sampled["nP_real"] != nP:
        raise RuntimeError(f"[fold {fold_idx}] sampler H/P mismatch")

    healthy_all = healthy_3x + healthy_extra3
    healthy_by_name = {r["output_filename"]: r for r in healthy_all}
    missing_base = [n for n in gold_names if n not in healthy_by_name]
    if missing_base:
        raise RuntimeError(
            f"[fold {fold_idx}] {len(missing_base)} baseline files missing "
            f"from healthy pool, e.g. {missing_base[:5]}"
        )
    baseline_selected = [
        {**healthy_by_name[n], "role": "baseline_healthy", "label": "healthy"}
        for n in gold_names
    ]
    baseline_name_set = set(gold_names)
    n_tts_h_baseline = 2 * (nP - nH)
    if len(baseline_selected) != n_tts_h_baseline:
        raise RuntimeError(
            f"[fold {fold_idx}] baseline count {len(baseline_selected)} != "
            f"2(P-H)={n_tts_h_baseline}"
        )

    h_avail = [
        r for r in healthy_all
        if r["speaker_id"] in train_real_set and r["speaker_id"] not in test_set
    ]
    p_avail = [
        r for r in patho_1x
        if r["speaker_id"] in train_real_set and r["speaker_id"] not in test_set
    ]
    h_extra_cand = [r for r in h_avail if r["output_filename"] not in baseline_name_set]

    overlap_h = [r for r in h_avail if r["speaker_id"] in test_set]
    overlap_p = [r for r in p_avail if r["speaker_id"] in test_set]
    if overlap_h or overlap_p:
        raise RuntimeError(f"[fold {fold_idx}] test-speaker leaked into avail pool")

    rng_h = np.random.RandomState(int(fold_seed) + HEALTHY_EXTRA_RNG_OFFSET)
    rng_p = np.random.RandomState(int(fold_seed) + PATHO_RNG_OFFSET)
    extra_h_rank = speaker_balanced_nested_rank(h_extra_cand, rng_h)
    patho_rank = speaker_balanced_nested_rank(p_avail, rng_p)

    ranking_doc = {
        "fold_idx": fold_idx,
        "fold_seed": int(fold_seed),
        "protocol_seed": int(seed),
        "reseed_each_fold": bool(reseed_each_fold),
        "healthy_extra_rng": int(fold_seed) + HEALTHY_EXTRA_RNG_OFFSET,
        "patho_rng": int(fold_seed) + PATHO_RNG_OFFSET,
        "note": (
            "Healthy balancing (2(P-H)) is locked to vowel_9concat bal11 "
            "aug_sentence stage2_tts_sample.json. Extra healthy and "
            "pathological TTS are a single speaker-balanced ranking; "
            "1.1 ‚ä?1.2 ‚ä?1.3 are prefixes of this ranking."
        ),
        "nH_real": nH,
        "nP_real": nP,
        "n_tts_healthy_baseline": n_tts_h_baseline,
        "n_healthy_train_available": len(h_avail),
        "n_healthy_extra_candidates": len(h_extra_cand),
        "n_patho_train_available": len(p_avail),
        "n_test_speaker_tts_overlap": 0,
        "extra_healthy_ranking": [
            _compact_rec(r, "extra_healthy_rank") for r in extra_h_rank
        ],
        "pathological_ranking": [
            _compact_rec(r, "pathological_rank") for r in patho_rank
        ],
        "baseline_healthy": [_compact_rec(r, "baseline_healthy") for r in baseline_selected],
    }
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    with open(out_fold_dir / "tts_nested_ranking.json", "w", encoding="utf-8") as f:
        json.dump(ranking_doc, f, indent=2)

    scale_reports = []
    prev_extra_h = None
    prev_patho = None
    for scale in SCALES:
        frac = SCALE_FRACS[scale]
        B, Q = extra_quota_half_up_frac(nP, frac)
        healthy_required = n_tts_h_baseline + Q
        patho_required = Q
        entries = nP * (1 + NUM_AUGMENTED) + Q
        entries_h = nH * (1 + NUM_AUGMENTED) + healthy_required
        entries_p = nP * (1 + NUM_AUGMENTED) + patho_required
        issues = []
        if len(h_avail) < healthy_required:
            issues.append(
                f"healthy pool short: avail={len(h_avail)} need={healthy_required}"
            )
        if len(h_extra_cand) < Q:
            issues.append(
                f"extra healthy short: cand={len(h_extra_cand)} need Q={Q}"
            )
        if len(p_avail) < patho_required:
            issues.append(
                f"patho pool short: avail={len(p_avail)} need={patho_required}"
            )
        extra_h_sel = [
            {**r, "role": f"extra_healthy_{scale}"} for r in extra_h_rank[:Q]
        ]
        patho_sel = [
            {**r, "role": f"pathological_{scale}"} for r in patho_rank[:Q]
        ]
        extra_h_names = [r["output_filename"] for r in extra_h_sel]
        patho_names = [r["output_filename"] for r in patho_sel]
        all_sel = baseline_selected + extra_h_sel + patho_sel
        all_names = [r["output_filename"] for r in all_sel]
        all_paths = [r["path"] for r in all_sel]
        all_sids = [str(r["speaker_id"]) for r in all_sel]

        if set(extra_h_names) & baseline_name_set:
            issues.append("baseline/extra healthy filename overlap")
        if len(all_names) != len(set(all_names)):
            issues.append("duplicate TTS filename")
        if len(all_paths) != len(set(all_paths)):
            issues.append("duplicate TTS path")
        if any(sid in test_set for sid in all_sids):
            issues.append("test-speaker TTS in selection")
        if any(sid not in train_real_set for sid in all_sids):
            issues.append("TTS speaker not in current train speakers")
        if len(extra_h_sel) != Q or len(patho_sel) != Q:
            issues.append("selected extra/patho length != Q")
        if entries_h != entries_p:
            issues.append(f"H:P entries not 1:1 ({entries_h}:{entries_p})")
        if entries_h != entries or entries_p != entries:
            issues.append(
                f"entries != 2P+Q ({entries_h}/{entries_p} vs {entries})"
            )
        if abs(entries - scale * B) > 0.51:
            issues.append(
                f"entries {entries} not ‚â?scale*B={scale}*{B}={scale * B}"
            )
        if prev_extra_h is not None and extra_h_names[: len(prev_extra_h)] != prev_extra_h:
            issues.append("extra healthy not nested prefix of smaller scale")
        if prev_patho is not None and patho_names[: len(prev_patho)] != prev_patho:
            issues.append("patho not nested prefix of smaller scale")
        if gold_names != [r["output_filename"] for r in baseline_selected]:
            issues.append("healthy baseline changed with scale")

        ok = not issues
        report = {
            "fold_idx": fold_idx,
            "fold_seed": int(fold_seed),
            "tts_scale": scale,
            "num_augmented": NUM_AUGMENTED,
            "nH_real": nH,
            "nP_real": nP,
            "B": B,
            "Q": Q,
            "healthy_tts_required": healthy_required,
            "pathological_tts_required": patho_required,
            "n_tts_healthy_baseline": n_tts_h_baseline,
            "n_tts_healthy_extra": Q,
            "n_tts_healthy_total": healthy_required,
            "n_tts_pathological": Q,
            "n_healthy_train_available": len(h_avail),
            "n_healthy_extra_candidates": len(h_extra_cand),
            "n_patho_train_available": len(p_avail),
            "healthy_pool_ok": len(h_avail) >= healthy_required,
            "patho_pool_ok": len(p_avail) >= patho_required,
            "n_test_speaker_overlap": int(sum(sid in test_set for sid in all_sids)),
            "entries_healthy": entries_h,
            "entries_pathological": entries_p,
            "entries_equal": entries_h == entries_p,
            "entries_equals_scale_B": abs(entries - scale * B) <= 0.51,
            "scale_times_B": scale * B,
            "nested_prefix_ok": (
                prev_extra_h is None
                or extra_h_names[: len(prev_extra_h)] == prev_extra_h
            ) and (
                prev_patho is None
                or patho_names[: len(prev_patho)] == prev_patho
            ),
            "baseline_locked": True,
            "ok": ok,
            "issues": issues,
            "selected_synth": [_compact_rec(r, r["role"]) for r in all_sel],
        }
        tag = f"tts_scale{str(scale).replace('.', '')}"
        if tag.endswith("0"):
            tag = tag[:-1]
        man_path = out_fold_dir / f"{tag}_sample.json"
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(
            "[fold %d] scale=%s B=%d Q=%d Htts=%d/%d Ptts=%d/%d entries=%d:%d ok=%s",
            fold_idx, scale, B, Q,
            healthy_required, len(h_avail),
            patho_required, len(p_avail),
            entries_h, entries_p, ok,
        )
        if issues:
            for msg in issues:
                logger.error("[fold %d][%s] %s", fold_idx, scale, msg)
        scale_reports.append({k: v for k, v in report.items() if k != "selected_synth"})
        prev_extra_h = extra_h_names
        prev_patho = patho_names

    return {
        "fold_idx": fold_idx,
        "fold_seed": int(fold_seed),
        "nH_real": nH,
        "nP_real": nP,
        "n_test_speakers": len(test_set),
        "n_train_real_speakers": len(train_real_set),
        "ok": all(r["ok"] for r in scale_reports),
        "scales": scale_reports,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="vowel_9concat nested TTS-scale dry-run")
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_tts_vowel_9concat_bal11"),
    )
    ap.add_argument(
        "--new_tts_dir",
        type=Path,
        default=Path("./svd_cleaned_paper_tts_vowel_9concat_scale_candidates"),
    )
    ap.add_argument(
        "--bal11_baseline_dir",
        type=Path,
        default=Path(
            "experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11/"
            "vowel_9concat/feat8/tapt_then_selected_layers_L0-L8/aug_sentence"
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "experiments/svd_paper_mafr_tapt_orig_aug_disjoint_synth_bal11_vowel9concat_tts_scale_nested/"
            "feat8/tapt_then_selected_layers_L0-L8"
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--reseed_each_fold", action="store_true", default=True)
    ap.add_argument("--no_reseed_each_fold", action="store_true")
    args = ap.parse_args()
    reseed = not args.no_reseed_each_fold
    output_dir = _assert_new_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    new_dir = args.new_tts_dir.resolve()
    if data_dir == new_dir:
        raise ValueError("new_tts_dir must not be the old bal11 dir")

    (
        file_paths, labels, speaker_ids, rec_types, is_synthetic,
        label2id, id2label,
    ) = load_metadata_with_tts(
        str(data_dir), task="detection", recording_type="vowel",
    )
    n_real = sum(1 for s in is_synthetic if not s)
    n_synth = sum(1 for s in is_synthetic if s)
    rec_set = set(rec_types)
    if rec_set != {REC}:
        raise RuntimeError(f"expected only {REC}, got {rec_set}")
    logger.info("loaded real=%d synth_from_bal11=%d labels=%s", n_real, n_synth, label2id)

    healthy_3x = load_synth_pool(data_dir, "healthy")
    new_rows = load_synth_pool(new_dir, "healthy")
    healthy_extra3 = [r for r in new_rows if int(r["extra_idx"]) == 3]
    extra3_other = [r for r in new_rows if int(r["extra_idx"]) != 3]
    if extra3_other:
        raise RuntimeError(
            f"new dir has unexpected healthy extra_idx values: "
            f"{Counter(r['extra_idx'] for r in extra3_other)}"
        )
    patho_1x = load_synth_pool(new_dir, "pathological")
    logger.info(
        "pools: healthy3x=%d extra3=%d patho1x=%d",
        len(healthy_3x), len(healthy_extra3), len(patho_1x),
    )
    if len(healthy_3x) != 2061:
        logger.warning("expected 2061 healthy 3x, got %d", len(healthy_3x))
    if len(healthy_extra3) != 687:
        logger.warning("expected 687 healthy extra3, got %d", len(healthy_extra3))
    if len(patho_1x) != 1351:
        logger.warning("expected 1351 patho 1x, got %d", len(patho_1x))

    folds = create_cv_folds_with_tts(
        speaker_ids, labels, is_synthetic, n_folds=args.n_folds, seed=args.seed,
    )
    fold_summaries = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_summaries.append(
            audit_fold(
                fold_idx=fold_idx,
                n_folds=args.n_folds,
                seed=args.seed,
                reseed_each_fold=reseed,
                file_paths=file_paths,
                labels=labels,
                speaker_ids=speaker_ids,
                rec_types=rec_types,
                is_synthetic=is_synthetic,
                label2id=label2id,
                train_idx=train_idx,
                test_idx=test_idx,
                healthy_3x=healthy_3x,
                healthy_extra3=healthy_extra3,
                patho_1x=patho_1x,
                baseline_dir=args.bal11_baseline_dir,
                out_fold_dir=fold_dir,
            )
        )

    all_ok = all(f["ok"] for f in fold_summaries)
    summary = {
        "script": Path(__file__).name,
        "recording_type": REC,
        "num_augmented": NUM_AUGMENTED,
        "protocol_seed": args.seed,
        "reseed_each_fold": reseed,
        "n_folds": args.n_folds,
        "data_dir": str(data_dir),
        "new_tts_dir": str(new_dir),
        "bal11_baseline_dir": str(Path(args.bal11_baseline_dir).resolve()),
        "n_healthy_3x": len(healthy_3x),
        "n_healthy_extra3": len(healthy_extra3),
        "n_patho_1x": len(patho_1x),
        "all_folds_ok": all_ok,
        "folds": fold_summaries,
        "formula": {
            "B": "2P",
            "Q": "round_half_up((scale-1)*B)",
            "healthy_tts_required": "2(P-H)+Q",
            "pathological_tts_required": "Q",
            "entries_per_class": "2P+Q = scale*B (half-up Q)",
            "nested": "1.1 extra ‚ä?1.2 extra ‚ä?1.3 extra",
            "healthy_balancing": "locked to existing vowel_9concat bal11",
        },
        "did_not_train": True,
    }
    out_path = output_dir / "vowel9concat_tts_scale_nested_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s all_folds_ok=%s", out_path, all_ok)
    return 0 if all_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
