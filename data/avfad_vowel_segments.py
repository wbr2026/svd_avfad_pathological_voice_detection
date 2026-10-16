#!/usr/bin/env python3
"""Segment extraction for AVFAD Barche-3 (i â†?a â†?u concat).

Plan A (head): first ``segment_sec`` of each vowel wav (zero-pad if short).
Plan B (stable): librosa trim (top_db) â†?first voiced onset â†?skip post_onset_sec
â†?take ``segment_sec`` (zero-pad if short). Matches AVFAD cleaning trim + Praat-style
200 ms after phonation start.
Plan C (vad_rep): trim â†?librosa VAD split â†?if >=3 voiced chunks take middle,
else take longest â†?pad/truncate to ``segment_sec``.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

SegmentMode = Literal["head", "stable"]


def extract_segment(
    waveform: np.ndarray,
    sr: int,
    segment_sec: float,
    mode: SegmentMode = "stable",
    top_db: float = 30.0,
    post_onset_sec: float = 0.2,
) -> np.ndarray:
    n_target = int(round(segment_sec * sr))
    if n_target <= 0:
        raise ValueError("segment_sec must be positive")
    if waveform.size == 0:
        return np.zeros(n_target, dtype=np.float32)

    if mode == "head":
        return _pad_or_truncate(waveform, n_target)

    if not HAS_LIBROSA:
        return _pad_or_truncate(waveform, n_target)

    trimmed, _ = librosa.effects.trim(waveform.astype(np.float32), top_db=top_db)
    if trimmed.size == 0:
        trimmed = waveform.astype(np.float32)

    start = 0
    intervals = librosa.effects.split(trimmed, top_db=top_db)
    if intervals.shape[0] > 0:
        start = int(intervals[0, 0])

    skip = int(round(post_onset_sec * sr))
    start = min(start + skip, len(trimmed))
    return _pad_or_truncate(trimmed[start:], n_target)


def _pad_or_truncate(waveform: np.ndarray, n_target: int) -> np.ndarray:
    if len(waveform) >= n_target:
        return waveform[:n_target].astype(np.float32, copy=True)
    out = np.zeros(n_target, dtype=np.float32)
    out[: len(waveform)] = waveform
    return out


def _voiced_chunks(
    trimmed: np.ndarray,
    sr: int,
    top_db: float,
    min_rep_sec: float,
) -> list[np.ndarray]:
    if not HAS_LIBROSA or trimmed.size == 0:
        return []
    intervals = librosa.effects.split(trimmed.astype(np.float32), top_db=top_db)
    min_samples = int(round(min_rep_sec * sr))
    chunks: list[np.ndarray] = []
    for start, end in intervals:
        if int(end) - int(start) >= min_samples:
            chunks.append(trimmed[int(start): int(end)].astype(np.float32, copy=True))
    return chunks


def pick_vad_rep_chunk(
    chunks: list[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """>=3 chunks -> middle by time order; else -> longest by sample count."""
    n = len(chunks)
    if n == 0:
        raise ValueError("chunks must be non-empty")
    if n >= 3:
        idx = n // 2
        rule = "middle"
    else:
        idx = max(range(n), key=lambda i: len(chunks[i]))
        rule = "longest"
    return chunks[idx], {
        "n_vad_segments": n,
        "pick_rule": rule,
        "picked_index": idx,
    }


def extract_vad_rep_segment(
    waveform: np.ndarray,
    sr: int,
    segment_sec: float,
    top_db: float = 30.0,
    min_rep_sec: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Plan C: VAD split; >=3 voiced segments -> middle, else longest; then cap/pad."""
    n_target = int(round(segment_sec * sr))
    if n_target <= 0:
        raise ValueError("segment_sec must be positive")

    meta: Dict[str, object] = {
        "n_vad_segments": 0,
        "pick_rule": "none",
        "picked_index": -1,
    }

    if waveform.size == 0:
        return np.zeros(n_target, dtype=np.float32), meta

    if not HAS_LIBROSA:
        meta["pick_rule"] = "fallback_no_librosa"
        return _pad_or_truncate(waveform, n_target), meta

    trimmed, _ = librosa.effects.trim(waveform.astype(np.float32), top_db=top_db)
    if trimmed.size == 0:
        trimmed = waveform.astype(np.float32)

    chunks = _voiced_chunks(trimmed, sr, top_db, min_rep_sec)
    meta["n_vad_segments"] = len(chunks)

    if chunks:
        picked, pick_meta = pick_vad_rep_chunk(chunks)
        meta.update(pick_meta)
    else:
        meta["pick_rule"] = "fallback_trimmed"
        picked = trimmed

    return _pad_or_truncate(picked, n_target), meta
