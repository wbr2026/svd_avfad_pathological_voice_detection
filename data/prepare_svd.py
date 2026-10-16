#!/usr/bin/env python3

import os
import re
import sys
import csv
import json
import struct
import shutil
import zipfile
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from sklearn.model_selection import StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==== File patterns to KEEP ====
# Based on actual SVD filenames: {speaker_id}-phrase.nsp, {id}-a_n.nsp, etc.
KEEP_PATTERNS = {
    'sentence':       re.compile(r'^(\d+)-phrase\.(nsp|wav)$', re.IGNORECASE),
    'vowel_a_normal': re.compile(r'^(\d+)-a_n\.(nsp|wav)$', re.IGNORECASE),
    'vowel_i_normal': re.compile(r'^(\d+)-i_n\.(nsp|wav)$', re.IGNORECASE),
    'vowel_u_normal': re.compile(r'^(\d+)-u_n\.(nsp|wav)$', re.IGNORECASE),
}


def classify_file(filename):
    """Returns (recording_type, speaker_id) if file should be kept, else None."""
    for rec_type, pattern in KEEP_PATTERNS.items():
        m = pattern.match(filename)
        if m:
            return rec_type, m.group(1)
    return None


# ==== NSP to WAV conversion ====
def convert_nsp_to_wav(nsp_path, wav_path, sr=50000):
    """Convert .nsp (raw 16-bit PCM, 50kHz) to .wav"""
    try:
        with open(nsp_path, 'rb') as f:
            raw = f.read()
        n = len(raw) // 2
        if n == 0:
            return False
        samples = struct.unpack(f'<{n}h', raw[:n * 2])
        if HAS_SOUNDFILE:
            audio = np.array(samples, dtype=np.float32) / 32768.0
            sf.write(wav_path, audio, sr)
        else:
            data_size = n * 2
            with open(wav_path, 'wb') as f:
                f.write(b'RIFF')
                f.write(struct.pack('<I', 36 + data_size))
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
                f.write(b'data')
                f.write(struct.pack('<I', data_size))
                f.write(struct.pack(f'<{n}h', *samples))
        return True
    except Exception as e:
        logger.warning(f"NSP convert failed {nsp_path}: {e}")
        return False


# ==== Extract all zip files ====
def extract_all_zips(input_dir):
    """Auto-extract all .zip in input_dir to same-name folders."""
    input_path = Path(input_dir)
    zips = sorted(input_path.glob('*.zip'))
    if not zips:
        logger.info("No zip files to extract")
        return
    logger.info(f"Found {len(zips)} zip files, extracting...")
    for zp in zips:
        target = input_path / zp.stem
        if target.exists() and target.is_dir() and any(target.iterdir()):
            logger.info(f"  Skip {zp.name} (already extracted)")
            continue
        logger.info(f"  Extracting {zp.name} ...")
        try:
            with zipfile.ZipFile(str(zp), 'r') as z:
                z.extractall(str(target))
            logger.info(f"    Done")
        except Exception as e:
            logger.error(f"    Failed: {e}")
    logger.info("All zips extracted")


# ==== Find speaker directories (handles nested zip extraction) ====
def find_speaker_dirs(pathology_dir):
    """
    Find speaker dirs under a pathology folder.
    Handles nested cases like: Dysphonie/Dysphonie/1/sentences/...
    """
    subdirs = sorted([d for d in pathology_dir.iterdir() if d.is_dir()])
    if not subdirs:
        return []

    # Check if first subdir is numeric (speaker id)
    if subdirs[0].name.isdigit():
        return subdirs

    # Try one level deeper (common with zip extraction nesting)
    for sd in subdirs:
        deeper = sorted([d for d in sd.iterdir() if d.is_dir()])
        if deeper and deeper[0].name.isdigit():
            return deeper

    # Try two levels deeper
    for sd in subdirs:
        for ssd in sorted(sd.iterdir()):
            if ssd.is_dir():
                deeper = sorted([d for d in ssd.iterdir() if d.is_dir()])
                if deeper and deeper[0].name.isdigit():
                    return deeper

    return subdirs


# ==== Scan all SVD data ====
def scan_svd_data(input_dir):
    """Scan directory and collect all target recordings."""
    input_path = Path(input_dir)
    records = []
    skip_ext = Counter()

    # Get all pathology category folders (skip hidden dirs)
    dirs = sorted([d for d in input_path.iterdir()
                   if d.is_dir() and not d.name.startswith('.')])
    logger.info(f"Found {len(dirs)} pathology folders")

    for pdir in dirs:
        pname = pdir.name
        is_healthy = pname.lower() == 'healthy'
        label = 'healthy' if is_healthy else 'pathological'

        speaker_dirs = find_speaker_dirs(pdir)
        if not speaker_dirs:
            logger.warning(f"  No speakers in {pname}/")
            continue

        spk_count = 0
        for sdir in speaker_dirs:
            # Look in sentences/ and vowels/ subdirectories
            scan = []
            for sub in ['sentences', 'vowels']:
                p = sdir / sub
                if p.exists():
                    scan.append(p)
            if not scan:
                scan = [sdir]  # fallback

            found = False
            for d in scan:
                if not d.is_dir():
                    continue
                for f in sorted(d.iterdir()):
                    if not f.is_file():
                        continue
                    result = classify_file(f.name)
                    if result is None:
                        skip_ext[f.suffix.lower()] += 1
                        continue
                    rtype, sid = result
                    records.append({
                        'speaker_id': sid,
                        'label': label,
                        'pathology': pname if not is_healthy else 'healthy',
                        'recording_type': rtype,
                        'original_path': str(f),
                        'filename': f.name,
                    })
                    found = True
            if found:
                spk_count += 1

        logger.info(f"  {pname}: {spk_count} speakers")

    logger.info(f"Skipped files by extension: {dict(skip_ext)}")
    return records


# ==== Main cleaning pipeline ====
def clean_svd_data(input_dir, output_dir, top_k=6, n_folds=10):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 0: Extract zips
    logger.info("=" * 60)
    logger.info("Step 0: Extracting zip files...")
    extract_all_zips(input_dir)

    # Step 1: Scan
    logger.info("=" * 60)
    logger.info("Step 1: Scanning data...")
    records = scan_svd_data(input_dir)
    logger.info(f"Total target recordings: {len(records)}")
    tc = Counter(r['recording_type'] for r in records)
    logger.info(f"Type distribution: {dict(tc)}")
    if not records:
        logger.error("No recordings found! Check directory structure.")
        sys.exit(1)

    # Step 2: Convert nsp -> wav
    logger.info("=" * 60)
    logger.info("Step 2: Converting nsp -> wav...")
    sdir = out / 'sentences'
    vdir = out / 'vowels'
    sdir.mkdir(exist_ok=True)
    vdir.mkdir(exist_ok=True)

    processed = []
    errs = 0
    for i, r in enumerate(records):
        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i + 1}/{len(records)}")
        src = Path(r['original_path'])
        tdir = sdir if r['recording_type'] == 'sentence' else vdir
        safe_p = re.sub(r'[^\w\-]', '_', r['pathology'])
        ofn = f"{r['speaker_id']}_{safe_p}_{r['recording_type']}.wav"
        opath = tdir / ofn

        ok = True
        if src.suffix.lower() == '.nsp':
            ok = convert_nsp_to_wav(str(src), str(opath))
        else:
            try:
                shutil.copy2(str(src), str(opath))
            except Exception:
                ok = False
        if ok:
            r['processed_path'] = str(opath)
            r['output_filename'] = ofn
            processed.append(r)
        else:
            errs += 1

    logger.info(f"Processed: {len(processed)}, Errors: {errs}")

    # Step 3: Stats
    logger.info("=" * 60)
    logger.info("Step 3: Statistics...")
    spks = defaultdict(lambda: {'label': None, 'pathology': None})
    for r in processed:
        spks[r['speaker_id']]['label'] = r['label']
        spks[r['speaker_id']]['pathology'] = r['pathology']

    nh = sum(1 for s in spks.values() if s['label'] == 'healthy')
    npat = sum(1 for s in spks.values() if s['label'] == 'pathological')
    ns = sum(1 for r in processed if r['recording_type'] == 'sentence')
    nv = sum(1 for r in processed if r['recording_type'] != 'sentence')
    logger.info(f"Speakers: {len(spks)} (Healthy={nh}, Pathological={npat})")
    logger.info(f"Sentences: {ns}, Vowels: {nv}")

    # Step 4: Pathology classes
    logger.info("=" * 60)
    logger.info("Step 4: Pathology class distribution...")
    pcounts = Counter()
    for s in spks.values():
        if s['label'] == 'pathological':
            pcounts[s['pathology']] += 1

    logger.info(f"Total pathology classes: {len(pcounts)}")
    for p, c in pcounts.most_common():
        logger.info(f"  {p}: {c}")

    topk = pcounts.most_common(top_k)
    topk_names = [n for n, _ in topk]
    logger.info(f"\nTop-{top_k} for classification task:")
    for n, c in topk:
        logger.info(f"  * {n}: {c}")

    # Step 5: K-fold CV (speaker-level stratified)
    logger.info("=" * 60)
    logger.info(f"Step 5: {n_folds}-fold CV split...")
    sids = sorted(spks.keys())
    slabels = [spks[s]['label'] for s in sids]
    folds = None

    if HAS_SKLEARN and len(set(slabels)) >= 2:
        minc = min(Counter(slabels).values())
        nf = min(n_folds, minc)
        if nf < n_folds:
            logger.warning(f"Adjusting folds {n_folds} -> {nf} (min class={minc})")
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
        folds = {}
        for fi, (tri, tei) in enumerate(skf.split(sids, slabels)):
            folds[f'fold_{fi}'] = {
                'train': [sids[i] for i in tri],
                'test': [sids[i] for i in tei],
            }
            trh = sum(1 for i in tri if slabels[i] == 'healthy')
            teh = sum(1 for i in tei if slabels[i] == 'healthy')
            logger.info(f"  Fold {fi}: Train({len(tri)}: H={trh}, P={len(tri)-trh}) | "
                        f"Test({len(tei)}: H={teh}, P={len(tei)-teh})")
        with open(out / 'cv_folds.json', 'w', encoding='utf-8') as f:
            json.dump(folds, f, indent=2)
        logger.info(f"Saved cv_folds.json")
    else:
        logger.warning("Skipping CV split (sklearn missing or insufficient classes)")

    # Step 6: Save metadata
    logger.info("=" * 60)
    logger.info("Step 6: Saving metadata...")

    # Full metadata
    fields = ['speaker_id', 'label', 'pathology', 'recording_type',
              'output_filename', 'processed_path', 'original_path']
    with open(out / 'metadata.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in processed:
            w.writerow({k: r.get(k, '') for k in fields})
    logger.info(f"  metadata.csv ({len(processed)} records)")

    # Detection task metadata (binary)
    with open(out / 'detection_metadata.csv', 'w', newline='', encoding='utf-8') as f:
        det_fields = ['speaker_id', 'label', 'recording_type', 'output_filename']
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        for r in processed:
            w.writerow({k: r[k] for k in det_fields})
    logger.info(f"  detection_metadata.csv ({len(processed)} records)")

    # Classification task metadata (top-K pathological only)
    cls_recs = [r for r in processed if r['pathology'] in topk_names]
    with open(out / 'classification_metadata.csv', 'w', newline='', encoding='utf-8') as f:
        cls_fields = ['speaker_id', 'pathology', 'recording_type', 'output_filename']
        w = csv.DictWriter(f, fieldnames=cls_fields)
        w.writeheader()
        for r in cls_recs:
            w.writerow({k: r[k] for k in cls_fields})
    logger.info(f"  classification_metadata.csv ({len(cls_recs)} records, top-{top_k})")

    # Config
    config = {
        'total_speakers': len(spks),
        'healthy_speakers': nh,
        'pathological_speakers': npat,
        'sentence_recordings': ns,
        'vowel_recordings': nv,
        'total_recordings': len(processed),
        'pathology_distribution': dict(pcounts.most_common()),
        'top_k_classes': topk_names,
        'n_folds': n_folds,
    }
    with open(out / 'dataset_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"  dataset_config.json")

    # Final summary
    logger.info("=" * 60)
    logger.info("CLEANING COMPLETE!")
    logger.info(f"Output directory: {out}")
    logger.info(f"  sentences/  ({ns} wav files)")
    logger.info(f"  vowels/     ({nv} wav files)")
    logger.info("")
    logger.info("Comparison with Paper Table 1 (SVD):")
    logger.info(f"                       Paper    Actual")
    logger.info(f"  #Healthy             687      {nh}")
    logger.info(f"  #Pathological        1356     {npat}")
    logger.info(f"  #Sentences           2043     {ns}")
    logger.info(f"  #Vowels              6129     {nv}")
    logger.info(f"  #PathClasses         71       {len(pcounts)}")
    logger.info("=" * 60)


def main():
    p = argparse.ArgumentParser(
        description='SVD Data Cleaning - Voice Disorder Analysis (Interspeech 2024)')
    p.add_argument('--input_dir', required=True, help='Raw SVD data directory')
    p.add_argument('--output_dir', required=True, help='Output directory')
    p.add_argument('--top_k', type=int, default=6, help='Top-K classes (default 6)')
    p.add_argument('--n_folds', type=int, default=10, help='CV folds (default 10)')
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"Not found: {args.input_dir}")
        sys.exit(1)

    if not HAS_SOUNDFILE:
        logger.warning("soundfile not installed (pip install soundfile)")
    if not HAS_SKLEARN:
        logger.warning("scikit-learn not installed (pip install scikit-learn)")

    clean_svd_data(args.input_dir, args.output_dir, args.top_k, args.n_folds)


if __name__ == '__main__':
    main()