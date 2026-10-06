# Voice disorder detection on SVD and AVFAD

Whisper-small experiments for pathological-voice detection on SVD (German) and AVFAD (Portuguese). Each corpus is used as read sentences and as one concatenated vowel clip per speaker. Extra training audio is cloned from the real clip with XTTS-v2. Test folds contain real recordings only, split by speaker into 10 folds.

Raw SVD and AVFAD audio are not in this repository.

Scripts live in two trees:

- `AI4Voice/src/` — SVD cleaning, AVFAD cleaning, AVFAD sentence TTS
- `AI4Voice copy/src/` — vowel concatenation, the other TTS scripts, and training

## Prepare real audio

| Step | Script |
|---|---|
| SVD sentences and normal-pitch `/a i u/` | `AI4Voice/src/svd_data_cleaning.py` |
| SVD 9-vowel clip (`i/a/u` × low/normal/high) | `AI4Voice copy/src/build_iau_barche9_speaker_aligned.py` |
| Package that clip as `vowel_9concat` | `AI4Voice copy/src/build_vowel9_concat.py` |
| AVFAD sentences (six CAPE-V sentences, three repetitions) | `AI4Voice/src/avfad_clean_paper_aligned.py` |
| AVFAD sustained vowels | `AI4Voice/src/avfad_clean_paper_aligned_sv.py` |
| AVFAD `i → a → u` clip (8 s per vowel) | `AI4Voice copy/src/build_avfad_iau_barche3_stable.py` |

```bash
python svd_data_cleaning.py --input_dir /path/to/svd_raw --output_dir ./svd_cleaned

python avfad_clean_paper_aligned.py \
  --raw_dir /path/to/AVFAD \
  --xlsx /path/to/AVFAD_01_00_00.xlsx \
  --output_dir ./avfad_sentences

python avfad_clean_paper_aligned_sv.py \
  --raw_dir /path/to/AVFAD \
  --xlsx /path/to/AVFAD_01_00_00.xlsx \
  --output_dir ./avfad_vowels
