# Pathological voice detection using masked reconstruction fine-tuning of Whisper and synthetic speech generation

Whisper-small experiments for pathological-voice detection on SVD (German) and AVFAD (Portuguese). Each corpus is used as read sentences and as one concatenated vowel clip per speaker. Extra training audio is cloned from the real clip with XTTS-v2. Test folds contain real recordings only, split by speaker into 10 folds.

Raw SVD and AVFAD audio are not in this repository.

Run every command from the repository root. `train/whisper_train_utils.py` and `data/avfad_vowel_segments.py` are helpers. Do not run them on their own.

## Pipeline

```text
data/prepare_svd.py              data/concat_svd_vowels.py -> data/package_svd_vowels.py
data/prepare_avfad_sentences.py  data/concat_avfad_vowels.py
data/prepare_avfad_vowels.py
        |                                |
        v                                v
tts/synthesize_svd_sentences.py  tts/synthesize_svd_vowels.py
tts/synthesize_avfad_sentences.py tts/synthesize_avfad_vowels.py
        |
        v
train/probe_frozen_layers.py
train/train_whisper_masked_svd.py
train/train_whisper_masked_avfad.py
train/train_whisper_peft.py
train/train_whisper_head_*.py
train/train_*_tts_scale*.py
```

## Requirements

- Python 3.10
- PyTorch with CUDA
- `transformers`, `soundfile`, `librosa`, `scikit-learn`, `numpy`, `pandas`
- [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) for the `tts/synthesize_*.py` scripts
- `mamba-ssm` only if you run `train/train_whisper_head_mamba3.py`

See `requirements.txt`. Checkpoints: `openai/whisper-small`. The frozen probe also loads HuBERT, wav2vec2, and WavLM base models.

## 1. Prepare real audio

Obtain SVD and AVFAD separately and point the scripts at those directories. Write cleaned audio under `./datasets/`, not into `data/` (that folder holds scripts).

```bash
python data/prepare_svd.py --input_dir /path/to/svd_raw --output_dir ./datasets/svd

python data/prepare_avfad_sentences.py --raw_dir /path/to/AVFAD --xlsx /path/to/AVFAD_01_00_00.xlsx --output_dir ./datasets/avfad_sentences

python data/prepare_avfad_vowels.py --raw_dir /path/to/AVFAD --xlsx /path/to/AVFAD_01_00_00.xlsx --output_dir ./datasets/avfad_vowels
```

`data/prepare_svd.py` keeps the read sentence and normal-pitch `/a i u/`. The 9-vowel clip is built next from raw multipitch files.

```bash
python data/concat_svd_vowels.py --paper_dir ./datasets/svd --multi_dir ./datasets/svd_multipitch --synth_root ./datasets/svd_vowel_tts_multipitch --raw_root /path/to/svd_raw --out_dir ./datasets/svd_vowel_barche9

python data/package_svd_vowels.py --src_dir ./datasets/svd_vowel_barche9 --out_dir ./datasets/svd_vowel_9concat
```

Order in the SVD clip: `i/a/u x low/normal/high`. AVFAD clips are `i -> a -> u` (8 s per vowel by default).

```bash
python data/concat_avfad_vowels.py --aligned_sv_dir ./datasets/avfad_vowels --vowel_tts_dir ./datasets/avfad_vowel_tts --out_dir ./datasets/avfad_vowel_concat
```

Do not rely on the default paths inside `data/concat_*.py`. Always pass the directories above.

## 2. Synthesize training audio

XTTS-v2. `speaker_wav` is the real clip. Existing wavs are kept unless `--force` is set.

```bash
python tts/synthesize_svd_sentences.py --dataset svd --data_dir ./datasets/svd --output_dir ./datasets/svd_tts --sentences_only --target_synth_ratio 4

python tts/synthesize_svd_vowels.py --data_dir ./datasets/svd_vowel_9concat --target_synth_ratio 3

python tts/synthesize_avfad_sentences.py --data_dir ./datasets/avfad_sentences --target_synth_ratio 1

python tts/synthesize_avfad_vowels.py --data_dir ./datasets/avfad_vowel_concat --output_dir ./datasets/avfad_vowel_tts --target_synth_ratio 1
```

`--target_synth_ratio 1` writes one synthetic file per eligible real file. The SVD sentence experiments in this repo use a 4x healthy pool.

## 3. Frozen layer probe

`train/probe_frozen_layers.py` freezes Whisper, HuBERT, wav2vec2, or WavLM and trains a linear head on one encoder layer. `--run_all_layers` sweeps every layer.

```bash
python train/probe_frozen_layers.py --dataset svd --recording_type sentence --backbone_type wavlm --model_name microsoft/wavlm-base-plus --run_all_layers --num_epochs 30 --n_folds 10 --output_dir ./experiments/probe_wavlm
```

`--backbone_type` is `whisper`, `hubert`, `wav2vec2`, or `wavlm`. Pass the matching `--model_name`.

## 4. Fine-tune Whisper-small

Common flags: `--data_dir`, `--recording_type`, `--feature_layer`, `--strategy_layers`, `--n_folds 10`. Test folds are real audio only. `--exclude_test_speaker_synthetic` drops synthetic clips whose speaker is in the current test fold.

### Masked reconstruction, then selected layers

Stage 1 reconstructs masked acoustic features on real training audio (encoder layers `0..feature_layer`). Stage 2 fine-tunes those layers for detection. Reported depth: SVD layer 8, AVFAD layer 10.

SVD sentences (1:1 healthy TTS after the test-speaker filter):

```bash
python train/train_whisper_masked_svd.py --data_dir ./datasets/svd_tts --task detection --recording_type sentence --method tapt_then_selected_layers --model_name openai/whisper-small --feature_layer 8 --strategy_layers 0,1,2,3,4,5,6,7,8 --n_folds 10 --exclude_test_speaker_synthetic --balance_synth_to_ones --class_weight none --output_dir ./experiments/svd_sentence_L0-8
```

AVFAD sentences (no TTS):

```bash
python train/train_whisper_masked_avfad.py --data_dir ./datasets/avfad_sentences --task detection --recording_type sentence --method tapt_then_selected_layers --model_name openai/whisper-small --feature_layer 10 --strategy_layers 0,1,2,3,4,5,6,7,8,9,10 --max_synth_ratio 0 --max_length_sec 24.0 --n_folds 10 --output_dir ./experiments/avfad_sentence_L0-10
```

Sweep depth by changing `--feature_layer` and `--strategy_layers`.

### Fine-tuning method at layers 0-8 (no masked reconstruction)

```bash
python train/train_whisper_peft.py --data_dir ./datasets/svd_tts --task detection --recording_type sentence --method e2e --strategy adapter --feature_layer 8 --strategy_layers 0,1,2,3,4,5,6,7,8 --exclude_test_speaker_synthetic --output_dir ./experiments/svd_sentence_adapter
```

`--strategy` is `selected_layers`, `adapter`, or `bitfit`.

### Classifier head at layers 0-8

Masked reconstruction stays on. Only the head on the frame sequence changes.

```bash
python train/train_whisper_head_lstm.py --data_dir ./datasets/svd_tts --task detection --recording_type sentence --method tapt_then_selected_layers --classifier_head lstm --feature_layer 8 --strategy_layers 0,1,2,3,4,5,6,7,8 --exclude_test_speaker_synthetic --output_dir ./experiments/svd_sentence_lstm
```

| Script | `--classifier_head` |
|---|---|
| `train/train_whisper_head_lstm.py` | `lstm` |
| `train/train_whisper_head_transformer.py` | `vanilla_transformer` |
| `train/train_whisper_head_mamba3.py` | `mamba3` |

## 5. TTS scale 1.1 / 1.2 / 1.3

These scripts do not use `--max_synth_ratio`. They take a nested speaker-balanced prefix of a locked ranking.

### SVD sentences

Write the ranking first with `train/build_svd_tts_ranking15.py` and `train/build_svd_tts_ranking125.py`. You do not need to train 1.25x or 1.5x. Then:

```bash
python train/train_svd_tts_scale11.py --data_dir ./datasets/svd_tts --scale15_source_dir ./experiments/svd_tts_scale15 --scale125_source_dir ./experiments/svd_tts_scale125 --task detection --recording_type sentence --method tapt_then_selected_layers --feature_layer 8 --strategy_layers 0,1,2,3,4,5,6,7,8 --exclude_test_speaker_synthetic --output_dir ./experiments/svd_tts_scale11
```

`train/train_svd_tts_scale12.py` is the same pattern. `train/train_svd_tts_scale13.py` also needs `--scale12_source_dir` pointing at the 1.2 run.

### AVFAD sentences

```bash
python train/train_avfad_tts_scale.py --data_dir ./datasets/avfad_sentences --audit_tts_scale_only --shared_ranking_dir ./experiments/avfad_tts_rankings

python train/train_avfad_tts_scale.py --data_dir ./datasets/avfad_sentences --tts_scale 1.1 --feature_layer 10 --strategy_layers 0,1,2,3,4,5,6,7,8,9,10 --max_length_sec 24.0 --output_dir ./experiments/avfad_tts_scale11
```

1.2 and 1.3 use `train/train_avfad_tts_scale_12_13.py` with `--tts_scale 1.2` or `--tts_scale 1.3`.

### Vowels (SVD 1.2x, AVFAD 1.1x)

Run `train/build_svd_vowel_tts_ranking.py` first.

```bash
python train/train_svd_vowel_tts_scale12.py --data_dir ./datasets/svd_vowel_9concat --recording_type vowel --max_length_sec 20.0 --method tapt_then_selected_layers --feature_layer 8 --strategy_layers 0,1,2,3,4,5,6,7,8 --output_dir ./experiments/svd_vowel_tts_scale12

python train/train_avfad_vowel_tts_scale.py --data_dir ./datasets/avfad_vowel_concat --recording_type vowel --max_length_sec 24.0 --tts_scale 1.1 --feature_layer 10 --strategy_layers 0,1,2,3,4,5,6,7,8,9,10 --output_dir ./experiments/avfad_vowel_tts_scale11
```

## What each comparison changes

| Question | Script |
|---|---|
| Which frozen layer of which encoder? | `train/probe_frozen_layers.py` |
| How many Whisper layers on sentences? | `train/train_whisper_masked_svd.py` / `train/train_whisper_masked_avfad.py` |
| Adapter / BitFit / selected layers? | `train/train_whisper_peft.py` |
| LSTM / Transformer / Mamba-3 head? | `train/train_whisper_head_*.py` |
| Sentence TTS scale 1.1 / 1.2 / 1.3? | `train/train_svd_tts_scale*.py`, `train/train_avfad_tts_scale*.py` |
| Same scale on vowels? | `train/train_svd_vowel_tts_scale12.py`, `train/train_avfad_vowel_tts_scale.py` |

## Reference

Sentence cleaning and the XTTS cloning setup follow Koudounas et al., *Voice Disorder Analysis: a Transformer-based Approach*, Interspeech 2024.
