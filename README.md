# Pathological voice detection using masked reconstruction fine-tuning of Whisper and synthetic speech generation

This repository contains the code for binary pathological voice detection using Whisper and TTS-generated speech. Experiments are conducted on the Saarbrücken Voice Database (SVD) and the Advanced Voice Function Assessment Databases (AVFAD).

## Dataset

| Dataset | Recording | Speakers | Samples |
| --- | --- | ---: | ---: |
| SVD | Sentence | 1,988 | 2,256 |
| SVD | Vowel | 2,038 | 2,038 |
| AVFAD | Sentence | 708 | 2,124 |
| AVFAD | Vowel | 708 | 708 |

SVD vowel samples concatenate `/i/`, `/a/`, and `/u/` at three pitch levels. AVFAD vowel samples concatenate the three vowels at normal pitch. The datasets are evaluated using speaker-level 10-fold cross-validation.

The original recordings are not included in this repository. Data preparation scripts are in [`data/`](data/).

## Model

The main model uses [Whisper-small](https://huggingface.co/openai/whisper-small) with masked acoustic feature reconstruction for fine-tuning, followed by temporal mean pooling and a linear classifier (Mean+Linear).

The repository also includes comparisons of frozen Whisper, HuBERT, WavLM, and wav2vec 2.0 representations, other fine-tuning strategies and downstream classifiers, and XTTS-v2 synthetic speech for class balancing and data expansion.

## Code

- [`data/`](data/): preprocessing and vowel concatenation
- [`train/`](train/): frozen-layer evaluation, Whisper fine-tuning, classifiers, and TTS-scale experiments
- [`tts/`](tts/): synthetic speech generation

## Environment

The experiments use Python 3.10, PyTorch, CUDA, and Hugging Face Transformers. Install the basic dependencies with:

```bash
pip install -r requirements.txt
pip install audiomentations
```

Additional packages such as `peft` and `mamba-ssm` are needed for the corresponding model comparisons.

## Note

The repository does not include the raw datasets, generated audio, or all intermediate TTS pools and fold manifests. Some experiment-specific scripts, including the final SVD vowel 1.1× training configuration, are not publicly available. See [Reproduction notes](REPRODUCIBILITY_NOTES.md) for detailed commands and known limitations.
