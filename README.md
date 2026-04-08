# Wav2Vec2 Fine-Tuning (Without Pre-training) — ASR with CTC

> **Note:** This is a project from back in 2024, just putting it on GitHub for reference for future use.

This notebook explores fine-tuning a **Wav2Vec2** model for **Automatic Speech Recognition (ASR)** using a **CTC (Connectionist Temporal Classification)** objective — trained from scratch without leveraging pre-trained weights. The experiment was run on a Kaggle GPU (T4) environment in April 2024.

---

## Overview

The goal was to train a Wav2Vec2-based speech-to-text model using the HuggingFace `transformers` library, building a custom vocabulary tokenizer from the dataset rather than relying on a pre-trained checkpoint. Audio inputs are processed at 16 kHz and transcriptions are cleaned (special characters and digits removed) before training.

---

## Pipeline

1. **Dataset Loading** — Loads an audio dataset (CommonVoice-style) with `test` and `validation` splits (~7,016 examples each)
2. **Text Preprocessing** — Strips special characters (`,.?!--,`) and digits via regex; lowercases all transcriptions
3. **Vocabulary Construction** — Extracts a character-level vocabulary from the dataset; adds `[UNK]` and `[PAD]` tokens (29 tokens total)
4. **Tokenizer Setup** — `Wav2Vec2CTCTokenizer` initialized from the custom `vocab.json`; pushed to HuggingFace Hub (`Aviral2412/fineturning-without-pretraining-2`)
5. **Feature Extraction** — `Wav2Vec2FeatureExtractor` configured for 16 kHz mono audio, with mean normalization
6. **Processor** — Combines tokenizer + feature extractor into `Wav2Vec2Processor`
7. **Audio Preprocessing** — Loads `.mp3` files via `librosa`, resamples to 16 kHz, extracts input values and label IDs; filters to max 5 seconds
8. **Data Collation** — Custom `DataCollatorCTCWithPadding` for dynamic padding of audio inputs and label sequences separately
9. **Model** — `Wav2Vec2ForCTC` initialized without pre-trained weights
10. **Training** — HuggingFace `Trainer` with `WER` (Word Error Rate) as evaluation metric

---

## Training Configuration

| Parameter              | Value          |
|------------------------|----------------|
| Epochs                 | 35             |
| Per-device batch size  | 16             |
| Gradient accumulation  | 2 steps        |
| Learning rate          | 3e-4           |
| LR scheduler           | Linear         |
| Warmup steps           | 500            |
| Weight decay           | 0.01           |
| FP16 training          | ✅ Enabled      |
| Gradient checkpointing | ✅ Enabled      |
| Eval/save steps        | 500            |
| Platform               | Kaggle (GPU T4)|

---

## Results

Training was run for ~35 epochs (~4,060 steps total). Because the model was trained **from scratch** (no pre-trained weights), the WER remained high (~0.9999), confirming that Wav2Vec2 strongly benefits from pre-trained representations for downstream ASR tasks. This experiment serves as a useful baseline/ablation reference.

| Step | Training Loss | Validation Loss | WER    |
|------|---------------|-----------------|--------|
| 500  | 1829.24       | 781.05          | 0.9999 |
| 1000 | 1459.40       | 777.18          | 0.9999 |
| 1500 | 1454.83       | 777.35          | 0.9999 |
| 2000 | 1448.89       | 788.01          | 0.9999 |
| 4000 | 1442.62       | 779.55          | 0.9999 |

---

## Key Takeaway

Training Wav2Vec2 for ASR **without pre-training** does not converge to useful representations within a reasonable number of epochs. For practical ASR use cases, always start from a pre-trained checkpoint (e.g., `facebook/wav2vec2-base` or `facebook/wav2vec2-large-960h`).

---

## Requirements

```
torch>=2.1.2
torchaudio
transformers>=4.39.3
datasets
librosa
huggingface_hub
accelerate
jiwer          # for WER metric
```

---

## File Structure

```
.
└── Wave2VecFineTuning.ipynb   # Main experiment notebook
```

---

## HuggingFace Hub

The tokenizer was pushed to: [`Aviral2412/fineturning-without-pretraining-2`](https://huggingface.co/Aviral2412/fineturning-without-pretraining-2)

---

## References

- [HuggingFace Wav2Vec2 Docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [Fine-Tuning Wav2Vec2 for ASR (HF Blog)](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [CTC Loss (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
