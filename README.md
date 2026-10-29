# Medical Speech Transcription with Wav2Vec2

Fine-tuning pretrained **Wav2Vec2-base** for medical speech transcription using PyTorch and Hugging Face Transformers.

The saved experiment achieved a **best logged validation word error rate (WER) of 20.26%** on **668 validation clips**.

## Approach

The model starts from the pretrained [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base) checkpoint and is fine-tuned using a **Connectionist Temporal Classification (CTC)** objective.

The pipeline includes:

- Resampling audio to **16 kHz** and normalizing waveforms.
- Cleaning transcripts by lowercasing and removing specified punctuation and digits.
- Building a custom character-level vocabulary and tokenizer.
- Filtering recordings to **under five seconds** and removing empty label sequences.
- Dynamically padding audio inputs and transcript labels, with label padding excluded from the loss.
- Fine-tuning with the convolutional feature encoder frozen.
- Evaluating greedy CTC transcriptions using WER.

## Dataset

The notebook loads medical audio recordings and paired transcripts from training and validation CSV files.

Each CSV contains two columns:

| Column | Description |
|---|---|
| `path` | Relative path to an audio recording |
| `transcript` | Corresponding transcription |

Expected Kaggle directory structure:

```text
/kaggle/input/dataset/
├── Training_dataset/
│   ├── Training_dataset.csv
│   └── <audio files and subdirectories>
└── Validation_dataset/
    ├── Validation_dataset.csv
    └── <audio files and subdirectories>
```

Dataset sizes recorded in the saved notebook:

| Split | Initially loaded | Used in the saved training run |
|---|---:|---:|
| Training | 4,662 | 3,139 |
| Validation | 1,000 | 668 |

## Training Configuration

| Parameter | Setting |
|---|---|
| Starting checkpoint | `facebook/wav2vec2-base` |
| Training objective | CTC |
| Convolutional feature encoder | Frozen |
| Configured epochs | 35 |
| Per-device batch size | 16 |
| Gradient accumulation | 2 steps |
| Learning rate | `3e-4` |
| Learning rate scheduler | Linear |
| Warmup steps | 500 |
| Weight decay | `0.01` |
| Mixed precision | FP16 |
| Gradient checkpointing | Enabled |
| Length-based batching | Enabled |
| Evaluation interval | 500 steps |

## Results

The following results come from the notebook’s saved training output. Lower WER is better.

| Training step | Validation loss | Validation WER |
|---:|---:|---:|
| 500 | 0.5657 | 37.42% |
| 1,000 | 0.4064 | 26.65% |
| 1,500 | 0.4500 | 24.37% |
| 2,000 | 0.4261 | 22.85% |
| 2,500 | 0.3828 | 20.93% |
| 3,000 | 0.3803 | **20.26%** |

The saved run completed **3,430 optimization steps** in approximately **2 hours and 3 minutes**.

The best logged validation WER was **0.202556**, recorded at step 3,000. These are validation results from one training run. The notebook does not report a separate held-out test evaluation or a pre-fine-tuning baseline.

## Running the Experiment

1. Open the notebook in a Kaggle GPU environment.
2. Attach the dataset using the directory structure above, or update the paths.
3. Prepare the notebook’s dependencies: PyTorch, torchaudio, Transformers, Datasets, Accelerate, NumPy, pandas, librosa, and jiwer.
4. Run the preprocessing, model setup, training, and evaluation cells in order.

The notebook uses historical Hugging Face APIs, including `datasets.load_metric` and `processor.as_target_processor`. Reproduction requires compatible package versions or updates to those API calls.
