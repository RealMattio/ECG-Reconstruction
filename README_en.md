# PPG → ECG Reconstruction — PINN Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-MIMIC--III%20WDB-blueviolet)
![Approach](https://img.shields.io/badge/Approach-Physics--Informed%20NN-orange)
![SLURM](https://img.shields.io/badge/HPC-SLURM-informational?logo=linux&logoColor=white)

| [🇬🇧 Read in English](README_en.md) | [🇮🇹 Leggi in Italiano](README.md) |
| :--- | :--- |

---

## Objective

Reconstruction of the **ECG (Lead II)** signal starting strictly from the **PPG (photoplethysmography)** signal acquired via a non-invasive sensor. This is achieved using a Physics-Informed Neural Network (PINN) that incorporates the constraints of the McSharry differential equation, which governs cardiac dynamics, directly into the loss function.

---

## 1. Dataset — MIMIC-III Waveform Database (Matched)

### Clinical Patient Selection

The source dataset is the **MIMIC-III Waveform Database Matched** (`mimic3wdb-matched`), accessible on [PhysioNet](https://physionet.org/content/mimic3wdb-matched/1.0/) upon registration and signing of the Data Use Agreement.

Patient selection occurs in two phases:

1. **Clinical Filtering** (`scripts/filter_no_cardiac_patologies_patients.py`, `scripts/filter_ecg_patologies.py`): Patients are extracted from the MIMIC-III clinical database and filtered by excluding those with ICD-9 diagnoses related to structural heart diseases (heart failure, infarction, atrial fibrillation, etc.). The result is an `allowed_patients.csv` file.

2. **Selective Download** (`scripts/download_healthy_waveforms.py`): Downloads from PhysioNet only the WFDB waveforms of the approved patients, further filtering by a minimum duration (≥ 3 minutes per session) and balancing the contribution per patient (≤ 36 hours total). The download automatically pauses if free disk space drops below 4 GB. It supports parallelization via SLURM using `--worker_id` and `--total_workers`.

The downloaded files are `.hea` + `.dat` pairs in standard WFDB format, organized by patient:

```text
mimic3wdb-matched_healthy_data/
├── p00/
│   └── p000052/
│       ├── 3533390_0011.hea
│       └── 3533390_0011.dat
├── p01/
│   └── ...
```

Each record contains at least the **`II`** (ECG Lead II) and **`PLETH`** (PPG) channels.

---

## 2. Preprocessing and Manifest Creation

The preprocessing is entirely **zero-copy**: no signal is duplicated on disk. The only output is a `dataset_manifest.json` file (a few KBs) that indexes the valid segments with their original WFDB paths.

### Execution

```bash
sbatch run_pinn_manifest.sh
# or locally:
python scripts/preprocess_mimic_pinn_manifest.py \
    --input_dir  ./mimic3wdb-matched_healthy_data \
    --output_dir ./mimic3wdb-matched_healthy_data \
    --target_fs 125 --ppg_thr 0.9 --ecg_thr 0.9
```

### Internal Steps

```
WFDB File (.hea + .dat)
      │
      ▼
  Resampling to 125 Hz (scipy.signal.resample)
      │
      ▼
  Butterworth bandpass filter (order 4)
      ├── PPG:  0.5 – 5.0 Hz
      └── ECG:  0.5 – 40.0 Hz
      │
      ▼
  Sliding window for quality evaluation (4 s, step 1 s)
      ├── PPG: Spectral SQI + morphology + polarity  →  Pearson score
      └── ECG: kurtosis + QRS morphology             →  Pearson score
      │
      ▼
  Merging valid PPG intervals ∩ valid ECG intervals
      │
      ▼
  Extraction of contiguous segments ≥ 7 s
      │
      ▼
  dataset_manifest.json  (WFDB path + segment indices + PPG inversion flag)
```

---

## 3. Window Filtering — Signal Quality Criteria

Each 4-second window is accepted only if it simultaneously passes the checks for both PPG **and** ECG.

### 3.1 PPG — Signal Quality Index (SQI)

| Criterion | Method | Threshold |
|---|---|---|
| **Non-flat signal** | standard deviation | `std > 1e-6` |
| **Spectral SQI** | Welch PSD — ratio of power in HR peak (0.5–3 Hz) to total (0.5–10 Hz) | `≥ 0.5` |
| **Beat morphology** | Pearson correlation between individual detected beats and median template | `≥ 0.9` |
| **Polarity** | Amplitude asymmetry: `max(x – mean) > |min(x – mean)|` → if false, the signal is inverted | — |

### 3.2 ECG — Signal Quality Index

| Criterion | Method | Threshold |
|---|---|---|
| **Non-flat signal** | standard deviation | `std > 1e-4` |
| **Kurtosis** | Pearson kurtosis (non-Fisher) — a clean ECG has impulsive R peaks → high kurtosis | `≥ 5.0` |
| **QRS morphology** | Pearson correlation between extracted QRS complexes and median template | `≥ 0.9` |

### 3.3 Intersection and Segmentation

Intervals that pass the PPG filtering and those that pass the ECG filtering are merged (overlapping intervals are combined) and intersected. Only the resulting contiguous segments with a length of **≥ 7 seconds** are added to the manifest.

### 3.4 Per-Window Normalization During Training

At the time of extracting each training window (in `MimicSmartDataset.__getitem__`), both the PPG and ECG are **independently** normalized in the `[0, 1]` range using per-window min-max scaling:

```
x_norm = (x − min(x)) / (max(x) − min(x) + ε)
```

- **PPG (7 s)**: Normalized independently, after polarity correction.
- **ECG (7 s = past + target)**: Past and target are normalized **together** with the same min/max, so the model doesn't see artificial discontinuities at the boundary between known context and the target to predict.
- **theta, omega** (PINN physical signals): **Not normalized** — they encode phase (rad) and angular frequency (rad/s).

---

## 4. Training Pipeline Architecture

```
dataset_manifest.json
        │
        ▼
  MimicSmartDataset.__init__
  (loads WFDB → resamples → bandpass filter
   → computes theta/omega → caches in RAM)
        │
        ▼
  MimicSmartDataset.__getitem__
  (for each window: PPG polarity correction
   → [0,1] normalization for PPG and ECG → PPG derivative)
        │
        ▼
  Model Input: X = [PPG, ΔPPGₜ, ECG_past]  shape: (B, 3, 875)
  Target:      Y = ECG_target                shape: (B, 1, 125)
        │
        ▼
      PINN Model
        │
        ▼
  Hybrid PINN Loss
  ├── Amplitude   (MAE / HUBER / RMSE weighted on peaks)   weight 1.0
  ├── Morphology  (1 − predicted vs target Pearson)        weight 0.4
  └── McSharry ODE (dz/dt residual − f(z,θ,ω))            weight 0.1
        │
        ▼
  K-Fold CV (k=1 or k=5) per patient
  Early Stopping (patience=20) + LR Scheduler (÷2 every 40 epochs)
```

### Model Inputs

| Channel | Signal | Size |
|---|---|---|
| 0 | Normalized PPG [0,1] | 875 samples (7 s × 125 Hz) |
| 1 | First derivative of PPG (ΔPPGₜ) | 875 samples |
| 2 | Past ECG with last-value padding | 875 samples (6 s actual + 1 s padding) |

### Target

| Signal | Size |
|---|---|
| Normalized future ECG [0,1] | 125 samples (1 s × 125 Hz) |

### PINN Loss — McSharry ODE

The physical component of the loss enforces that the empirical derivative of the generated ECG respects the McSharry differential equation (2003):

```
dz/dt = −Σᵢ aᵢ · Δθᵢ · exp(−Δθᵢ²/2bᵢ²) − (z − z₀)

where:  Δθᵢ = (θ − θᵢ) mod 2π
        θ = instantaneous cardiac phase (from find_peaks on the ECG signal)
        ω = instantaneous angular frequency (2π / RR period)
```

The ODE loss penalizes the discrepancy between the empirical derivative of the generated signal and the theoretical one:

```
L_ODE = mean((dz_emp − dz_phys)²)
```

---

## 5. Models

All models are located in `src/mimic_generation_PINN/models/` and can be selected via `--model`.
The models deliberately cover **five distinct architectural paradigms** to ensure an exhaustive comparison for publication purposes:

| Paradigm | Model(s) |
|---|---|
| CNN + recurrent (LSTM) | `lightweight_hybrid`, `ha_cnn_bilstm_ar` |
| Multi-scale Transformer | `bio_transformer` (PhysioFormer) |
| Pure TCN with gating (WaveNet-style) | `ppg_wavenet` |
| Encoder-Decoder with skip connections | `ecg_unet1d` |
| Dual-branch CNN 2D + LSTM | `dual_branch_hybrid` |

---

### 5.1 `lightweight_hybrid` *(default)*

Compact architecture designed for fast training. Best-performing model among the CNN+LSTM baselines.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform (J=2, Q=8)
    │
    ▼
1D CNN Encoder
    ├── Conv1d(3→32, k=7) + BN + LeakyReLU + MaxPool(2)
    └── Conv1d(32→64, k=5) + BN + LeakyReLU + MaxPool(2)
    │
    ▼
BiLSTM (hidden=64, bidirectional → 128 features)
    │
    ▼
Linear Upsample → 125 samples
    │
    ▼
Decoder: Dropout(0.2) + Conv1d(128→64, k=3) + Conv1d(64→1, k=1)
    │
    ▼
Output (B, 1, 125)
```

### 5.2 `ha_cnn_bilstm_ar`

HA-CNN-BiLSTM autoregressive model with Attention Gate, taken from the literature.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ▼
CNN Stage
    ├── Conv1d(3→64, k=7) + BN + LeakyReLU + MaxPool(2)
    └── Conv1d(64→128, k=5) + BN + LeakyReLU + MaxPool(2)
    │
    ▼
BiLSTM (hidden=128, bidirectional → 256 features)
    │
    ▼
Linear Upsample → 125 samples
    │
    ▼
Attention Gate: Conv1d(256→256, k=1) + Sigmoid  (element-wise gating)
    │
    ▼
Regression Head: Conv1d(256→128) + Conv1d(128→64) + Conv1d(64→1)
    │
    ▼
Output (B, 1, 125)
```

### 5.3 `bio_transformer` — *PhysioFormer*

Original Transformer architecture with multi-scale tokenization and residual PPG bypass.
Compared to the original version: Pre-LN, 4 layers, learnable positional embedding, progressive decoder, residual bypass from the PPG channel directly to the output.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ▼
Multi-Scale Tokenizer
    ├── Conv1d(3 → d/3,  k=1,  padding=0)  ← fine scale
    ├── Conv1d(3 → d/3,  k=5,  padding=2)  ← medium scale
    └── Conv1d(3 → d/3+, k=15, padding=7)  ← coarse scale
    Concat → BN → GELU  →  (B, d_model=128, 875)
    │
    ▼
Learnable Positional Embedding
    │
    ▼
Transformer Encoder Pre-LN (4 layers, 8 heads, FF=256, dropout=0.1)
    │
    ▼
Progressive Decoder
    Interpolate → Conv1d(128→64, k=3) → Conv1d(64→32, k=3) → Conv1d(32→1, k=1)
    │
    ├── + Residual PPG Bypass: Conv1d(1→8→1) on the original PPG channel
    ▼
Output (B, 1, 125)
```

### 5.4 `ppg_wavenet` — *PPGWaveNet*

Purely convolutional architecture inspired by WaveNet. No LSTM or Transformer.
Highly parallelizable on GPU; very large receptive field through cyclic exponential dilations.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ▼
Input Embedding: Conv1d(3→64, k=1) + BN + GELU
    │
    ▼
Stack of 12 WaveNetResidualBlocks (2 cycles × [d=1,2,4,8,16,32])
    │
    Each block:
    ├── filter_conv: Conv1d(64→64, k=3, dilation=d)
    ├── gate_conv:   Conv1d(64→64, k=3, dilation=d)
    ├── h = tanh(filter) ⊙ σ(gate)        ← gated activation
    ├── residual = Conv1d(h) + x           → passed to next block
    └── skip     = Conv1d(h)               → accumulated in skip_sum
    │
    ▼
skip_sum + final_x → Output Head
    GELU → Conv1d(64→64, k=1) → GELU → Conv1d(64→32, k=3) → Conv1d(32→1, k=1)
    │
    ▼
Interpolate → 125 samples
    │
    ▼
Output (B, 1, 125)
```

### 5.5 `ecg_unet1d` — *ECGUNet1D*

Adaptation of U-Net (originally designed for biomedical image segmentation) to 1D signal regression.
Skip connections preserve precise temporal morphology (R-peak position, QRS duration) during reconstruction.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ▼  Encoder
    E1: ConvBlock(3→32)    → MaxPool(2)   →  (B, 32,  437)
    E2: ConvBlock(32→64)   → MaxPool(2)   →  (B, 64,  218)
    E3: ConvBlock(64→128)  → MaxPool(2)   →  (B, 128, 109)
                              MaxPool(2)  →  (B, 128,  54)   Bottleneck input
    │
    ▼  Bottleneck
    BiLSTM(128, hidden=128, 2 layers, bidirectional) + Conv1d(256→128)
    │
    ▼  Decoder  (Upsample ×2 + skip concatenation)
    D3: Upsample → concat(E3) → ConvBlock(256→128)
    D2: Upsample → concat(E2) → ConvBlock(192→64)
    D1: Upsample → concat(E1) → ConvBlock(96→32)
    │
    ▼
Conv1d(32→1, k=1) → Interpolate → 125 samples
    │
    ▼
Output (B, 1, 125)
```

### 5.6 `dual_branch_hybrid`

Dual-branch architecture that separates PPG processing from the contextual ECG.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ├─────────────────┬──────────────────────────┐
    │                 │                          │
    ▼                 ▼                          ▼
2D CNN BRANCH    PPG LSTM BRANCH         ECG LSTM BRANCH
(4 Conv2d blocks  2-layer BiLSTM          2-layer BiLSTM
on spectrogram)   hidden=128 bidirect.    hidden=128 bidirect.
                  on PPG + ΔPPG           on ECG_past
    │                 └──────────┬───────────────┘
    │                            │
    │                      Fusion Linear(512→256)
    │                            │
    └────────────── Attention Gate (Softmax) ────┘
                                 │
                          Final Regression
                        Conv1d(512→128→64→1)
                                 │
                          Output (B, 1, 125)
```

### Wavelet Scattering Transform (WST)

When `apply_wst=True` (default configuration), all models apply a **Wavelet Scattering Transform** ([Kymatio](https://www.kymat.io/) library) with parameters `J=2, Q=8` before their respective encoders. The WST produces time-frequency representations that are stable to deformations, useful for capturing the morphological characteristics of physiological signals independently of small phase and amplitude variations.

---

## 6. Execution

### Prerequisites

```bash
pip install -r requirements.txt
# Requires PhysioNet access (registration and DUA for MIMIC-III)
```

### Execution Order

```bash
# 1. Clinical filtering (one-time, requires access to the MIMIC-III clinical DB)
python scripts/filter_no_cardiac_patologies_patients.py
python scripts/filter_ecg_patologies.py

# 2. Waveform download (SLURM cluster, can be parallelized)
sbatch job_ecg_gen_download_dataset.sh

# 3. Manifest creation (CPU-only, ~1-4 hours on 497 records)
sbatch run_pinn_manifest.sh

# 4. PINN Training
sbatch run_pinn_ecg.sh
# or locally:
python src/main_onlyPPG_PINN_mimic3wdb.py --model lightweight_hybrid --epochs 100
```

### Main Arguments for `main_onlyPPG_PINN_mimic3wdb.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | `lightweight_hybrid` | Architecture: `lightweight_hybrid`, `ha_cnn_bilstm_ar`, `dual_branch_hybrid`, `bio_transformer`, `ppg_wavenet`, `ecg_unet1d` |
| `--epochs` | `100` | Epochs per fold |
| `--batch_size` | `256` | Batch size |
| `--val_step` | `500` | Intermediate validation step |
| `--use_raw` | `False` | Use raw WFDB data instead of the preprocessed manifest |
| `--start_fold` | `1` | Fold to resume from (SLURM resume) |
| `--base_loss` | `MAE` | Base loss function: `MAE`, `RMSE`, `HUBER` |

---

## 7. Experiment Outputs

Results are saved in `src/experiments/mimic_pinn_results/healthy_patients/<LOSS>_loss/<model>_<timestamp>/`:

```text
<run>/
├── fold_1/
│   ├── best_lightweight_hybrid.pth      # Weights of the best model
│   ├── performances.json                # RMSE, MAE, Pearson, SNR, BPM error
│   ├── training_history.csv
│   ├── train_loss.png / val_loss.png
│   ├── val_epoch_*.png                  # Snapshot predicted vs true ECG
│   └── epoch_*_autoregressive.png       # Autoregressive generation comparison
└── k_fold_final_report.json             # Aggregated metrics across all folds
```

---

## 8. Code Structure

```text
PPG2ECG_Workstation/
├── src/
│   ├── main_onlyPPG_PINN_mimic3wdb.py       # Main entry point
│   ├── data_loader/
│   │   └── mimic3wdb_data_loader.py          # Lazy/full WFDB loading
│   ├── preprocessing/
│   │   └── mimic_autoregressive_preprocessor.py
│   ├── mimic_generation_PINN/
│   │   ├── pipeline.py                       # MimicSmartDataset + K-Fold loop
│   │   ├── trainer.py                        # Training loop + PINN loss
│   │   ├── model_factory.py
│   │   └── models/
│   │       ├── lightweight_hybrid.py          # 1D CNN + BiLSTM (best baseline)
│   │       ├── ha_cnn_bilstm_autoregressive.py# 1D CNN + BiLSTM + Attention Gate
│   │       ├── dual_branch_hybrid.py          # 2D CNN + Dual BiLSTM
│   │       ├── bio_transformer.py             # PhysioFormer: multi-scale Transformer
│   │       ├── ppg_wavenet.py                 # WaveNet-style dilated TCN with gating
│   │       └── ecg_unet1d.py                  # 1D U-Net encoder-decoder with skip connections
│   └── evaluation/
│       ├── evaluation.py                     # RMSE, MAE, Pearson, SNR, BPM
│       └── visualization.py                  # Plotting + autoregressive generation
├── scripts/
│   ├── filter_no_cardiac_patologies_patients.py
│   ├── filter_ecg_patologies.py
│   ├── download_healthy_waveforms.py
│   ├── preprocess_mimic_pinn_manifest.py     # Zero-copy manifest builder
│   └── count_model_complexity.py             # Parameter count + FLOPs profiler
├── mimic3wdb-matched_healthy_data/           # WFDB data (not tracked by git)
│   └── dataset_manifest.json                 # Generated by preprocess_mimic_pinn_manifest.py
├── run_pinn_manifest.sh                      # SLURM: preprocessing (CPU)
├── run_pinn_ecg.sh                           # SLURM: training (GPU A40)
├── Q1_REVIEW_CHECKLIST.md                    # Checklist for Q1 journal submission
├── requirements.txt                          # GPU dependencies (workstation)
├── requirements-local.txt                    # CPU-only dependencies (local dev)
└── tests/
    └── test_new_models_pipeline.py           # 49 end-to-end tests (pytest)
```

---

## 9. Model Computational Complexity

Measured with `scripts/count_model_complexity.py` on input `(3, 875) → (125,)`, without WST.
FLOPs are estimated via `torchinfo` (MACs × 2); the PhysioFormer column is underestimated because `torchinfo` does not fully count multi-head attention operations — on the A40 GPU the Transformer is much faster than on CPU thanks to native parallelization.

| Model | Parameters | FLOPs (M) | Weight VRAM¹ | CPU fwd (batch=256) |
|---|---:|---:|---:|---:|
| LightweightHybrid | 102,465 | 45 | ~2 MB | 133 ms |
| PPGWaveNet | 408,577 | 712 | ~6 MB | 3,153 ms² |
| HA-CNN-BiLSTM-AR | 496,001 | 201 | ~8 MB | 366 ms |
| ECGUNet1D | 999,745 | 335 | ~15 MB | 702 ms |
| PhysioFormer (BioTransformer) | 1,204,266 | ~15³ | ~18 MB | 16,784 ms² |
| DualBranchHybrid | 1,753,795 | 2,232 | ~27 MB | 4,103 ms |

> ¹ VRAM = fp32 weights + gradients + Adam states. Excludes activations and data batch.
> ² CPU values are not representative: PPGWaveNet and PhysioFormer benefit greatly from GPU parallelization.
> ³ `torchinfo` underestimates Transformer FLOPs (does not count multi-head attention).

**Estimated total VRAM on A40 (batch=256):** all models require < 4 GB including activations. The A40 GPU (48 GB) can train any model with batch 256 without issues.

### Recommended Training Order on GPU A40

Ordered by increasing computational risk (fastest models first to validate the pipeline):

| Priority | Model | Rationale |
|---|---|---|
| 1 | `lightweight_hybrid` | Lightest (102K params, 45M FLOPs) — fast pipeline validation |
| 2 | `ha_cnn_bilstm_ar` | Similar structure, useful as a direct baseline |
| 3 | `ecg_unet1d` | Stable architecture, no problematic components |
| 4 | `bio_transformer` | Transformer: slow on CPU but fast on A40, monitor early stopping |
| 5 | `ppg_wavenet` | 712M FLOPs per step — verify throughput before 5 full folds |
| 6 | `dual_branch_hybrid` | Heaviest (2.2B FLOPs) — run last |

```bash
# Recommended sequence on SLURM cluster (A40)
python src/main_onlyPPG_PINN_mimic3wdb.py --model lightweight_hybrid  --epochs 100
python src/main_onlyPPG_PINN_mimic3wdb.py --model ha_cnn_bilstm_ar    --epochs 100
python src/main_onlyPPG_PINN_mimic3wdb.py --model ecg_unet1d          --epochs 100
python src/main_onlyPPG_PINN_mimic3wdb.py --model bio_transformer     --epochs 100
python src/main_onlyPPG_PINN_mimic3wdb.py --model ppg_wavenet         --epochs 100
python src/main_onlyPPG_PINN_mimic3wdb.py --model dual_branch_hybrid  --epochs 100
```

---

## 10. Experimental Results

> **This section is updated progressively as trainings complete.**
> Metrics computed on the validation set of the 5-fold CV on MIMIC-III (healthy patients, Lead II).
> Format: `mean ± std` over 5 folds. Base loss: MAE. WST: enabled (`apply_wst=True`).

### 10.1 Signal Reconstruction Metrics

| Model | RMSE ↓ | MAE ↓ | Pearson ↑ | SNR (dB) ↑ | BPM Error ↓ |
|---|---|---|---|---|---|
| LightweightHybrid | — | — | — | — | — |
| HA-CNN-BiLSTM-AR | — | — | — | — | — |
| ECGUNet1D | — | — | — | — | — |
| PhysioFormer | — | — | — | — | — |
| PPGWaveNet | — | — | — | — | — |
| DualBranchHybrid | — | — | — | — | — |

### 10.2 Training Loss by Component (best fold)

| Model | Total loss | MAE (weighted) | Pearson loss | ODE loss |
|---|---|---|---|---|
| LightweightHybrid | — | — | — | — |
| HA-CNN-BiLSTM-AR | — | — | — | — |
| ECGUNet1D | — | — | — | — |
| PhysioFormer | — | — | — | — |
| PPGWaveNet | — | — | — | — |
| DualBranchHybrid | — | — | — | — |

### 10.3 Methodological Notes

- **Dataset:** MIMIC-III Waveform Database Matched — healthy patients (no structural cardiac pathologies)
- **Split:** 85% CV (5-fold per patient) + 15% invisible hold-out test set
- **Normalization:** per-window min-max [0,1], computed separately on PPG and ECG (6s past + 1s target together)
- **Hardware:** NVIDIA A40 (48 GB VRAM), CUDA 12.4
- **Seed:** 45 (reproducible via `set_reproducibility`)
- **Fixed hyperparameters for all models:** lr=0.001, Adam, batch=256, patience=20, LR decay ÷2 every 40 epochs
