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
  └── McSharry ODE (dz/dt residual − f(z,θ,ω))             weight 0.1
        │
        ▼
  K-Fold CV (k=1 or k=4) per patient
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

### 5.1 `lightweight_hybrid` *(default)*

Compact architecture designed for fast training on mid-range GPUs.

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

HA-CNN-BiLSTM autoregressive model with Attention Gate.

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

### 5.3 `dual_branch_hybrid`

Dual-branch architecture that separates PPG processing from the contextual ECG.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ├─────────────────┬──────────────────────────┐
    │                 │                          │
    ▼                 ▼                          ▼
2D CNN BRANCH    PPG LSTM BRANCH            ECG LSTM BRANCH
(4 Conv2d blocks  2-layer BiLSTM             2-layer BiLSTM
on spectrogram)   hidden=128 bidirect.       hidden=128 bidirect.
                  on PPG + ΔPPG              on ECG_past
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

### 5.4 `bio_transformer`

Transformer-based model with Sinusoidal Positional Encoding.

```
Input (B, 3, 875)
    │
    ├── [optional] Wavelet Scattering Transform
    │
    ▼
CNN Projector: Conv1d → d_model (projection into token space)
    │
    ▼
Positional Encoding (sinusoidal)
    │
    ▼
Transformer Encoder (2 layers, 4 heads, FF=128, dropout=0.2)
    │
    ▼
Upsample + Conv1d(d_model→1)
    │
    ▼
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
| `--model` | `lightweight_hybrid` | Architecture: `lightweight_hybrid`, `ha_cnn_bilstm_ar`, `dual_branch_hybrid`, `bio_transformer` |
| `--epochs` | `100` | Epochs per fold |
| `--batch_size` | `256` | Batch size |
| `--val_step` | `500` | Intermediate validation step |
| `--use_raw` | `False` | Use raw WFDB data instead of the preprocessed manifest |

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
│   │       ├── lightweight_hybrid.py
│   │       ├── ha_cnn_bilstm_autoregressive.py
│   │       ├── dual_branch_hybrid.py
│   │       └── bio_transformer.py
│   └── evaluation/
│       ├── evaluation.py                     # RMSE, MAE, Pearson, SNR, BPM
│       └── visualization.py                  # Plotting + autoregressive generation
├── scripts/
│   ├── filter_no_cardiac_patologies_patients.py
│   ├── filter_ecg_patologies.py
│   ├── download_healthy_waveforms.py
│   └── preprocess_mimic_pinn_manifest.py     # Zero-copy manifest builder
├── mimic3wdb-matched_healthy_data/           # WFDB data (not tracked by git)
│   └── dataset_manifest.json                 # Generated by preprocess_mimic_pinn_manifest.py
├── run_pinn_manifest.sh                      # SLURM: preprocessing (CPU)
└── run_pinn_ecg.sh                           # SLURM: training (GPU A100)
```
