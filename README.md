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

## Obiettivo

Ricostruzione del segnale **ECG (Lead II)** a partire dal solo segnale **PPG (fotopletismografia)** acquisito da sensore non invasivo, usando una rete neurale Physics-Informed (PINN) che incorpora nella funzione di loss i vincoli dell'equazione differenziale di McSharry che governa la dinamica cardiaca.

---

## 1. Dataset — MIMIC-III Waveform Database (Matched)

### Selezione clinica dei pazienti

Il dataset di partenza è il **MIMIC-III Waveform Database Matched** (`mimic3wdb-matched`), accessibile su [PhysioNet](https://physionet.org/content/mimic3wdb-matched/1.0/) previa registrazione e firma del Data Use Agreement.

La selezione dei pazienti avviene in due fasi:

1. **Filtraggio clinico** (`scripts/filter_no_cardiac_patologies_patients.py`, `scripts/filter_ecg_patologies.py`): i pazienti vengono estratti dal database clinico MIMIC-III e filtrati escludendo chi presenta diagnosi ICD-9 legate a patologie cardiache strutturali (insufficienza cardiaca, infarto, fibrillazione atriale, ecc.). Il risultato è un file `allowed_patients.csv`.

2. **Download selettivo** (`scripts/download_healthy_waveforms.py`): scarica da PhysioNet solo i waveform WFDB dei pazienti approvati, filtrando ulteriormente per durata minima (≥ 3 minuti per sessione) e bilanciando il contributo per paziente (≤ 36 ore totali). Il download si interrompe automaticamente se lo spazio libero su disco scende sotto i 4 GB. Supporta parallelizzazione via SLURM con `--worker_id` e `--total_workers`.

I file scaricati sono coppie `.hea` + `.dat` in formato WFDB standard, organizzati per paziente:

```text
mimic3wdb-matched_healthy_data/
├── p00/
│   └── p000052/
│       ├── 3533390_0011.hea
│       └── 3533390_0011.dat
├── p01/
│   └── ...
```

Ogni record contiene almeno i canali **`II`** (ECG Lead II) e **`PLETH`** (PPG).

---

## 2. Preprocessing e Creazione del Manifest

Il preprocessing è interamente **zero-copia**: nessun segnale viene duplicato su disco. L'unico output è un file `dataset_manifest.json` (pochi KB) che indicizza i segmenti validi con i path WFDB originali.

### Esecuzione

```bash
sbatch run_pinn_manifest.sh
# oppure in locale:
python scripts/preprocess_mimic_pinn_manifest.py \
    --input_dir  ./mimic3wdb-matched_healthy_data \
    --output_dir ./mimic3wdb-matched_healthy_data \
    --target_fs 125 --ppg_thr 0.9 --ecg_thr 0.9
```

### Passi interni

```
File WFDB (.hea + .dat)
      │
      ▼
  Ricampionamento a 125 Hz (scipy.signal.resample)
      │
      ▼
  Filtro bandpass Butterworth (ordine 4)
      ├── PPG:  0.5 – 5.0 Hz
      └── ECG:  0.5 – 40.0 Hz
      │
      ▼
  Sliding window di valutazione qualità (4 s, passo 1 s)
      ├── PPG: SQI spettrale + morfologia + polarità  →  score Pearson
      └── ECG: kurtosis + morfologia QRS             →  score Pearson
      │
      ▼
  Merging intervalli PPG validi ∩ ECG validi
      │
      ▼
  Estrazione segmenti contigui ≥ 7 s
      │
      ▼
  dataset_manifest.json  (path WFDB + indici segmento + flag inversione PPG)
```

---

## 3. Filtraggio delle Finestre — Criteri di Qualità del Segnale

Ogni finestra da 4 secondi viene accettata solo se supera simultaneamente i controlli su PPG **e** ECG.

### 3.1 PPG — Signal Quality Index (SQI)

| Criterio | Metodo | Soglia |
|---|---|---|
| **Segnale non piatto** | deviazione standard | `std > 1e-6` |
| **SQI spettrale** | Welch PSD — rapporto potenza nel picco HR (0.5–3 Hz) su totale (0.5–10 Hz) | `≥ 0.5` |
| **Morfologia battito** | Correlazione di Pearson tra i singoli battiti rilevati e il template mediano | `≥ 0.9` |
| **Polarità** | Asimmetria di ampiezza: `max(x – mean) > \|min(x – mean)\|` → se falso, il segnale viene invertito | — |

### 3.2 ECG — Signal Quality Index

| Criterio | Metodo | Soglia |
|---|---|---|
| **Segnale non piatto** | deviazione standard | `std > 1e-4` |
| **Kurtosis** | Kurtosis di Pearson (non-Fisher) — un ECG pulito ha picchi R impulsivi → alta kurtosis | `≥ 5.0` |
| **Morfologia QRS** | Correlazione di Pearson tra i complessi QRS estratti e il template mediano | `≥ 0.9` |

### 3.3 Intersezione e segmentazione

Gli intervalli che superano il filtraggio su PPG e quelli su ECG vengono uniti (merge di intervalli sovrapposti) e intersecati. Solo i segmenti contigui risultanti di lunghezza **≥ 7 secondi** vengono inseriti nel manifest.

### 3.4 Normalizzazione per finestra durante il training

Al momento dell'estrazione di ogni finestra di training (in `MimicSmartDataset.__getitem__`), sia il PPG che l'ECG vengono normalizzati **indipendentemente** nell'intervallo `[0, 1]` con min-max per finestra:

```
x_norm = (x − min(x)) / (max(x) − min(x) + ε)
```

- **PPG (7 s)**: normalizzato indipendentemente, dopo la correzione di polarità.
- **ECG (7 s = past + target)**: past e target vengono normalizzati **insieme** con lo stesso min/max, così il modello non vede discontinuità artificiali al confine tra contesto noto e target da predire.
- **theta, omega** (segnali fisici PINN): **non normalizzati** — codificano fase (rad) e frequenza angolare (rad/s).

---

## 4. Architettura della Pipeline di Training

```
dataset_manifest.json
        │
        ▼
  MimicSmartDataset.__init__
  (carica WFDB → ricampiona → filtra bandpass
   → calcola theta/omega → cache in RAM)
        │
        ▼
  MimicSmartDataset.__getitem__
  (per ogni finestra: correzione polarità PPG
   → normalizzazione [0,1] PPG e ECG → derivata PPG)
        │
        ▼
  Input al modello: X = [PPG, ΔPPGₜ, ECG_past]  shape: (B, 3, 875)
  Target:          Y = ECG_target                shape: (B, 1, 125)
        │
        ▼
      Modello PINN
        │
        ▼
  Loss ibrida PINN
  ├── Ampiezza  (MAE / HUBER / RMSE weighted sui picchi)   peso 1.0
  ├── Morfologia (1 − Pearson predetto vs target)           peso 0.4
  └── ODE McSharry (residuo dz/dt − f(z,θ,ω))             peso 0.1
        │
        ▼
  K-Fold CV (k=1 o k=4) per paziente
  Early Stopping (patience=20) + LR Scheduler (÷2 ogni 40 epoche)
```

### Input al modello

| Canale | Segnale | Dimensione |
|---|---|---|
| 0 | PPG normalizzato [0,1] | 875 campioni (7 s × 125 Hz) |
| 1 | Derivata prima del PPG (ΔPPGₜ) | 875 campioni |
| 2 | ECG passato con last-value padding | 875 campioni (6 s reali + 1 s padding) |

### Target

| Segnale | Dimensione |
|---|---|
| ECG futuro normalizzato [0,1] | 125 campioni (1 s × 125 Hz) |

### Loss PINN — McSharry ODE

La componente fisica della loss impone che la derivata empirica dell'ECG generato rispetti l'equazione differenziale di McSharry (2003):

```
dz/dt = −Σᵢ aᵢ · Δθᵢ · exp(−Δθᵢ²/2bᵢ²) − (z − z₀)

dove:  Δθᵢ = (θ − θᵢ) mod 2π
       θ = fase cardiaca istantanea (da find_peaks sul segnale ECG)
       ω = frequenza angolare istantanea (2π / periodo RR)
```

La loss ODE penalizza la discrepanza tra la derivata empirica del segnale generato e quella teorica:

```
L_ODE = mean((dz_emp − dz_phys)²)
```

---

## 5. Modelli

Tutti i modelli si trovano in `src/mimic_generation_PINN/models/` e sono selezionabili via `--model`.

### 5.1 `lightweight_hybrid` *(default)*

Architettura compatta pensata per training rapido su GPU di fascia media.

```
Input (B, 3, 875)
    │
    ├── [opzionale] Wavelet Scattering Transform (J=2, Q=8)
    │
    ▼
CNN 1D Encoder
    ├── Conv1d(3→32, k=7) + BN + LeakyReLU + MaxPool(2)
    └── Conv1d(32→64, k=5) + BN + LeakyReLU + MaxPool(2)
    │
    ▼
BiLSTM (hidden=64, bidirectional → 128 features)
    │
    ▼
Upsample lineare → 125 campioni
    │
    ▼
Decoder: Dropout(0.2) + Conv1d(128→64, k=3) + Conv1d(64→1, k=1)
    │
    ▼
Output (B, 1, 125)
```

### 5.2 `ha_cnn_bilstm_ar`

Modello autoregressivo HA-CNN-BiLSTM con Attention Gate.

```
Input (B, 3, 875)
    │
    ├── [opzionale] Wavelet Scattering Transform
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
Upsample lineare → 125 campioni
    │
    ▼
Attention Gate: Conv1d(256→256, k=1) + Sigmoid  (gating element-wise)
    │
    ▼
Regression Head: Conv1d(256→128) + Conv1d(128→64) + Conv1d(64→1)
    │
    ▼
Output (B, 1, 125)
```

### 5.3 `dual_branch_hybrid`

Architettura a doppio ramo che separa l'elaborazione PPG dall'ECG contestuale.

```
Input (B, 3, 875)
    │
    ├── [opzionale] Wavelet Scattering Transform
    │
    ├─────────────────┬──────────────────────────┐
    │                 │                          │
    ▼                 ▼                          ▼
RAMO CNN 2D    RAMO LSTM PPG            RAMO LSTM ECG
(4 blocchi     BiLSTM a 2 layer         BiLSTM a 2 layer
Conv2d su      hidden=128 bidirez.      hidden=128 bidirez.
spettrogramma) su PPG + ΔPPG           su ECG_past
    │                 └──────────┬───────────────┘
    │                            │
    │                       Fusion Linear(512→256)
    │                            │
    └────────────── Attention Gate (Softmax) ────┘
                                 │
                          Final Regression
                    Conv1d(512→128→64→1)
                                 │
                          Output (B, 1, 125)
```

### 5.4 `bio_transformer`

Modello basato su Transformer con Positional Encoding sinusoidale.

```
Input (B, 3, 875)
    │
    ├── [opzionale] Wavelet Scattering Transform
    │
    ▼
CNN Projector: Conv1d → d_model (proiezione nel token space)
    │
    ▼
Positional Encoding (sinusoidale)
    │
    ▼
Transformer Encoder (2 layer, 4 head, FF=128, dropout=0.2)
    │
    ▼
Upsample + Conv1d(d_model→1)
    │
    ▼
Output (B, 1, 125)
```

### Wavelet Scattering Transform (WST)

Quando `apply_wst=True` (configurazione di default), tutti i modelli applicano una **Wavelet Scattering Transform** (libreria [Kymatio](https://www.kymat.io/)) con parametri `J=2, Q=8` prima del proprio encoder. La WST produce rappresentazioni tempo-frequenza stabili alle deformazioni, utili per catturare le caratteristiche morfologiche dei segnali fisiologici indipendentemente da piccole variazioni di fase e ampiezza.

---

## 6. Esecuzione

### Prerequisiti

```bash
pip install -r requirements.txt
# Richiede accesso PhysioNet (registrazione e DUA per MIMIC-III)
```

### Ordine di esecuzione

```bash
# 1. Filtraggio clinico (una tantum, richiede accesso al DB clinico MIMIC-III)
python scripts/filter_no_cardiac_patologies_patients.py
python scripts/filter_ecg_patologies.py

# 2. Download waveform (cluster SLURM, può essere parallelizzato)
sbatch job_ecg_gen_download_dataset.sh

# 3. Creazione manifest (CPU-only, ~1-4 ore su 497 record)
sbatch run_pinn_manifest.sh

# 4. Addestramento PINN
sbatch run_pinn_ecg.sh
# oppure in locale:
python src/main_onlyPPG_PINN_mimic3wdb.py --model lightweight_hybrid --epochs 100
```

### Argomenti principali di `main_onlyPPG_PINN_mimic3wdb.py`

| Argomento | Default | Descrizione |
|---|---|---|
| `--model` | `lightweight_hybrid` | Architettura: `lightweight_hybrid`, `ha_cnn_bilstm_ar`, `dual_branch_hybrid`, `bio_transformer` |
| `--epochs` | `100` | Epoche per fold |
| `--batch_size` | `256` | Batch size |
| `--val_step` | `500` | Step di validazione intermedia |
| `--use_raw` | `False` | Usa dati WFDB raw invece del manifest preprocessato |

---

## 7. Output degli Esperimenti

I risultati vengono salvati in `src/experiments/mimic_pinn_results/healthy_patients/<LOSS>_loss/<modello>_<timestamp>/`:

```text
<run>/
├── fold_1/
│   ├── best_lightweight_hybrid.pth      # Pesi del modello migliore
│   ├── performances.json                # RMSE, MAE, Pearson, SNR, BPM error
│   ├── training_history.csv
│   ├── train_loss.png / val_loss.png
│   ├── val_epoch_*.png                  # Snapshot predicted vs true ECG
│   └── epoch_*_autoregressive.png       # Generazione autoregressiva a confronto
└── k_fold_final_report.json             # Metriche aggregate su tutti i fold
```

---

## 8. Struttura del Codice

```text
PPG2ECG_Workstation/
├── src/
│   ├── main_onlyPPG_PINN_mimic3wdb.py       # Entry point principale
│   ├── data_loader/
│   │   └── mimic3wdb_data_loader.py          # Caricamento WFDB lazy/full
│   ├── preprocessing/
│   │   └── mimic_autoregressive_preprocessor.py
│   ├── mimic_generation_PINN/
│   │   ├── pipeline.py                       # MimicSmartDataset + K-Fold loop
│   │   ├── trainer.py                        # Loop training + loss PINN
│   │   ├── model_factory.py
│   │   └── models/
│   │       ├── lightweight_hybrid.py
│   │       ├── ha_cnn_bilstm_autoregressive.py
│   │       ├── dual_branch_hybrid.py
│   │       └── bio_transformer.py
│   └── evaluation/
│       ├── evaluation.py                     # RMSE, MAE, Pearson, SNR, BPM
│       └── visualization.py                  # Plot + generazione autoregressiva
├── scripts/
│   ├── filter_no_cardiac_patologies_patients.py
│   ├── filter_ecg_patologies.py
│   ├── download_healthy_waveforms.py
│   └── preprocess_mimic_pinn_manifest.py     # Zero-copy manifest builder
├── mimic3wdb-matched_healthy_data/           # Dati WFDB (non tracciati da git)
│   └── dataset_manifest.json                 # Generato da preprocess_mimic_pinn_manifest.py
├── run_pinn_manifest.sh                      # SLURM: preprocessing (CPU)
└── run_pinn_ecg.sh                           # SLURM: training (GPU A100)
```
