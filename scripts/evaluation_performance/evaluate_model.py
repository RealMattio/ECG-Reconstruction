#!/usr/bin/env python3
"""
scripts/evaluation_performance/evaluate_model.py

Valutazione completa delle performance del modello lightweight_hybrid su MIMIC-III.

Riapre il test set ESATTAMENTE come lo ha visto il training:
  - stesso dataset_split.json (hold-out 15%)
  - stessa MimicSmartDataset con identico preprocessing
  - stesso seed, shuffle=False

Metriche calcolate per ogni finestra temporale:
  Pearson r, RMSE, rRMSE, MSE, DTW, MAE  (+ deviazioni standard aggregate)

Output
------
  performance_complete.json  →  FINAL_MODEL_DIR/  (tutte le finestre + aggregati + top-5)
  evaluation_summary.json    →  scripts/evaluation_performance/  (aggregati + top-5)
"""

import os
import sys
import json
import time

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader

# ── Radice del progetto ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mimic_generation_PINN.pipeline      import MimicSmartDataset, set_reproducibility
from src.mimic_generation_PINN.model_factory import ModelFactory


# =============================================================================
# CONFIGURAZIONE — identica a main_onlyPPG_PINN_mimic3wdb.py
# =============================================================================
MODEL_NAME = "lightweight_hybrid"  # nome del modello (per i pesi salvati)
PREPROCESSED_DATA = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data")

FINAL_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "src/experiments/final_mimic_pinn_results/MAE_loss/lightweight_hybrid_20260717_125410/final_full_model"
)
if not os.path.exists(FINAL_MODEL_DIR):
    raise FileNotFoundError(
)
WEIGHTS_PATH = os.path.join(FINAL_MODEL_DIR, f"best_{MODEL_NAME}.pth")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/evaluation_performance/

CONFIGS = {
    'model_type':     'lightweight_hybrid',
    'apply_wst':      True,
    'target_fs':      125,
    'x_sec':          7,
    'gen_sec':        1,
    'input_channels': 3,
    'actual_seq_len': 875,   # 7 s × 125 Hz
    'target_len':     125,   # 1 s × 125 Hz
    'normalize_01':   False,
    'batch_size':     256,
    'seed':           45,
}
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# DTW — traversal a diagonali anti-parallele (numpy-vectorized)
#
# Sulla stessa anti-diagonale (i+j=k) non esistono dipendenze reciproche:
# D[i,j] dipende solo da D[i-1,j], D[i,j-1], D[i-1,j-1] (diagonali k-1 e k-2).
# Questo permette di aggiornare tutti i punti di una diagonale con una singola
# operazione numpy, riducendo i cicli Python da O(n*m) a O(n+m).
# ─────────────────────────────────────────────────────────────────────────────
def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for k in range(1, n + m + 1):
        i_arr = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j_arr = k - i_arr
        mask  = (j_arr >= 1) & (j_arr <= m)
        i_arr, j_arr = i_arr[mask], j_arr[mask]
        if len(i_arr) == 0:
            continue
        cost = np.abs(x[i_arr - 1] - y[j_arr - 1])
        prev = np.minimum(
            np.minimum(D[i_arr - 1, j_arr], D[i_arr, j_arr - 1]),
            D[i_arr - 1, j_arr - 1],
        )
        D[i_arr, j_arr] = cost + prev
    return float(D[n, m])


def _compute_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcola tutte le metriche richieste per una singola finestra (array 1-D).

    rRMSE = RMSE / (max(y_true) - min(y_true))
    """
    mse   = float(np.mean((y_true - y_pred) ** 2))
    rmse  = float(np.sqrt(mse))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    y_rng = float(y_true.max() - y_true.min())
    rrmse = float(rmse / (y_rng + 1e-8))

    # Pearson r: protezione da segnale piatto
    if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
        r = 0.0
    else:
        r = float(scipy.stats.pearsonr(y_true, y_pred)[0])

    dtw = _dtw_distance(y_true, y_pred)

    return {'r': r, 'rmse': rmse, 'rrmse': rrmse, 'mse': mse, 'dtw': dtw, 'mae': mae}


def _aggregate(records: list) -> dict:
    """Media e deviazione standard di ogni metrica su tutti i campioni."""
    keys   = ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']
    result = {}
    for k in keys:
        vals            = np.array([rec['metrics'][k] for rec in records])
        result[k]       = float(np.mean(vals))
        result[f'{k}_std'] = float(np.std(vals))
    return result


# =============================================================================
# MAIN
# =============================================================================
def run_evaluation():
    t0 = time.time()
    print("=" * 65)
    print(f"  VALUTAZIONE COMPLETA — {CONFIGS['model_type']}  MIMIC-III")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_reproducibility(CONFIGS['seed'])

    # ── 1. Carica lo split (hold-out test 15%) ────────────────────────────────
    split_path = os.path.join(PREPROCESSED_DATA, "dataset_split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file non trovato: {split_path}\n"
            "Assicurati di puntare alla stessa directory usata durante il training."
        )
    with open(split_path) as f:
        split = json.load(f)
    test_patients = set(split["test_patients"])
    print(f"Pazienti nel test set: {len(test_patients)}")

    # ── 2. Filtra il manifest per i soli pazienti di test ────────────────────
    manifest_path = os.path.join(PREPROCESSED_DATA, "dataset_manifest.json")
    with open(manifest_path) as f:
        full_manifest = json.load(f)
    test_subset = [m for m in full_manifest if m['subject_id'] in test_patients]
    print(f"Entry manifest nel test set: {len(test_subset)}")

    # ── 3. Dataset identico alla pipeline (stesso preprocessing + windowing) ──
    test_ds = MimicSmartDataset(test_subset, PREPROCESSED_DATA, CONFIGS)
    n_windows = len(test_ds)
    print(f"Finestre totali: {n_windows}")

    # Ricava le dimensioni reali dall'effettivo campione del dataset
    sample_x, sample_y, _, _, _ = test_ds[0]
    CONFIGS['input_channels'] = int(sample_x.shape[0])
    CONFIGS['actual_seq_len'] = int(sample_x.shape[-1])
    CONFIGS['target_len']     = int(sample_y.shape[-1])
    print(f"Input: {list(sample_x.shape)}   Output: {list(sample_y.shape)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIGS['batch_size'],
        shuffle=False,          # identico alla pipeline
        num_workers=0,          # anti-OOM leak, identico alla pipeline
        pin_memory=(device.type == 'cuda'),
    )

    # ── 4. Modello + pesi ─────────────────────────────────────────────────────
    model = ModelFactory.get_model(CONFIGS).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"Pesi caricati da: {WEIGHTS_PATH}\n")

    # ── 5. Inferenza con metriche per-campione ────────────────────────────────
    # Ogni record mantiene l'info_string del dataset (soggetto + posizione)
    # così possiamo risalire al paziente e all'intervallo esatto.
    print("Avvio inferenza + calcolo metriche per campione...")
    all_records = []   # lista di {info, patient_id, metrics}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            X, Y, _, _, info_strings = batch
            outputs = model(X.to(device))

            y_pred = outputs.detach().cpu().numpy().reshape(outputs.shape[0], -1)
            y_true = Y.detach().cpu().numpy().reshape(Y.shape[0], -1)

            for i in range(y_pred.shape[0]):
                info       = info_strings[i]
                # info_string: "<subject_id>/<record>_s<s>_e<e> | Pos: <idx>"
                patient_id = info.split("/")[0].strip() if "/" in info else "unknown"
                metrics    = _compute_sample_metrics(y_true[i], y_pred[i])
                all_records.append({
                    'info':       info,
                    'patient_id': patient_id,
                    'metrics':    metrics,
                })

            # Progresso ogni 20 batch (o all'ultimo)
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
                done    = len(all_records)
                elapsed = time.time() - t0
                eta     = (elapsed / done) * (n_windows - done) if done else 0.0
                print(f"  [{batch_idx + 1:3d}/{len(test_loader)}]  "
                      f"{done}/{n_windows} campioni  "
                      f"elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s")

    print(f"\nCampioni valutati: {len(all_records)}")

    # ── 6. Metriche aggregate (media ± std) ───────────────────────────────────
    agg = _aggregate(all_records)

    print(f"\n── Risultati aggregati — {CONFIGS['model_type']} ──────────────────────────────────────────")
    for k in ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']:
        print(f"  {k:6s}: {agg[k]:.6f}  ±  {agg[k + '_std']:.6f}")
    print("─" * 65)

    # ── 7a. Top-5 finestre per Pearson r ─────────────────────────────────────
    sorted_all = sorted(all_records, key=lambda rec: rec['metrics']['r'], reverse=True)
    top5_windows = [
        {
            'rank':       i + 1,
            'patient_id': rec['patient_id'],
            'info':       rec['info'],
            'metrics':    rec['metrics'],
        }
        for i, rec in enumerate(sorted_all[:5])
    ]

    print(f"\n── Top-5 finestre  (Pearson r più alto) — {CONFIGS['model_type']} ────────────────────────")
    for e in top5_windows:
        m = e['metrics']
        print(
            f"  #{e['rank']}  [{e['patient_id']:12s}]  "
            f"r={m['r']:.4f}  RMSE={m['rmse']:.4f}  "
            f"MAE={m['mae']:.4f}  DTW={m['dtw']:.2f}"
        )
        print(f"       {e['info']}")

    # ── 7b. Top-5 pazienti per media Pearson r ────────────────────────────────
    # Raggruppa le finestre per paziente e calcola media + std per ognuno
    patient_buckets: dict = {}
    for rec in all_records:
        pid = rec['patient_id']
        patient_buckets.setdefault(pid, []).append(rec['metrics'])

    patient_stats = []
    for pid, metrics_list in patient_buckets.items():
        keys = ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']
        avg  = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
        std  = {f'{k}_std': float(np.std([m[k] for m in metrics_list])) for k in keys}
        patient_stats.append({
            'patient_id':    pid,
            'num_windows':   len(metrics_list),
            'avg_metrics':   avg,
            'std_metrics':   std,
        })

    patient_stats.sort(key=lambda p: p['avg_metrics']['r'], reverse=True)
    top5_patients = [{'rank': i + 1, **p} for i, p in enumerate(patient_stats[:5])]

    print(f"\n── Top-5 pazienti  (media Pearson r più alta) — {CONFIGS['model_type']} ──────────────────")
    for e in top5_patients:
        m = e['avg_metrics']
        print(
            f"  #{e['rank']}  [{e['patient_id']:12s}]  "
            f"finestre={e['num_windows']:3d}  "
            f"r={m['r']:.4f}  RMSE={m['rmse']:.4f}  "
            f"MAE={m['mae']:.4f}  DTW={m['dtw']:.2f}"
        )

    elapsed_total = round(time.time() - t0, 1)

    # ── 8. performance_complete.json nella cartella del modello ───────────────
    performance_complete = {
        "model":                   CONFIGS['model_type'],
        "weights_path":            WEIGHTS_PATH,
        "test_patients_count":     len(test_patients),
        "total_windows_evaluated": len(all_records),
        "generation_params": {
            "fs":                   CONFIGS['target_fs'],
            "num_predicted_points": CONFIGS['target_len'],
            "input_window_sec":     CONFIGS['x_sec'],
            "generation_sec":       CONFIGS['gen_sec'],
        },
        "performance_metrics": {k: agg[k] for k in ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']},
        "performance_std":     {k: agg[k] for k in [f'{m}_std' for m in ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']]},
        "top5_best_windows":   top5_windows,
        "top5_best_patients":  top5_patients,
    }
    complete_path = os.path.join(FINAL_MODEL_DIR, "performance_complete.json")
    with open(complete_path, 'w') as f:
        json.dump(performance_complete, f, indent=2)
    print(f"\n✅ performance_complete.json  →  {complete_path}")

    # ── 9. evaluation_summary.json in scripts/evaluation_performance/ ─────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "model":                   CONFIGS['model_type'],
        "weights_path":            WEIGHTS_PATH,
        "test_patients_count":     len(test_patients),
        "total_windows_evaluated": len(all_records),
        "performance_metrics": {k: agg[k] for k in ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']},
        "performance_std":     {k: agg[k] for k in [f'{m}_std' for m in ['r', 'rmse', 'rrmse', 'mse', 'dtw', 'mae']]},
        "top5_best_windows":       top5_windows,
        "top5_best_patients":      top5_patients,
        "elapsed_seconds":         elapsed_total,
    }
    summary_path = os.path.join(OUTPUT_DIR, f"evaluation_summary_{CONFIGS['model_type']}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ evaluation_summary.json       →  {summary_path}")
    print(f"\nTempo totale: {elapsed_total}s")


if __name__ == "__main__":
    run_evaluation()
