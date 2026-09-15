#!/usr/bin/env python3
"""
scripts/clinical_fidelty_evaluation/debug/regenerate_worst_hr_cases.py

Debug: isola le finestre con il maggiore errore di frequenza cardiaca
(hr_abs_error > HR_ERROR_THRESHOLD) da clinical_fidelity_results.json e
produce un plot per ciascuna.

A differenza delle versioni precedenti di questo script (quando
evaluate_clinical_fidelity.py rigenerava il segnale al volo con
modello+MimicSmartDataset), qui non serve più ricostruire nulla: vero e
generato sono già entrambi salvati per intero nei file .npz di
scripts/autoregressive_drift_evaluation/experiments/drift_generation_clean/
(uno per record WFDB, con il segnale ripulito + generato in autoregressione
VERA su tutta la sua durata). Basta quindi risalire dal file_info
("<record_path>_s<start>_e<end>") al file .npz giusto e ritagliare la
finestra corrispondente, più un po' di contesto visivo precedente.

NOTA SCALA: ecg_generated resta nello spazio di generazione del modello
(circa [0,1]) e NON viene mai riportato alla scala fisica di ecg_target — a
differenza delle vecchie versioni teacher-forced, dove le due scale
combaciavano per costruzione. È quindi normale/atteso che il pannello ECG
reale e quello ECG generato abbiano range dell'asse Y molto diversi: si
guarda forma/tempistica dei picchi, non l'ampiezza assoluta.

Output (in questa cartella)
----------------------------
  worst_hr_cases_signals.npz   → per ogni finestra: <key>__true, <key>__pred,
                                  <key>__ppg_input, <key>__ppg_diff,
                                  <key>__window_start_sec, <key>__fs
  worst_hr_cases_metadata.json → file_info, hr_true/pred, hr_abs_error,
                                  n_true_peaks/n_pred_peaks, window_start_sec
  worst/<sample>.png           → 4 subplot (PPG, derivata PPG, ECG vero,
                                  ECG generato) con contesto precedente,
                                  titolo con hr_abs_error
"""

import os
import re
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEBUG_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.clinical_fidelty_evaluation.evaluate_clinical_fidelity import (
    GENERATION_DIR, _extract_r_peaks,
)

RESULTS_PATH = os.path.join(os.path.dirname(DEBUG_DIR), "clinical_fidelity_results.json")
WORST_PLOTS_DIR = os.path.join(DEBUG_DIR, "worst")
SIGNALS_PATH = os.path.join(DEBUG_DIR, "worst_hr_cases_signals.npz")
METADATA_PATH = os.path.join(DEBUG_DIR, "worst_hr_cases_metadata.json")

HR_ERROR_THRESHOLD = 60.0
CONTEXT_BUFFER_SEC = 5  # secondi di segnale precedente mostrati per orientamento visivo

FILE_INFO_RE = re.compile(r'^(?P<record_path>.+)_s(?P<start>\d+)_e(?P<end>\d+)$')


def _safe_key(file_info: str) -> str:
    return file_info.replace('/', '__')


def _find_worst_records(results_path: str, threshold: float):
    with open(results_path) as f:
        data = json.load(f)
    per_segment = data['per_segment']
    worst = [r for r in per_segment if r.get('hr_abs_error') is not None and r['hr_abs_error'] > threshold]
    worst.sort(key=lambda r: r['hr_abs_error'], reverse=True)
    return worst


def _load_window_with_context(file_info: str, buffer_sec: float):
    m = FILE_INFO_RE.match(file_info)
    if not m:
        return None
    record_path = m.group('record_path')
    start, end = int(m.group('start')), int(m.group('end'))
    npz_path = os.path.join(GENERATION_DIR, record_path + ".npz")
    if not os.path.exists(npz_path):
        return None

    with np.load(npz_path) as cached:
        ecg_target = cached['ecg_target'].astype(np.float32)
        ecg_generated = cached['ecg_generated'].astype(np.float32)
        ppg_input = cached['ppg_input'].astype(np.float32)
        fs = int(cached['fs']) if 'fs' in cached else 125

    buffer_samples = int(buffer_sec * fs)
    ctx_start = max(0, start - buffer_samples)

    true_sig = ecg_target[ctx_start:end]
    pred_sig = ecg_generated[ctx_start:end]
    ppg_sig = ppg_input[ctx_start:end]
    ppg_diff_sig = np.zeros_like(ppg_sig)
    ppg_diff_sig[1:] = ppg_sig[1:] - ppg_sig[:-1]

    window_start_sec = (start - ctx_start) / fs  # dove, nel plot, inizia la finestra analizzata
    return ppg_sig, ppg_diff_sig, true_sig, pred_sig, fs, window_start_sec


def _plot_case(ppg_sig, ppg_diff_sig, true_sig, pred_sig, fs, window_start_sec,
               hr_true, hr_pred, hr_abs_error, file_info, save_path):
    n = len(true_sig)
    t = np.arange(n) / fs

    true_peaks = _extract_r_peaks(true_sig, fs)
    pred_peaks = _extract_r_peaks(pred_sig, fs)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t, ppg_sig, color='#2b6cb0', linewidth=1.2)
    axes[0].set_title("PPG input (clean, raw)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, ppg_diff_sig, color='#805ad5', linewidth=1.2)
    axes[1].set_title("PPG derivative (raw)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, true_sig, color='#2f855a', linewidth=1.2)
    axes[2].scatter(true_peaks / fs, true_sig[true_peaks], color='black', marker='x', s=50, label='R-peak')
    axes[2].set_title(f"Real ECG  (HR = {hr_true:.1f} bpm, {len(true_peaks)} peaks)")
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, pred_sig, color='#c53030', linewidth=1.2)
    axes[3].scatter(pred_peaks / fs, pred_sig[pred_peaks], color='black', marker='x', s=50, label='R-peak')
    axes[3].set_title(f"Generated ECG — autoregressive, model scale  (HR = {hr_pred:.1f} bpm, {len(pred_peaks)} peaks)")
    axes[3].set_xlabel('Time [s]')
    axes[3].legend(loc='upper right')
    axes[3].grid(alpha=0.3)

    for ax in axes:
        ax.axvspan(0, window_start_sec, color='gray', alpha=0.12)
        ax.axvline(window_start_sec, color='black', linestyle='--', linewidth=1.0)
    axes[0].text(window_start_sec / 2, axes[0].get_ylim()[1], 'preceding context',
                 ha='center', va='top', fontsize=8, color='dimgray')
    axes[0].text(window_start_sec + (t[-1] - window_start_sec) / 2, axes[0].get_ylim()[1], 'analyzed window',
                 ha='center', va='top', fontsize=8, color='dimgray')

    fig.suptitle(f"{file_info}  —  hr_abs_error = {hr_abs_error:.2f} bpm", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(WORST_PLOTS_DIR, exist_ok=True)

    print(f"Carico risultati da: {RESULTS_PATH}")
    worst_records = _find_worst_records(RESULTS_PATH, HR_ERROR_THRESHOLD)
    print(f"Finestre con hr_abs_error > {HR_ERROR_THRESHOLD}: {len(worst_records)}")
    if not worst_records:
        print("Nessuna finestra da analizzare, esco.")
        return

    signals_to_save = {}
    metadata = []

    for rec_metrics in worst_records:
        file_info = rec_metrics['file_info']
        loaded = _load_window_with_context(file_info, CONTEXT_BUFFER_SEC)
        if loaded is None:
            print(f"[WARN] Impossibile caricare: {file_info}")
            continue
        ppg_sig, ppg_diff_sig, true_sig, pred_sig, fs, window_start_sec = loaded

        key = _safe_key(file_info)
        signals_to_save[f'{key}__true'] = true_sig
        signals_to_save[f'{key}__pred'] = pred_sig
        signals_to_save[f'{key}__ppg_input'] = ppg_sig
        signals_to_save[f'{key}__ppg_diff'] = ppg_diff_sig
        signals_to_save[f'{key}__window_start_sec'] = np.array([window_start_sec])
        signals_to_save[f'{key}__fs'] = np.array([fs])

        metadata.append({
            'file_info': file_info,
            'key': key,
            'hr_true': rec_metrics.get('hr_true'),
            'hr_pred': rec_metrics.get('hr_pred'),
            'hr_abs_error': rec_metrics.get('hr_abs_error'),
            'n_true_peaks': rec_metrics.get('n_true_peaks'),
            'n_pred_peaks': rec_metrics.get('n_pred_peaks'),
            'window_start_sec': window_start_sec,
        })

        plot_path = os.path.join(WORST_PLOTS_DIR, f"{key}.png")
        _plot_case(ppg_sig, ppg_diff_sig, true_sig, pred_sig, fs, window_start_sec,
                   rec_metrics['hr_true'], rec_metrics['hr_pred'],
                   rec_metrics['hr_abs_error'], file_info, plot_path)
        print(f"  ✓ {file_info}  (hr_abs_error={rec_metrics['hr_abs_error']:.2f})  →  {plot_path}")

    np.savez_compressed(SIGNALS_PATH, **signals_to_save)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Segnali salvati in: {SIGNALS_PATH}")
    print(f"✅ Metadata salvati in: {METADATA_PATH}")
    print(f"✅ Plot salvati in: {WORST_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
