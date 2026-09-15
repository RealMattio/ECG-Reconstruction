#!/usr/bin/env python3
"""
scripts/clinical_fidelty_evaluation/debug/plot_hr_error_length_histograms.py

AGGIORNATO per la sorgente dati "segnali puliti + autoregressione vera"
(drift_generation_clean/): ora ogni finestra di analisi ha per costruzione
la STESSA durata fissa (CHUNK_SEC secondi, vedi evaluate_clinical_fidelity.py)
— non c'è più variabilità nella lunghezza della finestra come quando le
finestre venivano ritagliate da segmenti WFDB grezzi di durata arbitraria.
L'analisi "errore vs lunghezza finestra" delle versioni precedenti di questo
script non avrebbe quindi più nulla da mostrare (tutte le finestre
cadrebbero in un unico bin).

La domanda interessante ora è un'altra, resa possibile proprio dal nuovo
paradigma: dato che la generazione è autoregressiva VERA (self-feeding), più
un chunk è lontano nel tempo dal seed iniziale, più a lungo il modello ha
generato usando le proprie uscite passate come contesto — è quindi la
variabile naturale su cui cercare drift temporale. Questo script produce
quindi, per ogni soglia di errore in HR_ERROR_THRESHOLDS, un istogramma:
  asse x = secondi di generazione autoregressiva trascorsi dal seed prima
           dell'inizio di questo chunk (posizione nel record, non lunghezza
           del chunk che è sempre CHUNK_SEC)
  asse y = numero di finestre a quella posizione il cui hr_abs_error
           è >= soglia

Non serve ricaricare dataset/modello: tutto ciò che serve (file_info,
hr_abs_error) è già in clinical_fidelity_results.json.

Per ogni istogramma viene salvato anche un .json con, per ciascun valore di
posizione (bin dell'asse x):
  position_sec    → secondi di generazione trascorsi dal seed
  n_abs           → numero assoluto di finestre in quell'intervallo con
                     hr_abs_error >= soglia (stessa quantità del grafico)
  n_rel           → numero relativo = n_abs / (numero totale di finestre a
                     quella posizione, indipendentemente dalla soglia)

Output (in questa cartella)
----------------------------
  hist_hr_error_ge_15.png   hist_hr_error_ge_15.json
  hist_hr_error_ge_20.png   hist_hr_error_ge_20.json
  hist_hr_error_ge_40.png   hist_hr_error_ge_40.json
  hist_hr_error_ge_60.png   hist_hr_error_ge_60.json
"""

import os
import re
import sys
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEBUG_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.clinical_fidelty_evaluation.evaluate_clinical_fidelity import FS, SEED_SEC

RESULTS_PATH = os.path.join(os.path.dirname(DEBUG_DIR), "clinical_fidelity_results.json")
OUTPUT_DIR = DEBUG_DIR

HR_ERROR_THRESHOLDS = [15, 20, 40, 60]

FILE_INFO_RE = re.compile(r'_s(?P<start>\d+)_e(?P<end>\d+)$')


def _elapsed_generation_sec(file_info: str, fs: int, seed_sec: float):
    """Secondi di generazione autoregressiva trascorsi dal seed prima
    dell'inizio di questo chunk (posizione nel record)."""
    m = FILE_INFO_RE.search(file_info)
    if not m:
        return None
    start = int(m.group('start'))
    return start / fs - seed_sec


def main():
    fs = FS

    print(f"Carico risultati da: {RESULTS_PATH}")
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    recs = data['per_segment']
    print(f"Segmenti totali: {len(recs)}")

    lengths, errors = [], []
    n_skipped = 0
    for r in recs:
        hr_err = r.get('hr_abs_error')
        if hr_err is None:
            continue
        position_sec = _elapsed_generation_sec(r['file_info'], fs, SEED_SEC)
        if position_sec is None:
            n_skipped += 1
            continue
        lengths.append(round(position_sec))
        errors.append(hr_err)

    lengths = np.array(lengths)
    errors = np.array(errors)
    print(f"Segmenti con hr_abs_error disponibile: {len(errors)}  (non parsabili: {n_skipped})")

    # Totale campioni per posizione, indipendente dalla soglia (denominatore
    # del numero relativo).
    total_counts_by_length = Counter(lengths.tolist())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for thr in HR_ERROR_THRESHOLDS:
        mask = errors >= thr
        sel_lengths = lengths[mask]
        n_sel = int(mask.sum())
        print(f"  hr_abs_error >= {thr:>3d} bpm: {n_sel} finestre")

        counts = Counter(sel_lengths.tolist())
        if counts:
            xs = list(range(min(counts), max(counts) + 1))
        else:
            xs = []
        ys = [counts.get(x, 0) for x in xs]

        fig, ax = plt.subplots(figsize=(12, 6))
        if xs:
            ax.bar(xs, ys, width=0.9, color='#c53030', edgecolor='black', linewidth=0.3)
        ax.set_xlabel('Elapsed autoregressive generation time since seed [s]')
        ax.set_ylabel(f'# windows with hr_abs_error ≥ {thr} bpm')
        ax.set_title(f'Error vs. position in record — hr_abs_error ≥ {thr} bpm  (N={n_sel})')
        ax.grid(alpha=0.3)
        fig.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"hist_hr_error_ge_{thr}.png")
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"    → {save_path}")

        bins_data = []
        for x in xs:
            n_abs = counts.get(x, 0)
            n_total = total_counts_by_length.get(x, 0)
            n_rel = (n_abs / n_total) if n_total > 0 else 0.0
            bins_data.append({
                'position_sec': x,
                'n_abs': n_abs,
                'n_rel': n_rel,
            })

        json_path = os.path.join(OUTPUT_DIR, f"hist_hr_error_ge_{thr}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'threshold_bpm': thr,
                'n_total_selected': n_sel,
                'bins': bins_data,
            }, f, indent=2)
        print(f"    → {json_path}")

    print("\n✅ Istogrammi e json salvati.")


if __name__ == "__main__":
    main()
