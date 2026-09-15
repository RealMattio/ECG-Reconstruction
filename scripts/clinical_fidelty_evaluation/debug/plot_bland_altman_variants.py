#!/usr/bin/env python3
"""
scripts/clinical_fidelty_evaluation/debug/plot_bland_altman_variants.py

Il Bland-Altman "classico" prodotto da evaluate_clinical_fidelity.py
(scripts/clinical_fidelty_evaluation/bland_altman_hr.png) è uno scatter
plot puro: con ~55.000 punti sovrapposti diventa una macchia solida e non
si vede più dove si concentra davvero la massa dei dati.

Questo script rigenera lo stesso Bland-Altman (stessa sorgente dati,
clinical_fidelity_results.json, stesso bias / limits of agreement) in 3
varianti pensate per dataset di queste dimensioni:

  1. bland_altman_hexbin.png     — istogramma 2D esagonale, colore in
                                    scala logaritmica (densità di punti)
  2. bland_altman_scatter_alpha.png — scatter con alpha estremo (0.03) e
                                    marker minuscoli: un outlier isolato
                                    è quasi invisibile, la zona centrale
                                    satura per sovrapposizione
  3. bland_altman_marginals.png  — stesso scatter a bassa opacità,
                                    affiancato da un istogramma + KDE
                                    della distribuzione degli errori
                                    (differenza pred - reale) sull'asse Y

Nessun ricalcolo di modello/dataset: i dati (hr_true, hr_pred per
segmento) sono già in clinical_fidelity_results.json.
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEBUG_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_PATH = os.path.join(os.path.dirname(DEBUG_DIR), "clinical_fidelity_results.json")
OUTPUT_DIR = DEBUG_DIR


def _load_hr_pairs(results_path: str):
    with open(results_path) as f:
        data = json.load(f)
    recs = data['per_segment']
    means, diffs = [], []
    for r in recs:
        hr_true, hr_pred = r.get('hr_true'), r.get('hr_pred')
        if hr_true is not None and hr_pred is not None:
            means.append((hr_true + hr_pred) / 2.0)
            diffs.append(hr_pred - hr_true)
    return np.array(means), np.array(diffs)


def _ba_stats(diffs: np.ndarray):
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs))
    return bias, sd, bias + 1.96 * sd, bias - 1.96 * sd


def _draw_ba_lines(ax, bias, loa_upper, loa_lower, with_labels=True):
    ax.axhline(bias, color='black', linestyle='-', linewidth=1.5,
               label=f'Bias = {bias:.2f} bpm' if with_labels else None)
    ax.axhline(loa_upper, color='red', linestyle='--', linewidth=1.2,
               label=f'+1.96 SD = {loa_upper:.2f} bpm' if with_labels else None)
    ax.axhline(loa_lower, color='red', linestyle='--', linewidth=1.2,
               label=f'-1.96 SD = {loa_lower:.2f} bpm' if with_labels else None)


def plot_hexbin(means, diffs, bias, loa_upper, loa_lower, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(means, diffs, gridsize=70, cmap='inferno', bins='log', mincnt=1)
    _draw_ba_lines(ax, bias, loa_upper, loa_lower)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('count (log scale)')
    ax.set_xlabel('Mean HR (real, predicted) [bpm]')
    ax.set_ylabel('HR difference (predicted - real) [bpm]')
    ax.set_title(f'Bland-Altman — hexbin density  (N={len(means)})')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_scatter_alpha(means, diffs, bias, loa_upper, loa_lower, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(means, diffs, s=18, alpha=0.08, color='#2b6cb0', edgecolors='face')
    _draw_ba_lines(ax, bias, loa_upper, loa_lower)
    ax.set_xlabel('Mean HR (real, predicted) [bpm]')
    ax.set_ylabel('HR difference (predicted - real) [bpm]')
    ax.set_title(f'Bland-Altman Plot — Heart Rate Agreement (N={len(means)})')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_marginals(means, diffs, bias, loa_upper, loa_lower, save_path):
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_marg = fig.add_subplot(gs[1], sharey=ax_main)

    ax_main.scatter(means, diffs, s=18, alpha=0.5, color='#2b6cb0', edgecolors='none')
    _draw_ba_lines(ax_main, bias, loa_upper, loa_lower)
    ax_main.set_xlabel('Mean HR (real, predicted) [bpm]')
    ax_main.set_ylabel('HR difference (predicted - real) [bpm]')
    ax_main.set_title(f'Bland-Altman with marginal error distribution  (N={len(means)})')
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_main.grid(alpha=0.2)

    ax_marg.hist(diffs, bins=150, orientation='horizontal', color='#2b6cb0',
                 alpha=0.6, edgecolor='none', density=True)
    y_grid = np.linspace(diffs.min(), diffs.max(), 400)
    kde = gaussian_kde(diffs)
    ax_marg.plot(kde(y_grid), y_grid, color='#1a365d', linewidth=1.5)
    _draw_ba_lines(ax_marg, bias, loa_upper, loa_lower, with_labels=False)
    ax_marg.set_xlabel('Density')
    plt.setp(ax_marg.get_yticklabels(), visible=False)
    ax_marg.tick_params(axis='y', length=0)

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    print(f"Carico risultati da: {RESULTS_PATH}")
    means, diffs = _load_hr_pairs(RESULTS_PATH)
    print(f"Coppie HR (reale, predetta) disponibili: {len(means)}")

    bias, sd, loa_upper, loa_lower = _ba_stats(diffs)
    print(f"Bias = {bias:.3f} bpm, SD = {sd:.3f} bpm, "
          f"LoA = [{loa_lower:.3f}, {loa_upper:.3f}] bpm")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hexbin_path = os.path.join(OUTPUT_DIR, "bland_altman_hexbin.png")
    plot_hexbin(means, diffs, bias, loa_upper, loa_lower, hexbin_path)
    print(f"✅ {hexbin_path}")

    alpha_path = os.path.join(OUTPUT_DIR, "bland_altman_scatter_alpha.png")
    plot_scatter_alpha(means, diffs, bias, loa_upper, loa_lower, alpha_path)
    print(f"✅ {alpha_path}")

    marginals_path = os.path.join(OUTPUT_DIR, "bland_altman_marginals.png")
    plot_marginals(means, diffs, bias, loa_upper, loa_lower, marginals_path)
    print(f"✅ {marginals_path}")


if __name__ == "__main__":
    main()
