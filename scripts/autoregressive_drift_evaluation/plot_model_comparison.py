"""
Confronto diretto fra piu' modelli sulla valutazione del drift a finestre
(vedi run_windowed_drift_test.py / run_pat_analysis.py): sovrappone le curve
MAE/RMSE per secondo generato e MAE del PAT per orizzonte di piu' model_id
sullo stesso grafico, per rispondere alla domanda "il nuovo modello e' piu' o
meno stabile del precedente?".

Riusa i risultati gia' calcolati da ciascun modello (drift_windowed_results.json
e, se presente, pat_analysis/pat_pairs.json): non genera nulla, e' solo lettura
+ plotting.

Uso:
  python scripts/autoregressive_drift_evaluation/plot_model_comparison.py \
      --model_ids lightweight_hybrid_20260608_192241 lightweight_hybrid_20260716_145739

  # senza --model_ids: confronta automaticamente tutti i model_id trovati
  # sotto experiments/drift_evaluation_windowed/
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_windowed")
COMPARISON_DIR = os.path.join(BASE_OUTPUT_DIR, "model_comparison")

FS = 125
FIRST_X_SEC = 7
MIN_COUNT = 5  # per punto per-modello: sotto questa soglia il punto non e' abbastanza affidabile da plottare

# Palette categorica (colorblind-safe, ordine fisso: il primo modello e'
# sempre lo stesso colore a prescindere da quanti se ne confrontano)
PALETTE = ["#4E79A7", "#B03A2E", "#59A14F", "#F28E2B", "#8E5C9E", "#76B7B2"]
GRID = "#D9D9D9"
INK = "#333333"


def discover_model_ids():
    if not os.path.isdir(BASE_OUTPUT_DIR):
        return []
    ids = []
    for name in sorted(os.listdir(BASE_OUTPUT_DIR)):
        results_path = os.path.join(BASE_OUTPUT_DIR, name, "drift_windowed_results.json")
        if os.path.isfile(results_path):
            ids.append(name)
    return ids


def load_segments(model_id):
    path = os.path.join(BASE_OUTPUT_DIR, model_id, "drift_windowed_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("segments", [])


def aggregate_per_second(segs, metric):
    buckets = defaultdict(list)
    for s in segs:
        arr = s.get("per_second", {}).get(metric)
        if not arr:
            continue
        for j, v in enumerate(arr):
            if v is not None and np.isfinite(v):
                buckets[FIRST_X_SEC + j].append(v)
    xs = sorted(x for x, vals in buckets.items() if len(vals) >= MIN_COUNT)
    x = np.array(xs, dtype=float)
    mean = np.array([np.mean(buckets[k]) for k in xs])
    std = np.array([np.std(buckets[k]) for k in xs])
    count = np.array([len(buckets[k]) for k in xs])
    return x, mean, std, count


def load_pat_cumulative_mae(model_id, min_count=10):
    """Riusa la cache PAT (pat_analysis/pat_pairs.json) se presente, e ricalcola
    la curva cumulativa del MAE del PAT per orizzonte (stessa logica di
    run_pat_analysis.cumulative_curves, ma solo per MAE, per non duplicare
    l'intero modulo)."""
    cache_path = os.path.join(BASE_OUTPUT_DIR, model_id, "pat_analysis", "pat_pairs.json")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        seg_pats = json.load(f)["segments"]

    buckets = defaultdict(list)
    for s in seg_pats:
        bt = np.asarray(s["beat_times"], dtype=float)
        if len(bt) == 0:
            continue
        pat_real = np.asarray(s["pat_real"], dtype=float)
        pat_gen = np.asarray(s["pat_gen"], dtype=float)
        d = pat_gen - pat_real
        order = np.argsort(bt)
        bt, d = bt[order], d[order]

        abs_cum = np.concatenate([[0.0], np.cumsum(np.abs(d))])
        h_max = int(np.floor(min(s["length_sec"], bt[-1])))
        h_min = max(FIRST_X_SEC, int(np.ceil(bt[0])))
        if h_max < h_min:
            continue
        hors = np.arange(h_min, h_max + 1)
        k = np.searchsorted(bt, hors, side="right")
        valid = k >= 1
        hors, k = hors[valid], k[valid]
        mae = abs_cum[k] / k
        for h, a in zip(hors, mae):
            buckets[int(h)].append(a)

    xs = sorted(x for x, v in buckets.items() if len(v) >= min_count)
    x = np.array(xs, dtype=float)
    mean = np.array([np.mean(buckets[k]) for k in xs])
    std = np.array([np.std(buckets[k]) for k in xs])
    count = np.array([len(buckets[k]) for k in xs])
    return x, mean, std, count


def _style_axes(ax, xlabel):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK)
    ax.set_xlabel(xlabel, color=INK)


def plot_comparison(curves_by_model, ylabel, title, filename, nonneg=True):
    """curves_by_model: {model_id: (x, mean, std, count)}."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    any_data = False

    for i, (model_id, (x, mean, std, count)) in enumerate(curves_by_model.items()):
        if len(x) == 0:
            continue
        any_data = True
        color = PALETTE[i % len(PALETTE)]
        lower = mean - std
        if nonneg:
            lower = np.maximum(lower, 0.0)
        ax.fill_between(x, lower, mean + std, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.8, label=model_id, zorder=3)

    if not any_data:
        print(f"[WARN] Nessun dato per '{title}', salto {filename}.")
        plt.close(fig)
        return

    _style_axes(ax, "Lunghezza della finestra = secondo generato [s]")
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()

    os.makedirs(COMPARISON_DIR, exist_ok=True)
    out = os.path.join(COMPARISON_DIR, filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✓ Salvato: {out}")


def main():
    parser = argparse.ArgumentParser(description="Confronta il drift (e il PAT) di piu' modelli sullo stesso grafico.")
    parser.add_argument("--model_ids", nargs="*", default=None,
                        help="model_id da confrontare (default: tutti quelli trovati sotto drift_evaluation_windowed/).")
    args = parser.parse_args()

    model_ids = args.model_ids or discover_model_ids()
    if len(model_ids) < 2:
        print(f"[ERRORE] Servono almeno 2 modelli da confrontare, trovati: {model_ids}")
        print(f"         (cercati sotto {BASE_OUTPUT_DIR})")
        sys.exit(1)

    print("=" * 60)
    print(" CONFRONTO FRA MODELLI — drift a finestre + PAT")
    print("=" * 60)
    print(f"-> Modelli: {model_ids}")

    segs_by_model = {}
    for mid in model_ids:
        segs = load_segments(mid)
        segs_by_model[mid] = segs
        print(f"   {mid}: {len(segs)} segmenti")

    mae_curves = {mid: aggregate_per_second(segs, "MAE") for mid, segs in segs_by_model.items()}
    rmse_curves = {mid: aggregate_per_second(segs, "RMSE") for mid, segs in segs_by_model.items()}

    plot_comparison(mae_curves, "MAE sul secondo generato",
                     "Confronto MAE per secondo generato\n(atteso: in salita con la lunghezza — piu' basso e' meglio)",
                     "compare_MAE.png")
    plot_comparison(rmse_curves, "RMSE sul secondo generato",
                     "Confronto RMSE per secondo generato\n(atteso: in salita con la lunghezza — piu' basso e' meglio)",
                     "compare_RMSE.png")

    pat_curves = {}
    for mid in model_ids:
        c = load_pat_cumulative_mae(mid)
        if c is not None:
            pat_curves[mid] = c
        else:
            print(f"[INFO] {mid}: nessuna cache PAT trovata (esegui prima run_pat_analysis.py --model_id {mid}), "
                  f"escluso dal confronto PAT.")

    if len(pat_curves) >= 2:
        plot_comparison(pat_curves, "MAE del PAT [ms] (cumulativo)",
                         "Confronto MAE del Pulse Arrival Time vs orizzonte\n(atteso: in salita con l'orizzonte — piu' basso e' meglio)",
                         "compare_PAT_MAE.png")
    else:
        print("[INFO] Confronto PAT saltato: meno di 2 modelli con cache PAT disponibile.")

    print("\n" + "=" * 60)
    print(f"✅ Finito. Grafici di confronto in: {COMPARISON_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
