"""
Diagnostica di un singolo record: mostra PPG input ed ECG reale/generato per
capire *come* si arriva a una combinazione MAE basso + Pearson r basso (o
negativo) sulla finestra finale di 10s (le metriche Gruppo 1 di
run_autoregressive_drift_test.py). Non serve a produrre figure per il paper,
ma a ispezionare interattivamente un record specifico.

Uso:
    python plot_single_patient_diagnostic.py --record p05/p052972/3368521_0032
"""
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_quality import compute_quality_mask, quality_fraction  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_diagnostics")

FS = 125
SEED_SEC = 6
EVAL_WINDOW_SEC = 10
EVAL_SAMPLES = EVAL_WINDOW_SEC * FS


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def _time_axis_formatter(x, _pos=None):
    if x < 60:
        return f"{x:.0f}s"
    if x < 3600:
        return f"{x / 60:.0f}m"
    return f"{x / 3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Diagnostica PPG/ECG per un singolo record del drift test.")
    parser.add_argument("--record", required=True, help="record_path come in drift_performance_results.json, es. p05/p052972/3368521_0032")
    args = parser.parse_args()

    npz_path = os.path.join(GENERATION_DIR, args.record + ".npz")
    if not os.path.exists(npz_path):
        print(f"[ERRORE] Non trovato: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)
    ppg = data["ppg_input"].astype(np.float64)
    ecg_t = data["ecg_target"].astype(np.float64)
    ecg_g = data["ecg_generated"].astype(np.float64)

    total_sec = len(ecg_t) / FS
    quality_mask = compute_quality_mask(ecg_t, ppg, FS)

    # --- Riproduce ESATTAMENTE le metriche Gruppo 1 (MAE_10s / Pearson_10s) ---
    n_samples = len(ecg_t)
    win_true = ecg_t[n_samples - EVAL_SAMPLES: n_samples]
    win_pred = ecg_g[n_samples - EVAL_SAMPLES: n_samples]  # gia' in scala del modello (~[0,1])
    win_true_norm = _minmax_norm(win_true)
    mae_10s = float(np.mean(np.abs(win_true_norm - win_pred)))
    pearson_10s, _ = pearsonr(win_true_norm, win_pred)
    q_10s = quality_fraction(quality_mask, n_samples - EVAL_SAMPLES, n_samples)

    print(f"Record: {args.record}  (durata {total_sec:.0f}s)")
    print(f"MAE_10s ricalcolato      : {mae_10s:.4f}")
    print(f"Pearson_10s ricalcolato  : {pearson_10s:.4f}")
    print(f"Std ECG reale (finestra) : {win_true_norm.std():.4f}")
    print(f"Std ECG generato (fin.)  : {win_pred.std():.4f}")
    print(f"Frazione qualita' SQI (ultimi 10s): {q_10s:.2%}")

    # --- Normalizzazioni per il plot ---
    ppg_full_norm = _minmax_norm(ppg)
    ecg_t_full_norm = _minmax_norm(ecg_t)  # scala globale, coerente con la pipeline
    ecg_g_full_norm = _minmax_norm(ecg_g)

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 1.6, 1.6, 0.3], hspace=0.55)
    t_full = np.arange(len(ecg_t)) / FS

    # --- Riga 1: PPG input ---
    ax_ppg = fig.add_subplot(gs[0])
    ax_ppg.axvspan(0, SEED_SEC, color="green", alpha=0.08)
    ax_ppg.axvspan(SEED_SEC, total_sec, color="red", alpha=0.05)
    ax_ppg.plot(t_full, ppg_full_norm, color="steelblue", linewidth=0.9)
    ax_ppg.set_ylabel("PPG input\n(norm. globale)")
    ax_ppg.set_title(f"Record {args.record} — PPG input e ricostruzione ECG (durata {total_sec:.0f}s)",
                      fontsize=13, fontweight="bold")
    ax_ppg.grid(True, alpha=0.3)

    # --- Riga 2: ECG reale vs generato, intera durata, scala globale ---
    ax_full = fig.add_subplot(gs[1], sharex=ax_ppg)
    ax_full.axvspan(0, SEED_SEC, color="green", alpha=0.08, label="Seed (contesto reale)")
    ax_full.axvspan(SEED_SEC, total_sec, color="red", alpha=0.05, label="Generazione autoregressiva")
    ax_full.axvspan(total_sec - EVAL_WINDOW_SEC, total_sec, color="black", alpha=0.06, label="Finestra MAE_10s/Pearson_10s")
    ax_full.plot(t_full, ecg_t_full_norm, color="black", linestyle="--", alpha=0.6, linewidth=1, label="ECG reale")
    ax_full.plot(t_full, ecg_g_full_norm, color="red", alpha=0.85, linewidth=1, label="ECG generato")
    ax_full.axvline(SEED_SEC, color="blue", linestyle=":", linewidth=1)
    ax_full.set_ylabel("ECG\n(norm. globale sull'intero record)")
    ax_full.legend(loc="upper right", fontsize=8, ncol=2)
    ax_full.grid(True, alpha=0.3)
    ax_full.set_xlabel("Secondi")

    # --- Riga 3: zoom sulla finestra esatta usata per MAE_10s / Pearson_10s ---
    ax_zoom = fig.add_subplot(gs[2])
    t_zoom = np.arange(EVAL_SAMPLES) / FS
    ax_zoom.plot(t_zoom, win_true_norm, color="black", linestyle="--", alpha=0.7, linewidth=1.3, label="ECG reale")
    ax_zoom.plot(t_zoom, win_pred, color="red", alpha=0.9, linewidth=1.3, label="ECG generato")
    ax_zoom.fill_between(t_zoom, win_pred.mean() - win_pred.std(), win_pred.mean() + win_pred.std(),
                          color="red", alpha=0.08, label=f"±1 std generato ({win_pred.std():.3f})")
    ax_zoom.set_title(
        f"Finestra finale ({EVAL_WINDOW_SEC}s) usata per le metriche — "
        f"MAE={mae_10s:.3f}, Pearson r={pearson_10s:.3f}, "
        f"std reale={win_true_norm.std():.3f} vs std generato={win_pred.std():.3f}",
        fontsize=11
    )
    ax_zoom.set_xlabel(f"Secondi (finestra locale, t={total_sec - EVAL_WINDOW_SEC:.0f}s–{total_sec:.0f}s del record)")
    ax_zoom.set_ylabel("Ampiezza\n(norm. locale alla finestra)")
    ax_zoom.legend(loc="upper right", fontsize=8)
    ax_zoom.grid(True, alpha=0.3)

    # --- Riga 4: striscia qualita' SQI ---
    ax_q = fig.add_subplot(gs[3], sharex=ax_ppg)
    stride = max(1, len(quality_mask) // 3000)
    tq = t_full[::stride]
    ax_q.fill_between(tq, 0, 1, where=quality_mask[::stride], color="seagreen", alpha=0.6, step="mid")
    ax_q.fill_between(tq, 0, 1, where=~quality_mask[::stride], color="dimgray", alpha=0.6, step="mid")
    ax_q.set_yticks([])
    ax_q.set_ylabel("Qualita'\nSQI", fontsize=8)
    ax_q.set_xlabel("Secondi")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = args.record.replace("/", "_")
    save_path = os.path.join(OUTPUT_DIR, f"diagnostic_{safe_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\nSalvato: {save_path}")


if __name__ == "__main__":
    main()
