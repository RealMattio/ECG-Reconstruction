"""
Due grafici a linea, uno per MAE e uno per RMSE (metriche Gruppo 1, finestra
finale di 10s), con sull'asse X tutti gli orizzonti temporali (1m, 30m, 1h,
6h, 12h, 24h) e sull'asse Y il valore medio della metrica su tutti i record
disponibili per quell'orizzonte (il numero di record disponibili varia da
un orizzonte all'altro — e' un aspetto atteso, non un problema: gli
orizzonti piu' lunghi hanno naturalmente meno record idonei).

Usa i risultati del nuovo paradigma (run_autoregressive_drift_test_clean.py):
segnale ripulito via splicing SQI PRIMA della generazione.
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_clean", "drift_performance_results_clean.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_clean")

HORIZONS = ["1m", "30m", "1h", "6h", "12h", "24h"]


def load_metric_series(results_path):
    with open(results_path) as f:
        all_results = json.load(f)

    rows = []
    for horizon in HORIZONS:
        for entry in all_results.get(horizon, []):
            metrics = entry.get("metrics", {})
            mae = metrics.get("MAE_10s")
            rmse = metrics.get("RMSE_10s")
            if mae is None or rmse is None or pd.isna(mae) or pd.isna(rmse):
                continue
            rows.append({"Horizon": horizon, "Subject": entry["subject_id"], "MAE": float(mae), "RMSE": float(rmse)})

    return pd.DataFrame(rows)


def plot_metric_vs_horizon(df: pd.DataFrame, metric_col: str, color: str, filename: str, y_label: str):
    agg = df.groupby("Horizon").agg(
        mean=(metric_col, "mean"),
        std=(metric_col, "std"),
        n=(metric_col, "count"),
    ).reindex(HORIZONS).dropna(subset=["mean"])

    x = np.arange(len(agg))
    y = agg["mean"].to_numpy()
    yerr = agg["std"].fillna(0).to_numpy()
    n = agg["n"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, linewidth=2, markersize=7,
                capsize=4, ecolor=color, elinewidth=1, alpha=0.9, label=f"{metric_col} medio ± std")

    for xi, yi, ni in zip(x, y, n):
        ax.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(agg.index.tolist())
    ax.set_xlabel("Orizzonte temporale di generazione")
    ax.set_ylabel(y_label)
    ax.set_title(f"{metric_col} (finestra finale 10s) al variare dell'orizzonte\n"
                 f"(segnale SQI-pulito, generazione autoregressiva)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"📊 Salvato: {save_path}")
    print(agg[["mean", "std", "n"]])


def main():
    print("=" * 60)
    print(" MAE / RMSE vs ORIZZONTE (nuovo paradigma, segnale SQI-pulito)")
    print("=" * 60)

    if not os.path.exists(RESULTS_PATH):
        print(f"[ERRORE] {RESULTS_PATH} non trovato.")
        print("         Esegui prima run_autoregressive_drift_test_clean.py (o ar_drif_test_clean.sh su Slurm).")
        return

    df = load_metric_series(RESULTS_PATH)
    if df.empty:
        print("[ERRORE] Nessun dato valido trovato.")
        return

    plot_metric_vs_horizon(df, "MAE", "steelblue", "drift_mae_vs_horizon.png", "MAE (segnale normalizzato)")
    plot_metric_vs_horizon(df, "RMSE", "crimson", "drift_rmse_vs_horizon.png", "RMSE (segnale normalizzato)")

    print("\n✅ Generazione completata con successo!")


if __name__ == "__main__":
    main()
