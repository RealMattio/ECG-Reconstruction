import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURAZIONI E PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_clean", "drift_performance_results_clean.json")
GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation_clean")
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_clean", "visual_context")

FS = 125
PLOT_WINDOW_SEC = 10
PLOT_SAMPLES = PLOT_WINDOW_SEC * FS

# Finestra candidata cercata dopo il confine dell'orizzonte, e passo di scansione.
SEARCH_WINDOW_SEC = 300
SEARCH_STRIDE_SEC = 1

HORIZONS_ORDER = ["1m", "30m", "1h", "6h", "12h", "24h"]
HORIZON_SEC = {"1m": 60, "30m": 1800, "1h": 3600, "6h": 21600, "12h": 43200, "24h": 86400}
SEED_SEC = 6

# Metriche su cui ripetere l'intera selezione (pazienti migliori/peggiori +
# ricerca della finestra): entrambe "piu' basso e' meglio".
METRICS = {
    "MAE": {"results_key": "MAE_10s"},
    "RMSE": {"results_key": "RMSE_10s"},
}


def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Impossibile trovare {RESULTS_PATH}")
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)


def find_24h_patients(all_results):
    return [entry["subject_id"] for entry in all_results.get("24h", [])]


def analyze_performances(all_results, results_key):
    max_horizon_perf = {}

    for horizon in HORIZONS_ORDER:
        for entry in all_results.get(horizon, []):
            subj_id = entry["subject_id"]
            val = entry.get("metrics", {}).get(results_key)
            if val is None or pd.isna(val):
                continue
            max_horizon_perf[subj_id] = float(val)

    sorted_perf = sorted(max_horizon_perf.items(), key=lambda item: item[1])

    best_patients = [p[0] for p in sorted_perf[:2]]
    worst_patients = [p[0] for p in sorted_perf[-2:]] if len(sorted_perf) >= 2 else []
    worst_patients = [p for p in worst_patients if p not in best_patients]

    return best_patients, worst_patients


def get_patient_record_path(subj_id, all_results):
    """Cerca dall'orizzonte piu' lungo a scendere, cosi' se il paziente ha
    piu' record peschiamo quello con la generazione piu' lunga."""
    for horizon in reversed(HORIZONS_ORDER):
        for entry in all_results.get(horizon, []):
            if entry["subject_id"] == subj_id:
                return entry["record_path"]
    return None


def _minmax_norm(signal: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(signal), np.nanmax(signal)
    if pd.isna(mn) or pd.isna(mx) or (mx - mn) < 1e-8:
        return np.zeros_like(signal)
    return (signal - mn) / (mx - mn)


def _find_best_window_start(ecg_true_full, ecg_gen_full, search_start, search_end, win_samples, stride, metric):
    """Tra tutte le finestre candidate in [search_start, search_end], sceglie
    quella che minimizza la metrica scelta (MAE o RMSE) tra reale e generato."""
    best_start = search_start
    best_score = np.inf

    for start in range(search_start, search_end + 1, stride):
        t = ecg_true_full[start:start + win_samples]
        g = ecg_gen_full[start:start + win_samples]
        if len(t) < win_samples or len(g) < win_samples:
            continue
        if np.isnan(g).any() or np.isinf(g).any():
            continue

        t_norm = _minmax_norm(t)
        diff = t_norm - g
        if metric == "MAE":
            score = float(np.mean(np.abs(diff)))
        else:  # RMSE
            score = float(np.sqrt(np.mean(diff ** 2)))

        if score < best_score:
            best_score = score
            best_start = start

    return best_start


def load_signal_data(horizon, record_path, metric):
    npz_path = os.path.join(GENERATION_DIR, record_path + ".npz")

    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path)
        ecg_true_full = data["ecg_target"]
        ecg_gen_full = data["ecg_generated"]
        ppg_full = data["ppg_input"]

        horizon_end_sample = int((HORIZON_SEC[horizon] + SEED_SEC) * FS)
        total_len = min(len(ecg_true_full), len(ecg_gen_full), len(ppg_full))
        if horizon_end_sample > total_len:
            return None  # questo record non raggiunge questo orizzonte

        search_start = horizon_end_sample
        search_end = min(horizon_end_sample + SEARCH_WINDOW_SEC * FS, total_len) - PLOT_SAMPLES
        stride = SEARCH_STRIDE_SEC * FS

        if search_end < search_start:
            start_sample = max(0, horizon_end_sample - PLOT_SAMPLES)
        else:
            start_sample = _find_best_window_start(
                ecg_true_full, ecg_gen_full, search_start, search_end, PLOT_SAMPLES, stride, metric
            )

        end_sample = start_sample + PLOT_SAMPLES
        return {
            "ppg": ppg_full[start_sample:end_sample],
            "ecg_true": ecg_true_full[start_sample:end_sample],
            "ecg_gen": ecg_gen_full[start_sample:end_sample],
        }
    except Exception as e:
        print(f"Errore nel caricamento di {npz_path}: {e}")
        return None


def plot_patient_horizons(subj_id, record_path, title_prefix, filename_prefix, metric, output_dir):
    available_data = {}
    for horizon in HORIZONS_ORDER:
        data = load_signal_data(horizon, record_path, metric)
        if data is not None:
            available_data[horizon] = data

    if not available_data:
        print(f"Nessun dato generato trovato per il paziente {subj_id}.")
        return

    n_horizons = len(available_data)

    fig, axes = plt.subplots(nrows=n_horizons, ncols=2, figsize=(15, 3 * n_horizons), squeeze=False)
    fig.suptitle(f"{title_prefix} - Paziente: {subj_id}\nContestualizzazione per Orizzonte", fontsize=16, fontweight='bold', y=0.98)

    time_axis = np.linspace(0, PLOT_WINDOW_SEC, PLOT_SAMPLES)

    for i, (horizon, signals) in enumerate(available_data.items()):
        ppg = signals["ppg"]
        ecg_t = signals["ecg_true"]
        ecg_g = signals["ecg_gen"]

        if np.isnan(ecg_g).any() or np.isinf(ecg_g).any():
            ecg_g_clean = np.zeros_like(ecg_g)
            is_exploded = True
        else:
            ecg_g_clean = ecg_g
            is_exploded = False

        ppg_norm = _minmax_norm(ppg)
        ecg_t_norm = _minmax_norm(ecg_t)
        ecg_g_norm = _minmax_norm(ecg_g_clean) if not is_exploded else ecg_g_clean

        # --- Colonna 1: PPG Input ---
        ax_ppg = axes[i, 0]
        ax_ppg.plot(time_axis, ppg_norm, color='blue', label='PPG Input')
        ax_ppg.set_title(f"Orizzonte: {horizon} - Input")
        ax_ppg.set_ylabel("Norm [0,1]")
        ax_ppg.grid(True, alpha=0.3)
        if i == n_horizons - 1:
            ax_ppg.set_xlabel("Secondi")

        # --- Colonna 2: ECG Confronto ---
        ax_ecg = axes[i, 1]
        ax_ecg.plot(time_axis, ecg_t_norm, color='black', alpha=0.4, label='ECG Reale', linestyle='--')

        if is_exploded:
            ax_ecg.plot(time_axis, ecg_g_norm, color='red', label='ECG Generato (COLLASSO NUMERICO)', linewidth=2)
            ax_ecg.text(5, 0.5, "DIVERGENZA NUMERICA (NaN/Inf)", color='red', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax_ecg.plot(time_axis, ecg_g_norm, color='red', label='ECG Generato', alpha=0.8)

        ax_ecg.set_title(f"Orizzonte: {horizon} - Ricostruzione")
        ax_ecg.grid(True, alpha=0.3)
        ax_ecg.legend(loc='upper right')
        if i == n_horizons - 1:
            ax_ecg.set_xlabel("Secondi")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename_prefix}_{subj_id}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✓ Grafico salvato: {save_path}")


def run_for_metric(metric, all_results):
    results_key = METRICS[metric]["results_key"]
    output_dir = os.path.join(OUTPUT_PLOTS_DIR, metric)

    print(f"\n--- Metrica: {metric} ({results_key}) ---")

    patients_24h = find_24h_patients(all_results)[:2]
    best_patients, worst_patients = analyze_performances(all_results, results_key)

    print(f"-> Pazienti 24h individuati: {patients_24h}")
    print(f"-> Pazienti TOP Performance individuati: {best_patients}")
    print(f"-> Pazienti WORST Performance individuati: {worst_patients}")

    jobs = []

    for p in patients_24h:
        jobs.append({"id": p, "prefix": "Target_24h", "title": "Paziente Lungo (24h)"})

    for p in best_patients:
        jobs.append({"id": p, "prefix": "Best_Perf", "title": "Miglior Performance"})

    for p in worst_patients:
        jobs.append({"id": p, "prefix": "Worst_Perf", "title": "Peggior Performance"})

    if not jobs:
        print("Nessun paziente trovato per la visualizzazione.")
        return

    for job in jobs:
        record_path = get_patient_record_path(job["id"], all_results)
        if record_path:
            plot_patient_horizons(job["id"], record_path, job["title"], job["prefix"], metric, output_dir)
        else:
            print(f"Impossibile trovare il path del record per il paziente {job['id']}")

    print(f"-> Figure di {metric} salvate in: {output_dir}")


def main():
    print("=" * 60)
    print(" GENERAZIONE GRAFICI CONTESTUALI (segnale SQI-pulito)")
    print("=" * 60)

    try:
        all_results = load_results()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    for metric in METRICS:
        run_for_metric(metric, all_results)

    print("\n" + "=" * 60)
    print(f"✅ Finito! Tutti i grafici si trovano in: {OUTPUT_PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
