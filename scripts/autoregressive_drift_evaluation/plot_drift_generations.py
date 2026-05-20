import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURAZIONI E PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation", "drift_performance_results.json")
GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation")
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation", "visual_context")

FS = 125
PLOT_WINDOW_SEC = 10
PLOT_SAMPLES = PLOT_WINDOW_SEC * FS

HORIZONS_ORDER = ["1m", "30m", "1h", "6h", "12h", "24h"]

def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Impossibile trovare {RESULTS_PATH}")
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)

def find_24h_patients(all_results):
    patients = []
    if "24h" in all_results:
        for entry in all_results["24h"]:
            patients.append(entry["subject_id"])
    return patients

def analyze_performances(all_results):
    best_candidates = []
    exploded_patients = []
    max_horizon_perf = {}
    
    for horizon in HORIZONS_ORDER:
        if horizon not in all_results:
            continue
            
        for entry in all_results[horizon]:
            subj_id = entry["subject_id"]
            metrics = entry.get("metrics", {})
            mae = metrics.get("MAE_10s")
            
            if pd.isna(mae) or mae is None:
                if subj_id not in exploded_patients:
                    exploded_patients.append(subj_id)
            else:
                max_horizon_perf[subj_id] = {
                    "horizon": horizon,
                    "mae": float(mae)
                }
                
    valid_perfs = {k: v for k, v in max_horizon_perf.items() if k not in exploded_patients}
    sorted_best = sorted(valid_perfs.items(), key=lambda item: item[1]["mae"])
    
    best_patients = [p[0] for p in sorted_best[:2]]
    worst_patients = exploded_patients[:2]
    
    return best_patients, worst_patients

def get_patient_record_path(subj_id, all_results):
    """
    Cerca nel json il 'record_path' originale.
    FIX: Cerca dall'orizzonte PIÙ LUNGO (24h) a scendere. 
    Così se il paziente ha più record, peschiamo sicuramente quello che è durato di più!
    """
    for horizon in reversed(HORIZONS_ORDER):
        if horizon in all_results:
            for entry in all_results[horizon]:
                if entry["subject_id"] == subj_id:
                    return entry["record_path"]
    return None

def load_signal_data(horizon, record_path):
    file_name = os.path.basename(record_path) + ".npz"
    dir_path = os.path.dirname(record_path)
    npz_path = os.path.join(GENERATION_DIR, horizon, dir_path, file_name)
    
    if not os.path.exists(npz_path):
        return None
        
    try:
        data = np.load(npz_path)
        return {
            "ppg": data["ppg_input"][-PLOT_SAMPLES:],
            "ecg_true": data["ecg_target"][-PLOT_SAMPLES:],
            "ecg_gen": data["ecg_generated"][-PLOT_SAMPLES:]
        }
    except Exception as e:
        print(f"Errore nel caricamento di {npz_path}: {e}")
        return None

def _minmax_norm(signal: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(signal), np.nanmax(signal)
    if pd.isna(mn) or pd.isna(mx) or (mx - mn) < 1e-8:
        return np.zeros_like(signal)
    return (signal - mn) / (mx - mn)

def plot_patient_horizons(subj_id, record_path, title_prefix, filename_prefix):
    available_data = {}
    for horizon in HORIZONS_ORDER:
        data = load_signal_data(horizon, record_path)
        if data is not None:
            available_data[horizon] = data
            
    if not available_data:
        print(f"Nessun dato generato trovato per il paziente {subj_id}.")
        return
        
    n_horizons = len(available_data)
    
    fig, axes = plt.subplots(nrows=n_horizons, ncols=2, figsize=(15, 3 * n_horizons), squeeze=False)
    fig.suptitle(f"{title_prefix} - Paziente: {subj_id}\nContestualizzazione Ultimi 10s per Orizzonte", fontsize=16, fontweight='bold', y=0.98)
    
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
            ax_ppg.set_xlabel("Secondi (Ultima Finestra)")

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
            ax_ecg.set_xlabel("Secondi (Ultima Finestra)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    # Aggiungiamo un identificativo univoco (il prefix) per evitare sovrascritture se un paziente appartiene a due categorie
    save_path = os.path.join(OUTPUT_PLOTS_DIR, f"{filename_prefix}_{subj_id}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✓ Grafico salvato: {save_path}")

def main():
    print("=" * 60)
    print(" GENERAZIONE GRAFICI CONTESTUALI (SEGNALI TEMPORALI)")
    print("=" * 60)
    
    try:
        all_results = load_results()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    patients_24h = find_24h_patients(all_results)[:2] 
    best_patients, worst_patients = analyze_performances(all_results)
    
    print(f"-> Pazienti 24h individuati: {patients_24h}")
    print(f"-> Pazienti TOP Performance individuati: {best_patients}")
    print(f"-> Pazienti WORST (Collassati) individuati: {worst_patients}")
    
    jobs = []
    
    # FIX: Nessun controllo anti-duplicati. Aggiungiamo sempre tutti i job.
    for p in patients_24h:
        jobs.append({"id": p, "prefix": "Target_24h", "title": "Paziente Lungo (24h)"})
        
    for p in best_patients:
        jobs.append({"id": p, "prefix": "Best_Perf", "title": "Miglior Performance"})
            
    for p in worst_patients:
        jobs.append({"id": p, "prefix": "Worst_Perf", "title": "Peggior Performance (Collasso)"})

    if not jobs:
        print("Nessun paziente trovato per la visualizzazione.")
        return
        
    print("\n-> Generazione figure in corso...")
    for job in jobs:
        record_path = get_patient_record_path(job["id"], all_results)
        if record_path:
            plot_patient_horizons(job["id"], record_path, job["title"], job["prefix"])
        else:
            print(f"Impossibile trovare il path del record per il paziente {job['id']}")

    print("=" * 60)
    print(f"✅ Finito! Tutti i grafici si trovano in: {OUTPUT_PLOTS_DIR}")

if __name__ == "__main__":
    main()