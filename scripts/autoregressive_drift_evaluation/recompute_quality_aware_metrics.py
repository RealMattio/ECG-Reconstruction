"""
Annota i risultati del drift test (gia' generati/calcolati da
run_autoregressive_drift_test.py) con la frazione di segnale "pulito"
secondo lo stesso criterio SQI (kurtosi ECG / skewness PPG) usato per
costruire i dataset di training/valutazione "normali" del modello (vedi
signal_quality.py). Non rigenera nulla: legge i .npz gia' su disco e il
drift_performance_results.json esistente, e scrive una copia annotata.

Aggiunge a ciascuna voce "metrics":
  - Quality_Valid_Frac_10s: frazione di campioni puliti nella finestra
    finale di 10s usata per le metriche Gruppo 1.
  - Quality_Valid_Frac_Cumulative: frazione di campioni puliti sull'intero
    intervallo [0, orizzonte] usato per le metriche cumulative Gruppo 2/3.

Le metriche esistenti NON vengono ricalcolate/alterate: la logica di
filtraggio (quali record/finestre scartare) e' demandata agli script di
plotting (plot_drift_boxplots.py, plot_drift_temporal_analysis.py), cosi'
si puo' confrontare "raw" vs "quality-filtered" senza perdere nulla.
"""
import os
import sys
import json
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from signal_quality import compute_quality_mask, quality_fraction  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_performance_results.json")
ANNOTATED_RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_performance_results_quality_annotated.json")
GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation")

FS = 125
EVAL_WINDOW_SEC = 10
EVAL_SAMPLES = EVAL_WINDOW_SEC * FS
SEED_SEC = 6
HORIZON_SEC = {
    "1m": 60 + SEED_SEC,
    "30m": 1800 + SEED_SEC,
    "1h": 3600 + SEED_SEC,
    "6h": 21600 + SEED_SEC,
    "12h": 43200 + SEED_SEC,
    "24h": 86400 + SEED_SEC,
}
HORIZONS_ORDER = ["1m", "30m", "1h", "6h", "12h", "24h"]


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[ERRORE] {RESULTS_PATH} non trovato. Esegui prima run_autoregressive_drift_test.py.")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    # Raggruppiamo per record_path cosi' calcoliamo la maschera di qualita'
    # UNA sola volta per record (e' costosa quanto basta da non rifarla per
    # ognuno dei suoi orizzonti).
    records_to_horizons = {}
    for h in HORIZONS_ORDER:
        for entry in all_results.get(h, []):
            records_to_horizons.setdefault(entry["record_path"], []).append(h)

    print(f"-> {len(records_to_horizons)} record unici da annotare "
          f"({sum(len(v) for v in records_to_horizons.values())} voci orizzonte totali).")

    n_missing_npz = 0
    for rec_path, horizons in tqdm(records_to_horizons.items(), desc="Annotazione qualita'"):
        npz_path = os.path.join(GENERATION_DIR, rec_path + ".npz")
        if not os.path.exists(npz_path):
            n_missing_npz += 1
            continue

        with np.load(npz_path) as data:
            ecg_target = data["ecg_target"].astype(np.float64)
            ppg_input = data["ppg_input"].astype(np.float64)

        mask = compute_quality_mask(ecg_target, ppg_input, FS)

        for h in horizons:
            n_samples = min(int(HORIZON_SEC[h] * FS), len(mask))
            valid_10s = quality_fraction(mask, max(0, n_samples - EVAL_SAMPLES), n_samples)
            valid_cumulative = quality_fraction(mask, 0, n_samples)

            for entry in all_results[h]:
                if entry["record_path"] == rec_path:
                    entry["metrics"]["Quality_Valid_Frac_10s"] = valid_10s
                    entry["metrics"]["Quality_Valid_Frac_Cumulative"] = valid_cumulative

    if n_missing_npz:
        print(f"[WARN] {n_missing_npz} record avevano metriche ma nessun .npz corrispondente (saltati).")

    with open(ANNOTATED_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"-> Risultati annotati salvati in: {ANNOTATED_RESULTS_PATH}")


if __name__ == "__main__":
    main()
