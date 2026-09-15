"""
Ricostruisce drift_windowed_results.json nel NUOVO formato (profilo per-secondo)
a partire dagli .npz gia' generati da run_windowed_drift_test.py, SENZA rieseguire
la generazione sulla GPU.

Perche' serve:
  Una prima versione del run script salvava, per ogni segmento, UNA sola metrica
  scalare sull'intera porzione generata ("metrics"). La versione corretta salva
  invece il profilo SECONDO-PER-SECONDO ("per_second"). Poiche' la generazione
  NON e' cambiata (stesso modello, stessa segmentazione SQI) e ogni .npz contiene
  gia' i segnali `segN_true` e `segN_gen`, e' sufficiente RICALCOLARE le metriche
  dai segnali salvati: nessuna inferenza GPU da ripetere.

Uso tipico:
  # dopo che il job di generazione ha finito (o anche solo per i record gia' fatti)
  python scripts/autoregressive_drift_evaluation/recompute_per_second_from_npz.py

Sovrascrive drift_windowed_results.json con il nuovo formato. Poi:
  python scripts/autoregressive_drift_evaluation/plot_windowed_drift.py
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pat_hr_utils as PU  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
BASE_GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation_windowed")
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_windowed")

FS = 125
SEED_SAMPLES = 6 * FS
GEN_SAMPLES = 1 * FS


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def compute_per_second_metrics(true_gen: np.ndarray, pred_gen: np.ndarray, gen_samples: int = GEN_SAMPLES) -> dict:
    """Un valore per secondo generato (indice j -> asse X = 7 + j).
    Versione vettorizzata (equivalente numericamente a scipy.pearsonr per
    finestra ma senza loop Python: necessaria per processare decine di migliaia
    di segmenti in tempi ragionevoli)."""
    n_windows = len(true_gen) // gen_samples
    if n_windows == 0:
        return {"MAE": [], "RMSE": [], "Pearson": [], "Cosine": []}

    true_norm = _minmax_norm(true_gen)[: n_windows * gen_samples]
    pred = pred_gen[: n_windows * gen_samples].astype(np.float64)

    T = true_norm.reshape(n_windows, gen_samples)
    P = pred.reshape(n_windows, gen_samples)

    diff = T - P
    mae = np.mean(np.abs(diff), axis=1)
    rmse = np.sqrt(np.mean(diff ** 2, axis=1))

    # Pearson per finestra (r = cov / (std_t std_p)); 0 dove una serie e' costante
    tc = T - T.mean(axis=1, keepdims=True)
    pc = P - P.mean(axis=1, keepdims=True)
    num = np.sum(tc * pc, axis=1)
    den = np.sqrt(np.sum(tc ** 2, axis=1) * np.sum(pc ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.where(den > 1e-12, num / den, 0.0)

    # Cosine per finestra su fette min-max normalizzate (come Cumulative_CosineSim)
    Tn = (T - T.min(axis=1, keepdims=True)) / (T.max(axis=1, keepdims=True) - T.min(axis=1, keepdims=True) + 1e-8)
    Pn = (P - P.min(axis=1, keepdims=True)) / (P.max(axis=1, keepdims=True) - P.min(axis=1, keepdims=True) + 1e-8)
    cos = np.sum(Tn * Pn, axis=1) / (
        (np.sqrt(np.sum(Tn ** 2, axis=1)) + 1e-8) * (np.sqrt(np.sum(Pn ** 2, axis=1)) + 1e-8)
    )

    return {
        "MAE": mae.astype(float).tolist(),
        "RMSE": rmse.astype(float).tolist(),
        "Pearson": pearson.astype(float).tolist(),
        "Cosine": cos.astype(float).tolist(),
    }


def _subject_id_from_path(record_path: str) -> str:
    # record_path es. 'p00/p006299/3591341_0015' -> 'p006299'
    parts = record_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


def _load_old_metadata(path):
    """Recupera i metadati per (record_path, segment_index) dal vecchio JSON, se c'e'."""
    meta = {}
    if not os.path.exists(path):
        return meta
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return meta
    for s in data.get("segments", []):
        meta[(s["record_path"], s.get("segment_index"))] = {
            k: s[k] for k in ("seg_start_sample", "seg_end_sample", "seed_ecg_score", "ppg_inverted")
            if k in s
        }
    return meta


def main():
    parser = argparse.ArgumentParser(description="Ricalcola il profilo per-secondo dagli .npz (no GPU).")
    parser.add_argument("--model_id", required=True,
                        help="Sotto-cartella del modello (es. lightweight_hybrid_20260716_145739), "
                             "la stessa usata da run_windowed_drift_test.py --model_weights_path.")
    parser.add_argument("--gen_dir", default=None, help="Sovrascrive la cartella .npz dedotta da --model_id.")
    parser.add_argument("--out", default=None, help="Sovrascrive il file di output dedotto da --model_id.")
    parser.add_argument("--old_json", default=None, help="JSON da cui recuperare i metadati (default: --out stesso).")
    parser.add_argument("--no_hr", action="store_true",
                        help="Salta il calcolo dell'errore Heart Rate (piu' veloce, ma niente 4a curva).")
    args = parser.parse_args()

    gen_dir = args.gen_dir or os.path.join(BASE_GENERATION_DIR, args.model_id)
    out_path = args.out or os.path.join(BASE_OUTPUT_DIR, args.model_id, "drift_windowed_results.json")
    old_json = args.old_json or out_path
    args.gen_dir, args.out, args.old_json = gen_dir, out_path, old_json

    npz_files = sorted(glob.glob(os.path.join(args.gen_dir, "**", "*.npz"), recursive=True))
    print(f"-> Trovati {len(npz_files)} file .npz in {args.gen_dir}")
    if not npz_files:
        print("[ERRORE] Nessun .npz: la generazione non ha ancora prodotto nulla.")
        sys.exit(1)

    old_meta = _load_old_metadata(args.old_json)
    if old_meta:
        print(f"-> Metadati recuperati dal vecchio JSON per {len(old_meta)} segmenti.")

    segments = []
    n_records = 0
    for npz_path in npz_files:
        rec_path = os.path.relpath(npz_path, args.gen_dir)[:-4]  # toglie '.npz'
        subj_id = _subject_id_from_path(rec_path)
        try:
            with np.load(npz_path) as z:
                seg_ids = sorted(set(int(k[3:].split("_")[0]) for k in z.files if k.startswith("seg")))
                for i in seg_ids:
                    tkey, gkey = f"seg{i}_true", f"seg{i}_gen"
                    if tkey not in z.files or gkey not in z.files:
                        continue
                    true_full = z[tkey].astype(np.float32)
                    gen_full = z[gkey].astype(np.float32)
                    true_gen = true_full[SEED_SAMPLES:]
                    pred_gen = gen_full[SEED_SAMPLES:]
                    if len(true_gen) < GEN_SAMPLES:
                        continue
                    per_second = compute_per_second_metrics(true_gen, pred_gen)

                    if not args.no_hr:
                        n_windows = len(per_second["MAE"])
                        r_real = PU.detect_r_peaks(true_full, FS)
                        r_gen = PU.detect_r_peaks(gen_full, FS)
                        per_second["HR"] = PU.hr_error_per_second(
                            r_real, r_gen, FS, n_windows,
                            seed_sec=SEED_SAMPLES // FS, gen_sec=GEN_SAMPLES // FS,
                        )

                    entry = {
                        "subject_id": subj_id,
                        "record_path": rec_path,
                        "segment_index": i,
                        "length_sec": float(len(true_full)) / FS,
                        "gen_length_sec": float((len(true_gen) // GEN_SAMPLES) * GEN_SAMPLES) / FS,
                        "per_second": per_second,
                    }
                    entry.update(old_meta.get((rec_path, i), {}))
                    segments.append(entry)
            n_records += 1
        except Exception as e:
            print(f"[WARN] Errore su {npz_path}: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"segments": segments}, f, indent=2)
    os.replace(tmp, args.out)

    print(f"✅ Ricostruito {args.out}")
    print(f"   Record: {n_records} | Segmenti: {len(segments)} (formato per-secondo)")


if __name__ == "__main__":
    main()
