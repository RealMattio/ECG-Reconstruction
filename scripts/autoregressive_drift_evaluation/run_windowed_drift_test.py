"""
Valutazione del drift autoregressivo — PARADIGMA A FINESTRE (senza splicing).

Motivazione (vedi anche windowed_signal_quality.py):
  L'approccio precedente (run_autoregressive_drift_test_clean.py) ripuliva il
  segnale RIMUOVENDO i tratti rumorosi e INCOLLANDO quelli puliti in un unico
  segnale continuo. Ogni giunzione introduce pero' un gradino artificiale e
  altera la frequenza cardiaca: rumore che il modello non ha mai visto.

Nuovo approccio:
  1. Per ogni record si legge il segnale grezzo (PPG + ECG), lo si filtra e lo
     si scandisce in finestre di 4s valutando l'SQI (windowed_signal_quality).
     - La PPG deve essere ESTREMAMENTE pulita su OGNI finestra (soglia > training).
     - L'ECG e' giudicato SOLO sui primi 6s (il seed reale dato al modello).
     Si conservano INTERI i tratti in cui piu' finestre consecutive superano
     l'SQI: ne risultano tanti SEGMENTI contigui di lunghezza diversa (>= 7s),
     senza alcuna giunzione.
  2. Su OGNI segmento si esegue la generazione autoregressiva in modo
     indipendente (seed = primi 6s di ECG reale, PPG del segmento come input).
     Non si incolla nulla: cio' che non supera l'SQI viene semplicemente
     scartato.
  3. Per ogni segmento si calcolano MAE, RMSE, Pearson r e Cosine similarity
     sulla SOLA porzione GENERATA (da 6s in poi). Ogni segmento produce un
     punto (lunghezza_segmento -> metrica): al crescere della lunghezza ci si
     aspetta un drift crescente (MAE/RMSE su, correlazioni giu').

Output:
  - drift_windowed_results.json : lista piatta di segmenti con le metriche.
  - (opzionale) un .npz per record con i segnali dei segmenti, per ispezione.

Lo script e' RESUMABLE e multi-GPU come il precedente: ogni record completato
viene salvato subito (lock atomico), quindi e' sicuro rilanciarlo dopo un
timeout dello scheduler.
"""
import os
import sys
import json
import time
import signal
import fcntl
import queue as std_queue
import multiprocessing as mp
import torch
import wfdb
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt

# --- GESTIONE PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mimic_generation_PINN.model_factory import ModelFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windowed_signal_quality import (  # noqa: E402
    find_clean_segments,
    DEFAULT_WIN_SEC, DEFAULT_STEP_SEC, DEFAULT_PPG_THR, DEFAULT_ECG_THR,
    DEFAULT_SEED_SEC, DEFAULT_MIN_SEG_SEC,
)

# --- CONFIGURAZIONI ---
DATA_DIR = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data")
# Riusiamo il manifest esistente solo come ELENCO dei record del test set e
# della loro durata grezza disponibile: la segmentazione SQI viene ricalcolata.
LEGACY_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_evaluation")
MANIFEST_PATH = os.path.join(LEGACY_OUTPUT_DIR, "drift_test_manifest.json")

# Le performance di ogni modello vengono salvate in una sotto-cartella
# nominata come la cartella in cui il modello stesso e' salvato (es.
# "lightweight_hybrid_20260608_192241"), cosi' da poter confrontare piu'
# modelli senza che i risultati si sovrascrivano a vicenda. I path effettivi
# vengono risolti da resolve_paths() in base a --model_weights_path e settati
# come globali (vedi main()/gpu_worker(): con multiprocessing "spawn" ogni
# processo figlio re-importa il modulo da zero, quindi i path vanno
# ripropagati esplicitamente in ciascun processo, non solo nel padre).
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_evaluation_windowed")
BASE_GENERATION_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_generation_windowed")

DEFAULT_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "src", "experiments", "final_mimic_pinn_results", "MAE_loss", "lightweight_hybrid_20260608_192241", "final_full_model", "best_lightweight_hybrid.pth")

FS = 125
SEED_SEC = int(DEFAULT_SEED_SEC)          # 6s di seed reale
SAVE_GENERATIONS = True                    # salva i segnali dei segmenti (.npz) per ispezione

# Placeholder module-level (sovrascritti da resolve_paths()/set_paths() prima
# di qualunque uso reale, sia nel processo padre che in ciascun worker).
MODEL_WEIGHTS_PATH = DEFAULT_MODEL_WEIGHTS_PATH
MODEL_ID = None
OUTPUT_DIR = BASE_OUTPUT_DIR
GENERATION_DIR = BASE_GENERATION_DIR
RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_windowed_results.json")
RESULTS_LOCK_PATH = RESULTS_PATH + ".lock"
DISCARDED_PATH = os.path.join(OUTPUT_DIR, "discarded_records.json")
DISCARDED_LOCK_PATH = DISCARDED_PATH + ".lock"


def derive_model_id(model_weights_path):
    """'.../MAE_loss/lightweight_hybrid_20260608_192241/final_full_model/best_x.pth'
    -> 'lightweight_hybrid_20260608_192241' (la cartella in cui il modello e'
    salvato, due livelli sopra il file .pth)."""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(model_weights_path))))


def resolve_paths(model_weights_path):
    model_id = derive_model_id(model_weights_path)
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_id)
    generation_dir = os.path.join(BASE_GENERATION_DIR, model_id)
    results_path = os.path.join(output_dir, "drift_windowed_results.json")
    discarded_path = os.path.join(output_dir, "discarded_records.json")
    return {
        "MODEL_WEIGHTS_PATH": model_weights_path,
        "MODEL_ID": model_id,
        "OUTPUT_DIR": output_dir,
        "GENERATION_DIR": generation_dir,
        "RESULTS_PATH": results_path,
        "RESULTS_LOCK_PATH": results_path + ".lock",
        "DISCARDED_PATH": discarded_path,
        "DISCARDED_LOCK_PATH": discarded_path + ".lock",
    }


def set_paths(paths):
    """Applica un dict di resolve_paths() come globali di questo modulo, nel
    processo CORRENTE (va richiamata sia nel padre che in ogni worker)."""
    global MODEL_WEIGHTS_PATH, MODEL_ID, OUTPUT_DIR, GENERATION_DIR
    global RESULTS_PATH, RESULTS_LOCK_PATH, DISCARDED_PATH, DISCARDED_LOCK_PATH
    MODEL_WEIGHTS_PATH = paths["MODEL_WEIGHTS_PATH"]
    MODEL_ID = paths["MODEL_ID"]
    OUTPUT_DIR = paths["OUTPUT_DIR"]
    GENERATION_DIR = paths["GENERATION_DIR"]
    RESULTS_PATH = paths["RESULTS_PATH"]
    RESULTS_LOCK_PATH = paths["RESULTS_LOCK_PATH"]
    DISCARDED_PATH = paths["DISCARDED_PATH"]
    DISCARDED_LOCK_PATH = paths["DISCARDED_LOCK_PATH"]


def _minmax_norm(signal: np.ndarray) -> np.ndarray:
    mn, mx = signal.min(), signal.max()
    return (signal - mn) / (mx - mn + 1e-8)


def apply_bandpass_filter(sig, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
    clean_sig = np.nan_to_num(sig, nan=np.nanmean(sig))
    return filtfilt(b, a, clean_sig)


# =====================================================================
# GENERAZIONE AUTOREGRESSIVA (identica al paradigma precedente, per segmento)
# =====================================================================
def fast_autoregressive_inference(model, ppg_full, ecg_seed, configs, device):
    model.eval()
    fs = configs['target_fs']
    win_samples = int(configs['x_sec'] * fs)
    gen_samples = int(configs['gen_sec'] * fs)
    seed_samples = win_samples - gen_samples

    total_samples = len(ppg_full)
    gen_ecg = np.zeros(total_samples, dtype=np.float32)
    gen_ecg[:seed_samples] = _minmax_norm(ecg_seed)

    with torch.no_grad():
        for cursor in range(seed_samples, total_samples - gen_samples + 1, gen_samples):
            start_win = cursor - seed_samples
            end_win = cursor + gen_samples

            curr_ppg = ppg_full[start_win: end_win].copy()
            curr_ppg -= curr_ppg.mean()
            if curr_ppg.max() < -curr_ppg.min():
                curr_ppg = -curr_ppg
            curr_ppg_norm = _minmax_norm(curr_ppg)

            ppg_diff = np.zeros_like(curr_ppg_norm)
            ppg_diff[1:] = curr_ppg_norm[1:] - curr_ppg_norm[:-1]

            ecg_real_part = gen_ecg[start_win: cursor]
            padding = np.full((gen_samples,), ecg_real_part[-1])
            curr_ecg_past = np.concatenate([ecg_real_part, padding])

            ppg_t = torch.tensor(curr_ppg_norm, dtype=torch.float32)
            ppg_diff_t = torch.tensor(ppg_diff, dtype=torch.float32)
            ecg_p_t = torch.tensor(curr_ecg_past, dtype=torch.float32)

            X = torch.stack([ppg_t, ppg_diff_t, ecg_p_t], dim=0).unsqueeze(0).to(device)
            gen_ecg[cursor: end_win] = model(X).cpu().squeeze().numpy()

    # Ultimo campione realmente generato (multiplo intero di gen_samples oltre il seed)
    n_gen_windows = (total_samples - seed_samples) // gen_samples
    last_generated = seed_samples + n_gen_windows * gen_samples
    return gen_ecg, seed_samples, last_generated


# =====================================================================
# METRICHE per segmento — PROFILO SECONDO-PER-SECONDO sulla porzione GENERATA
# =====================================================================
# Per ogni segmento si calcola l'errore su OGNI singolo secondo generato:
#   indice 0 -> 1o secondo generato (intervallo [6s, 7s], asse X = 7s)
#   indice 1 -> 2o secondo generato (intervallo [7s, 8s], asse X = 8s)
#   indice j -> intervallo [6+j, 7+j], asse X = 7 + j
# In fase di plotting, a ciascuna posizione X si aggregano (media +/- std) i
# valori di TUTTI i segmenti che raggiungono quel secondo (i piu' lunghi
# contribuiscono anche alle X piu' grandi): il conteggio di segmenti diminuisce
# man mano che X cresce. La normalizzazione min-max del ground truth e' fatta
# UNA volta sull'intera porzione generata (non per-secondo), cosi' un eventuale
# drift di ampiezza del modello resta visibile invece di essere riassorbito.
def compute_per_second_metrics(true_gen: np.ndarray, pred_gen: np.ndarray, gen_samples: int) -> dict:
    """Ritorna {metric: [v_sec7, v_sec8, ...]} con un valore per secondo generato.
    true_gen/pred_gen coprono la sola porzione generata (multiplo di gen_samples).
    Vettorizzata (equivalente numericamente a scipy.pearsonr per finestra ma
    senza loop Python)."""
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

    tc = T - T.mean(axis=1, keepdims=True)
    pc = P - P.mean(axis=1, keepdims=True)
    num = np.sum(tc * pc, axis=1)
    den = np.sqrt(np.sum(tc ** 2, axis=1) * np.sum(pc ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.where(den > 1e-12, num / den, 0.0)

    # cosine su entrambe le fette min-max normalizzate (valore limitato e
    # confrontabile fra secondi, come Cumulative_CosineSim del vecchio codice)
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


# =====================================================================
# ELENCO RECORD (dal manifest legacy)
# =====================================================================
def build_unique_records(manifest):
    """Deduplica il manifest per record_path e ricava quanto segnale GREZZO
    leggere per ciascun record (la durata massima raggiunta tra gli orizzonti).
    La segmentazione SQI effettiva viene poi ricalcolata da find_clean_segments."""
    records = {}
    for entries in manifest.values():
        for e in entries:
            key = e['record_path']
            if key not in records:
                records[key] = {"subject_id": e['subject_id'], "record_path": key, "max_raw_end_sec": 0.0}
            records[key]["max_raw_end_sec"] = max(records[key]["max_raw_end_sec"], float(e['end_sec']))
    return sorted(records.values(), key=lambda r: r["max_raw_end_sec"])


def load_record_signals(record_path, total_req_samples):
    abs_rec_path = os.path.join(DATA_DIR, record_path)
    record = wfdb.rdrecord(abs_rec_path)
    idx_ecg = record.sig_name.index('II')
    idx_ppg = record.sig_name.index('PLETH')

    ecg_raw = record.p_signal[:total_req_samples, idx_ecg]
    ppg_raw = record.p_signal[:total_req_samples, idx_ppg]

    ecg_f = apply_bandpass_filter(ecg_raw, FS, 0.5, 40.0)
    ppg_f = apply_bandpass_filter(ppg_raw, FS, 0.5, 5.0)
    return ecg_f, ppg_f


# =====================================================================
# PERSISTENZA (lock atomico, resumabile)
# =====================================================================
def _read_results_snapshot():
    if not os.path.exists(RESULTS_PATH):
        return {"segments": []}
    try:
        with open(RESULTS_PATH, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}
    data.setdefault("segments", [])
    return data


def _atomic_append_segments(rec_path, new_segments):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_LOCK_PATH, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            data = _read_results_snapshot()
            already = {s["record_path"] for s in data["segments"]}
            if rec_path in already:
                return  # gia' salvato (doppia esecuzione): non duplicare
            data["segments"].extend(new_segments)
            tmp_path = RESULTS_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, RESULTS_PATH)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def _read_discarded_snapshot():
    if not os.path.exists(DISCARDED_PATH):
        return {}
    try:
        with open(DISCARDED_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _mark_discarded(rec_path, reason, **details):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DISCARDED_LOCK_PATH, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            discarded = _read_discarded_snapshot()
            discarded[rec_path] = {"reason": reason, **details}
            tmp_path = DISCARDED_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(discarded, f, indent=2)
            os.replace(tmp_path, DISCARDED_PATH)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


# =====================================================================
# ELABORAZIONE DI UN RECORD
# =====================================================================
def process_one_record(rec, model, device, configs, log=print):
    rec_path = rec["record_path"]
    subj_id = rec["subject_id"]

    try:
        total_req_samples = int(rec["max_raw_end_sec"] * FS)
        ecg_f, ppg_f = load_record_signals(rec_path, total_req_samples)
    except Exception as e:
        log(f"Errore lettura {rec_path}: {e}")
        _mark_discarded(rec_path, "read_error", error=str(e))
        return

    segments, seg_stats = find_clean_segments(ecg_f, ppg_f, FS)
    if not segments:
        log(f"[{device}] {rec_path}: nessun segmento pulito (PPG ok su "
            f"{seg_stats['n_ppg_windows_ok']} finestre, {seg_stats['n_ppg_intervals']} intervalli, "
            f"{seg_stats['n_seg_too_short']} troppo corti, {seg_stats['n_seg_bad_ecg_seed']} seed ECG scarso).")
        _mark_discarded(rec_path, "no_valid_segments", **seg_stats)
        return

    seg_results = []
    npz_arrays = {}
    t0 = time.time()
    for i, seg in enumerate(segments):
        ppg_seg = ppg_f[seg["start"]:seg["end"]].astype(np.float32)
        ecg_seg = ecg_f[seg["start"]:seg["end"]].astype(np.float32)

        seed_samples = SEED_SEC * FS
        gen_ecg, seed_s, last_gen = fast_autoregressive_inference(
            model, ppg_seg, ecg_seg[:seed_samples], configs, device
        )
        if last_gen <= seed_s:
            continue  # segmento troppo corto per generare anche solo 1s (non dovrebbe capitare)

        true_gen = ecg_seg[seed_s:last_gen]
        pred_gen = gen_ecg[seed_s:last_gen]
        gen_samples = int(configs['gen_sec'] * FS)
        per_second = compute_per_second_metrics(true_gen, pred_gen, gen_samples)

        used_len_samples = last_gen  # seed + porzione generata effettivamente usata
        seg_results.append({
            "subject_id": subj_id,
            "record_path": rec_path,
            "segment_index": i,
            "seg_start_sample": seg["start"],
            "seg_end_sample": seg["end"],
            "length_sec": float(used_len_samples) / FS,        # 7s .. lunghezza del segmento
            "gen_length_sec": float(last_gen - seed_s) / FS,   # durata realmente generata
            "seed_ecg_score": seg["seed_ecg_score"],
            "ppg_inverted": seg["ppg_inverted"],
            # per_second[m][j] = errore sul secondo generato j (asse X = 7 + j)
            "per_second": per_second,
        })

        if SAVE_GENERATIONS:
            npz_arrays[f"seg{i}_ppg"] = ppg_seg[:last_gen]
            npz_arrays[f"seg{i}_true"] = ecg_seg[:last_gen]
            npz_arrays[f"seg{i}_gen"] = gen_ecg[:last_gen]

    elapsed = time.time() - t0
    if not seg_results:
        _mark_discarded(rec_path, "no_generable_segments", **seg_stats)
        return

    lengths = [round(s["length_sec"]) for s in seg_results]
    log(f"[{device}] {rec_path}: {len(seg_results)} segmenti puliti "
        f"(lunghezze {lengths}s), generati in {elapsed:.1f}s")

    if SAVE_GENERATIONS and npz_arrays:
        save_dir_path = os.path.join(GENERATION_DIR, os.path.dirname(rec_path))
        os.makedirs(save_dir_path, exist_ok=True)
        np.savez_compressed(os.path.join(GENERATION_DIR, rec_path + ".npz"), fs=FS, **npz_arrays)

    _atomic_append_segments(rec_path, seg_results)


# =====================================================================
# WORKER MULTI-GPU
# =====================================================================
def gpu_worker(rank, device_str, model_weights_path, configs, task_queue, progress_queue, paths):
    # "spawn" re-importa il modulo da zero in questo processo: i path
    # namespaced-per-modello vanno riapplicati qui, non solo nel padre.
    set_paths(paths)

    device = torch.device(device_str)
    model = ModelFactory.get_model(configs).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    def log(msg):
        print(f"[worker{rank}] {msg}", flush=True)

    log(f"pronto su {device_str}")
    while True:
        rec = task_queue.get()
        if rec is None:
            break
        try:
            process_one_record(rec, model, device, configs, log=log)
        except Exception as e:
            log(f"ERRORE non gestito su {rec['record_path']}: {e}")
        finally:
            progress_queue.put(1)

    log("nessun altro record da processare, termino.")


def _resolve_devices(n_workers_arg):
    try:
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]

    if n_workers_arg is not None:
        if n_workers_arg <= len(devices):
            devices = devices[:n_workers_arg]
        elif devices == ["cpu"]:
            devices = ["cpu"] * n_workers_arg
        else:
            print(f"[WARN] --n_workers={n_workers_arg} > GPU disponibili ({len(devices)}); uso {len(devices)} worker.")
    return devices


def _pending_records(unique_records):
    data = _read_results_snapshot()
    processed = {s["record_path"] for s in data["segments"]}
    discarded = set(_read_discarded_snapshot().keys())
    done = processed | discarded
    return [r for r in unique_records if r["record_path"] not in done], len(processed), len(discarded)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoregressive Drift — paradigma a finestre (no splicing)")
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--model_weights_path", type=str, default=DEFAULT_MODEL_WEIGHTS_PATH,
                         help="Pesi del modello da valutare. I risultati vengono salvati in una sotto-cartella "
                              "nominata come la cartella del modello (es. lightweight_hybrid_20260716_145739), "
                              "cosi' da poter confrontare piu' modelli senza sovrascriverne i risultati.")
    args = parser.parse_args()

    paths = resolve_paths(args.model_weights_path)
    set_paths(paths)

    print("=" * 60)
    print(" DRIFT EVALUATION — PARADIGMA A FINESTRE (SEGMENTI CONTIGUI, NO SPLICING)")
    print("=" * 60)
    print(f" SQI: PPG>={DEFAULT_PPG_THR} su finestre di {DEFAULT_WIN_SEC:.0f}s (passo {DEFAULT_STEP_SEC:.0f}s) | "
          f"ECG>={DEFAULT_ECG_THR} solo sul seed di {DEFAULT_SEED_SEC:.0f}s | segmento min {DEFAULT_MIN_SEG_SEC:.0f}s")
    print(f" Modello: {MODEL_ID}")
    print(f" Output:  {OUTPUT_DIR}")

    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERRORE] Manifest non trovato: {MANIFEST_PATH}. Esegui prima build_drift_manifest.py.")
        sys.exit(1)

    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    configs = {
        'model_type': 'lightweight_hybrid',
        'target_fs': FS,
        'x_sec': 7,
        'gen_sec': 1,
        'apply_wst': True,
        'input_channels': 3,
        'actual_seq_len': 875,
        'target_len': 125,
    }

    devices = _resolve_devices(args.n_workers)
    print(f"-> Device worker: {devices}")
    print(f"-> Pesi modello: {MODEL_WEIGHTS_PATH}")

    unique_records = build_unique_records(manifest)
    pending_records, n_processed, n_discarded = _pending_records(unique_records)
    print(f"-> Record unici nel manifest: {len(unique_records)}")
    print(f"-> Da (ri)processare: {len(pending_records)} "
          f"({n_processed} gia' con segmenti, {n_discarded} gia' scartati)")

    if not pending_records:
        print("✅ Nessun record da fare: la valutazione e' gia' completa.")
        return

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    for rec in pending_records:
        task_queue.put(rec)
    for _ in devices:
        task_queue.put(None)

    processes = []
    for rank, device_str in enumerate(devices):
        p = ctx.Process(
            target=gpu_worker,
            args=(rank, device_str, MODEL_WEIGHTS_PATH, configs, task_queue, progress_queue, paths),
            daemon=False,
        )
        p.start()
        processes.append(p)

    def _handle_sigterm(signum, frame):
        print(f"\n[SIGTERM] Segnale di terminazione ricevuto: chiudo i {len(processes)} worker...", flush=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=30)
        task_queue.cancel_join_thread()
        progress_queue.cancel_join_thread()
        print("[SIGTERM] Worker terminati, esco (i record gia' completati restano salvati).", flush=True)
        sys.stdout.flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    total = len(pending_records)
    completed = 0
    pbar = tqdm(total=total, desc="Record completati")
    while completed < total and any(p.is_alive() for p in processes):
        try:
            progress_queue.get(timeout=1.0)
            completed += 1
            pbar.update(1)
        except std_queue.Empty:
            continue
    pbar.close()

    for p in processes:
        p.join()

    print("\n" + "=" * 60)
    remaining, n_processed, n_discarded = _pending_records(unique_records)
    if remaining:
        print(f"⚠️  Job terminato (probabilmente per il time-limit dello scheduler): "
              f"{len(unique_records) - len(remaining)}/{len(unique_records)} record completati.")
        print("    Rilancia lo stesso script: riprendera' da dove si e' interrotto.")
    else:
        data = _read_results_snapshot()
        print(f"✅ VALUTAZIONE DRIFT (PARADIGMA A FINESTRE) COMPLETATA AL 100%!")
        print(f"   Segmenti totali raccolti: {len(data['segments'])}")
    print(f"Risultati metriche in: {RESULTS_PATH}")
    if SAVE_GENERATIONS:
        print(f"Segnali dei segmenti in: {GENERATION_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
