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
import neurokit2 as nk
from tqdm import tqdm
from scipy.stats import pearsonr, wasserstein_distance
from scipy.signal import butter, filtfilt

# --- GESTIONE PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mimic_generation_PINN.model_factory import ModelFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_quality import compute_quality_mask, quality_fraction  # noqa: E402

# --- CONFIGURAZIONI ---
DATA_DIR = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_evaluation")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "drift_test_manifest.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_performance_results.json")

# Directory di salvataggio dei segnali generati: UN SOLO FILE per record,
# generato fino all'orizzonte piu' lungo che quel record supporta. Essendo
# la generazione autoregressiva deterministica (nessuna randomness), gli
# orizzonti piu' corti sono semplicemente prefissi di questa unica run:
# le metriche per ciascun orizzonte vengono quindi ricavate per slicing,
# senza dover rigenerare da zero (fino a 6x meno calcolo GPU).
GENERATION_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_generation")

# Inserisci il path del tuo modello addestrato
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "src", "experiments", "final_mimic_pinn_results", "MAE_loss", "lightweight_hybrid_20260608_192241", "final_full_model", "best_lightweight_hybrid.pth")

FS = 125
EVAL_WINDOW_SEC = 10  # Per le metriche del Gruppo 1
EVAL_SAMPLES = EVAL_WINDOW_SEC * FS
HORIZONS_ORDER = ["1m", "30m", "1h", "6h", "12h", "24h"]


def _minmax_norm(signal: np.ndarray) -> np.ndarray:
    """Normalizza [0, 1]"""
    mn, mx = signal.min(), signal.max()
    return (signal - mn) / (mx - mn + 1e-8)


def apply_bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    clean_sig = np.nan_to_num(signal, nan=np.nanmean(signal))
    return filtfilt(b, a, clean_sig)


# =====================================================================
# METRICHE GRUPPO 1: Sulla singola finestra all'orizzonte (10 secondi)
# =====================================================================
def calculate_advanced_metrics_group1(true_10s, pred_10s):
    metrics = {}
    metrics['MAE_10s'] = float(np.mean(np.abs(true_10s - pred_10s)))

    if np.std(true_10s) < 1e-6 or np.std(pred_10s) < 1e-6:
        metrics['Pearson_10s'] = 0.0
    else:
        p_val, _ = pearsonr(true_10s, pred_10s)
        metrics['Pearson_10s'] = float(p_val)

    metrics['Wasserstein_10s'] = float(wasserstein_distance(true_10s, pred_10s))

    try:
        se_true, _ = nk.entropy_sample(true_10s, delay=1, dimension=2)
        se_pred, _ = nk.entropy_sample(pred_10s, delay=1, dimension=2)
        metrics['SampEn_Error_10s'] = float(abs(se_true - se_pred))
    except Exception:
        metrics['SampEn_Error_10s'] = None

    try:
        lz_true, _ = nk.complexity_lempelziv(true_10s)
        lz_pred, _ = nk.complexity_lempelziv(pred_10s)
        metrics['LZ_Complexity_Error_10s'] = float(abs(lz_true - lz_pred))
    except Exception:
        metrics['LZ_Complexity_Error_10s'] = None

    return metrics


# =====================================================================
# METRICHE GRUPPI 2 & 3: Sull'intero segnale generato (Cumulativo)
# =====================================================================
def calculate_clinical_and_cumulative_metrics(true_full, pred_full, fs):
    metrics = {}

    t_norm = _minmax_norm(true_full)
    p_norm = _minmax_norm(pred_full)

    metrics['Cumulative_RMSD'] = float(np.sqrt(np.mean((t_norm - p_norm) ** 2)))

    dot_product = np.dot(t_norm, p_norm)
    norm_t = np.linalg.norm(t_norm) + 1e-8
    norm_p = np.linalg.norm(p_norm) + 1e-8
    metrics['Cumulative_CosineSim'] = float(dot_product / (norm_t * norm_p))

    def extract_ecg_features(signal):
        feats = {'HRV_RMSSD': None, 'QRS_ms': None, 'PR_ms': None, 'QT_ms': None}
        try:
            _, info = nk.ecg_peaks(signal, sampling_rate=fs)
            rpeaks = info["ECG_R_Peaks"]

            if len(rpeaks) > 3:
                hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=fs)
                if 'HRV_RMSSD' in hrv_metrics.columns:
                    feats['HRV_RMSSD'] = float(hrv_metrics['HRV_RMSSD'].iloc[0])

                _, waves = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="dwt")

                if 'ECG_R_Onsets' in waves and 'ECG_R_Offsets' in waves:
                    onsets = np.array(waves['ECG_R_Onsets'])
                    offsets = np.array(waves['ECG_R_Offsets'])
                    valid = ~np.isnan(onsets) & ~np.isnan(offsets)
                    if np.any(valid):
                        feats['QRS_ms'] = float(np.nanmean(offsets[valid] - onsets[valid]) / fs * 1000)

                if 'ECG_P_Onsets' in waves and 'ECG_R_Onsets' in waves:
                    p_on = np.array(waves['ECG_P_Onsets'])
                    r_on = np.array(waves['ECG_R_Onsets'])
                    valid = ~np.isnan(p_on) & ~np.isnan(r_on)
                    if np.any(valid):
                        feats['PR_ms'] = float(np.nanmean(r_on[valid] - p_on[valid]) / fs * 1000)

                if 'ECG_R_Onsets' in waves and 'ECG_T_Offsets' in waves:
                    r_on = np.array(waves['ECG_R_Onsets'])
                    t_off = np.array(waves['ECG_T_Offsets'])
                    valid = ~np.isnan(r_on) & ~np.isnan(t_off)
                    if np.any(valid):
                        feats['QT_ms'] = float(np.nanmean(t_off[valid] - r_on[valid]) / fs * 1000)
        except Exception:
            pass
        return feats

    true_feats = extract_ecg_features(true_full)
    pred_feats = extract_ecg_features(pred_full)

    for k in true_feats.keys():
        if true_feats[k] is not None and pred_feats[k] is not None:
            metrics[f'Error_Cumulative_{k}'] = abs(true_feats[k] - pred_feats[k])
        else:
            metrics[f'Error_Cumulative_{k}'] = None

    return metrics


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

    return gen_ecg


def build_unique_records(manifest):
    """Deduplica il manifest per record_path e determina, per ciascun record
    unico, tutti gli orizzonti che supporta e quello massimo (piu' lungo)."""
    records = {}
    for h_label, entries in manifest.items():
        for e in entries:
            key = e['record_path']
            if key not in records:
                records[key] = {
                    "subject_id": e['subject_id'],
                    "record_path": key,
                    "horizons": {},
                }
            records[key]["horizons"][h_label] = e['end_sec']

    for rec in records.values():
        max_label = max(rec["horizons"], key=lambda h: rec["horizons"][h])
        rec["max_horizon_label"] = max_label
        rec["max_end_sec"] = rec["horizons"][max_label]

    # Ordiniamo per durata massima crescente: i record piu' brevi (quindi
    # piu' veloci da generare) vengono completati per primi, garantendo
    # una copertura ampia del test set anche se il job viene interrotto
    # prima di finire i record piu' lunghi (6h/12h/24h).
    return sorted(records.values(), key=lambda r: r["max_end_sec"])


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


def compute_metrics_for_horizon(ecg_f, gen_ecg_full, h_end_sec, quality_mask=None):
    """Estrae le metriche Gruppo 1/2/3 per un dato orizzonte come slice
    della generazione (potenzialmente piu' lunga) gia' calcolata.

    Se quality_mask e' fornita (stessa risoluzione di campione di ecg_f,
    vedi signal_quality.compute_quality_mask), aggiunge anche la frazione di
    ground truth "pulito" secondo lo stesso criterio SQI usato per i dataset
    di training: la pipeline di drift, a differenza del resto del progetto,
    pesca il segnale WFDB grezzo e continuo (nessun filtro di qualita' a
    monte), quindi va segnalato quanto di ogni orizzonte e' effettivamente
    un riferimento affidabile per giudicare il drift del modello.
    """
    n_samples = int(round(h_end_sec * FS))
    n_samples = min(n_samples, len(ecg_f), len(gen_ecg_full))

    true_10s_norm = _minmax_norm(ecg_f[n_samples - EVAL_SAMPLES: n_samples])
    pred_10s_norm = gen_ecg_full[n_samples - EVAL_SAMPLES: n_samples]
    metrics_g1 = calculate_advanced_metrics_group1(true_10s_norm, pred_10s_norm)
    metrics_g23 = calculate_clinical_and_cumulative_metrics(ecg_f[:n_samples], gen_ecg_full[:n_samples], FS)
    metrics = {**metrics_g1, **metrics_g23}

    if quality_mask is not None:
        metrics['Quality_Valid_Frac_10s'] = quality_fraction(quality_mask, n_samples - EVAL_SAMPLES, n_samples)
        metrics['Quality_Valid_Frac_Cumulative'] = quality_fraction(quality_mask, 0, n_samples)

    return metrics


RESULTS_LOCK_PATH = RESULTS_PATH + ".lock"


def _read_results_snapshot():
    """Lettura non lockata (best-effort) usata solo per decidere se un record
    va (ri)processato: eventuali race vengono comunque risolte alla scrittura,
    che e' protetta da lock esclusivo tramite _atomic_update_results."""
    if not os.path.exists(RESULTS_PATH):
        return {h: [] for h in HORIZONS_ORDER}
    try:
        with open(RESULTS_PATH, 'r') as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, OSError):
        all_results = {}
    for h in HORIZONS_ORDER:
        all_results.setdefault(h, [])
    return all_results


def _atomic_update_results(update_fn):
    """Applica update_fn(all_results) -> bool (changed) sotto lock esclusivo
    a livello di file, cosi' piu' processi worker possono scrivere in
    sicurezza sullo stesso RESULTS_PATH condiviso. Scrittura atomica via
    file temporaneo + os.replace."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_LOCK_PATH, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            all_results = _read_results_snapshot()
            changed = update_fn(all_results)
            if changed:
                tmp_path = RESULTS_PATH + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(all_results, f, indent=4)
                os.replace(tmp_path, RESULTS_PATH)
            return all_results
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def process_one_record(rec, model, device, configs, log=print):
    """Genera (se necessario) e valuta un singolo record. Sicuro da chiamare
    concorrentemente da piu' processi worker: la generazione/lettura del
    .npz e il calcolo metriche avvengono senza lock (ognuno lavora su un
    record diverso, nessuna scrittura condivisa), mentre l'unica sezione
    critica (append dei risultati su RESULTS_PATH) e' protetta da lock."""
    rec_path = rec["record_path"]
    subj_id = rec["subject_id"]
    max_end_sec = rec["max_end_sec"]

    snapshot = _read_results_snapshot()
    processed_by_horizon = {h: {r['record_path'] for r in snapshot[h]} for h in HORIZONS_ORDER}
    missing_horizons = [h for h in rec["horizons"] if rec_path not in processed_by_horizon[h]]

    npz_path = os.path.join(GENERATION_DIR, rec_path + ".npz")
    npz_exists_and_complete = os.path.exists(npz_path)
    if npz_exists_and_complete:
        try:
            with np.load(npz_path) as cached:
                npz_exists_and_complete = len(cached["ecg_generated"]) >= int(max_end_sec * FS)
        except Exception:
            npz_exists_and_complete = False

    if not missing_horizons and npz_exists_and_complete:
        return  # Tutto gia' fatto per questo record

    # --- A. Caricamento Dati / Generazione (solo se serve) ---
    if npz_exists_and_complete:
        with np.load(npz_path) as cached:
            ecg_f = cached["ecg_target"].astype(np.float32)
            gen_ecg_full = cached["ecg_generated"].astype(np.float32)
            ppg_f = cached["ppg_input"].astype(np.float32)
    else:
        try:
            total_req_samples = int(max_end_sec * FS)
            ecg_f, ppg_f = load_record_signals(rec_path, total_req_samples)
        except Exception as e:
            log(f"Errore lettura {rec_path}: {e}")
            return

        seed_samples = 6 * FS
        ecg_seed = ecg_f[:seed_samples]
        t0 = time.time()
        gen_ecg_full = fast_autoregressive_inference(model, ppg_f, ecg_seed, configs, device)
        elapsed = time.time() - t0
        log(f"[{device}] {rec_path}: generati {max_end_sec:.0f}s in {elapsed:.1f}s "
            f"({max_end_sec / max(elapsed, 1e-6):.1f}x realtime)")

        save_dir_path = os.path.join(GENERATION_DIR, os.path.dirname(rec_path))
        os.makedirs(save_dir_path, exist_ok=True)
        np.savez_compressed(
            npz_path,
            ppg_input=ppg_f.astype(np.float32),
            ecg_target=ecg_f.astype(np.float32),
            ecg_generated=gen_ecg_full.astype(np.float32),
            fs=FS,
            max_end_sec=max_end_sec,
        )

    # --- B. Calcolo metriche per ogni orizzonte ancora mancante (fuori dal lock: puo' essere lento) ---
    new_metrics_by_horizon = {}
    remaining_horizons = [h for h in rec["horizons"] if rec_path not in processed_by_horizon[h]]
    quality_mask = compute_quality_mask(ecg_f.astype(np.float64), ppg_f.astype(np.float64), FS) if remaining_horizons else None
    for h_label, h_end_sec in sorted(rec["horizons"].items(), key=lambda kv: kv[1]):
        if rec_path in processed_by_horizon[h_label]:
            continue
        new_metrics_by_horizon[h_label] = compute_metrics_for_horizon(ecg_f, gen_ecg_full, h_end_sec, quality_mask)

    if not new_metrics_by_horizon:
        return

    # --- C. Scrittura atomica lock-protetta (ricontrolla sotto lock per sicurezza) ---
    def _update(all_results):
        changed = False
        existing = {h: {r['record_path'] for r in all_results[h]} for h in HORIZONS_ORDER}
        for h_label, full_metrics in new_metrics_by_horizon.items():
            if rec_path in existing[h_label]:
                continue
            all_results[h_label].append({
                "subject_id": subj_id,
                "record_path": rec_path,
                "metrics": full_metrics,
            })
            changed = True
        return changed

    _atomic_update_results(_update)


def gpu_worker(rank, device_str, model_weights_path, configs, task_queue, progress_queue):
    device = torch.device(device_str)
    model = ModelFactory.get_model(configs).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    def log(msg):
        print(f"[worker{rank}] {msg}", flush=True)

    log(f"pronto su {device_str}")
    while True:
        # get() bloccante (non get_nowait): il productor (main process) puo'
        # ancora star riempiendo la coda in background (feeder thread) quando
        # i worker partono, get_nowait() vedrebbe erroneamente la coda vuota.
        # Un sentinel None per worker segnala la fine effettiva del lavoro.
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
    """Determina la lista di device su cui far girare i worker: una GPU per
    worker se disponibili (torch.cuda.device_count()), altrimenti un solo
    worker su CPU. --n_workers permette di forzare il numero di processi."""
    try:
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    if n_gpus > 0:
        devices = [f"cuda:{i}" for i in range(n_gpus)]
    else:
        devices = ["cpu"]

    if n_workers_arg is not None:
        if n_workers_arg <= len(devices):
            devices = devices[:n_workers_arg]
        elif devices == ["cpu"]:
            devices = ["cpu"] * n_workers_arg  # piu' processi CPU concorrenti
        else:
            print(f"[WARN] --n_workers={n_workers_arg} > GPU disponibili ({len(devices)}); uso {len(devices)} worker.")

    return devices


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoregressive Drift - Generazione + valutazione (Sezione 4.4.2)")
    parser.add_argument("--n_workers", type=int, default=None,
                         help="Numero di processi worker paralleli (default: 1 per GPU disponibile, o 1 su CPU).")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 2: FULL AUTOREGRESSIVE DRIFT EVALUATION & SIGNAL SAVING")
    print(" (generazione UNICA per record fino all'orizzonte massimo,")
    print("  parallelizzata su piu' GPU/processi)")
    print("=" * 60)

    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERRORE] Manifest non trovato. Esegui prima build_drift_manifest.py.")
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
        'target_len': 125
    }

    devices = _resolve_devices(args.n_workers)
    print(f"-> Device worker: {devices}")
    print(f"-> Pesi modello: {MODEL_WEIGHTS_PATH}")

    unique_records = build_unique_records(manifest)
    total_gen_seconds_needed = sum(r["max_end_sec"] for r in unique_records)
    print(f"-> Record unici nel manifest: {len(unique_records)}")
    print(f"-> Totale secondi di ECG da generare (somma su tutti i record): {total_gen_seconds_needed / 3600:.1f}h "
          f"(su {len(devices)} worker in parallelo)")

    snapshot = _read_results_snapshot()
    processed_by_horizon = {h: {r['record_path'] for r in snapshot[h]} for h in HORIZONS_ORDER}

    def _is_pending(rec):
        rec_path = rec["record_path"]
        if any(rec_path not in processed_by_horizon[h] for h in rec["horizons"]):
            return True
        npz_path = os.path.join(GENERATION_DIR, rec_path + ".npz")
        if not os.path.exists(npz_path):
            return True
        try:
            with np.load(npz_path) as cached:
                return len(cached["ecg_generated"]) < int(rec["max_end_sec"] * FS)
        except Exception:
            return True

    pending_records = [r for r in unique_records if _is_pending(r)]
    print(f"-> Record da (ri)processare in questa esecuzione: {len(pending_records)}/{len(unique_records)}")

    if not pending_records:
        print("✅ Nessun record da fare: la valutazione e' gia' completa al 100%.")
        return

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    for rec in pending_records:
        task_queue.put(rec)
    for _ in devices:
        task_queue.put(None)  # sentinel di fine-lavoro, uno per worker

    processes = []
    for rank, device_str in enumerate(devices):
        p = ctx.Process(
            target=gpu_worker,
            args=(rank, device_str, MODEL_WEIGHTS_PATH, configs, task_queue, progress_queue),
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
        # Alla terminazione anticipata la task_queue puo' avere centinaia di
        # record ancora non consumati: senza cancel_join_thread() l'atexit
        # hook di multiprocessing prova a flushare quel buffer e il processo
        # principale resta appeso indefinitamente invece di uscire.
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
    final_snapshot = _read_results_snapshot()
    final_processed_by_horizon = {h: {r['record_path'] for r in final_snapshot[h]} for h in HORIZONS_ORDER}
    n_done = sum(1 for r in unique_records if r["record_path"] in final_processed_by_horizon[r["max_horizon_label"]])
    if n_done < len(unique_records):
        print(f"⚠️  Job terminato (probabilmente per il time-limit dello scheduler): "
              f"{n_done}/{len(unique_records)} record completati.")
        print("    Rilancia lo stesso script (es. tramite ar_drif_test.sh): riprendera' da dove si e' interrotto.")
    else:
        print(f"✅ VALUTAZIONE DRIFT E SALVATAGGIO SEGNALI COMPLETATI AL 100%!")
    print(f"Risultati metriche in: {RESULTS_PATH}")
    print(f"Segnali completi in: {GENERATION_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
