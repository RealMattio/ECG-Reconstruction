"""
Cambio di paradigma per la valutazione del drift (vedi run_autoregressive_drift_test.py
per la versione precedente, mantenuta come riferimento storico): invece di
generare sul segnale WFDB grezzo e poi pesare/escludere a posteriori le
finestre rumorose, qui il segnale viene ripulito PRIMA della generazione.

Per ogni record:
  1. Si carica il segnale grezzo (PPG + ECG) fino alla durata massima che il
     manifest indica disponibile per quel record (come nella pipeline
     precedente).
  2. Si calcola la maschera SQI (stesso criterio del resto del progetto,
     vedi signal_quality.py) e si rimuovono INTERAMENTE i tratti che non la
     superano, incollando i tratti puliti rimasti in un unico segnale piu'
     corto ma interamente pulito (signal_quality.splice_clean_signal).
  3. Gli orizzonti (1m/30m/1h/6h/12h/24h) vengono ridefiniti sulla durata del
     segnale PULITO risultante (non piu' sulla durata grezza originale): un
     record puo' quindi non raggiungere piu' gli stessi orizzonti di prima.
  4. La generazione autoregressiva (stessa fast_autoregressive_inference di
     prima) viene eseguita sul segnale pulito.
  5. Le metriche Gruppo 1 (finestra finale 10s) usano ora solo: RMSE, rRMSE,
     MAE, DTW, Pearson r. Le metriche Gruppo 2/3 (cumulative/cliniche)
     restano invariate.

Essendo il segnale pulito piu' corto di quello grezzo, ci si aspetta un
numero di record idonei per ciascun orizzonte inferiore a prima: e' una
conseguenza attesa e corretta del nuovo approccio, non un bug.
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
from fastdtw import fastdtw

# --- GESTIONE PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mimic_generation_PINN.model_factory import ModelFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_quality import splice_clean_signal  # noqa: E402

# --- CONFIGURAZIONI ---
DATA_DIR = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data")
# Riusiamo il manifest esistente (definisce solo quanto segnale GREZZO
# leggere per record; gli orizzonti effettivi vengono ridecisi dopo lo
# splicing, vedi sopra) — non serve rigenerarlo.
LEGACY_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_evaluation")
MANIFEST_PATH = os.path.join(LEGACY_OUTPUT_DIR, "drift_test_manifest.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_evaluation_clean")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_performance_results_clean.json")
GENERATION_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_generation_clean")

MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "src", "experiments", "final_mimic_pinn_results", "MAE_loss", "lightweight_hybrid_20260608_192241", "final_full_model", "best_lightweight_hybrid.pth")

FS = 125
EVAL_WINDOW_SEC = 10  # Per le metriche del Gruppo 1
EVAL_SAMPLES = EVAL_WINDOW_SEC * FS
SEED_SEC = 6
HORIZONS_ORDER = ["1m", "30m", "1h", "6h", "12h", "24h"]
DTW_RADIUS = 10  # fastdtw: piu' alto = piu' preciso ma piu' lento (10 e' un buon compromesso su 1250 campioni)


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
# METRICHE GRUPPO 1 (nuovo paradigma): RMSE, rRMSE, MAE, DTW, Pearson r
# =====================================================================
def calculate_group1_metrics(true_10s: np.ndarray, pred_10s: np.ndarray) -> dict:
    metrics = {}
    diff = true_10s - pred_10s

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    metrics['RMSE_10s'] = rmse

    # NOTA: true_10s arriva gia' min-max normalizzato a [0,1] (vedi
    # compute_metrics_for_horizon), quindi il suo range e' sempre ~1 per
    # costruzione: dividere per il range renderebbe rRMSE == RMSE, un
    # duplicato inutile. Normalizziamo invece per la RMS del segnale vero
    # (definizione standard di NRMSE), che non degenera con questo
    # preprocessing e resta comunque interpretabile come "errore relativo
    # all'energia tipica del segnale".
    true_rms = float(np.sqrt(np.mean(true_10s ** 2)))
    metrics['rRMSE_10s'] = float(rmse / (true_rms + 1e-8))

    metrics['MAE_10s'] = float(np.mean(np.abs(diff)))

    if np.std(true_10s) < 1e-6 or np.std(pred_10s) < 1e-6:
        metrics['Pearson_10s'] = 0.0
    else:
        p_val, _ = pearsonr(true_10s, pred_10s)
        metrics['Pearson_10s'] = float(p_val)

    dtw_dist, _ = fastdtw(true_10s, pred_10s, radius=DTW_RADIUS)
    # normalizzato per lunghezza del percorso di warping: rende il valore
    # confrontabile anche se in futuro si usassero finestre di lunghezza diversa
    metrics['DTW_10s'] = float(dtw_dist) / len(true_10s)

    return metrics


# =====================================================================
# METRICHE GRUPPI 2 & 3: Sull'intero segnale generato (Cumulativo) — invariate
# =====================================================================
def calculate_clinical_and_cumulative_metrics(true_full, pred_full, fs):
    import neurokit2 as nk
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
    unico, quanto segnale GREZZO e' disponibile e per quali orizzonti (in
    base alla durata grezza) vale la pena tentare la lettura. L'idoneita'
    EFFETTIVA per ciascun orizzonte viene ridecisa dopo lo splicing."""
    records = {}
    for h_label, entries in manifest.items():
        for e in entries:
            key = e['record_path']
            if key not in records:
                records[key] = {
                    "subject_id": e['subject_id'],
                    "record_path": key,
                    "raw_horizons": {},
                }
            records[key]["raw_horizons"][h_label] = e['end_sec']

    for rec in records.values():
        max_label = max(rec["raw_horizons"], key=lambda h: rec["raw_horizons"][h])
        rec["max_raw_horizon_label"] = max_label
        rec["max_raw_end_sec"] = rec["raw_horizons"][max_label]

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


def eligible_horizons_for_duration(raw_horizons: dict, duration_sec: float) -> dict:
    """Tra gli orizzonti che il segnale GREZZO poteva soddisfare, tiene solo
    quelli che il segnale PULITO (spliced) soddisfa ancora."""
    return {h: sec for h, sec in raw_horizons.items() if sec <= duration_sec}


def compute_metrics_for_horizon(ecg_f, gen_ecg_full, h_end_sec):
    n_samples = int(round(h_end_sec * FS))
    n_samples = min(n_samples, len(ecg_f), len(gen_ecg_full))

    true_10s_norm = _minmax_norm(ecg_f[n_samples - EVAL_SAMPLES: n_samples])
    pred_10s_norm = gen_ecg_full[n_samples - EVAL_SAMPLES: n_samples]
    metrics_g1 = calculate_group1_metrics(true_10s_norm, pred_10s_norm)
    metrics_g23 = calculate_clinical_and_cumulative_metrics(ecg_f[:n_samples], gen_ecg_full[:n_samples], FS)
    return {**metrics_g1, **metrics_g23}


RESULTS_LOCK_PATH = RESULTS_PATH + ".lock"

# Record che, dopo lo splicing, non raggiungono nemmeno l'orizzonte minimo
# '1m' (o che falliscono in lettura WFDB) non producono mai un .npz: senza
# una traccia persistente di questo esito, ad ogni resubmission verrebbero
# ritentati da capo all'infinito e il controllo di fine-job li conterebbe
# erroneamente come "ancora da fare", facendo credere a un'interruzione per
# timeout anche quando il job ha in realta' terminato tutto il suo lavoro.
DISCARDED_PATH = os.path.join(OUTPUT_DIR, "discarded_records.json")
DISCARDED_LOCK_PATH = DISCARDED_PATH + ".lock"


def _read_results_snapshot():
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
                json.dump(discarded, f, indent=4)
            os.replace(tmp_path, DISCARDED_PATH)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def process_one_record(rec, model, device, configs, log=print):
    rec_path = rec["record_path"]
    subj_id = rec["subject_id"]

    npz_path = os.path.join(GENERATION_DIR, rec_path + ".npz")

    if os.path.exists(npz_path):
        with np.load(npz_path) as cached:
            ecg_f = cached["ecg_target"].astype(np.float32)
            gen_ecg_full = cached["ecg_generated"].astype(np.float32)
            clean_duration_sec = float(len(ecg_f)) / FS
    else:
        # --- A1. Caricamento segnale GREZZO + splicing ---
        try:
            total_req_samples = int(rec["max_raw_end_sec"] * FS)
            ecg_raw, ppg_raw = load_record_signals(rec_path, total_req_samples)
        except Exception as e:
            log(f"Errore lettura {rec_path}: {e}")
            _mark_discarded(rec_path, "read_error", error=str(e))
            return

        ecg_clean, ppg_clean, splice_stats = splice_clean_signal(ecg_raw, ppg_raw, FS)
        clean_duration_sec = len(ecg_clean) / FS

        eligible = eligible_horizons_for_duration(rec["raw_horizons"], clean_duration_sec)
        if not eligible:
            log(f"[{device}] {rec_path}: scartato, dopo lo splicing restano solo "
                f"{clean_duration_sec:.0f}s puliti (< 66s, sotto l'orizzonte minimo '1m'). "
                f"Frazione pulita: {splice_stats['kept_fraction']:.1%}")
            _mark_discarded(rec_path, "too_short_after_splicing",
                             clean_duration_sec=clean_duration_sec,
                             kept_fraction=splice_stats['kept_fraction'])
            return

        # --- A2. Generazione autoregressiva sul segnale PULITO ---
        seed_samples = SEED_SEC * FS
        if len(ecg_clean) < seed_samples:
            log(f"[{device}] {rec_path}: scartato, segnale pulito troppo corto per il seed.")
            _mark_discarded(rec_path, "too_short_for_seed", clean_duration_sec=clean_duration_sec)
            return
        ecg_seed = ecg_clean[:seed_samples]
        t0 = time.time()
        gen_ecg_full = fast_autoregressive_inference(model, ppg_clean, ecg_seed, configs, device)
        elapsed = time.time() - t0
        log(f"[{device}] {rec_path}: grezzo {rec['max_raw_end_sec']:.0f}s -> pulito "
            f"{clean_duration_sec:.0f}s ({splice_stats['kept_fraction']:.1%} tenuto, "
            f"{splice_stats['n_splices']} giunzioni), generati in {elapsed:.1f}s "
            f"({clean_duration_sec / max(elapsed, 1e-6):.1f}x realtime)")

        ecg_f = ecg_clean
        save_dir_path = os.path.join(GENERATION_DIR, os.path.dirname(rec_path))
        os.makedirs(save_dir_path, exist_ok=True)
        np.savez_compressed(
            npz_path,
            ppg_input=ppg_clean.astype(np.float32),
            ecg_target=ecg_clean.astype(np.float32),
            ecg_generated=gen_ecg_full.astype(np.float32),
            fs=FS,
            clean_duration_sec=clean_duration_sec,
            kept_fraction=splice_stats['kept_fraction'],
            n_splices=splice_stats['n_splices'],
        )

    # --- B. Calcolo metriche per ogni orizzonte ancora mancante ---
    eligible = eligible_horizons_for_duration(rec["raw_horizons"], clean_duration_sec)
    if not eligible:
        return

    snapshot = _read_results_snapshot()
    processed_by_horizon = {h: {r['record_path'] for r in snapshot[h]} for h in HORIZONS_ORDER}

    new_metrics_by_horizon = {}
    for h_label, h_end_sec in sorted(eligible.items(), key=lambda kv: kv[1]):
        if rec_path in processed_by_horizon[h_label]:
            continue
        new_metrics_by_horizon[h_label] = compute_metrics_for_horizon(ecg_f, gen_ecg_full, h_end_sec)

    if not new_metrics_by_horizon:
        return

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

    if n_gpus > 0:
        devices = [f"cuda:{i}" for i in range(n_gpus)]
    else:
        devices = ["cpu"]

    if n_workers_arg is not None:
        if n_workers_arg <= len(devices):
            devices = devices[:n_workers_arg]
        elif devices == ["cpu"]:
            devices = ["cpu"] * n_workers_arg
        else:
            print(f"[WARN] --n_workers={n_workers_arg} > GPU disponibili ({len(devices)}); uso {len(devices)} worker.")

    return devices


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoregressive Drift (nuovo paradigma: splicing SQI prima della generazione)")
    parser.add_argument("--n_workers", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(" DRIFT EVALUATION — NUOVO PARADIGMA: SPLICING SQI PRIMA DELLA GENERAZIONE")
    print("=" * 60)

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
        'target_len': 125
    }

    devices = _resolve_devices(args.n_workers)
    print(f"-> Device worker: {devices}")
    print(f"-> Pesi modello: {MODEL_WEIGHTS_PATH}")

    unique_records = build_unique_records(manifest)
    print(f"-> Record unici nel manifest: {len(unique_records)} "
          f"(l'idoneita' per orizzonte verra' ridecisa dopo lo splicing SQI)")

    def _is_pending(rec, discarded):
        rec_path = rec["record_path"]
        if rec_path in discarded:
            return False  # gia' scartato in modo permanente (troppo corto/errore lettura): non ritentare
        npz_path = os.path.join(GENERATION_DIR, rec_path + ".npz")
        if not os.path.exists(npz_path):
            return True
        try:
            with np.load(npz_path) as cached:
                clean_duration_sec = len(cached["ecg_generated"]) / FS
        except Exception:
            return True
        eligible = eligible_horizons_for_duration(rec["raw_horizons"], clean_duration_sec)
        if not eligible:
            return False  # gia' processato e scartato (troppo corto dopo lo splicing)
        snapshot = _read_results_snapshot()
        processed_by_horizon = {h: {r['record_path'] for r in snapshot[h]} for h in HORIZONS_ORDER}
        return any(rec_path not in processed_by_horizon[h] for h in eligible)

    discarded = _read_discarded_snapshot()
    pending_records = [r for r in unique_records if _is_pending(r, discarded)]
    print(f"-> Record da (ri)processare in questa esecuzione: {len(pending_records)}/{len(unique_records)} "
          f"({len(discarded)} gia' scartati in run precedenti)")

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
    discarded = _read_discarded_snapshot()
    remaining = [r for r in unique_records if _is_pending(r, discarded)]
    if remaining:
        print(f"⚠️  Job terminato (probabilmente per il time-limit dello scheduler): "
              f"{len(unique_records) - len(remaining)}/{len(unique_records)} record completati.")
        print("    Rilancia lo stesso script (es. tramite ar_drif_test_clean.sh): riprendera' da dove si e' interrotto.")
    else:
        print("✅ VALUTAZIONE DRIFT (NUOVO PARADIGMA) COMPLETATA AL 100%!")
    print(f"Risultati metriche in: {RESULTS_PATH}")
    print(f"Segnali puliti+generati in: {GENERATION_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
