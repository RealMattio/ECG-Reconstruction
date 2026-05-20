import os
import sys
import json
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

# --- CONFIGURAZIONI ---
DATA_DIR = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_evaluation")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "drift_test_manifest.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_performance_results.json")

# Nuova directory per il salvataggio dei segnali generati
GENERATION_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation", "experiments", "drift_generation")

# Inserisci il path del tuo modello addestrato
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "src", "experiments", "final_mimic_pinn_results", "HUBER_loss", "lightweight_hybrid_20260324_155805", "final_full_model", "best_lightweight_hybrid.pth")

FS = 125
EVAL_WINDOW_SEC = 10 # Per le metriche del Gruppo 1
EVAL_SAMPLES = EVAL_WINDOW_SEC * FS

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
    except:
        metrics['SampEn_Error_10s'] = None
        
    try:
        lz_true, _ = nk.complexity_lempelziv(true_10s)
        lz_pred, _ = nk.complexity_lempelziv(pred_10s)
        metrics['LZ_Complexity_Error_10s'] = float(abs(lz_true - lz_pred))
    except:
        metrics['LZ_Complexity_Error_10s'] = None

    return metrics

# =====================================================================
# METRICHE GRUPPI 2 & 3: Sull'intero segnale generato (Cumulativo)
# =====================================================================
def calculate_clinical_and_cumulative_metrics(true_full, pred_full, fs):
    metrics = {}
    
    t_norm = _minmax_norm(true_full)
    p_norm = _minmax_norm(pred_full)
    
    metrics['Cumulative_RMSD'] = float(np.sqrt(np.mean((t_norm - p_norm)**2)))
    
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
            
            curr_ppg = ppg_full[start_win : end_win].copy()
            curr_ppg -= curr_ppg.mean()
            if curr_ppg.max() < -curr_ppg.min():
                curr_ppg = -curr_ppg
            curr_ppg_norm = _minmax_norm(curr_ppg)
            
            ppg_diff = np.zeros_like(curr_ppg_norm)
            ppg_diff[1:] = curr_ppg_norm[1:] - curr_ppg_norm[:-1]
            
            ecg_real_part = gen_ecg[start_win : cursor]
            padding = np.full((gen_samples,), ecg_real_part[-1])
            curr_ecg_past = np.concatenate([ecg_real_part, padding])
            
            ppg_t = torch.tensor(curr_ppg_norm, dtype=torch.float32)
            ppg_diff_t = torch.tensor(ppg_diff, dtype=torch.float32)
            ecg_p_t = torch.tensor(curr_ecg_past, dtype=torch.float32)
            
            X = torch.stack([ppg_t, ppg_diff_t, ecg_p_t], dim=0).unsqueeze(0).to(device)
            gen_ecg[cursor : end_win] = model(X).cpu().squeeze().numpy()

    return gen_ecg

def main():
    print("=" * 60)
    print(" FASE 2: FULL AUTOREGRESSIVE DRIFT EVALUATION & SIGNAL SAVING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERRORE] Manifest non trovato. Esegui prima lo script 1.")
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
    
    print(f"-> Inizializzazione Modello e caricamento pesi...")
    model = ModelFactory.get_model(configs).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            all_results = json.load(f)
        print("-> Continuo l'aggiornamento dei risultati precedenti...")
    else:
        all_results = {h: [] for h in manifest.keys()}

    horizons_order = ["1m", "30m", "1h", "6h", "12h", "24h"]
    
    for horizon in horizons_order:
        entries = manifest.get(horizon, [])
        if not entries: continue
            
        processed_records = [res['record_path'] for res in all_results[horizon]]
        entries_to_do = [e for e in entries if e['record_path'] not in processed_records]
        
        if not entries_to_do: continue
            
        print(f"\n>>> Avvio Orizzonte: {horizon} ({len(entries_to_do)} segmenti rimanenti)")
        pbar = tqdm(entries_to_do, desc=f"Generazione {horizon}")
        
        for entry in pbar:
            subj_id = entry['subject_id']
            rec_path = entry['record_path']
            req_sec = entry['end_sec']
            
            # --- A. Caricamento Dati ---
            abs_rec_path = os.path.join(DATA_DIR, rec_path)
            try:
                record = wfdb.rdrecord(abs_rec_path)
                idx_ecg = record.sig_name.index('II')
                idx_ppg = record.sig_name.index('PLETH')
                
                total_req_samples = int(req_sec * FS)
                ecg_raw = record.p_signal[:total_req_samples, idx_ecg]
                ppg_raw = record.p_signal[:total_req_samples, idx_ppg]
                
                ecg_f = apply_bandpass_filter(ecg_raw, FS, 0.5, 40.0)
                ppg_f = apply_bandpass_filter(ppg_raw, FS, 0.5, 5.0)
            except Exception as e:
                pbar.write(f"Errore lettura {rec_path}: {e}")
                continue

            # --- B. Inferenza Autoregressiva ---
            seed_samples = 6 * FS
            ecg_seed = ecg_f[:seed_samples]
            gen_ecg_full = fast_autoregressive_inference(model, ppg_f, ecg_seed, configs, device)
            
            # --- C. Calcolo Metriche ---
            true_10s_norm = _minmax_norm(ecg_f[-EVAL_SAMPLES:])
            pred_10s_norm = gen_ecg_full[-EVAL_SAMPLES:]
            metrics_g1 = calculate_advanced_metrics_group1(true_10s_norm, pred_10s_norm)
            metrics_g23 = calculate_clinical_and_cumulative_metrics(ecg_f, gen_ecg_full, FS)
            
            full_metrics = {**metrics_g1, **metrics_g23}
            
            # --- D. Salvataggio Metriche ---
            res_entry = {
                "subject_id": subj_id,
                "record_path": rec_path,
                "metrics": full_metrics
            }
            all_results[horizon].append(res_entry)
            
            with open(RESULTS_PATH, 'w') as f:
                json.dump(all_results, f, indent=4)
                
            # --- E. SALVATAGGIO SEGNALI GENERATI ---
            # Crea il path replicando la struttura originale, es: experiments/drift_generation/1h/p02/p023875/3153224_0018.npz
            save_dir_path = os.path.join(GENERATION_DIR, horizon, os.path.dirname(rec_path))
            os.makedirs(save_dir_path, exist_ok=True)
            
            file_name = os.path.basename(rec_path) + ".npz"
            full_save_path = os.path.join(save_dir_path, file_name)
            
            # Utilizziamo numpy compress per salvare spazio pur mantenendo un'altissima velocità di lettura
            np.savez_compressed(
                full_save_path,
                ppg_input=ppg_f.astype(np.float32),
                ecg_target=ecg_f.astype(np.float32),
                ecg_generated=gen_ecg_full.astype(np.float32),
                fs=FS
            )
            
            pbar.set_postfix({
                "Wass(10s)": f"{full_metrics.get('Wasserstein_10s', 0):.2f}", 
                "CS(Cum)": f"{full_metrics.get('Cumulative_CosineSim', 0):.2f}"
            })

    print("\n" + "=" * 60)
    print(f"✅ VALUTAZIONE DRIFT E SALVATAGGIO SEGNALI COMPLETATI AL 100%!")
    print(f"Risultati metriche in: {RESULTS_PATH}")
    print(f"Segnali completi in: {GENERATION_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()