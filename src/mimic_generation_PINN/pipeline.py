import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import json
import random
import pandas as pd
from scipy.signal import butter, sosfilt, resample as scipy_resample, find_peaks
from sklearn.model_selection import KFold, train_test_split
import gc
import psutil

# Import specifici
from src.data_loader.mimic3wdb_data_loader import MimicDataLoader
from src.preprocessing.mimic_autoregressive_preprocessor import MimicAutoregressivePreprocessor
from src.mimic_generation_PINN.trainer import Trainer
from src.mimic_generation_PINN.model_factory import ModelFactory
from src.evaluation.visualization import save_extended_reports, plot_training_history_metrics
from src.evaluation.evaluation import save_training_history, evaluate_test_set_performance

# ─────────────────────────────────────────────────────────────
# UTILITÀ WFDB  (per il formato manifest no-cache)
# ─────────────────────────────────────────────────────────────

def _extract_physics_signals(ecg_np, fs=125):
    """
    Calcola theta (fase cardiaca) e omega (freq. angolare) dall'ECG.
    Identico a extract_physics_signals in scripts/create_pinn_dataset.py.
    """
    sig_norm = (ecg_np - np.mean(ecg_np)) / (np.std(ecg_np) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs * 0.4), height=1.0)
    N = len(ecg_np)
    omega = np.full(N, 2.0 * np.pi, dtype=np.float32)
    theta = np.zeros(N, dtype=np.float32)
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            T_sec = (p2 - p1) / fs
            w = 2.0 * np.pi / T_sec
            omega[p1:p2] = w
            theta[p1:p2] = np.linspace(0, 2 * np.pi, p2 - p1, endpoint=False)
        p_first = peaks[0]
        if p_first > 0:
            omega[:p_first] = omega[p_first]
            t_arr = np.arange(-p_first, 0) / fs
            theta[:p_first] = (t_arr * omega[p_first]) % (2 * np.pi)
        p_last = peaks[-1]
        if p_last < N:
            omega[p_last:] = omega[p_last - 1]
            t_arr = np.arange(0, N - p_last) / fs
            theta[p_last:] = (t_arr * omega[p_last - 1]) % (2 * np.pi)
    else:
        t_arr = np.arange(N) / fs
        theta = (t_arr * omega) % (2 * np.pi)
    return theta, omega


def _load_wfdb_segments(entry, target_fs):
    """
    Carica un record WFDB dal path originale, lo ricampiona, filtra e restituisce
    i segmenti validi come lista di dict pronti per MimicSmartDataset.file_cache.

    Chiamato solo per manifest entries con chiave 'wfdb_path' (formato no-cache).
    """
    import wfdb

    wfdb_path = entry['wfdb_path']
    subject_id = entry['subject_id']
    results = []

    try:
        record = wfdb.rdrecord(wfdb_path)
        idx_ecg = record.sig_name.index('II')
        idx_ppg = record.sig_name.index('PLETH')
        ecg_raw = np.nan_to_num(record.p_signal[:, idx_ecg]).astype(np.float32)
        ppg_raw = np.nan_to_num(record.p_signal[:, idx_ppg]).astype(np.float32)

        orig_fs = record.fs
        if orig_fs != target_fs:
            n = int(len(ecg_raw) / orig_fs * target_fs)
            ecg_raw = scipy_resample(ecg_raw, n).astype(np.float32)
            ppg_raw = scipy_resample(ppg_raw, n).astype(np.float32)

        n_samples = min(len(ecg_raw), len(ppg_raw))
        ecg_raw = ecg_raw[:n_samples]
        ppg_raw = ppg_raw[:n_samples]

        # Filtro bandpass (identico a preprocess_mimic_smart.py)
        nyq = 0.5 * target_fs
        sos_ppg = butter(4, [0.5 / nyq, 5.0 / nyq], btype='band', output='sos')
        sos_ecg = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band', output='sos')
        ppg_filt = sosfilt(sos_ppg, ppg_raw).astype(np.float32)
        ecg_filt = sosfilt(sos_ecg, ecg_raw).astype(np.float32)

        for seg_info in entry['segments']:
            s, e = seg_info['start'], seg_info['end']
            ppg_seg = ppg_filt[s:e].copy()
            ecg_seg = ecg_filt[s:e].copy()

            if seg_info.get('ppg_inverted', False):
                ppg_seg = ppg_seg * -1.0

            theta, omega = _extract_physics_signals(ecg_seg, target_fs)
            rec_name = os.path.basename(wfdb_path)
            results.append({
                'ppg':       torch.tensor(ppg_seg, dtype=torch.float32),
                'ecg':       torch.tensor(ecg_seg, dtype=torch.float32),
                'theta':     torch.tensor(theta,   dtype=torch.float32),
                'omega':     torch.tensor(omega,   dtype=torch.float32),
                'file_info': f"{subject_id}/{rec_name}_s{s}_e{e}",
            })

    except Exception as e:
        print(f"   [WARN] Errore WFDB {wfdb_path}: {e}")

    return results


# --- UTILS ---
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed globale impostato a: {seed}")

# --- NUOVA CLASSE DATASET PER DATI PINN (GIA' PREPROCESSATI) ---
class MimicSmartDataset(Dataset):
    def __init__(self, manifest_subset, data_root, configs):
        self.data_root = data_root
        self.configs = configs
        self.samples_map = [] 
        self.file_cache = []  
        
        self.win_samples = int(configs['target_fs'] * configs['x_sec']) 
        self.gen_samples = int(configs['target_fs'] * configs['gen_sec']) 
        step_size = self.gen_samples 
        
        print(f"   [SmartDataset] Caricamento in RAM di {len(manifest_subset)} entry...")

        for entry in manifest_subset:
            if 'wfdb_path' in entry:
                # ── Formato no-cache: carica direttamente da WFDB ──
                segments = _load_wfdb_segments(entry, configs['target_fs'])
                for data in segments:
                    cache_idx = len(self.file_cache)
                    self.file_cache.append(data)
                    total_len = len(data['ppg'])
                    for start_idx in range(0, total_len - self.win_samples + 1, step_size):
                        self.samples_map.append((cache_idx, start_idx))
            else:
                # ── Formato legacy: carica da file .pt ──
                rel_path = os.path.join(entry['subject_id'], os.path.basename(entry['file_path']))
                full_path = os.path.join(self.data_root, rel_path)
                try:
                    data = torch.load(full_path, map_location='cpu')
                    data['file_info'] = rel_path
                    cache_idx = len(self.file_cache)
                    self.file_cache.append(data)
                    total_len = len(data['ppg'])
                    for start_idx in range(0, total_len - self.win_samples + 1, step_size):
                        self.samples_map.append((cache_idx, start_idx))
                except Exception:
                    pass

        print(f"   [SmartDataset] Pronto. {len(self.samples_map)} finestre in memoria.")

    def __len__(self):
        return len(self.samples_map)

    def __getitem__(self, idx):
        cache_idx, start_idx = self.samples_map[idx]
        data = self.file_cache[cache_idx]

        target_start = start_idx + self.win_samples - self.gen_samples
        target_end   = start_idx + self.win_samples

        # 1. PPG — correzione polarità poi normalizzazione [0,1] per finestra
        ppg_win = data['ppg'][start_idx : start_idx + self.win_samples].float()

        # Asimmetria di ampiezza: il picco sistolico è la deviazione POSITIVA dominante.
        # Se la deviazione negativa massima supera quella positiva, il segnale è invertito.
        ppg_centered = ppg_win - ppg_win.mean()
        if ppg_centered.max() < -ppg_centered.min():
            ppg_win = ppg_win * -1.0

        ppg_min = ppg_win.min()
        ppg_max = ppg_win.max()
        ppg_win = (ppg_win - ppg_min) / (ppg_max - ppg_min + 1e-8)

        # 2. ECG — normalizzazione [0,1] sull'intera finestra di 7s (past + target insieme)
        #    così past e target condividono la stessa scala e il modello non vede
        #    discontinuità artificiali al confine.
        ecg_full = data['ecg'][start_idx : target_end].float()
        ecg_min  = ecg_full.min()
        ecg_max  = ecg_full.max()
        ecg_full_norm = (ecg_full - ecg_min) / (ecg_max - ecg_min + 1e-8)

        ecg_real_part = ecg_full_norm[: self.win_samples - self.gen_samples]
        target        = ecg_full_norm[self.win_samples - self.gen_samples :]

        last_val  = ecg_real_part[-1]
        padding   = torch.full((self.gen_samples,), last_val)
        ecg_past  = torch.cat([ecg_real_part, padding], dim=0)

        # 3. Target Fisici (theta/omega NON normalizzati: codificano grandezze fisiche)
        theta_target = data['theta'][target_start : target_end].float()
        omega_target = data['omega'][target_start : target_end].float()

        # 4. Derivata PPG (calcolata dopo normalizzazione e correzione polarità)
        ppg_diff = torch.zeros_like(ppg_win)
        ppg_diff[1:] = ppg_win[1:] - ppg_win[:-1]

        X = torch.stack([ppg_win, ppg_diff, ecg_past], dim=0)
        Y = target.unsqueeze(0)

        info_string = f"{data['file_info']} | Pos: {start_idx}"
        return X, Y, theta_target, omega_target, info_string

# --- VECCHIA FUNZIONE (LEGACY/FALLBACK) ---
def _prepare_data_legacy(subject_keys, raw_data, preprocessor, configs):
    all_ppg, all_ecg_past, all_target = [], [], []
    temp_loader = MimicDataLoader(data_dir="", target_fs=configs['target_fs'])
    temp_loader.subjects_index = raw_data['subjects_data'] 

    for s_key in subject_keys:
        meta = raw_data['subjects_data'][s_key]
        try:
            signal_data = temp_loader.load_signal(s_key, meta)
        except: continue
        
        ppg_full, ecg_full = signal_data['PPG'], signal_data['ECG']
        if len(ppg_full) == 0: continue

        processed = preprocessor.process_subject(ppg_full, ecg_full, configs, is_training=True)
        del ppg_full, ecg_full, signal_data
        
        if processed and len(processed['ppg']) > 0:
            all_ppg.append(processed['ppg'])
            all_ecg_past.append(processed['ecg_past'])
            all_target.append(processed['target'])
        del processed
    
    gc.collect()
    if not all_ppg: return None, 0

    try:
        X_ppg = np.concatenate(all_ppg, axis=0)
        del all_ppg; X_past = np.concatenate(all_ecg_past, axis=0)
        del all_ecg_past; Y_target = np.concatenate(all_target, axis=0)
        del all_target; gc.collect()

        if X_ppg.ndim == 2: X_combined = np.stack([X_ppg, X_past], axis=1)
        else: X_combined = np.concatenate([X_ppg, X_past], axis=1)
        del X_ppg, X_past
        
        X_tensor = torch.tensor(X_combined).float()
        Y_tensor = torch.tensor(Y_target).float().unsqueeze(1)
        
        theta_dummy = torch.zeros_like(Y_tensor).squeeze(1)
        omega_dummy = torch.full_like(Y_tensor, 2.0 * np.pi).squeeze(1)
        
        return TensorDataset(X_tensor, Y_tensor, theta_dummy, omega_dummy), len(Y_tensor)
    except MemoryError:
        return None, 0

# --- PIPELINE PRINCIPALE MODIFICATA ---
def run_k_fold_pipeline(base_path_unused, configs):
    main_save_dir = configs['model_save_path'] 
    os.makedirs(main_save_dir, exist_ok=True)
    device = configs['device']
    k_folds = configs.get('k_folds', 5)
    seed = configs.get('seed', 42)
    set_reproducibility(seed)

    preprocessor = MimicAutoregressivePreprocessor(
        fs=configs['target_fs'], window_sec=configs['x_sec'], gen_sec=configs['gen_sec']
    )

    if configs['apply_preprocessing']:
        loader = MimicDataLoader(data_dir=configs['raw_data'], target_fs=configs['target_fs'])
        raw_data = loader.load_data(lazy=True)
        unique_patient_ids = sorted(list(set([v['subject_id'] for k,v in raw_data['subjects_data'].items()])))
    else:
        manifest_file = configs.get('manifest_name', 'dataset_manifest.json')
        manifest_path = os.path.join(configs['preprocessed_data'], manifest_file)
        with open(manifest_path, 'r') as f: full_manifest = json.load(f)
        unique_patient_ids = sorted(list(set([m['subject_id'] for m in full_manifest])))

    unique_patient_ids = [pid for pid in unique_patient_ids if pid not in configs['excluded_subjects_ids']]
    print(f"[PIPELINE] Pazienti totali nel dataset: {len(unique_patient_ids)}")

    patient_ids_arr = np.array(unique_patient_ids)
    fold_results = []
    
    # --- FIX ROBUSTO PER LA GESTIONE DEGLI SPLIT ---
    if k_folds <= 1:
        print("[INFO] k_folds impostato a 1. Esecuzione split singolo Train/Val (80/20).")
        indices = np.arange(len(patient_ids_arr))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        splits = [(train_idx, val_idx)]
        k_folds = 1 # Normalizziamo a 1 per i log
    else:
        print(f"[INFO] Esecuzione {k_folds}-Fold Cross Validation.")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = kf.split(patient_ids_arr)
    # -----------------------------------------------

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}\n>>> AVVIO FOLD {fold_idx + 1}/{k_folds}\n{'='*50}")
        
        fold_dir = os.path.join(main_save_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)
        configs['model_save_path'] = fold_dir
        
        train_patients = patient_ids_arr[train_idx]
        val_patients = patient_ids_arr[val_idx]

        if configs['apply_preprocessing']:
            train_keys = [k for k,v in raw_data['subjects_data'].items() if v['subject_id'] in train_patients]
            val_keys = [k for k,v in raw_data['subjects_data'].items() if v['subject_id'] in val_patients]
            train_ds, n_train = _prepare_data_legacy(train_keys, raw_data, preprocessor, configs)
            val_ds, n_val = _prepare_data_legacy(val_keys, raw_data, preprocessor, configs)
        else:
            train_subset = [m for m in full_manifest if m['subject_id'] in train_patients]
            val_subset = [m for m in full_manifest if m['subject_id'] in val_patients]
            train_ds = MimicSmartDataset(train_subset, configs['preprocessed_data'], configs)
            val_ds = MimicSmartDataset(val_subset, configs['preprocessed_data'], configs)
            n_train, n_val = len(train_ds), len(val_ds)

        print(f"   Dataset: Train={n_train} finestre, Val={n_val} finestre")

        num_cpus = os.cpu_count()
        workers = min(num_cpus, 8) if device.type == 'cuda' else 0
        train_loader = DataLoader(train_ds, batch_size=configs['batch_size'], shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True if workers > 0 else False)
        val_loader = DataLoader(val_ds, batch_size=configs['batch_size'], shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True if workers > 0 else False)
        print(f"   [DEBUG] Batch Size: {configs['batch_size']} | Iterazioni per Epoca: {len(train_loader)} | Workers: {workers}")
        
        try:
            sample_x, sample_y, _, _, _ = train_ds[0] 
            configs['input_channels'] = sample_x.shape[0] 
            configs['actual_seq_len'] = sample_x.shape[-1]
            configs['target_len'] = sample_y.shape[-1]
            
            model = ModelFactory.get_model(configs).to(device)
        except Exception as e:
            print(f"[ERROR] Init Modello Fallita: {e}"); continue

        trainer = Trainer(model, device, configs, preprocessor=preprocessor)
        history = trainer.fit(train_loader, val_loader, epochs=configs['epochs'], patience=configs['patience'])
        
        save_training_history(history, fold_dir) 
        plot_training_history_metrics(history, fold_dir) 
        
        best_path = os.path.join(fold_dir, f'best_{configs.get("model_type", "model")}.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            
        metrics = evaluate_test_set_performance(model, val_loader, device, fold_dir, configs)
        
        fold_results.append({
            "fold": fold_idx + 1,
            "train_patients": train_patients.tolist(),
            "val_patients": val_patients.tolist(),
            "metrics": metrics
        })
        
        del model, trainer, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    report_path = os.path.join(main_save_dir, "k_fold_final_report.json")
    with open(report_path, 'w') as f: json.dump(fold_results, f, indent=4)
    print(f"\n[PIPELINE COMPLETATA] Risultati salvati in: {main_save_dir}")

    return fold_results