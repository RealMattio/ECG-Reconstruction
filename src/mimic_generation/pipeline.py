import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import json
import random
import pandas as pd
from sklearn.model_selection import KFold
import gc
import psutil

# Import specifici
from src.data_loader.mimic3wdb_data_loader import MimicDataLoader
from src.preprocessing.mimic_autoregressive_preprocessor import MimicAutoregressivePreprocessor
from src.mimic_generation.trainer import Trainer
from src.mimic_generation.model_factory import ModelFactory
from src.evaluation.visualization import save_extended_reports, plot_training_history_metrics
from src.evaluation.evaluation import save_training_history, evaluate_test_set_performance

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

# --- NUOVA CLASSE DATASET PER DATI PREPROCESSATI ---
class MimicSmartDataset(Dataset):
    """
    Versione OTTIMIZZATA: Carica tutti i file .pt in RAM all'inizializzazione.
    Elimina la latenza del disco durante il training.
    """
    def __init__(self, manifest_subset, data_root, configs):
        self.data_root = data_root
        self.configs = configs
        self.samples_map = [] # Lista di tuple (idx_file_in_cache, start_index)
        self.file_cache = []  # Lista contenente i dati veri caricati in RAM
        
        self.win_samples = int(configs['target_fs'] * configs['x_sec']) # 875
        self.gen_samples = int(configs['target_fs'] * configs['gen_sec']) # 125
        
        print(f"   [SmartDataset] Caricamento in RAM di {len(manifest_subset)} file sessione...")
        
        # Carichiamo tutti i file in memoria una volta sola
        for entry in manifest_subset:
            rel_path = os.path.join(entry['subject_id'], os.path.basename(entry['file_path']))
            full_path = os.path.join(self.data_root, rel_path)
            
            try:
                # Carica il tensore in memoria (CPU)
                data = torch.load(full_path, map_location='cpu')
                
                # Salviamo il dato nella cache
                cache_idx = len(self.file_cache)
                self.file_cache.append(data)
                
                # Mappiamo gli indici validi a questo file nella cache
                valid_idx = data['valid_indices'].numpy()
                for start_idx in valid_idx:
                    self.samples_map.append((cache_idx, start_idx))
                    
            except Exception as e:
                print(f"   [WARN] Errore lettura {full_path}: {e}")
                continue
                
        print(f"   [SmartDataset] Pronto. {len(self.samples_map)} finestre totali in memoria.")

    def __len__(self):
        return len(self.samples_map)

    def __getitem__(self, idx):
        cache_idx, start_idx = self.samples_map[idx]
        data = self.file_cache[cache_idx]
        
        # 1. PPG Completo (7s)
        ppg_win = data['ppg'][start_idx : start_idx + self.win_samples].float()
        
        # 2. ECG Passato Sincronizzato (6s reali + 1s padding)
        # Invece di shiftare di 1 campione, partiamo dallo stesso T del PPG
        ecg_real_part = data['ecg'][start_idx : start_idx + self.win_samples - self.gen_samples].float()
        
        # Padding con l'ultimo valore noto (Last-Value Padding) per evitare sbalzi
        last_val = ecg_real_part[-1]
        padding = torch.full((self.gen_samples,), last_val)
        ecg_past = torch.cat([ecg_real_part, padding], dim=0)
        
        # 3. Target (L'ultimo secondo dell'ECG)
        target = data['ecg'][start_idx + self.win_samples - self.gen_samples : start_idx + self.win_samples].float()

        # 4. Derivata PPG (aiuta a trovare i picchi)
        ppg_diff = torch.zeros_like(ppg_win)
        ppg_diff[1:] = ppg_win[1:] - ppg_win[:-1]

        # Stack Finale (3 canali)
        X = torch.stack([ppg_win, ppg_diff, ecg_past], dim=0) # (3, 875)
        Y = target.unsqueeze(0) # (1, 125)
        
        return X, Y

# --- VECCHIA FUNZIONE (LEGACY/FALLBACK) ---
def _prepare_data_legacy(subject_keys, raw_data, preprocessor, configs):
    # Questa funzione viene chiamata SOLO se apply_preprocessing=True
    # Manteniamo il codice precedente per compatibilità
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
        return TensorDataset(X_tensor, Y_tensor), len(Y_tensor)
    except MemoryError:
        return None, 0

# --- PIPELINE PRINCIPALE MODIFICATA ---
def run_k_fold_pipeline(base_path_unused, configs):
    main_save_dir = configs['model_save_path'] # Es: experiments/exp_2026...
    os.makedirs(main_save_dir, exist_ok=True)
    device = configs['device']
    k_folds = configs.get('k_folds', 5)
    set_reproducibility(configs.get('seed', 42))

    # Inizializzazione Preprocessor globale per i plot del Trainer
    preprocessor = MimicAutoregressivePreprocessor(
        fs=configs['target_fs'], window_sec=configs['x_sec'], gen_sec=configs['gen_sec']
    )

    # 1. Caricamento Metadati (Smart o Legacy)
    if configs['apply_preprocessing']:
        # Modalità RAW
        loader = MimicDataLoader(data_dir=configs['raw_data'], target_fs=configs['target_fs'])
        raw_data = loader.load_data(lazy=True)
        unique_patient_ids = sorted(list(set([v['subject_id'] for k,v in raw_data['subjects_data'].items()])))
    else:
        # Modalità SMART
        manifest_path = os.path.join(configs['preprocessed_data'], 'dataset_manifest.json')
        with open(manifest_path, 'r') as f: full_manifest = json.load(f)
        unique_patient_ids = sorted(list(set([m['subject_id'] for m in full_manifest])))

    # Filtro esclusi
    unique_patient_ids = [pid for pid in unique_patient_ids if pid not in configs['excluded_subjects_ids']]
    print(f"[PIPELINE] Pazienti totali per {k_folds}-Fold: {len(unique_patient_ids)}")

    # 2. K-Fold Loop
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=configs.get('seed', 42))
    fold_results = []
    patient_ids_arr = np.array(unique_patient_ids)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids_arr)):
        print(f"\n{'='*50}\n>>> AVVIO FOLD {fold_idx + 1}/{k_folds}\n{'='*50}")
        
        # SETUP CARTELLA FOLD
        fold_dir = os.path.join(main_save_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # AGGIORNAMENTO DINAMICO PATH NEL CONFIG
        configs['model_save_path'] = fold_dir
        
        # 3. Dataset Setup
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
        # 4. Inizializzazione Modello per questo Fold
        try:
            sample_x, sample_y = train_ds[0]
            configs['input_channels'], configs['actual_seq_len'] = sample_x.shape[0], sample_x.shape[-1]
            configs['target_len'] = sample_y.shape[-1]
            
            model = ModelFactory.get_model(configs).to(device)
        except Exception as e:
            print(f"[ERROR] Init Modello Fallita: {e}"); continue

        # 5. Training
        trainer = Trainer(model, device, configs, preprocessor=preprocessor)
        history = trainer.fit(train_loader, val_loader, epochs=configs['epochs'], patience=configs['patience'])
        
        # 6. Salvataggio Performance Fold
        save_training_history(history, fold_dir) # Salva JSON
        plot_training_history_metrics(history, fold_dir) # Salva Curve PNG
        
        # Valutazione finale sul Best Model del Fold
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
        
        # Pulizia rigorosa
        del model, trainer, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    # Report Finale Globale
    report_path = os.path.join(main_save_dir, "k_fold_final_report.json")
    with open(report_path, 'w') as f: json.dump(fold_results, f, indent=4)
    print(f"\n[PIPELINE COMPLETATA] Risultati salvati in: {main_save_dir}")

    return fold_results