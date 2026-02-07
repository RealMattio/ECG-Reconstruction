import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import random
import pandas as pd

# Import dei moduli autoregressivi
from src.preprocessing.bidmc_autoregressive_preprocessor import BidmcAutoregressivePreprocessor
from src.bidmc_generation.models.ha_cnn_bilstm_autoregressive import HACNNBiLSTM_AR
from src.bidmc_generation.trainer import HACNNBiLSTMTrainer
from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.bidmc_generation.model_factory import ModelFactory

# Import delle funzioni di valutazione aggiornate
from src.evaluation.visualization import (
    save_extended_reports, 
    save_inference_plot, 
    plot_training_history_metrics
)
from src.evaluation.evaluation import save_training_history, evaluate_test_set_performance

def set_reproducibility(seed):
    """Imposta il seed per tutte le librerie per garantire riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Rende l'algoritmo di convoluzione deterministico (leggermente più lento)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed globale impostato a: {seed}")

def _prepare_data(subject_keys, raw_data, preprocessor, configs):
    """
    Trasforma i dati grezzi in tensori per il modello AR.
    Input: Stack di [PPG, ECG_Past] -> (Batch, 2, Seq_Len)
    Target: ECG_Future -> (Batch, 1, Target_Len)
    """
    all_ppg, all_ecg_past, all_target = [], [], []
    
    for s_key in subject_keys:
        data = raw_data['subjects_data'][s_key]
        # Il preprocessor AR restituisce un dizionario con i tre segmenti
        processed = preprocessor.process_subject(data['PPG'], data['ECG'], configs, is_training=True)
        
        if processed is not None and len(processed['ppg']) > 0:
            all_ppg.append(processed['ppg'])
            all_ecg_past.append(processed['ecg_past'])
            all_target.append(processed['target'])
    
    if not all_ppg:
        raise ValueError(f"Nessun dato estratto dai soggetti: {subject_keys}")

    # Concatenazione dei dati di tutti i soggetti
    X_ppg = np.concatenate(all_ppg, axis=0)
    X_past = np.concatenate(all_ecg_past, axis=0)
    Y_target = np.concatenate(all_target, axis=0)
    
    # GESTIONE DIMENSIONI WST vs RAW
    if X_ppg.ndim == 2: 
        # Caso RAW: (N, T). Stackiamo su un nuovo asse canali
        # Output: (N, 2, T) -> Canale 0: PPG, Canale 1: ECG
        X_combined = np.stack([X_ppg, X_past], axis=1)
    else:
        # Caso WST: (N, C, T). Concateniamo lungo i canali esistenti
        # Output: (N, 38, T) -> Canali 0-18: PPG, Canali 19-37: ECG
        X_combined = np.concatenate([X_ppg, X_past], axis=1)
        
    X_tensor = torch.tensor(X_combined).float()
    Y_tensor = torch.tensor(Y_target).float().unsqueeze(1)
    
    return TensorDataset(X_tensor, Y_tensor)

def run_ha_cnn_bilstm_pipeline(base_path, configs):
    set_reproducibility(configs.get('seed', 45))
    device = configs['device']
    save_dir = configs['model_save_path']
    os.makedirs(save_dir, exist_ok=True)

    # 1. CARICAMENTO DATI
    loader = BidmcDataLoader()
    # Carichiamo tutti i 53 soggetti disponibili
    raw_data = loader.load_subjects([str(i).zfill(2) for i in range(1, 54)], base_path)
    
    # Filtraggio soggetti (Escludiamo quelli rumorosi specificati)
    exclude_ids = configs.get('excluded_subjects_ids', [])
    exclude_keys = [f"S{str(i).zfill(2)}" for i in exclude_ids]
    available_keys = [k for k in raw_data['subjects_data'].keys() if k not in exclude_keys]

    # 2. INIZIALIZZAZIONE PREPROCESSORE AUTOREGRESSIVO
    # Passiamo x_sec (es. 8) per definire la finestra di input
    preprocessor = BidmcAutoregressivePreprocessor(
        fs=configs['target_fs'], 
        window_sec=configs.get('x_sec', 7),
        gen_sec=configs.get('gen_sec', 1)
    )

    # Splitting dei soggetti (Subject-Wise)
    train_keys, val_keys, test_keys = preprocessor.split_subjects(
        available_keys,
        train_ratio=configs.get('train_ratio', 0.8),
        val_ratio=configs.get('val_ratio', 0.1),
        seed=configs.get('seed', 45)
    )

    # 3. PREPARAZIONE DATASET (Train, Val, Test)
    print("\n--- Generazione Dataset Autoregressivi ---")
    train_ds = _prepare_data(train_keys, raw_data, preprocessor, configs)
    val_ds = _prepare_data(val_keys, raw_data, preprocessor, configs)
    test_ds = _prepare_data(test_keys, raw_data, preprocessor, configs)

    # 4. AGGIORNAMENTO DINAMICO CONFIGS
    # Rileviamo le dimensioni reali dai tensori appena creati
    configs['input_channels'] = train_ds[0][0].shape[0] # Sarà 2
    configs['actual_seq_len'] = train_ds[0][0].shape[-1] # fs * x_sec
    configs['target_len'] = train_ds[0][1].shape[-1]     # fs * 1
    
    # Salvataggio configurazione finale
    serializable_configs = configs.copy()
    serializable_configs['device'] = str(device)
    serializable_configs['excluded_subjects'] = exclude_keys

    serializable_configs['dataset_splits'] = {
        'train_subjects': train_keys,
        'val_subjects': val_keys,
        'test_subjects': test_keys,
        'count_train': len(train_keys),
        'count_val': len(val_keys),
        'count_test': len(test_keys)
    }

    with open(os.path.join(save_dir, 'experiment_configs.json'), 'w') as f:
        json.dump(serializable_configs, f, indent=4)

    # 5. DATALOADERS
    train_loader = DataLoader(train_ds, batch_size=configs['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=configs['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=configs['batch_size'], shuffle=False)

    # =========================================================
    # 6. MODELLO E TRAINER (MODIFICATO CON FACTORY)
    # =========================================================
    # Usiamo la factory per ottenere il modello specificato nel main
    try:
        model = ModelFactory.get_model(configs)
    except ValueError as e:
        print(e)
        return None # Interrompiamo se il modello non esiste

    # Il trainer rimane lo stesso, dato che l'interfaccia di training è comune
    trainer = HACNNBiLSTMTrainer(model, device, configs)
    # =========================================================

    # 7. TRAINING
    history = trainer.fit(train_loader, val_loader, epochs=configs['epochs'], patience=configs['patience'])

    model_name = configs.get('model_type', 'ha_cnn_bilstm_ar')
    # 8. SALVATAGGIO E REPORTISTICA
    final_model_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    save_training_history(history, save_dir)
    plot_training_history_metrics(history, save_dir)
    
    # Carichiamo il miglior modello per i grafici finali
    best_model_path = os.path.join(save_dir, f"best_{model_name}.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    print("\n--- Generazione Report Autoregressivi (Iterative) ---")
    save_extended_reports(model, train_keys, raw_data, preprocessor, device, configs, "train")
    save_extended_reports(model, val_keys, raw_data, preprocessor, device, configs, "val")
    save_extended_reports(model, test_keys, raw_data, preprocessor, device, configs, "test")
    
    # Grafico di inferenza semplice su batch
    save_inference_plot(model, test_loader, device, os.path.join(save_dir, "final_test_batch_inference.png"))

    evaluate_test_set_performance(model, test_loader, device, save_dir, configs)

    print(f"\n[PIPELINE COMPLETE] Risultati salvati in: {save_dir}")
    return history

def run_k_fold_pipeline(base_path, configs):
    # 0. SETUP GLOBALE
    main_save_dir = configs['model_save_path']
    os.makedirs(main_save_dir, exist_ok=True)
    device = configs['device']
    
    # Parametro K-Fold
    k_folds = configs.get('k_folds', 5)
    global_seed = configs.get('seed', 42)
    
    # 1. CARICAMENTO DATI (Una volta sola)
    loader = BidmcDataLoader()
    raw_data = loader.load_subjects([str(i).zfill(2) for i in range(1, 54)], base_path)
    
    exclude_ids = configs.get('excluded_subjects_ids', [])
    exclude_keys = [f"S{str(i).zfill(2)}" for i in exclude_ids]
    available_keys = [k for k in raw_data['subjects_data'].keys() if k not in exclude_keys]
    
    # Preprocessor (istanziato una volta per configurazione)
    preprocessor = BidmcAutoregressivePreprocessor(
        fs=configs['target_fs'], 
        window_sec=configs.get('x_sec', 7),
        gen_sec=configs.get('gen_sec', 1)
    )

    # 2. LOGICA DI SELEZIONE SOGGETTI (Shuffle Iniziale)
    set_reproducibility(global_seed)
    shuffled_subjects = sorted(list(available_keys))
    random.shuffle(shuffled_subjects)
    
    # Per essere sicuri di avere abbastanza coppie uniche, usiamo una finestra scorrevole
    # Se K > N/2, alcuni soggetti si ripeteranno, ma le coppie saranno diverse.
    if len(shuffled_subjects) < 2:
        raise ValueError("Non ci sono abbastanza soggetti per fare validazione.")

    # Struttura per salvare i risultati aggregati
    cross_validation_results = []

    print(f"\n{'='*60}")
    print(f"AVVIO CROSS-VALIDATION A {k_folds} FOLD (Random Sub-Sampling)")
    print(f"{'='*60}")

    # =========================================================================
    # CICLO DEI FOLD
    # =========================================================================
    for fold_idx in range(k_folds):
        print(f"\n>>> INIZIO FOLD {fold_idx + 1}/{k_folds}")
        
        # A. Selezione Soggetti per questo Fold
        # Prendiamo 2 soggetti in modo circolare dalla lista mescolata
        idx_val = (fold_idx * 2) % len(shuffled_subjects)
        idx_test = (fold_idx * 2 + 1) % len(shuffled_subjects)
        
        val_subj = shuffled_subjects[idx_val]
        test_subj = shuffled_subjects[idx_test]
        
        # Tutti gli altri sono training
        train_subjects = [s for s in shuffled_subjects if s != val_subj and s != test_subj]
        
        val_subjects = [val_subj]
        test_subjects = [test_subj]
        
        print(f"   Validation Subject: {val_subj}")
        print(f"   Test Subject:       {test_subj}")
        print(f"   Training Subjects:  {len(train_subjects)}")

        # B. Creazione Sottocartella per il Fold
        fold_dir = os.path.join(main_save_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)
        # Aggiorniamo temporaneamente il path nei configs per il trainer
        configs['model_save_path'] = fold_dir 
        
        # C. Preparazione Dataset
        train_ds = _prepare_data(train_subjects, raw_data, preprocessor, configs)
        val_ds = _prepare_data(val_subjects, raw_data, preprocessor, configs)
        test_ds = _prepare_data(test_subjects, raw_data, preprocessor, configs)
        
        # Aggiornamento Dimensioni Input (solo al primo fold o sempre per sicurezza)
        configs['input_channels'] = train_ds[0][0].shape[0] 
        configs['actual_seq_len'] = train_ds[0][0].shape[-1] 
        configs['target_len'] = train_ds[0][1].shape[-1]
        
        # D. DataLoaders
        train_loader = DataLoader(train_ds, batch_size=configs['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=configs['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=configs['batch_size'], shuffle=False)
        
        # E. Init Modello (Fresco per ogni fold)
        try:
            model = ModelFactory.get_model(configs)
            model.to(device)
        except ValueError as e:
            print(e); return

        trainer = HACNNBiLSTMTrainer(model, device, configs)
        
        # F. Training
        history = trainer.fit(train_loader, val_loader, epochs=configs['epochs'], patience=configs['patience'])
        
        # G. Salvataggio e Valutazione Fold
        # Salviamo la history del fold
        save_training_history(history, fold_dir)
        plot_training_history_metrics(history, fold_dir)
        
        # Carichiamo il best model del fold
        best_path = os.path.join(fold_dir, f'best_ha_cnn_bilstm.pth') # Il nome è standard nel trainer
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        
        # Generazione Grafici
        print(f"   Generazione grafici Fold {fold_idx+1}...")
        save_extended_reports(model, train_subjects, raw_data, preprocessor, device, configs, "train")
        save_extended_reports(model, val_subjects, raw_data, preprocessor, device, configs, "val")
        save_extended_reports(model, test_subjects, raw_data, preprocessor, device, configs, "test")
        
        # Calcolo Metriche Test (Quelle che contano)
        fold_metrics = evaluate_test_set_performance(model, test_loader, device, fold_dir, configs)
        
        # H. Archiviazione Risultati
        fold_record = {
            "fold": fold_idx + 1,
            "train_subjects_count": len(train_subjects),
            "val_subject": val_subj,
            "test_subject": test_subj,
            "metrics": fold_metrics
        }
        cross_validation_results.append(fold_record)
        
        # Pulizia memoria GPU tra un fold e l'altro
        del model, trainer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # FINE CICLO - AGGREGAZIONE DATI
    # =========================================================================
    print(f"\n{'='*60}")
    print("CALCOLO RISULTATI AGGREGATI")
    print(f"{'='*60}")
    
    # Creiamo un DataFrame per calcolare medie e std facilmente
    df_metrics = pd.DataFrame([f['metrics'] for f in cross_validation_results])
    
    aggregated_stats = {
        "mean": df_metrics.mean().to_dict(),
        "std": df_metrics.std().to_dict(),
        "min": df_metrics.min().to_dict(),
        "max": df_metrics.max().to_dict()
    }
    
    final_report = {
        "experiment_config": configs,
        "aggregated_performance": aggregated_stats,
        "fold_details": cross_validation_results
    }
    
    # Rimuoviamo oggetti non serializzabili dai config prima di salvare
    final_report['experiment_config']['device'] = str(configs['device'])
    
    report_path = os.path.join(main_save_dir, "cross_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Report completo salvato in: {report_path}")
    print(f"RMSE Medio: {aggregated_stats['mean'].get('rmse', 0):.4f} ± {aggregated_stats['std'].get('rmse', 0):.4f}")
    print(f"Pearson Medio: {aggregated_stats['mean'].get('pearson', 0):.4f} ± {aggregated_stats['std'].get('pearson', 0):.4f}")
    
    return cross_validation_results