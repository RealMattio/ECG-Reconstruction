import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.preprocessing.bidmc_preprocessor import BidmcPreprocessor
from src.bidmc_generation.models.ha_cnn_bilstm import HACNNBiLSTM
from src.bidmc_generation.trainer import HACNNBiLSTMTrainer
from src.evaluation.visualization import save_inference_plot, save_extended_reports, plot_training_history_metrics
from src.evaluation.evaluation import save_training_history

def _prepare_data(subject_keys, raw_data, preprocessor, configs):
    all_ppg, all_ecg = [], []
    
    for s_key in subject_keys:
        data = raw_data['subjects_data'][s_key]
        ppg_b, ecg_b = preprocessor.process_subject(data['PPG'], data['ECG'], configs)
        
        if len(ppg_b) > 0:
            all_ppg.append(ppg_b)
            all_ecg.append(ecg_b)
    
    if not all_ppg:
        raise ValueError(f"Nessun dato estratto dai soggetti: {subject_keys}")

    X = np.concatenate(all_ppg, axis=0)
    y = np.concatenate(all_ecg, axis=0)
    
    # Tensori (WST o Raw)
    X_tensor = torch.tensor(X).float()
    if not configs.get('apply_wst', False):
        X_tensor = X_tensor.unsqueeze(1)
    
    y_tensor = torch.tensor(y).float()
    if len(y_tensor.shape) == 2:
        y_tensor = y_tensor.unsqueeze(1)
    
    return TensorDataset(X_tensor, y_tensor)

def run_ha_cnn_bilstm_pipeline(base_path, configs):
    device = configs['device']
    save_dir = configs['model_save_path']
    os.makedirs(save_dir, exist_ok=True)

    # --- NOVITÀ: LISTA DI ESCLUSIONE ---
    # Definiamo gli ID da escludere (formato intero per comodità)
    exclude_ids = [4, 10, 13, 15, 32, 33, 44]
    exclude_keys = [f"S{str(i).zfill(2)}" for i in exclude_ids]
    
    # 1. CARICAMENTO DATI
    loader = BidmcDataLoader()
    # Carichiamo inizialmente tutti i 53 soggetti
    raw_data = loader.load_subjects([str(i).zfill(2) for i in range(1, 54)], base_path)
    
    # Filtraggio dei soggetti prima dello split
    all_keys = list(raw_data['subjects_data'].keys())
    available_keys = [k for k in all_keys if k not in exclude_keys]
    
    print(f"Soggetti totali caricati: {len(all_keys)}")
    print(f"Soggetti esclusi: {exclude_keys}")
    print(f"Soggetti utilizzati per la pipeline: {len(available_keys)}")

    # 2. SPLITTING DEI SOGGETTI (su lista filtrata)
    preprocessor = BidmcPreprocessor(fs=configs['target_fs'], beat_len=configs['beat_len'])
    train_keys, val_keys, test_keys = preprocessor.split_subjects(
        available_keys,
        train_ratio=configs.get('train_ratio', 0.8),
        val_ratio=configs.get('val_ratio', 0.1),
        seed=configs.get('seed', 42)
    )

    # 3. PREPROCESSING PER SET
    print("Creazione Train Dataset...")
    train_ds = _prepare_data(train_keys, raw_data, preprocessor, configs)
    
    print("Creazione Val Dataset...")
    val_ds = _prepare_data(val_keys, raw_data, preprocessor, configs)
    
    print("Creazione Test Dataset...")
    test_ds = _prepare_data(test_keys, raw_data, preprocessor, configs)

    # Rilevamento dimensioni
    actual_input_channels = train_ds[0][0].shape[0] if configs.get('apply_wst', False) else 1
    actual_seq_len = train_ds[0][0].shape[-1]
    configs['input_channels'] = actual_input_channels
    configs['actual_seq_len'] = actual_seq_len

    # Salvataggio Config (include i soggetti esclusi per memoria futura)
    serializable_configs = configs.copy()
    serializable_configs['device'] = str(device)
    serializable_configs['excluded_subjects'] = exclude_keys
    with open(os.path.join(save_dir, 'experiment_configs.json'), 'w') as f:
        json.dump(serializable_configs, f, indent=4)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=configs['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=configs['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=configs['batch_size'], shuffle=False)

    # 4. MODELLO E TRAINER
    model = HACNNBiLSTM(configs=configs, seq_len=configs['beat_len'])
    trainer = HACNNBiLSTMTrainer(model, device, configs)

    # 5. TRAINING
    history = trainer.fit(train_loader, val_loader, epochs=configs['epochs'], patience=configs['patience'])

    # 6. SALVATAGGIO E REPORTISTICA
    final_model_path = os.path.join(save_dir, "ha_cnn_bilstm_final.pth")
    torch.save(model.state_dict(), final_model_path)
    save_training_history(history, save_dir)
    plot_training_history_metrics(history, save_dir)
    
    best_model_path = os.path.join(save_dir, 'best_ha_cnn_bilstm.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    print("\n--- Generazione Report Grafici Estesi ---")
    save_extended_reports(model, train_keys, raw_data, preprocessor, device, configs, "train")
    save_extended_reports(model, val_keys, raw_data, preprocessor, device, configs, "val")
    save_extended_reports(model, test_keys, raw_data, preprocessor, device, configs, "test")
    save_inference_plot(model, test_loader, device, os.path.join(save_dir, "final_test_batch_inference.png"))

    print(f"Pipeline completata. Risultati in: {save_dir}")
    return history