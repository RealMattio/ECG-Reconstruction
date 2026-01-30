import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import json # Necessario per il salvataggio dei parametri

# Importazione dei moduli precedentemente definiti
from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.preprocessing.bidmc_preprocessor import BidmcPreprocessor
from src.bidmc_generation.models.ha_cnn_bilstm import HACNNBiLSTM
from src.bidmc_generation.trainer import HACNNBiLSTMTrainer
from src.evaluation.visualization import save_inference_plot

def run_ha_cnn_bilstm_pipeline(base_path, configs):
    """
    Pipeline principale aggiornata con salvataggio automatico dei parametri.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")
    
    # Creazione cartella di destinazione
    save_dir = os.path.abspath(configs['model_save_path'])
    os.makedirs(save_dir, exist_ok=True)
    configs['model_save_path'] = save_dir 
    print(f"I modelli verranno salvati in: {save_dir}")

    # --- NOVITÀ: SALVATAGGIO CONFIGURAZIONE ---
    # Convertiamo torch.device in stringa per renderlo compatibile con JSON
    serializable_configs = configs.copy()
    if 'device' in serializable_configs:
        serializable_configs['device'] = str(serializable_configs['device'])
    
    config_save_path = os.path.join(save_dir, 'experiment_configs.json')
    with open(config_save_path, 'w') as f:
        json.dump(serializable_configs, f, indent=4)
    print(f"Parametri di addestramento salvati in: {config_save_path}")
    # ------------------------------------------

    # 1. CARICAMENTO DATI
    subject_ids = [str(i).zfill(2) for i in range(1, 54)]
    loader = BidmcDataLoader()
    raw_data = loader.load_subjects(subject_ids, base_path)

    # 2. PREPROCESSING MODULARE
    preprocessor = BidmcPreprocessor(fs=configs['target_fs'], beat_len=configs['beat_len'])
    all_ppg_beats, all_ecg_beats = [], []

    print(f"Inizio preprocessing (WST: {configs.get('apply_wst', False)}, Overlap: {configs.get('overlap_windows', False)})")
    for s_id, data in raw_data['subjects_data'].items():
        ppg_b, ecg_b = preprocessor.process_subject(data['PPG'], data['ECG'], configs)
        if len(ppg_b) > 0:
            all_ppg_beats.append(ppg_b)
            all_ecg_beats.append(ecg_b)

    X = np.concatenate(all_ppg_beats, axis=0)
    y = np.concatenate(all_ecg_beats, axis=0)
    
    # Rilevamento dimensioni post-preprocessing
    actual_input_channels = X.shape[1] if configs.get('apply_wst', False) else 1
    actual_seq_len = X.shape[-1] 
    
    # Aggiornamento parametri dinamici
    configs['input_channels'] = actual_input_channels
    configs['actual_seq_len'] = actual_seq_len

    # Creazione Tensori
    if configs.get('apply_wst', False):
        X_tensor = torch.tensor(X).float()
    else:
        X_tensor = torch.tensor(X).float().unsqueeze(1)
    
    y_tensor = torch.tensor(y).float()
    if len(y_tensor.shape) == 2: y_tensor = y_tensor.unsqueeze(1)

    print(f"Totale battiti: {X_tensor.shape[0]} | Input Channels: {actual_input_channels}")

    # 3. SPLIT DATASET
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.9 * len(dataset))
    test_dataset, train_dataset = random_split(dataset, [len(dataset)-train_size, train_size])

    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False)

    # 4. INIZIALIZZAZIONE MODELLO E TRAINER
    model = HACNNBiLSTM(configs=configs, seq_len=configs['beat_len'])
    trainer = HACNNBiLSTMTrainer(model, device, configs)

    # 5. ADDESTRAMENTO
    history = trainer.fit(
        train_loader, 
        test_loader, 
        epochs=configs['epochs'], 
        patience=configs['patience']
    )

    # 6. SALVATAGGIO FINALE
    save_path = os.path.join(configs['model_save_path'], "ha_cnn_bilstm_final.pth")
    torch.save(model.state_dict(), save_path)

    # Aggiornamento file JSON con i parametri rilevati dinamicamente (channels e seq_len)
    with open(config_save_path, 'w') as f:
        serializable_configs.update({
            'input_channels': actual_input_channels,
            'actual_seq_len': actual_seq_len
        })
        json.dump(serializable_configs, f, indent=4)

    print("Generazione grafico di inferenza...")
    plot_save_path = os.path.join(configs['model_save_path'], "final_validation_inference.png")
    model.load_state_dict(torch.load(os.path.join(configs['model_save_path'], 'best_ha_cnn_bilstm.pth')))
    save_inference_plot(model, test_loader, device, plot_save_path)

    return history