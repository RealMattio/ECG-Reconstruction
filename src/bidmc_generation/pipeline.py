import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.preprocessing.bidmc_preprocessor import BidmcPreprocessor
from src.bidmc_generation.models.ha_cnn_bilstm import HACNNBiLSTM
from src.bidmc_generation.trainer import HACNNBiLSTMTrainer
from src.evaluation.visualization import save_inference_plot

def run_ha_cnn_bilstm_pipeline(base_path, configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")
    save_dir = os.path.abspath(configs['model_save_path'])
    os.makedirs(save_dir, exist_ok=True)
    configs['model_save_path'] = save_dir # Aggiorniamo nelle configs per il trainer
    print(f"I modelli verranno salvati in: {save_dir}")

    # 1. CARICAMENTO DATI
    subject_ids = [str(i).zfill(2) for i in range(1, 54)]
    loader = BidmcDataLoader()
    raw_data = loader.load_subjects(subject_ids, base_path)

    # 2. PREPROCESSING MODULARE
    preprocessor = BidmcPreprocessor(fs=125, beat_len=120)
    all_ppg_beats, all_ecg_beats = [], []

    print(f"Inizio preprocessing (WST: {configs.get('apply_wst', False)}, Overlap: {configs.get('overlap_windows', False)})")
    for s_id, data in raw_data['subjects_data'].items():
        ppg_b, ecg_b = preprocessor.process_subject(data['PPG'], data['ECG'], configs)
        if len(ppg_b) > 0:
            all_ppg_beats.append(ppg_b)
            all_ecg_beats.append(ecg_b)

    X = np.concatenate(all_ppg_beats, axis=0)
    y = np.concatenate(all_ecg_beats, axis=0)
    
    # Rileviamo le dimensioni REALI post-preprocessing
    # Se WST è attivo, X ha forma (Batch, 8, 30). Se raw, (Batch, 120)
    actual_input_channels = X.shape[1] if configs.get('apply_wst', False) else 1
    actual_seq_len = X.shape[-1] 
    
    # Aggiorniamo le configs per il modello
    configs['input_channels'] = actual_input_channels
    configs['actual_seq_len'] = actual_seq_len

    # Creazione Tensori
    if configs.get('apply_wst', False):
        X_tensor = torch.tensor(X).float()
    else:
        X_tensor = torch.tensor(X).float().unsqueeze(1)
    
    # L'ECG target rimane (Batch, 1, 120) [o DWT se implementato]
    y_tensor = torch.tensor(y).float()
    if len(y_tensor.shape) == 2: y_tensor = y_tensor.unsqueeze(1)

    print(f"Totale battiti processati: {X_tensor.shape[0]} | Shape X: {X_tensor.shape}")

    # 3. SPLIT DATASET
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.9 * len(dataset))
    test_dataset, train_dataset = random_split(dataset, [len(dataset)-train_size, train_size]) # Inversione per test piccolo

    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False)

    # 4. INIZIALIZZAZIONE MODELLO (Passiamo configs)
    model = HACNNBiLSTM(configs=configs, seq_len=120)
    
    trainer = HACNNBiLSTMTrainer(model, device, configs)

    # 5. ADDESTRAMENTO
    history = trainer.fit(
        train_loader, 
        test_loader, 
        epochs=configs['epochs'], 
        patience=configs['patience']
    )

    # 6. SALVATAGGIO E INFERENZA
    save_path = os.path.join(configs['model_save_path'], "ha_cnn_bilstm_final.pth")
    torch.save(model.state_dict(), save_path)

    print("Generazione grafico di inferenza...")
    plot_save_path = os.path.join(configs['model_save_path'], "final_validation_inference.png")
    
    # Carichiamo il miglior modello per il plot
    model.load_state_dict(torch.load(os.path.join(configs['model_save_path'], 'best_ha_cnn_bilstm.pth')))
    save_inference_plot(model, test_loader, device, plot_save_path)

    return history