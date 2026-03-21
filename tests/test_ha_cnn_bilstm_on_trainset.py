import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import DataLoader, TensorDataset

# Importazione dei moduli del tuo progetto
from src.bidmc_generation.models.ha_cnn_bilstm import HACNNBiLSTM
from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.preprocessing.bidmc_preprocessor import BidmcPreprocessor

def generate_long_window_smooth(model, ppg_signal, preprocessor, device, configs):
    """
    Genera un ECG lungo gestendo dinamicamente overlap e WST.
    Risolve il problema dei campioni nulli alla fine coprendo l'intero segnale.
    """
    window_len = configs['beat_len']
    overlap_pct = configs.get('overlap_pct', 0.5)
    step = int(window_len * (1 - overlap_pct))
    
    total_samples = len(ppg_signal)
    output_ecg = np.zeros(total_samples)
    count_map = np.zeros(total_samples)
    
    model.eval()
    with torch.no_grad():
        # Generiamo la lista degli indici di inizio
        start_indices = list(range(0, total_samples - window_len, step))
        # Forziamo l'inclusione dell'ultima finestra per evitare campioni nulli
        if total_samples > window_len and start_indices[-1] + window_len < total_samples:
            start_indices.append(total_samples - window_len)
            
        for i in start_indices:
            seg = ppg_signal[i : i + window_len]
            
            if configs.get('apply_wst', False):
                seg_input = preprocessor.extract_wst_features(np.expand_dims(seg, axis=0))
                seg_t = torch.tensor(seg_input).float().to(device)
            else:
                seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(seg_t).cpu().numpy().flatten()
            
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    count_map[count_map == 0] = 1
    return output_ecg / count_map

def test_and_visualize():
    # 1. CONFIGURAZIONE PARAMETRI (MODULARI)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_fs = 125 # Frequenza di campionamento BIDMC
    
    # Percorso cartella timestamp corretta
    model_folder = 'src/bidmc_generation/models/ha_cnn_bilstm_withWST/20260130_154303/'
    model_path = os.path.join(model_folder, 'best_ha_cnn_bilstm.pth')
    data_path = 'bidmc_data'
    
    test_configs = {
        'beat_len': 120,            # Lunghezza finestra ridotta
        'overlap_pct': 0.5,        # Overlap usato nel training
        'apply_wst': True,         
        'overlap_windows': False    
    }
    
    print(f"--- Analisi Grafica e Performance su TRAINING SET ---")

    # 2. RILEVAMENTO DINAMICO DIMENSIONI
    preprocessor = BidmcPreprocessor(fs=target_fs, beat_len=test_configs['beat_len'])
    dummy_seg = np.zeros((1, test_configs['beat_len']))
    if test_configs['apply_wst']:
        dummy_wst = preprocessor.extract_wst_features(dummy_seg)
        test_configs['input_channels'] = dummy_wst.shape[1]
        test_configs['actual_seq_len'] = dummy_wst.shape[2]
    else:
        test_configs['input_channels'] = 1
        test_configs['actual_seq_len'] = test_configs['beat_len']

    # 3. CARICAMENTO MODELLO
    model = HACNNBiLSTM(configs=test_configs, seq_len=test_configs['beat_len']).to(device)
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. PREPROCESSING DATI DI TRAINING (Soggetti 01-47)
    loader = BidmcDataLoader()
    train_subject_ids = [str(i).zfill(2) for i in range(1, 48)]
    raw_data = loader.load_subjects(train_subject_ids, data_path)
    
    all_ppg_beats, all_ecg_beats = [], []
    for s_id, data in raw_data['subjects_data'].items():
        ppg_b, ecg_b = preprocessor.process_subject(data['PPG'], data['ECG'], test_configs)
        if len(ppg_b) > 0:
            all_ppg_beats.append(ppg_b)
            all_ecg_beats.append(ecg_b)
            
    X = np.concatenate(all_ppg_beats)
    y = np.concatenate(all_ecg_beats)
    X_train = torch.tensor(X).float().to(device)
    if not test_configs['apply_wst']: X_train = X_train.unsqueeze(1)
    y_train = torch.tensor(y).float().unsqueeze(1).to(device)
    
    # 5. INFERENZA E VISUALIZZAZIONE BATTITI SINGOLI
    with torch.no_grad():
        y_pred = model(X_train)
        rmse = torch.sqrt(torch.mean((y_pred - y_train)**2)).item()
        print(f"\n>> TRAINING SET GLOBAL RMSE: {rmse:.4f} <<\n")

    indices = random.sample(range(len(X_train)), 2)
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 4))
        # Asse X in secondi per il singolo segmento
        time_segment = np.arange(test_configs['beat_len']) / target_fs
        plt.plot(time_segment, y_train[idx, 0].cpu().numpy(), label='ECG Reale', color='black', alpha=0.4, linestyle='--')
        plt.plot(time_segment, y_pred[idx, 0].cpu().numpy(), label='ECG Generato', color='red')
        plt.title(f"Train Segment {i+1} ({test_configs['beat_len']} samples)")
        plt.xlabel("Tempo (s)")
        plt.legend(); plt.grid(alpha=0.3)
        
        save_name = os.path.join(model_folder, f"train_beat_{i+1}.png")
        plt.savefig(save_name); plt.close()

    # 6. FINESTRA LUNGA (Visualizzazione in Secondi e Fix campioni nulli)
    s_key = list(raw_data['subjects_data'].keys())[0]
    subj = raw_data['subjects_data'][s_key]
    ppg_norm = preprocessor.normalize_signal(preprocessor.apply_bandpass_filter(subj['PPG']))
    ecg_norm = preprocessor.normalize_signal(subj['ECG'])
    
    # Finestra di circa 7.2s (900 campioni)
    total_len = 900 
    start = random.randint(0, len(ppg_norm) - total_len)
    long_ppg = ppg_norm[start : start + total_len]
    long_ecg_real = ecg_norm[start : start + total_len]
    
    long_ecg_gen = generate_long_window_smooth(model, long_ppg, preprocessor, device, test_configs)
    
    # Creazione asse temporale corretto
    time_axis = np.arange(len(long_ppg)) / target_fs
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, long_ppg, color='blue', label='Input PPG (Train)')
    plt.title(f"Segnale PPG - Soggetto {s_key}")
    plt.ylabel("Ampiezza Norm.")
    plt.grid(alpha=0.3); plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, long_ecg_real, color='black', alpha=0.3, label='ECG Reale')
    plt.plot(time_axis, long_ecg_gen, color='red', label='ECG Ricostruito (Smooth Stitching)')
    plt.title("Ricostruzione ECG - Visualizzazione Temporale (Secondi)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza Norm.")
    plt.legend(); plt.grid(alpha=0.3)
    
    plt.tight_layout()
    long_save_name = os.path.join(model_folder, "train_long_window_seconds.png")
    plt.savefig(long_save_name)
    print(f"Analisi completata. Grafico finale salvato in: {long_save_name}")

if __name__ == "__main__":
    test_and_visualize()