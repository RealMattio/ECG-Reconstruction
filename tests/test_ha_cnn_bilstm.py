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

def generate_long_window_smooth(model, ppg_signal, preprocessor, device, configs, window_len=120, overlap_pct=0.5):
    """
    Genera un ECG lungo gestendo la Wavelet Scattering se attiva.
    """
    step = int(window_len * (1 - overlap_pct))
    output_ecg = np.zeros(len(ppg_signal))
    count_map = np.zeros(len(ppg_signal))
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ppg_signal) - window_len, step):
            seg = ppg_signal[i : i + window_len]
            
            # --- PREPARAZIONE INPUT (Raw o WST) ---
            if configs.get('apply_wst', False):
                # Trasforma il segmento raw in feature WST (es. 8x30)
                # Serve aggiungere dimensione batch per il preprocessor
                seg_input = preprocessor.extract_wst_features(np.expand_dims(seg, axis=0))
                seg_t = torch.tensor(seg_input).float().to(device) # Già (1, 8, 30)
            else:
                seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inferenza
            pred = model(seg_t).cpu().numpy().flatten()
            
            # Ricomposizione (L'output del modello è sempre 120 campioni)
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    count_map[count_map == 0] = 1
    return output_ecg / count_map

def test_and_visualize():
    # 1. CONFIGURAZIONE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'src/bidmc_generation/models/ha_cnn_bilstm_withWST/best_ha_cnn_bilstm.pth'
    data_path = 'bidmc_data'
    target_fs = 125 
    beat_len = 120 

    # IMPORTANTE: Definire configs coerenti con il modello addestrato
    test_configs = {
        'apply_wst': True,          # Imposta True se il modello usa Wavelet Scattering
        'overlap_windows': False,   # Per il caricamento batch standard
        'input_channels': 8,        # Numero canali rilevati durante il training WST
        'actual_seq_len': 30        # Lunghezza sequenza ridotta da WST (se J=2)
    }
    
    test_subject_ids = [str(i).zfill(2) for i in range(48, 54)]
    print(f"--- Testing HA-CNN-BILSTM (WST: {test_configs['apply_wst']}) ---")

    # 2. CARICAMENTO MODELLO
    # Correzione: Passiamo test_configs al costruttore
    model = HACNNBiLSTM(configs=test_configs, seq_len=beat_len).to(device)
    
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modello caricato correttamente.")

    # 3. CARICAMENTO E PREPROCESSING
    loader = BidmcDataLoader()
    preprocessor = BidmcPreprocessor(fs=target_fs, beat_len=beat_len)
    raw_data = loader.load_subjects(test_subject_ids, data_path)
    
    all_ppg_beats = []
    all_ecg_beats = []
    
    print("Elaborazione dati di test...")
    for s_id, data in raw_data['subjects_data'].items():
        ppg_beats, ecg_beats = preprocessor.process_subject(data['PPG'], data['ECG'], test_configs)
        if len(ppg_beats) > 0:
            all_ppg_beats.append(ppg_beats)
            all_ecg_beats.append(ecg_beats)
            
    # Trasformazione in Tensori
    X = np.concatenate(all_ppg_beats)
    y = np.concatenate(all_ecg_beats)

    if test_configs['apply_wst']:
        X_test = torch.tensor(X).float().to(device)
    else:
        X_test = torch.tensor(X).float().unsqueeze(1).to(device)
    
    y_test = torch.tensor(y).float().unsqueeze(1).to(device)
    
    # 4. INFERENZA GLOBALE
    with torch.no_grad():
        y_pred = model(X_test)
        rmse = torch.sqrt(torch.mean((y_pred - y_test)**2)).item()
        print(f"\n>> TEST SET GLOBAL RMSE: {rmse:.4f} <<\n")

    # 5. VISUALIZZAZIONE SAMPLE CASUALI
    indices = random.sample(range(len(X_test)), 2)
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[idx, 0].cpu().numpy(), label='ECG Reale', color='black', alpha=0.4, linestyle='--')
        plt.plot(y_pred[idx, 0].cpu().numpy(), label='ECG Generato', color='red')
        plt.title(f"Test Random Segment {i+1} (RMSE: {rmse:.4f})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f"test_beat_{i+1}.png"); plt.close()

    # 6. FINESTRA LUNGA (~7.2s)
    s_key = list(raw_data['subjects_data'].keys())[0]
    subj = raw_data['subjects_data'][s_key]
    
    ppg_norm = preprocessor.normalize_signal(preprocessor.apply_bandpass_filter(subj['PPG']))
    ecg_norm = preprocessor.normalize_signal(subj['ECG'])
    
    total_len = 900 
    start = random.randint(0, len(ppg_norm) - total_len)
    long_ppg = ppg_norm[start : start + total_len]
    long_ecg_real = ecg_norm[start : start + total_len]
    
    long_ecg_gen = generate_long_window_smooth(
        model, long_ppg, preprocessor, device, test_configs, window_len=beat_len, overlap_pct=0.5
    )
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(long_ppg, color='blue', label='Input PPG')
    plt.legend(); plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(long_ecg_real, color='black', alpha=0.3, label='ECG Reale')
    plt.plot(long_ecg_gen, color='red', label='ECG Ricostruito (Smooth)')
    plt.xlabel("Samples"); plt.legend(); plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("long_window_smooth_reconstruction.png")
    print("Test completato.")

if __name__ == "__main__":
    test_and_visualize()