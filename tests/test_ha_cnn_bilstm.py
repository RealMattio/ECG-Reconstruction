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

def generate_long_window_smooth(model, ppg_signal, device, window_len=120, overlap_pct=0.5):
    """
    Genera un ECG lungo usando una sliding window e facendo la media degli overlap.
    Questo rimuove le discontinuità ai bordi dei segmenti.
    """
    step = int(window_len * (1 - overlap_pct))
    output_ecg = np.zeros(len(ppg_signal))
    count_map = np.zeros(len(ppg_signal)) # Mappa per la media pesata
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ppg_signal) - window_len, step):
            seg = ppg_signal[i : i + window_len]
            # Prepariamo il tensore (Batch=1, Channel=1, Seq=120)
            seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inferenza: il modello restituisce (1, 1, 120)
            pred = model(seg_t).cpu().numpy().flatten()
            
            # Sommiamo i valori e incrementiamo il contatore per la media
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    # Evitiamo divisioni per zero e calcoliamo la media
    count_map[count_map == 0] = 1
    return output_ecg / count_map

def test_and_visualize():
    # 1. CONFIGURAZIONE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'src/bidmc_generation/models/ha_cnn_bilstm/best_ha_cnn_bilstm.pth'
    data_path = 'bidmc_data'
    target_fs = 125 # Frequenza MIMIC II 
    beat_len = 120 # Lunghezza battito Tabella 1 
    
    # Configurazione per il preprocessor
    test_configs = {'overlap_windows': True} 
    
    test_subject_ids = [str(i).zfill(2) for i in range(48, 54)]
    print(f"--- Testing HA-CNN-BILSTM on Subjects: {test_subject_ids} ---")

    # 2. CARICAMENTO MODELLO
    model = HACNNBiLSTM(input_dim=1, output_dim=1, seq_len=beat_len).to(device)
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modello caricato correttamente.")

    # 3. CARICAMENTO E PREPROCESSING DATI DI TEST
    loader = BidmcDataLoader()
    preprocessor = BidmcPreprocessor(fs=target_fs, beat_len=beat_len)
    raw_data = loader.load_subjects(test_subject_ids, data_path)
    
    all_ppg_beats = []
    all_ecg_beats = []
    
    print("Elaborazione battiti del test set (con overlap)...")
    for s_id, data in raw_data['subjects_data'].items():
        # Passiamo test_configs correttamente
        ppg_beats, ecg_beats = preprocessor.process_subject(data['PPG'], data['ECG'], test_configs)
        if len(ppg_beats) > 0:
            all_ppg_beats.append(ppg_beats)
            all_ecg_beats.append(ecg_beats)
            
    # Trasformazione in Tensori
    X_test = torch.tensor(np.concatenate(all_ppg_beats)).float().unsqueeze(1).to(device)
    y_test = torch.tensor(np.concatenate(all_ecg_beats)).float().unsqueeze(1).to(device)
    
    # 4. INFERENZA SU TUTTO IL TEST SET
    print(f"Esecuzione inferenza su {len(X_test)} segmenti...")
    with torch.no_grad():
        y_pred = model(X_test)
        # RMSE loss come da paper [cite: 305]
        rmse = torch.sqrt(torch.mean((y_pred - y_test)**2)).item()
        print(f"\n>> TEST SET GLOBAL RMSE: {rmse:.4f} <<\n")

    # 5. VISUALIZZAZIONE DI DUE BATTITI CASUALI
    indices = random.sample(range(len(X_test)), 2)
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[idx, 0].cpu().numpy(), label='ECG Reale', color='black', alpha=0.4, linestyle='--')
        plt.plot(y_pred[idx, 0].cpu().numpy(), label='ECG Generato', color='red')
        plt.title(f"Test Random Segment {i+1}")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f"test_beat_{i+1}.png"); plt.close()

    # 6. GENERAZIONE FINESTRA LUNGA (~7.2 secondi) CON STITCHING FLUIDO
    s_key = list(raw_data['subjects_data'].keys())[0]
    subject_data = raw_data['subjects_data'][s_key]
    
    # Preprocessing continuo
    ppg_filt = preprocessor.apply_bandpass_filter(subject_data['PPG'])
    ppg_norm = preprocessor.normalize_signal(ppg_filt)
    ecg_norm = preprocessor.normalize_signal(subject_data['ECG'])
    
    # Finestra di circa 7.2 secondi (900 campioni)
    total_len = 900 
    start_idx = random.randint(0, len(ppg_norm) - total_len)
    
    long_ppg = ppg_norm[start_idx : start_idx + total_len]
    long_ecg_real = ecg_norm[start_idx : start_idx + total_len]
    
    # Utilizzo della funzione smooth per evitare discontinuità
    long_ecg_gen = generate_long_window_smooth(
        model, long_ppg, device, window_len=beat_len, overlap_pct=0.5
    )
    
    # Plot Finale
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(long_ppg, color='blue', label='Input PPG')
    plt.title("Input PPG Continuo")
    plt.grid(alpha=0.3); plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(long_ecg_real, color='black', alpha=0.3, label='ECG Reale')
    plt.plot(long_ecg_gen, color='red', label='ECG Ricostruito (Smooth Stitching)')
    plt.title(f"Ricostruzione ECG Continua (~{total_len/target_fs:.1f}s)")
    plt.xlabel("Samples"); plt.legend(); plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("long_window_smooth_reconstruction.png")
    print("Test completato. Grafici salvati.")

if __name__ == "__main__":
    test_and_visualize()