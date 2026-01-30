import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import datetime
from torch.utils.data import DataLoader, TensorDataset

# Importazione dei moduli del tuo progetto
from src.bidmc_generation.models.ha_cnn_bilstm import HACNNBiLSTM
from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.preprocessing.bidmc_preprocessor import BidmcPreprocessor

def generate_long_window_smooth(model, ppg_signal, preprocessor, device, configs):
    """
    Genera un ECG lungo gestendo dinamicamente overlap e WST.
    """
    window_len = configs['beat_len']
    overlap_pct = configs.get('overlap_pct', 0.5)
    step = int(window_len * (1 - overlap_pct))
    
    output_ecg = np.zeros(len(ppg_signal))
    count_map = np.zeros(len(ppg_signal))
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ppg_signal) - window_len, step):
            seg = ppg_signal[i : i + window_len]
            
            if configs.get('apply_wst', False):
                # Trasforma il segmento raw in feature WST
                seg_input = preprocessor.extract_wst_features(np.expand_dims(seg, axis=0))
                seg_t = torch.tensor(seg_input).float().to(device)
            else:
                seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(seg_t).cpu().numpy().flatten()
            
            # L'output del modello è sempre della lunghezza originale della finestra (es. 60)
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    count_map[count_map == 0] = 1
    return output_ecg / count_map

def test_and_visualize():
    # 1. CONFIGURAZIONE PARAMETRI DI TEST (MODULARI)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # AGGIORNA QUESTO PERCORSO con la tua cartella timestamp
    model_folder = 'src/bidmc_generation/models/ha_cnn_bilstm_withWST/20260130_141822/'
    model_path = os.path.join(model_folder, 'best_ha_cnn_bilstm.pth')
    
    data_path = 'bidmc_data'
    
    # Configurazione speculare a quella usata nel main del training
    test_configs = {
        'beat_len': 60,            # Lunghezza usata nel training
        'overlap_pct': 0.1,        # Overlap usato nel training (10%)
        'apply_wst': True,         # Deve essere coerente con il modello salvato
        'overlap_windows': False    # Caricamento batch standard per il test globale
    }
    
    print(f"--- Testing HA-CNN-BILSTM (Modulare) ---")
    print(f"Parametri: Beat_Len={test_configs['beat_len']}, Overlap={test_configs['overlap_pct']*100}%")

    # 2. RILEVAMENTO DINAMICO DIMENSIONI (Pre-inizializzazione)
    # Creiamo un preprocessor temporaneo per "misurare" l'effetto della WST
    preprocessor = BidmcPreprocessor(fs=125, beat_len=test_configs['beat_len'])
    
    # Testiamo la trasformazione su un array di zeri per ottenere le shape reali
    dummy_seg = np.zeros((1, test_configs['beat_len']))
    if test_configs['apply_wst']:
        dummy_wst = preprocessor.extract_wst_features(dummy_seg)
        test_configs['input_channels'] = dummy_wst.shape[1]
        test_configs['actual_seq_len'] = dummy_wst.shape[2]
    else:
        test_configs['input_channels'] = 1
        test_configs['actual_seq_len'] = test_configs['beat_len']
    
    print(f"Dimensioni rilevate: Canali={test_configs['input_channels']}, Seq_Len={test_configs['actual_seq_len']}")

    # 3. CARICAMENTO MODELLO
    model = HACNNBiLSTM(configs=test_configs, seq_len=test_configs['beat_len']).to(device)
    
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modello e pesi caricati correttamente (No Size Mismatch).")

    # 4. PREPROCESSING DATI DI TEST
    loader = BidmcDataLoader()
    test_subject_ids = [str(i).zfill(2) for i in range(48, 54)]
    raw_data = loader.load_subjects(test_subject_ids, data_path)
    
    all_ppg_beats, all_ecg_beats = [], []
    for s_id, data in raw_data['subjects_data'].items():
        ppg_b, ecg_b = preprocessor.process_subject(data['PPG'], data['ECG'], test_configs)
        if len(ppg_b) > 0:
            all_ppg_beats.append(ppg_b)
            all_ecg_beats.append(ecg_b)
            
    X = np.concatenate(all_ppg_beats)
    y = np.concatenate(all_ecg_beats)

    X_test = torch.tensor(X).float().to(device)
    if not test_configs['apply_wst']: X_test = X_test.unsqueeze(1)
    y_test = torch.tensor(y).float().unsqueeze(1).to(device)
    
    # 5. INFERENZA E VISUALIZZAZIONE
    with torch.no_grad():
        y_pred = model(X_test)
        rmse = torch.sqrt(torch.mean((y_pred - y_test)**2)).item()
        print(f"\n>> TEST SET GLOBAL RMSE: {rmse:.4f} <<\n")

    # Plot sample casuali
    indices = random.sample(range(len(X_test)), 2)
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[idx, 0].cpu().numpy(), label='ECG Reale', color='black', alpha=0.4, linestyle='--')
        plt.plot(y_pred[idx, 0].cpu().numpy(), label='ECG Generato', color='red')
        plt.title(f"Test Random Segment {i+1} (60 samples)")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f"{model_folder}test_beat_{i+1}.png"); plt.close()

    # 6. FINESTRA LUNGA (Sempre 7.2s -> 900 campioni)
    s_key = list(raw_data['subjects_data'].keys())[0]
    subj = raw_data['subjects_data'][s_key]
    ppg_norm = preprocessor.normalize_signal(preprocessor.apply_bandpass_filter(subj['PPG']))
    ecg_norm = preprocessor.normalize_signal(subj['ECG'])
    
    start = random.randint(0, len(ppg_norm) - 900)
    long_ppg = ppg_norm[start : start + 900]
    long_ecg_real = ecg_norm[start : start + 900]
    
    # Generazione fluida con i nuovi parametri modulari
    long_ecg_gen = generate_long_window_smooth(
        model, long_ppg, preprocessor, device, test_configs
    )
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(long_ppg, color='blue', label='Input PPG')
    plt.grid(alpha=0.3); plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(long_ecg_real, color='black', alpha=0.3, label='ECG Reale')
    plt.plot(long_ecg_gen, color='red', label='ECG Ricostruito (Smooth Stitching)')
    plt.xlabel("Samples"); plt.legend(); plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_folder}long_window_reconstruction.png")
    print(f"Test completato. Risultati in: {model_folder}")

if __name__ == "__main__":
    test_and_visualize()