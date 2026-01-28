import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import resample
from src.data_loader.data_loader import DaliaDataLoader
from src.generation.approach3.approach3_model import Approach3Model

def run_inference_s1():
    # 1. CONFIGURAZIONE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = 'data' # Assicurati che il percorso sia corretto
    subject_id = 'S1'
    target_fs = 125
    window_sec = 7
    window_samples = window_sec * target_fs # 875 campioni

    # Percorsi dei modelli (Adeguare se i nomi file sono diversi)
    # Nota: Per l'inferenza "con ECG" serve il modello a 6 canali della Fase 1
    # Per l'inferenza "senza ECG" serve il modello a 5 canali della Fase 2
    model_path_final = f"src/generation/models/approach3/model_app3_test_sub_{subject_id}.pth"

    # 2. CARICAMENTO DATI SOGGETTO S1
    loader = DaliaDataLoader()
    raw_data = loader.load_subjects([subject_id], base_path, is_resampled=False)
    data = raw_data['subjects_data'][subject_id]

    # Preprocessing minimo (Z-score e Resampling)
    ppg = (data['PPG'] - data['PPG'].mean()) / (data['PPG'].std() + 1e-8)
    ecg = (data['ECG'] - data['ECG'].mean()) / (data['ECG'].std() + 1e-8)
    acc = data['ACC']
    eda = data['EDA']

    num_samples_target = int(len(ppg) * (target_fs / data['fs_ppg']))
    ppg_res = resample(ppg, num_samples_target)
    ecg_res = resample(ecg, num_samples_target)
    acc_res = np.array([resample(acc[:, i], num_samples_target) for i in range(3)]).T
    eda_res = resample(eda, num_samples_target)

    # Prendiamo una finestra casuale per il test
    idx = 1000 # Punto di inizio finestra
    p_win = torch.tensor(ppg_res[idx:idx+window_samples]).float().view(1, 1, -1)
    a_win = torch.tensor(acc_res[idx:idx+window_samples]).float().transpose(0, 1).unsqueeze(0)
    e_win = torch.tensor(eda_res[idx:idx+window_samples]).float().view(1, 1, -1)
    target_win = torch.tensor(ecg_res[idx:idx+window_samples]).float().view(1, 1, -1)

    # 3. CARICAMENTO MODELLI
    # Modello Fase 2 (Specializzato - 5 canali)
    factory_final = Approach3Model(pretrain_mode=False)
    gen_final, _ = factory_final.get_models()
    if os.path.exists(model_path_final):
        gen_final.load_state_dict(torch.load(model_path_final, map_location=device))
        print(f"✓ Modello finale (5 ch) caricato per {subject_id}")
    gen_final.to(device).eval()

    # Modello Fase 1 (Pre-trained - 6 canali)
    # Nota: Qui lo inizializziamo per vedere come 'avrebbe' performato con l'ECG. 
    # Idealmente dovresti caricare i pesi salvati a fine Fase 1.
    factory_pre = Approach3Model(pretrain_mode=True)
    gen_pre, _ = factory_pre.get_models()
    gen_pre.to(device).eval()

    # 4. INFERENZA
    with torch.no_grad():
        # Input senza ECG (5 canali): PPG + ACC + EDA
        input_5ch = torch.cat([p_win, a_win, e_win], dim=1).transpose(1, 2).to(device)
        out_no_ecg = gen_final(input_5ch).cpu().squeeze().numpy()

        # Input con ECG (6 canali): PPG + ACC + EDA + Real ECG
        input_6ch = torch.cat([p_win, a_win, e_win, target_win], dim=1).transpose(1, 2).to(device)
        out_with_ecg = gen_pre(input_6ch).cpu().squeeze().numpy()

    # 5. PLOTTING
    plt.figure(figsize=(15, 14))
    
    # Subplot 1: Segnali Sensori
    plt.subplot(4, 1, 1)
    plt.plot(ppg_res[idx:idx+window_samples], label='PPG', color='blue')
    plt.title(f"Inference Subject {subject_id} - Input Sensors")
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(acc_res[idx:idx+window_samples, 0], label='ACC X')
    plt.plot(acc_res[idx:idx+window_samples, 1], label='ACC Y')
    plt.plot(acc_res[idx:idx+window_samples, 2], label='ACC Z')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(eda_res[idx:idx+window_samples], label='EDA', color='green')
    plt.legend()

    # Subplot 2: Confronto Output
    plt.subplot(4, 1, 4)
    time_axis = np.linspace(0, window_sec, window_samples)
    plt.plot(time_axis, ecg_res[idx:idx+window_samples], label='Ground Truth (Real ECG)', color='black', alpha=0.3, linestyle='--')
    plt.plot(time_axis, out_with_ecg, label='Output with ECG Input (Phase 1)', color='purple', alpha=0.7)
    plt.plot(time_axis, out_no_ecg, label='Output WITHOUT ECG Input (Phase 2)', color='red', linewidth=1.5)
    
    plt.title("ECG Reconstruction Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = f"inference_comparison_{subject_id}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"✓ Grafico di confronto salvato in: {save_path}")

if __name__ == "__main__":
    run_inference_s1()