import pickle
import numpy as np
import os
from scipy import signal

def extract_original_data_with_ecg_resampling(input_path, output_path):
    """
    Carica i dati di un soggetto, raggruppa i segnali originali (PPG, ACC, EDA)
    ed esegue il downsampling dell'ECG a 256 Hz.
    """
    # 1. Caricamento del file originale
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Estrazione dei segnali originali
    ppg = data['signal']['wrist']['BVP'].flatten()  # 64 Hz
    acc = data['signal']['wrist']['ACC']            # 32 Hz
    eda = data['signal']['wrist']['EDA'].flatten()  # 4 Hz
    ecg_original = data['signal']['chest']['ECG'].flatten() # 700 Hz

    # 2. Definizione delle frequenze
    fs_original_ecg = 700
    fs_target_ecg = 256
    
    # Calcolo del numero di campioni per il downsampling dell'ECG
    # Formula: numero_campioni_originali * (fs_target / fs_original)
    num_samples_target_ecg = int(len(ecg_original) * fs_target_ecg / fs_original_ecg)

    print(f"Inizio elaborazione per {data['subject']}...")

    # 3. Downsampling solo dell'ECG
    ecg_resampled = signal.resample(ecg_original, num_samples_target_ecg)

    # 4. Creazione del dizionario finale (struttura invariata)
    original_dict = {
        'subject': data['subject'],
        'PPG': ppg,
        'ACC': acc,
        'EDA': eda,
        'ECG': ecg_resampled,
        'fs_ppg': 64,
        'fs_acc': 32,
        'fs_eda': 4,
        'fs_ecg': fs_target_ecg,
        'activity': data['activity'],
        'label': data['label']
    }

    # 5. Salvataggio in pkl
    with open(output_path, 'wb') as f:
        pickle.dump(original_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Salvataggio completato: {output_path}")
    print(f"Dimensioni: PPG - {len(ppg)}, EDA - {len(eda)}, ACC - {len(acc)}, ECG (Resampled) - {len(ecg_resampled)}")

# Esempio di utilizzo
if __name__ == "__main__":
    sample = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15"]
    
    for subject_id in sample:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_file = os.path.join(base_path, 'data', 'raw', f"{subject_id}.pkl")
        
        output_dir = os.path.join(base_path, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{subject_id}_original.pkl")
        
        if os.path.exists(input_file):
            extract_original_data_with_ecg_resampling(input_file, output_file)
        else:
            print(f"Errore: File {input_file} non trovato.")