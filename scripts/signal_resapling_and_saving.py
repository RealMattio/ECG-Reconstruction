import pickle
import numpy as np
from scipy import signal
import os

def resample_subject_data(input_path, output_path):
    """
    Carica i dati di un soggetto, esegue l'upsampling di EDA e ACC a 64 Hz,
    il downsampling dell'ECG a 256 Hz e salva il risultato.
    """
    # 1. Caricamento del file originale
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Estrazione dei segnali originali dal dizionario 'signal' [cite: 116]
    # PPG (BVP) è già a 64 Hz [cite: 21, 91]
    ppg = data['signal']['wrist']['BVP'].flatten() 
    
    # ACC (Wrist) è a 32 Hz [cite: 22, 89]
    acc = data['signal']['wrist']['ACC'] 
    
    # EDA è a 4 Hz [cite: 91]
    eda = data['signal']['wrist']['EDA'].flatten()
    
    # ECG è a 700 Hz [cite: 77, 80]
    ecg = data['signal']['chest']['ECG'].flatten()

    # 2. Definizione delle frequenze [cite: 21, 22, 80, 91]
    fs_original_ppg = 64
    fs_original_acc = 32
    fs_original_eda = 4
    fs_original_ecg = 700
    
    fs_target_input = 64   # Target per EDA e ACC
    fs_target_ecg = 256     # Target per ECG (Downsampling)

    # Calcolo della durata totale basata sul PPG (riferimento stabile a 64 Hz)
    total_seconds = len(ppg) / fs_original_ppg
    num_samples_target_input = int(total_seconds * fs_target_input)
    num_samples_target_ecg = int(total_seconds * fs_target_ecg)

    print(f"Inizio elaborazione per {data['subject']}...")

    # 3. Resampling
    
    # Upsampling ACC: da 32 Hz a 64 Hz (Fattore 2) [cite: 21, 22, 89]
    # Essendo l'ACC un segnale a 3 assi, lo ricampioniamo asse per asse
    acc_resampled = signal.resample(acc, num_samples_target_input)
    
    # Upsampling EDA: da 4 Hz a 64 Hz (Fattore 16) [cite: 21, 91]
    eda_resampled = signal.resample(eda, num_samples_target_input)
    
    # Downsampling ECG: da 700 Hz a 256 Hz 
    ecg_resampled = signal.resample(ecg, num_samples_target_ecg)

    # 4. Creazione del nuovo dizionario
    # Nota: Manteniamo il PPG originale poiché è già alla frequenza target di 64 Hz 
    resampled_dict = {
        'subject': data['subject'],
        'PPG': ppg,
        'ACC': acc_resampled,
        'EDA': eda_resampled,
        'ECG': ecg_resampled,
        'fs_input': fs_target_input,
        'fs_ecg': fs_target_ecg,
        'activity': data['activity'], # Spesso utile per la fase 2 [cite: 110]
        'label': data['label']        # Ground truth HR [cite: 112]
    }

    # 5. Salvataggio in pkl
    with open(output_path, 'wb') as f:
        pickle.dump(resampled_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Salvataggio completato: {output_path}")
    print(f"Dimensioni finali - PPG/EDA/ACC: PPG - {len(ppg)} campioni, EDA - {len(eda_resampled)} campioni, ACC - {len(acc_resampled)} campioni, ECG - {len(ecg_resampled)} campioni.")

# Esempio di utilizzo
if __name__ == "__main__":
    # Assicurati che i percorsi siano corretti
    sample = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15"]
    for subject_id in sample:
        # Risaliamo alla root del progetto partendo da scripts/
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # --- RIGHE MODIFICATE ---
        input_file = os.path.join(base_path, 'data', 'raw', f"{subject_id}.pkl")
        # Creiamo la cartella di output se non esiste
        output_dir = os.path.join(base_path, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{subject_id}_resampled.pkl")
        
        if os.path.exists(input_file):
            resample_subject_data(input_file, output_file)
        else:
            print(f"Errore: File {input_file} non trovato.")