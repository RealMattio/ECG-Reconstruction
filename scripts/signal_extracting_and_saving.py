import pickle
import numpy as np
import os

def extract_original_data(input_path, output_path):
    """
    Carica i dati di un soggetto e salva un dizionario con i segnali originali
    senza effettuare resampling.
    """
    # 1. Caricamento del file originale
    with open(input_path, 'rb') as f:
        # Il file .pkl include dati sincronizzati e labelled 
        data = pickle.load(f, encoding='latin1')
    
    # Estrazione dei segnali originali dal dizionario 'signal' 
    # Il segnale PPG (BVP) ha una frequenza di campionamento di 64 Hz 
    ppg = data['signal']['wrist']['BVP'].flatten() 
    
    # Il segnale ACC (Wrist) ha una frequenza di campionamento di 32 Hz 
    acc = data['signal']['wrist']['ACC'] 
    
    # Il segnale EDA ha una frequenza di campionamento di 4 Hz 
    eda = data['signal']['wrist']['EDA'].flatten()
    
    # Il segnale ECG ha una frequenza di campionamento di 700 Hz
    ecg = data['signal']['chest']['ECG'].flatten()

    # 2. Definizione delle frequenze originali 
    fs_ppg = 64
    fs_acc = 32
    fs_eda = 4
    fs_ecg = 700

    print(f"Inizio estrazione per {data['subject']}...")

    # 3. Creazione del dizionario con i segnali originali
    # Manteniamo i segnali così come estratti, senza manipolazioni temporali
    original_dict = {
        'subject': data['subject'],
        'PPG': ppg,
        'ACC': acc,
        'EDA': eda,
        'ECG': ecg,
        'fs_ppg': fs_ppg,
        'fs_acc': fs_acc,
        'fs_eda': fs_eda,
        'fs_ecg': fs_ecg,
        'activity': data['activity'], # IDs delle attività 0...8 
        'label': data['label']        # Ground truth heart rate 
    }

    # 4. Salvataggio in pkl
    with open(output_path, 'wb') as f:
        pickle.dump(original_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Salvataggio completato: {output_path}")
    print(f"Dimensioni originali - PPG: {len(ppg)}, EDA: {len(eda)}, ACC: {len(acc)}, ECG: {len(ecg)}")

# Esempio di utilizzo
if __name__ == "__main__":
    sample = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15"]
    
    for subject_id in sample:
        # Risaliamo alla root del progetto partendo da scripts/
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Percorso del file originale (assicurati che la struttura data/raw/ sia corretta)
        # Se i file sono dentro sottocartelle SX, usa: os.path.join(base_path, 'data', 'raw', subject_id, f"{subject_id}.pkl")
        input_file = os.path.join(base_path, 'data', 'raw', f"{subject_id}.pkl")
        
        # Creazione della cartella di output se non esiste
        output_dir = os.path.join(base_path, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{subject_id}_original.pkl")
        
        if os.path.exists(input_file):
            extract_original_data(input_file, output_file)
        else:
            print(f"Errore: File {input_file} non trovato.")