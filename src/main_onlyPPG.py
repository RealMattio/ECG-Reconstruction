import os
import sys
import torch

# 1. Gestione dei percorsi
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importazione della pipeline aggiornata per l'Approccio 2 (Solo PPG)
from src.generation.only_ppg.pipeline import run_approach2_pipeline

def main():
    # 2. Configurazione dei percorsi dati
    data_path = os.path.join(PROJECT_ROOT, 'data')
    
    # 3. Selezione dei soggetti (Dataset PPG-DaLiA)
    all_subjects = [f"S{i}" for i in range(1, 16)]
    
    # Suddivisione soggetti: 10 Training, 2 Validation, 3 Test
    train_subs = all_subjects[:10]
    val_subs = all_subjects[10:12]
    test_subs = all_subjects[12:]

    configs = {
        # Architettura Ibrida: Dilated CNN + BiLSTM + Attention [cite: 20, 25]
        'model_type': 'ha_cnn_bilstm_fourier', 
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
        # Parametri di segmentazione (7 secondi)
        'window_size': 7,
        'window_shift': 2,      # Sovrapposizione per aumentare il dataset
        'target_fs': 256,       # Frequenza ECG originale
        'target_len': 1792,     # 7s * 256Hz = 1792 campioni
        
        # Split del dataset
        'split': {
            'train': train_subs, 
            'val': val_subs,    
            'test': test_subs   
        },
        
        # Iperparametri basati sul paper [cite: 318, 340]
        'batch_size': 20,       # Valore suggerito dal paper 
        'optimizer_type': 'sgdm', # Il migliore per RMSE secondo il paper 
        'lr': 0.001,            # Learning rate ottimale per SGDM 
        'epochs': 100,          # Il paper ne usa 50, ma noi teniamo early stopping 
        'patience': 20,         # Aumentata per permettere a SGDM di convergere
        
        # Loss combinata: Bilanciamento tra valori assoluti e morfologia
        'loss_weights': {
            'l1': 0.5,          # Peso ridotto per dare spazio all forma
            'pearson': 2.5      # Peso aumentato per forzare la sincronia dei picchi
        }
    }

    print(f"\n--- Ricostruzione ECG: HA-CNN-BILSTM (Solo PPG) ---") 
    print(f"Ottimizzatore: {configs['optimizer_type'].upper()} | LR: {configs['lr']}")
    print(f"Target: {configs['window_size']}s @ {configs['target_fs']}Hz")

    # 4. Esecuzione della Pipeline
    trainer, test_data = run_approach2_pipeline(
        subject_ids=all_subjects, 
        base_path=data_path,
        configs=configs,
        show_random_sample=True
    )

    print("\n--- Pipeline Terminata con Successo ---")
    
    # Percorso di salvataggio
    model_path = os.path.join(PROJECT_ROOT, 'src', 'models', 'approach2_ppg_only')
    print(f"Modelli e grafici migliori salvati in: {model_path}")

if __name__ == "__main__":
    main()