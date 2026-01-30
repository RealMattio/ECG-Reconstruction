import os
import sys
import torch
import datetime

# 1. GESTIONE PERCORSI
# Assumiamo che il main sia nella cartella 'src'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importazione della pipeline modulare per l'Approccio 4
from bidmc_generation.pipeline import run_ha_cnn_bilstm_pipeline

def main():
    # 2. CONFIGURAZIONE PERCORSI DATI
    # Assicurati che 'bidmc_data' sia nella root del progetto
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(os.path.dirname(PROJECT_ROOT), 'bidmc_data')
    model_save_path = os.path.join(PROJECT_ROOT, 'bidmc_generation', 'models', 'ha_cnn_bilstm_withWST', timestamp)
    
    # 3. SELEZIONE SOGGETTI (Dataset BIDMC Respiratory)
    # 53 pazienti medici con segnali a 125Hz 
    all_subjects = [str(i).zfill(2) for i in range(1, 54)]

    # 4. CONFIGURAZIONE IPERPARAMETRI E FLAG MODULARI
    configs = {
        'model_type': 'ha_cnn_bilstm',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
        # Parametri del segnale (Fissi per BIDMC)
        'target_fs': 125,        # Frequenza di campionamento target
        'beat_len': 60,         # Lunghezza finestra come da Tabella 1 
        'overlap_pct': 0.1,     # 10% sovrapposizione per continuità
        # --- FLAG DI MODULARITÀ ---
        'overlap_windows': True, # Attiva sliding window per continuità temporale
        'apply_wst': True,       # Attiva Wavelet Scattering (19 canali in input)
        'apply_dwt': False,      # Opzionale: DWT sull'ECG target

        # --- NUOVI PARAMETRI LOSS MORFOLOGICA ---
        'use_morphological_loss': True, # Attiva la Pearson Loss
        'morph_loss_weight': 0.6,       # Peso (da 0 a 1). 0.6 spinge molto sulla forma.
        'peak_loss_weight': 3.0,        # Peso per la Weighted MSE sui picchi (Quanto penalizzare gli errori sui picchi)
        
        # --- PARAMETRI DI TRAINING ---
        'batch_size': 20,        # Minimo suggerito dal paper 
        'optimizer_type': 'SGDM', # RMSE migliore (0.031) secondo lo studio 
        'lr': 0.001,             # Learning Rate iniziale
        'epochs': 1000,           # Esteso con Early Stopping
        'patience': 15,          # "Simpatica" pazienza per la convergenza
        
        # Percorso salvataggio coerente con la tua struttura
        'model_save_path': model_save_path
    }

    print("-" * 50)
    print(f"AVVIO PIPELINE: HA-CNN-BILSTM (Approccio 4)")
    print(f"Dataset: BIDMC Respiratory | Soggetti: {len(all_subjects)}")
    print(f"Configurazione: WST={configs['apply_wst']}, Overlap={configs['overlap_windows']}")
    print(f"Ottimizzatore: {configs['optimizer_type']} | Device: {configs['device']}")
    print("-" * 50)

    # 5. ESECUZIONE DELLA PIPELINE
    try:
        history = run_ha_cnn_bilstm_pipeline(
            base_path=data_path,
            configs=configs
        )
        print("\n[SUCCESS] Pipeline terminata correttamente.")
        print(f"Modelli e grafici salvati in: {configs['model_save_path']}")
        
    except Exception as e:
        print(f"\n[ERROR] Si è verificato un errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()