import os
import sys

# 1. Gestione dei percorsi
# Poiché main.py è in src/, risaliamo di un livello per trovare la root del progetto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importazione della pipeline specifica per l'Approccio 1
from src.generation.approach1.pipeline import run_approach1_pipeline

def main():
    # 2. Configurazione dei percorsi dati e modelli
    data_path = os.path.join(PROJECT_ROOT, 'data')
    
    # 3. Selezione dei soggetti (Dataset PPG-DaLiA)
    # È consigliato isolare i soggetti per un test subject-independent [cite: 15, 39]
    all_subjects = [f"S{i}" for i in range(1, 16)]
    
    configs = {
        'split': {
            'train': ["S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9"],
            'val': ["S10", "S11"],
            'test': ["S12", "S13", "S14", "S15"] # Soggetti unseen [cite: 37]
        },
        
        # Parametri di segmentazione
        # Il paper usa finestre di ~4s (512 punti a 125Hz) con 50% overlap [cite: 119, 615]
        # Manteniamo la tua richiesta di 8s per catturare più variabilità HRV [cite: 517]
        'window_size': 8,
        'window_shift': 2,
        'target_len': 2048, # 8s * 256Hz target (downsampling consigliato) [cite: 115]
        
        # Iperparametri di training
        'batch_size': 32,
        'epochs': 1, # In linea con i training di CLEP-GAN e QRS-ED 
        'lr': 1e-4,
        'patience': 15,
        'use_scheduler': True,
        
        # Pesi della Loss Composita [cite: 461, 500]
        # Ispirati ai coefficienti alpha=30, beta=3, gamma=1 del paper [cite: 504]
        'loss_weights': {
            'alpha': 1.0,  # Ramo PPG (Segnale primario) [cite: 751]
            'beta': 0.3,   # Ramo ACC (Correzione artefatti) [cite: 33]
            'gamma': 0.1,  # Ramo EDA (Contesto lento) [cite: 740]
            'delta': 2.0   # Output Finale (Obiettivo primario) [cite: 64, 498]
        }
    }

    print("--- Progetto ECG-Reconstruction: Approccio 1 (Late Fusion) ---")
    print(f"Project Root: {PROJECT_ROOT}")

    # 4. Esecuzione della Pipeline
    # La funzione run_approach1_pipeline in pipeline.py gestisce il flusso end-to-end
    trainer, test_results = run_approach1_pipeline(
        subject_ids=all_subjects,
        base_path=data_path,
        configs=configs
    )

    print("\n--- Pipeline Terminata con Successo ---")
    print("\n--- Pipeline Terminata con Successo ---")
    # Calcolo del percorso per la stampa finale
    model_path = os.path.join(PROJECT_ROOT, 'src', 'generation', 'models', 'approach1')
    print(f"Modelli salvati in: {model_path}")
    

if __name__ == "__main__":
    main()