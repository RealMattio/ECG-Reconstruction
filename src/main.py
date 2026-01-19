import os
import sys

# 1. Gestione dei percorsi
# Poiché main.py è in src/, risaliamo di un livello per trovare la root del progetto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importazione della pipeline specifica per l'Approccio 2 (Encoder-Decoder)
from src.generation.approach2.pipeline import run_approach2_pipeline

def main():
    # 2. Configurazione dei percorsi dati
    data_path = os.path.join(PROJECT_ROOT, 'data')
    
    # 3. Selezione dei soggetti (Dataset PPG-DaLiA)
    all_subjects = [f"S{i}" for i in range(1, 16)]
    
    configs = {
        'split': {
            'train': ["S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9"],
            'val': ["S10", "S11"],
            'test': ["S12", "S13", "S14", "S15"] # Soggetti unseen per test finale
        },
        
        # Parametri di segmentazione (Sincronizzati con il preprocessing)
        'window_size': 8,
        'window_shift': 2,
        'target_len': 2048, # 8s * 256Hz target (ECG ricampionato a 256Hz)
        
        # Iperparametri di training
        'batch_size': 32,
        'epochs': 200, 
        'lr': 3e-3,          # Learning Rate alzato per l'architettura U-Net
        'patience': 15,      # Early stopping dopo 15 epoche senza miglioramenti
        'use_scheduler': True,
        
        # Nota: Nell'approccio 2 la fusione è interna (Feature Fusion), 
        # i pesi possono essere usati per eventuali loss ausiliarie sui rami latenti.
        'loss_weights': {
            'alpha': 1.0,  
            'beta': 1.0,   
            'gamma': 0.1,  
            'delta': 2.0   
        }
    }

    print("--- Progetto ECG-Reconstruction: Approccio 2 (Encoder-Decoder U-Net) ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dispositivo in uso: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 4. Esecuzione della Pipeline dell'Approccio 2
    # Gestisce caricamento, preprocessing, training e salvataggio history/plots
    trainer, test_results = run_approach2_pipeline(
        subject_ids=all_subjects,
        base_path=data_path,
        configs=configs
    )

    print("\n--- Pipeline Approccio 2 Terminata con Successo ---")
    
    # Calcolo del percorso per la stampa finale dove risiedono i modelli dell'app 2
    model_path = os.path.join(PROJECT_ROOT, 'src', 'generation', 'models', 'approach2')
    print(f"Modelli, grafici e history salvati in: {model_path}")
    

if __name__ == "__main__":
    import torch # Import necessario per il check cuda nel print
    main()