import os
import sys
import torch

# 1. Gestione dei percorsi
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importazione della pipeline specifica per l'Approccio 3 (P2E-WGAN Bi-LSTM)
from src.generation.approach3.pipeline import run_approach3_pipeline

def main():
    # 2. Configurazione dei percorsi dati
    data_path = os.path.join(PROJECT_ROOT, 'data')
    
    # 3. Selezione dei soggetti (Dataset PPG-DaLiA)
    all_subjects = [f"S{i}" for i in range(1, 16)]
    
    configs = {
        'model_type': 'p2e_wgan_bilstm', 
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
        # Parametri di segmentazione (7 secondi come richiesto)
        'window_size': 7,
        'target_fs': 125,      # Frequenza target del paper P2E-WGAN [cite: 120]
        'target_len': 875,     # 7s * 125Hz = 875 campioni
        
        # Iperparametri di training (basati sul paper P2E-WGAN) [cite: 213, 215]
        'batch_size': 32,      # Puoi alzare a 64 o 128 se la memoria GPU lo permette
        'pretrain_epochs': 50, # Fase 1: Teacher-Guided con ECG in input
        'train_epochs': 100,   # Fase 2: Specializzazione solo sui sensori
        'lr': 2e-4,            # Learning rate specifico per GAN (Adam beta1=0.5) 
        
        # Pesi della Loss specifici per P2E-WGAN [cite: 154, 156, 169]
        'loss_weights': {
            'lambda_gp': 10.0,      # Coefficiente Gradient Penalty per WGAN-GP [cite: 169]
            'lambda_extrema': 80.0, # Peso per la ricostruzione dei picchi (Extrema Loss) [cite: 154]
            'gamma': 1.0            # Peso addizionale per la derivative loss (opzionale)
        }
    }

    print(f"\n--- Progetto ECG-Reconstruction: Approccio 3 (P2E-WGAN Bi-LSTM) ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dispositivo in uso: {configs['device']}")
    print(f"Configurazione: Finestre di {configs['window_size']}s a {configs['target_fs']}Hz")

    # 4. Esecuzione della Pipeline dell'Approccio 3
    # Nota: Esegue internamente il ciclo Leave-One-Out (LOOCV) sui soggetti
    run_approach3_pipeline(
        base_path=data_path,
        subjects=all_subjects,
        configs=configs
    )

    print("\n--- Pipeline Approccio 3 Terminata con Successo ---")
    
    # Percorso di salvataggio dei risultati dell'approccio 3
    model_path = os.path.join(PROJECT_ROOT, 'src', 'generation', 'models', 'approach3')
    print(f"Modelli e risultati salvati in: {model_path}")

if __name__ == "__main__":
    main()