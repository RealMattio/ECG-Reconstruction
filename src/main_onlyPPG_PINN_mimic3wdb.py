import os
import sys
import torch
import datetime
import argparse

# 1. GESTIONE PERCORSI
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importazione della pipeline modulare
from src.mimic_generation_PINN.pipeline import run_k_fold_pipeline

def main():
    # --- 2. GESTIONE ARGOMENTI DA RIGA DI COMANDO ---
    parser = argparse.ArgumentParser(description="MIMIC-III rPPG Training Pipeline")
    
    # Argomento per il modello
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        default='lightweight_hybrid', # Cambiato al modello che stai testando ora
        choices=['ha_cnn_bilstm_ar', 'bio_transformer', 'dual_branch_hybrid', 'lightweight_hybrid'],
        help="Scegli l'architettura del modello da addestrare."
    )
    
    # Parametri operativi 
    parser.add_argument('--epochs', type=int, default=100, help="Numero di epoche per fold")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size ottimizzato per saturare la GPU")
    
    # Flag per forzare l'uso dei dati RAW 
    parser.add_argument('--use_raw', action='store_true', help="Se attivo, usa i dati RAW invece di quelli preprocessati.")
    parser.add_argument('--val_step', type=int, default=500, help="Esegui validazione ogni N batch (step)")

    args = parser.parse_args()
    # -----------------------------------------------

    # 3. CONFIGURAZIONE PERCORSI E DATI
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_data_path = os.path.join(os.path.dirname(PROJECT_ROOT), 'mimic3wdb-matched_raw_data')
    preprocessed_data_path = os.path.join(PROJECT_ROOT, '../mimic3wdb-matched_healthy_data')

    BASE_LOSS = 'HUBER'  # Scegli tra 'MAE', 'HUBER' o 'RMSE' a seconda di quale vuoi testare come base per la PINN

    # Path Output Esperimenti
    model_save_path = os.path.join(PROJECT_ROOT, 'experiments', 'mimic_pinn_results', 'healthy_patients' , f"{BASE_LOSS}_loss" ,f"{args.model}_{timestamp}")

    # 4. CONFIGURAZIONE COMPLETA (configs)
    configs = {
        # --- GESTIONE MODALITÀ DATI ---
        'apply_preprocessing': args.use_raw,      
        'raw_data': raw_data_path,                
        'preprocessed_data': preprocessed_data_path, 

        # --- SCELTA MODELLO E HARDWARE ---
        'model_type': args.model,       
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        
        # --- PARAMETRI SEGNALE ---
        'target_fs': 125,               
        'x_sec': 7,                     
        'gen_sec': 1,                   
        'stride_sec': 1,                
        
        # --- PREPROCESSING ON-THE-FLY ---
        'normalize_01': False,           # Fondamentale che sia False per la PINN
        'apply_wst': True,              
        
        # --- DATA AUGMENTATION (Anti-Overfitting per run lunghe) ---
        'apply_augmentation': True,     
        'aug_random_gain': True,        
        'aug_additive_noise': True,     
        'aug_context_noise': 0.02,      
        'apply_context_augmentation': True, 
        
        # --- LOSS FUNCTION HYBRID PINN ---
        'base_loss_type': BASE_LOSS,       # Scegli tra: 'MAE', 'HUBER' o 'RMSE'
        'use_morphological_loss': True, 
        'morph_loss_weight': 0.4,        # Pearson = 40%  IMPORTANTE: questo prima era 0.5 mentre ode 0.10
        'ode_loss_weight': 0.1,          # McSharry ODE = 10% (L'RMSE avrà il restante 50%)
        'peak_loss_weight': 3.0,         # Penalità aumentata per gli errori sui picchi R
        
        # --- TRAINING & SCHEDULER ---
        'batch_size': args.batch_size,  
        'optimizer_type': 'ADAM',       
        'lr': 0.001,                   
        'epochs': args.epochs,
        'use_early_stopping': True,     # Disattivato per vedere la curva intera
        'patience': 20,                  # (Ignorato se early stopping è False)
        
        'use_lr_scheduler': True,        # Attivato
        'lr_step_size': 40,              # Taglia il LR ogni 40 epoche
        'lr_gamma': 0.5,                 # Lo dimezza (1e-3 -> 5e-4 -> 2.5e-4...)
        
        # --- VALIDAZIONE ---
        'k_folds': 1,
        'val_step': args.val_step,                   
        'seed': 45,                   # Con 5 o 4 fold il seed è 45 altrimenti con 1 fold è 46    
        
        # --- PERCORSI OUTPUT ---
        'model_save_path': model_save_path, 
        'manifest_name': 'dataset_manifest.json',
        'excluded_subjects_ids': []     
    }
    
    print("-" * 60)
    print(f"AVVIO PIPELINE MIMIC-III - {timestamp}")
    print(f"Modalità Dati: {'RAW (Lento)' if configs['apply_preprocessing'] else 'SMART CACHE (Veloce)'}")
    if not configs['apply_preprocessing']:
        print(f"Cache Path: {configs['preprocessed_data']}")
    print(f"Modello: {configs['model_type'].upper()}") 
    print(f"Epoche: {configs['epochs']} | Batch Size: {configs['batch_size']}")
    print(f"Device: {configs['device']}")
    print("-" * 60)

    # 5. ESECUZIONE
    try:
        results = run_k_fold_pipeline(None, configs)
        
        print("\n[SUCCESS] Pipeline terminata con successo.")
        print(f"Tutti i risultati sono salvati in: {model_save_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Errore critico durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

 