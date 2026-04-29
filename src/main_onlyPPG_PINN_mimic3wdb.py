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
        default='lightweight_hybrid',
        choices=['ha_cnn_bilstm_ar', 'bio_transformer', 'dual_branch_hybrid',
                 'lightweight_hybrid', 'ppg_wavenet', 'ecg_unet1d'],
        help="Scegli l'architettura del modello da addestrare."
    )
    
    # Parametri operativi 
    parser.add_argument('--epochs', type=int, default=100, help="Numero di epoche per fold")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size ottimizzato per saturare la GPU")
    
    # Flag per forzare l'uso dei dati RAW 
    parser.add_argument('--use_raw', action='store_true', help="Se attivo, usa i dati RAW invece di quelli preprocessati.")
    parser.add_argument('--val_step', type=int, default=500, help="Esegui validazione ogni N batch (step)")
    
    # --- ARGOMENTO PER SLURM RESUME ---
    parser.add_argument('--start_fold', type=int, default=1, help="Specifica la fold da cui ripartire (es. 3)")

    # --- CARTELLA OUTPUT (obbligatoria se --start_fold > 1) ---
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help=(
            "Cartella in cui salvare (o dove continuare a salvare) i risultati. "
            "Se omessa, viene creata automaticamente con timestamp. "
            "OBBLIGATORIA quando --start_fold > 1."
        )
    )

    # Aggiungo la LOSS come parametro da scegliere da riga di comando
    parser.add_argument('--base_loss', type=str.upper, default='MAE', choices=['MAE', 'RMSE', 'HUBER'], help='Scegli la loss di base.')

    args = parser.parse_args()

    # Validazione: --output_dir è obbligatorio se si riprende da una fold intermedia
    if args.start_fold > 1 and args.output_dir is None:
        parser.error("--output_dir è obbligatorio quando si specifica --start_fold > 1. "
                     "Passa la stessa cartella usata all'avvio originale dell'esperimento.")
    # -----------------------------------------------

    # 3. CONFIGURAZIONE PERCORSI E DATI
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_data_path = os.path.join(os.path.dirname(PROJECT_ROOT), 'mimic3wdb-matched_raw_data')
    preprocessed_data_path = os.path.join(PROJECT_ROOT, '../mimic3wdb-matched_healthy_data')

    BASE_LOSS = args.base_loss

    # Path Output Esperimenti:
    # - se --output_dir è fornito, si usa quello (resume o cartella custom)
    # - altrimenti si genera un nuovo path con timestamp
    if args.output_dir is not None:
        model_save_path = args.output_dir
    else:
        model_save_path = os.path.join(
            PROJECT_ROOT, 'experiments', 'final_mimic_pinn_results',
            f"{BASE_LOSS}_loss", f"{args.model}_{timestamp}"
        )

    # 4. CONFIGURAZIONE COMPLETA (configs)
    configs = {
        'apply_preprocessing': args.use_raw,      
        'raw_data': raw_data_path,                
        'preprocessed_data': preprocessed_data_path, 
        'model_type': args.model,       
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        
        'target_fs': 125,               
        'x_sec': 7,                     
        'gen_sec': 1,                   
        'stride_sec': 1,                
        
        'normalize_01': False,           
        'apply_wst': True,              
        
        'apply_augmentation': True,     
        'aug_random_gain': True,        
        'aug_additive_noise': True,     
        'aug_context_noise': 0.02,      
        'apply_context_augmentation': True, 
        
        'base_loss_type': BASE_LOSS,       
        'use_morphological_loss': True, 
        'morph_loss_weight': 0.4,        
        'ode_loss_weight': 0.1,          
        'peak_loss_weight': 3.0,         
        
        'batch_size': args.batch_size,  
        'optimizer_type': 'ADAM',       
        'lr': 0.001,                   
        'epochs': args.epochs,
        'use_early_stopping': True,     
        'patience': 20,                  
        
        'use_lr_scheduler': True,        
        'lr_step_size': 40,              
        'lr_gamma': 0.5,                 
        
        'k_folds': 5,
        'start_fold': args.start_fold,   # Passiamo la fold di partenza
        'val_step': args.val_step,                   
        'seed': 45,                   
        
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
    print(f"Epoche massime per Fold: {configs['epochs']} | Batch Size: {configs['batch_size']}")
    print(f"Device: {configs['device']}")
    print(f"Output dir: {model_save_path}")
    if args.start_fold > 1:
        print(f"⚠️  RESUME ATTIVO: Ripresa dalla Fold {args.start_fold}")
    print("-" * 60)

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