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
from src.mimic_generation.pipeline import run_k_fold_pipeline

def main():
    # --- 2. GESTIONE ARGOMENTI DA RIGA DI COMANDO ---
    parser = argparse.ArgumentParser(description="MIMIC-III rPPG Training Pipeline")
    
    # Argomento per il modello
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        default='bio_transformer', 
        choices=['ha_cnn_bilstm_ar', 'bio_transformer', 'dual_branch_hybrid', 'lightweight_hybrid'],
        help="Scegli l'architettura del modello da addestrare."
    )
    
    # Parametri operativi
    parser.add_argument('--epochs', type=int, default=100, help="Numero di epoche per fold")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    
    # Flag per forzare l'uso dei dati RAW (utile per debug o se non hai preprocessato)
    parser.add_argument('--use_raw', action='store_true', help="Se attivo, usa i dati RAW invece di quelli preprocessati.")
    parser.add_argument('--val_step', type=int, default=500, help="Esegui validazione ogni N batch (step)")

    args = parser.parse_args()
    # -----------------------------------------------

    # 3. CONFIGURAZIONE PERCORSI E DATI
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Definiamo dove sono i dati
    # Path ai dati RAW (usato se args.use_raw è True)
    # Nota: Assicurati che punti alla root corretta dei dati raw (es. mimic3wdb-matched_raw_data)
    raw_data_path = os.path.join(os.path.dirname(PROJECT_ROOT), '../mimic3wdb-matched_raw_data')
    
    # Path ai dati PREPROCESSATI (usato se args.use_raw è False)
    preprocessed_data_path = os.path.join(PROJECT_ROOT, '../mimic3_smart_cache') 

    # Path Output Esperimenti
    model_save_path = os.path.join(PROJECT_ROOT, 'experiments', 'mimic_results', f"{args.model}_{timestamp}")

    # 4. CONFIGURAZIONE COMPLETA (configs)
    configs = {
        # --- GESTIONE MODALITÀ DATI ---
        'apply_preprocessing': args.use_raw,      # False = Usa .pt (Smart Dataset), True = Usa Raw (Lento)
        'raw_data': raw_data_path,                # Path Raw
        'preprocessed_data': preprocessed_data_path, # Path Smart Cache

        # --- SCELTA MODELLO E HARDWARE ---
        'model_type': args.model,       
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        
        # --- PARAMETRI SEGNALE (Devono coincidere col preprocessing fatto) ---
        'target_fs': 125,               
        'x_sec': 7,                     
        'gen_sec': 1,                   
        'stride_sec': 1,                
        
        # --- PREPROCESSING ON-THE-FLY (Se attivo WST nel training) ---
        'normalize_01': True,           
        'apply_wst': True,              
        
        # --- DATA AUGMENTATION (Solo Training) ---
        'apply_augmentation': True,     
        'aug_random_gain': True,        
        'aug_additive_noise': True,     
        'aug_context_noise': 0.05,      
        'apply_context_augmentation': True, 
        
        # --- LOSS FUNCTION ---
        'use_morphological_loss': True, 
        'morph_loss_weight': 0.5,       
        'peak_loss_weight': 3.0,        
        
        # --- TRAINING ---
        'batch_size': args.batch_size,  
        'optimizer_type': 'ADAM',       
        'lr': 0.0005,                   
        'epochs': args.epochs,          
        'patience': 15,                 
        
        # --- VALIDAZIONE ---
        'k_folds': 5,
        'val_step': args.val_step,                   
        'seed': 45,                     
        
        # --- PERCORSI OUTPUT ---
        'model_save_path': model_save_path, 
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
        # FIX: Rimosso 'base_path', passiamo tutto dentro 'configs'
        # In pipeline.py, la funzione deve essere def run_k_fold_pipeline(base_path_unused, configs): 
        # oppure def run_k_fold_pipeline(configs): (se l'hai aggiornata così).
        # Per sicurezza, passo None come primo argomento se la firma richiede ancora un argomento posizionale inutile.
        
        # Se la tua pipeline è: def run_k_fold_pipeline(base_path_unused, configs):
        results = run_k_fold_pipeline(None, configs)
        
        print("\n[SUCCESS] Pipeline terminata con successo.")
        print(f"Tutti i risultati sono salvati in: {model_save_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Errore critico durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()