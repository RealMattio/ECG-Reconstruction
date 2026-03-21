import os
import json
import torch
from tqdm import tqdm

def recover_json_manifest(data_dir, fs=125):
    manifest_data = []
    
    # 1. Trova tutte le cartelle dei pazienti nella nuova cache
    patient_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('p') and os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"🔍 Inizio scansione di {len(patient_dirs)} pazienti in {data_dir}...")
    
    for pat_id in tqdm(patient_dirs, desc="Generazione JSON"):
        pat_dir = os.path.join(data_dir, pat_id)
        # Trova tutti i frammenti salvati
        pt_files = sorted([f for f in os.listdir(pat_dir) if f.endswith('.pt')])
        
        for file_name in pt_files:
            file_path = os.path.join(pat_dir, file_name)
            
            try:
                # Carichiamo il tensore solo per leggerne la lunghezza
                data = torch.load(file_path, map_location='cpu')
                length_samples = len(data['ppg'])
                
                # CAST FONDAMENTALE: Convertiamo in tipi Python nativi per il JSON
                length_samples_int = int(length_samples)
                duration_sec_float = float(length_samples_int / fs)
                
                # Aggiungiamo al manifest
                manifest_data.append({
                    'subject_id': pat_id,
                    'file_path': file_name,
                    'length_samples': length_samples_int,
                    'duration_sec': duration_sec_float
                })
            except Exception as e:
                print(f"⚠️ Errore nella lettura del file {file_path}: {e}")
                
    # 2. Salvataggio sicuro del JSON
    manifest_path = os.path.join(data_dir, 'dataset_manifest.json')
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
        print("\n" + "="*60)
        print("✅ MANIFEST GENERATO CON SUCCESSO!")
        print("="*60)
        print(f"🔹 File salvato in: {manifest_path}")
        print(f"🔹 Totale segmenti registrati: {len(manifest_data)}")
    except Exception as e:
        print(f"\n❌ Errore durante il salvataggio del JSON: {e}")

if __name__ == "__main__":
    # Assicurati che questo sia il percorso della cartella dove ha salvato i nuovi file
    DEST_DATASET = "/home/mmerone/mattia/ecg_generation/mimic3_pinn_cache" 
    
    recover_json_manifest(DEST_DATASET)