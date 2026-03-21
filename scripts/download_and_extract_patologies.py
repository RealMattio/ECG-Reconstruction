import os
import gzip
import shutil
import pandas as pd

# --- CONFIGURAZIONE ---
DOWNLOAD_DIR = "./mimic_clinical_data"
TARGET_FILES = ["DIAGNOSES_ICD.csv.gz", "D_ICD_DIAGNOSES.csv.gz"]
OUTPUT_CSV = "unique_present_pathologies.csv"
# ----------------------

def extract_gz_if_needed(filename):
    """
    Controlla se il file .gz è presente in locale e lo estrae in .csv
    """
    gz_local_path = os.path.join(DOWNLOAD_DIR, filename)
    csv_local_path = os.path.join(DOWNLOAD_DIR, filename.replace('.gz', ''))

    # Se il CSV estratto esiste già, siamo a posto
    if os.path.exists(csv_local_path):
        print(f" -> [SKIP] Il file CSV {csv_local_path} è già estratto.")
        return csv_local_path

    # Verifica che l'utente abbia scaricato il .gz
    if not os.path.exists(gz_local_path):
        raise FileNotFoundError(f"Non trovo il file {gz_local_path}! Assicurati di averlo scaricato manualmente da PhysioNet e inserito in quella cartella.")

    print(f" -> Estrazione di {filename} in corso...")
    with gzip.open(gz_local_path, 'rb') as f_in:
        with open(csv_local_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    print(f" -> File estratto con successo: {csv_local_path}")
    return csv_local_path

def process_and_extract_unique_pathologies(diagnoses_csv, dictionary_csv):
    """
    Legge i CSV, trova i codici unici presenti nei pazienti e li mappa 
    con i loro nomi dal dizionario, salvando un file CSV pulito.
    """
    print("\nElaborazione dei dati per trovare le patologie uniche...")
    
    print(" -> Lettura di DIAGNOSES_ICD (può richiedere qualche secondo)...")
    df_diag = pd.read_csv(diagnoses_csv, dtype={'ICD9_CODE': str})
    
    unique_codes_present = df_diag['ICD9_CODE'].dropna().unique()
    print(f" -> Trovati {len(unique_codes_present)} codici patologia unici assegnati ai pazienti.")
    
    print(" -> Lettura del dizionario D_ICD_DIAGNOSES...")
    df_dict = pd.read_csv(dictionary_csv, dtype={'ICD9_CODE': str})
    
    df_filtered = df_dict[df_dict['ICD9_CODE'].isin(unique_codes_present)]
    
    final_df = df_filtered[['ICD9_CODE', 'LONG_TITLE']].sort_values(by='ICD9_CODE')
    
    out_path = os.path.join(DOWNLOAD_DIR, OUTPUT_CSV)
    final_df.to_csv(out_path, index=False)
    
    print(f"\n✅ FASE 1 COMPLETATA!")
    print(f"File salvato in: {out_path}")
    print(f"Il file contiene l'elenco esatto di {len(final_df)} patologie (senza duplicati).")

def main():
    print("--- MIMIC-III Clinical Database Local Processor ---")
    
    try:
        local_csv_paths = []
        for file_name in TARGET_FILES:
            csv_path = extract_gz_if_needed(file_name)
            local_csv_paths.append(csv_path)
            
        process_and_extract_unique_pathologies(local_csv_paths[0], local_csv_paths[1])
    except Exception as e:
        print(f"\n[ERRORE] {e}")

if __name__ == "__main__":
    main()