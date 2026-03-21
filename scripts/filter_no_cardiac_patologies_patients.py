import os
import pandas as pd
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATA_DIR = "./mimic_clinical_data"
DIAGNOSES_CSV = "DIAGNOSES_ICD.csv"
ALLOWED_PATHOLOGIES_CSV = "allowed_pathologies.csv"

OUTPUT_ALLOWED_PATIENTS = "allowed_patients.csv"
OUTPUT_EXCLUDED_PATIENTS = "excluded_patients.csv"
# ----------------------

def main():
    print("--- MIMIC-III: Selezione Coorte Pazienti (Patient-Level Exclusion) ---")
    
    diag_path = os.path.join(DATA_DIR, DIAGNOSES_CSV)
    allowed_path = os.path.join(DATA_DIR, ALLOWED_PATHOLOGIES_CSV)
    
    if not os.path.exists(diag_path) or not os.path.exists(allowed_path):
        print("[ERRORE] File necessari mancanti. Assicurati di aver completato Fase 1 e Fase 2.")
        return

    # 1. Caricamento dei codici ammessi
    print(f" -> Caricamento patologie ammesse da {ALLOWED_PATHOLOGIES_CSV}...")
    df_allowed = pd.read_csv(allowed_path, dtype={'ICD9_CODE': str})
    # Creiamo un set per una ricerca super veloce (O(1))
    allowed_codes = set(df_allowed['ICD9_CODE'].dropna().unique())
    print(f"    [OK] Caricati {len(allowed_codes)} codici patologia sicuri.")

    # 2. Caricamento della tabella diagnosi-pazienti
    print(f" -> Caricamento di tutte le diagnosi ospedaliere ({DIAGNOSES_CSV})...")
    df_diag = pd.read_csv(diag_path, dtype={'SUBJECT_ID': int, 'ICD9_CODE': str})
    
    # Pulizia righe vuote
    df_diag = df_diag.dropna(subset=['SUBJECT_ID', 'ICD9_CODE'])
    
    total_patients = df_diag['SUBJECT_ID'].nunique()
    print(f"    [OK] Trovati {total_patients} pazienti unici nell'intero database MIMIC-III.")

    # 3. Logica di Esclusione a livello di Paziente
    print("\n -> Incrocio dei dati e filtraggio dei pazienti in corso...")
    
    # Troviamo tutte le diagnosi che NON sono nella lista delle ammesse
    # (cioè le diagnosi cardiache, elettrolitiche, o altre escluse precedentemente)
    df_violations = df_diag[~df_diag['ICD9_CODE'].isin(allowed_codes)]
    
    # Tutti i pazienti che compaiono in df_violations sono "contaminati"
    excluded_patients = set(df_violations['SUBJECT_ID'].unique())
    
    # I pazienti ammessi sono semplicemente tutti i pazienti MENO quelli esclusi
    all_patients = set(df_diag['SUBJECT_ID'].unique())
    allowed_patients = all_patients - excluded_patients

    # 4. Salvataggio e Statistiche
    perc_allowed = (len(allowed_patients) / total_patients) * 100
    perc_excluded = (len(excluded_patients) / total_patients) * 100

    print("\n" + "="*50)
    print("✅ SELEZIONE COORTE COMPLETATA")
    print("="*50)
    print(f"👥 Totale pazienti valutati: {total_patients}")
    print(f"❌ Pazienti ESCLUSI (storia clinica incompatibile): {len(excluded_patients)} ({perc_excluded:.1f}%)")
    print(f"❤️ Pazienti AMMESSI (cuore e anamnesi sana): {len(allowed_patients)} ({perc_allowed:.1f}%)")
    print("="*50)

    # Salvataggio CSV
    out_allowed = os.path.join(DATA_DIR, OUTPUT_ALLOWED_PATIENTS)
    out_excluded = os.path.join(DATA_DIR, OUTPUT_EXCLUDED_PATIENTS)
    
    # Salviamo i risultati ordinati per SUBJECT_ID
    pd.DataFrame({'SUBJECT_ID': sorted(list(allowed_patients))}).to_csv(out_allowed, index=False)
    pd.DataFrame({'SUBJECT_ID': sorted(list(excluded_patients))}).to_csv(out_excluded, index=False)
    
    print(f"📁 Lista pazienti ammessi salvata in: {OUTPUT_ALLOWED_PATIENTS}")
    print(f"📁 Lista pazienti esclusi salvata in: {OUTPUT_EXCLUDED_PATIENTS}")

if __name__ == "__main__":
    main()