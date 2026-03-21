import os
import pandas as pd

# --- CONFIGURAZIONE ---
DATA_DIR = "./mimic_clinical_data"
INPUT_CSV = "unique_present_pathologies.csv"
OUTPUT_CSV = "allowed_pathologies.csv"
# ----------------------

# Parole chiave in inglese (MIMIC-III è in inglese) che indicano problemi cardiaci o ECG
EXCLUDED_KEYWORDS = [
    'heart', 'cardiac', 'myocard', 'infarct', 'ischemi', 'arrhythmia', 
    'fibrillat', 'tachycardia', 'bradycardia', 'bundle branch', 'block', 
    'pacemaker', 'defibrillator', 'valv', 'coronary', 'angina', 'ventric', 
    'atrial', 'potassium', 'hypokalemia', 'hyperkalemia', 'calcium', 
    'prolonged qt', 'st elevation', 'st depression', 'pericard'
]

def is_ecg_safe(icd9_code, title):
    """
    Ritorna True se la patologia è 'sicura' per l'ECG, False se deve essere esclusa.
    """
    code_str = str(icd9_code).strip().upper()
    title_lower = str(title).strip().lower()

    # 1. Filtro testuale di sicurezza (se contiene parole chiave cardiache/elettrolitiche)
    if any(keyword in title_lower for keyword in EXCLUDED_KEYWORDS):
        return False

    # 2. Gestione dei V-Codes (Fattori supplementari e anamnesi)
    if code_str.startswith('V'):
        # V42.1 (Trapianto cuore), V42.2 (Valvola), V43.3 (Sostituzione valvola)
        # V45.0 (Pacemaker), V53.3 (Regolazione Pacemaker)
        if code_str.startswith(('V421', 'V422', 'V433', 'V450', 'V533')):
            return False
        return True # Altri V-codes (es. anamnesi familiare di asma) sono ok

    # 3. Gestione E-Codes (Cause esterne come incidenti d'auto) -> Sono sicuri per l'ECG
    if code_str.startswith('E'):
        return True

    # 4. Filtro per Blocchi Numerici ICD-9
    try:
        # Prendi le prime 3 cifre che identificano la macro-categoria
        macro_category = int(code_str[:3])
        
        # Malattie del sistema circolatorio (Ipertensione, Ischemia, Fibrillazione, ecc.)
        if 390 <= macro_category <= 459:
            return False
            
        # Anomalie congenite cardiache
        if 745 <= macro_category <= 747:
            return False
            
        # Disordini elettrolitici (Alterano pesantemente l'onda T e il tratto ST)
        if macro_category == 276:
            return False
            
        # Sintomi relativi al sistema cardiovascolare (Palpitazioni, tachicardia)
        if macro_category == 785:
            return False
            
    except ValueError:
        # Se non riesce a fare il parsing numerico (caso raro), per sicurezza lo teniamo
        # ma ci affidiamo al filtro testuale
        pass

    return True

def main():
    print("--- MIMIC-III: Filtraggio Patologie ECG-Safe ---")
    
    input_path = os.path.join(DATA_DIR, INPUT_CSV)
    output_path = os.path.join(DATA_DIR, OUTPUT_CSV)
    
    if not os.path.exists(input_path):
        print(f"[ERRORE] File {input_path} non trovato. Esegui prima la Fase 1.")
        return
        
    print(f" -> Lettura del file: {input_path}")
    df = pd.read_csv(input_path, dtype={'ICD9_CODE': str})
    
    total_pathologies = len(df)
    
    # Applichiamo la funzione di filtro
    print(" -> Applicazione dei criteri clinici (ICD-9 e NLP di base)...")
    # Creiamo una maschera booleana applicando la funzione riga per riga
    mask = df.apply(lambda row: is_ecg_safe(row['ICD9_CODE'], row['LONG_TITLE']), axis=1)
    
    df_allowed = df[mask]
    df_excluded = df[~mask]
    
    # Salvataggio file
    df_allowed.to_csv(output_path, index=False)
    
    # Salviamo anche le escluse per controllo manuale (opzionale ma utile)
    excluded_path = os.path.join(DATA_DIR, "excluded_pathologies_log.csv")
    df_excluded.to_csv(excluded_path, index=False)
    
    print("\n✅ FASE 2 COMPLETATA!")
    print(f"Patologie totali: {total_pathologies}")
    print(f"Patologie escluse (Cardio/Elettrolitiche): {len(df_excluded)}")
    print(f"Patologie AMMESSE: {len(df_allowed)}")
    print("-" * 50)
    print(f"📁 Elenco patologie ammesse salvato in: {output_path}")
    print(f"📁 Log patologie scartate salvato in: {excluded_path}")

if __name__ == "__main__":
    main()