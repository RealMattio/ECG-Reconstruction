import wfdb
import numpy as np
import os
import shutil
import json
import requests
import sys
from requests.auth import HTTPBasicAuth

# --- CONFIGURAZIONE ---
ROOT_DIR = "./mimic3wdb-matched_raw_data" # Cartella radice dei dati
CREDENTIALS_FILE = "physionet_creds.json"  # Necessario per il riscaricamento

# Cartella da cui iniziare (es. "p00/p000298"). Se None, inizia dall'inizio.
START_FROM_FOLDER = None 

# Parametri Filtro
MIN_DURATION_SEC = 120

# URL Base PhysioNet per il riscaricamento
BASE_URL = "https://physionet.org/files/mimic3wdb-matched/1.0/"

# Se True, NON cancella nulla, stampa solo cosa farebbe.
DRY_RUN = False 
# ----------------------

def load_credentials(filepath):
    try:
        with open(filepath, 'r') as f:
            creds = json.load(f)
            return creds.get('username'), creds.get('password')
    except Exception as e:
        print(f"❌ Errore caricamento credenziali: {e}")
        sys.exit(1)

def redownload_record(record_path, rel_folder, record_name, user, pwd):
    """
    Tenta di scaricare nuovamente .hea e .dat sovrascrivendo quelli corrotti.
    rel_folder: es. "p00/p000298"
    """
    print(f"      🔄 TENTATIVO RIPRISTINO: Riscarico {record_name} da PhysioNet...")
    
    # Costruzione URL: base + cartella_paziente + nome_file
    # Esempio: .../1.0/p00/p000298/3533390_0011.hea
    base_record_url = f"{BASE_URL}{rel_folder}/{record_name}"
    
    extensions = ['.hea', '.dat']
    local_dir = os.path.dirname(record_path)
    
    success_count = 0
    
    for ext in extensions:
        file_url = base_record_url + ext
        local_file = os.path.join(local_dir, record_name + ext)
        
        try:
            if not DRY_RUN:
                with requests.get(file_url, auth=HTTPBasicAuth(user, pwd), stream=True, timeout=30) as r:
                    if r.status_code == 200:
                        with open(local_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        success_count += 1
                    else:
                        print(f"         ❌ Errore HTTP {r.status_code} su {ext}")
            else:
                print(f"         (Simulazione download {file_url})")
                success_count += 1
                
        except Exception as e:
            print(f"         ❌ Eccezione download {ext}: {e}")

    # Se abbiamo scaricato (o simulato) entrambi i file
    return success_count == 2

def check_quality_logic(record):
    """Logica di controllo qualità fornita dall'utente"""
    try:
        signals = record.sig_name
        
        if 'II' not in signals or 'PLETH' not in signals:
            return False, f"Canali mancanti (Trovati: {signals})"

        ecg = record.p_signal[:, signals.index('II')]
        ppg = record.p_signal[:, signals.index('PLETH')]

        # --- ECG CHECK ---
        if np.isnan(ecg).all(): status_ecg = "BAD"
        elif np.std(ecg[~np.isnan(ecg)]) < 0.005: status_ecg = "BAD"
        else:
            nan_pct = np.isnan(ecg).mean() * 100
            status_ecg = "BAD" if nan_pct > 50 else "OK"

        # --- PPG CHECK ---
        if np.isnan(ppg).all(): status_ppg = "BAD"
        elif np.std(ppg[~np.isnan(ppg)]) < 0.005: status_ppg = "BAD"
        else:
            nan_pct = np.isnan(ppg).mean() * 100
            status_ppg = "BAD" if nan_pct > 50 else "OK"

        if status_ecg == "OK" and status_ppg == "OK":
            return True, "Valid"
        else:
            return False, f"Qualità scarsa (ECG:{status_ecg}, PPG:{status_ppg})"
            
    except Exception as e:
        return False, f"Errore calcolo metriche: {e}"

def process_record(record_path, rel_folder, record_name, user, pwd):
    """
    Gestisce il flusso: Leggi -> (Errore? -> Riscarica -> Rileggi) -> Controlla Qualità
    Ritorna: True (Tieni), False (Cancella)
    """
    
    # 1. Primo Tentativo di Lettura
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"      ⚠️ Errore lettura iniziale: {e}")
        
        # 2. Tentativo di Ripristino (Riscaricamento)
        if redownload_record(record_path, rel_folder, record_name, user, pwd):
            try:
                if not DRY_RUN:
                    record = wfdb.rdrecord(record_path) # Riprova a leggere il file nuovo
                    print("         ✅ Ripristino riuscito! Il file ora è leggibile.")
                else:
                    return False # In dry run assumiamo fallimento per non crashare
            except Exception as e2:
                print(f"         ❌ Il file è ancora corrotto dopo il download: {e2}")
                return False # Cancella
        else:
            print("         ❌ Ripristino fallito (download incompleto).")
            return False # Cancella

    # 3. Controllo Durata
    duration = record.sig_len / record.fs
    if duration < MIN_DURATION_SEC:
        print(f"      ❌ Durata insufficiente: {duration:.1f}s")
        return False

    # 4. Controllo Qualità (ECG/PPG)
    is_good, reason = check_quality_logic(record)
    if not is_good:
        print(f"      ❌ {reason}")
        return False

    return True

def delete_files(base_path):
    hea = base_path + ".hea"
    dat = base_path + ".dat"
    try:
        if os.path.exists(hea): 
            if not DRY_RUN: os.remove(hea)
        if os.path.exists(dat): 
            if not DRY_RUN: os.remove(dat)
    except Exception as e:
        print(f"Errore cancellazione file: {e}")

def get_all_patient_folders(root_dir):
    patient_dirs = []
    if not os.path.exists(root_dir): return []
    for group in sorted(os.listdir(root_dir)): # p00
        g_path = os.path.join(root_dir, group)
        if os.path.isdir(g_path) and group.startswith('p'):
            for pat in sorted(os.listdir(g_path)): # p000123
                p_path = os.path.join(g_path, pat)
                if os.path.isdir(p_path):
                    patient_dirs.append(f"{group}/{pat}")
    return patient_dirs

def main():
    print("--- DATASET REPAIR & CLEAN TOOL ---")
    user, pwd = load_credentials(CREDENTIALS_FILE)
    
    all_patients = get_all_patient_folders(ROOT_DIR)
    print(f"Trovati {len(all_patients)} pazienti totali.")
    
    started = False
    if not START_FROM_FOLDER: started = True
    
    stats = {'kept': 0, 'deleted': 0, 'restored': 0}

    for rel_path in all_patients:
        # Resume Logic
        if not started:
            if rel_path == START_FROM_FOLDER:
                started = True
                print(f"--> Ripresa da: {rel_path}")
            else:
                continue

        full_dir = os.path.join(ROOT_DIR, rel_path)
        print(f"\n📂 Analisi: {rel_path}")
        
        # Trova record
        hea_files = sorted([f for f in os.listdir(full_dir) 
                            if f.endswith('.hea') and not f.startswith('p')])
        
        if not hea_files:
            print("   (Vuoto o solo layout)")
            continue

        for f in hea_files:
            record_name = f.replace('.hea', '')
            record_path_base = os.path.join(full_dir, record_name)
            
            # Processa (Analisi + Eventuale Ripristino)
            should_keep = process_record(record_path_base, rel_path, record_name, user, pwd)
            
            if should_keep:
                print(f"   ✅ OK: {record_name}")
                stats['kept'] += 1
            else:
                print(f"   🗑️ ELIMINAZIONE: {record_name}")
                delete_files(record_path_base)
                stats['deleted'] += 1

    print("\n" + "="*30)
    print("OPERAZIONE COMPLETATA")
    print(f"Record Mantenuti: {stats['kept']}")
    print(f"Record Eliminati: {stats['deleted']}")

if __name__ == "__main__":
    main()