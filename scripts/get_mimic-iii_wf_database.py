import requests
import os
import re
import json
import sys
import time
from urllib.parse import urljoin
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
import argparse

# --- CONFIGURAZIONE ---
DOWNLOAD_DIR = "./mimic3wdb-matched_raw_data"
CREDENTIALS_FILE = "physionet_creds.json"

# Se None scarica tutto. Se metti un numero, si ferma dopo averne scaricati TOT nuovi.
NUM_PATIENTS_TO_DOWNLOAD = None 
# Inserisci qui l'ultimo paziente processato per riprendere da lui (incluso).
# Esempio: "p00/p002345". Se None o stringa vuota, parte dall'inizio.
parser = argparse.ArgumentParser(description="Downloader MIMIC-III con Retry e Resume")
parser.add_argument('--resume_from', type=str, default=None, help="Paziente da cui riprendere (es. 'p00/p002345'). Se non specificato, parte dall'inizio.")
args = parser.parse_args()
RESUME_FROM_PATIENT = args.resume_from

BASE_URL = "https://physionet.org/files/mimic3wdb-matched/1.0/"
# ----------------------

def load_credentials(filepath):
    try:
        with open(filepath, 'r') as f:
            creds = json.load(f)
            return creds.get('username'), creds.get('password')
    except Exception as e:
        print(f"[ERRORE] Credenziali: {e}")
        sys.exit(1)

def get_patient_list(user, pwd):
    print("1. Recupero lista pazienti globale...")
    try:
        records_url = urljoin(BASE_URL, "RECORDS-waveforms")
        response = requests.get(records_url, auth=(user, pwd), stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"[ERRORE] Impossibile scaricare lista record (Status {response.status_code})")
            sys.exit(1)
        
        patient_dirs = set()
        for line in response.iter_lines():
            if line:
                parts = line.decode('utf-8').split('/')
                if len(parts) >= 2:
                    patient_dirs.add(f"{parts[0]}/{parts[1]}") 
        
        sorted_patients = sorted(list(patient_dirs))
        print(f"-> Trovati {len(sorted_patients)} pazienti unici.")
        return sorted_patients
    except Exception as e:
        print(f"[FATAL] Errore connessione iniziale: {e}")
        sys.exit(1)

def download_file_with_retry(url, local_path, user, pwd, max_retries=5):
    """
    Scarica un file gestendo disconnessioni e ChunkedEncodingError.
    Riprova fino a max_retries volte.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, auth=(user, pwd), stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: 
                            f.write(chunk)
            return True # Successo
        except (ChunkedEncodingError, ConnectionError, ReadTimeout, Exception) as e:
            attempt += 1
            wait_time = attempt * 2 # Backoff esponenziale (2s, 4s, 6s...)
            print(f"\n      ⚠️ Errore download ({e}). Riprovo {attempt}/{max_retries} tra {wait_time}s...", end='', flush=True)
            time.sleep(wait_time)
            
            # Se il file è rimasto incompleto o corrotto, verrà sovrascritto al prossimo tentativo
            
    print(f"\n      ❌ Fallito download di {os.path.basename(local_path)} dopo {max_retries} tentativi.")
    # Rimuoviamo il file parziale corrotto per non lasciare spazzatura
    if os.path.exists(local_path):
        os.remove(local_path)
    return False

def scan_and_download_patient(patient_rel_path, user, pwd):
    patient_url = urljoin(BASE_URL, patient_rel_path + "/")
    local_save_dir = os.path.join(DOWNLOAD_DIR, patient_rel_path)
    
    try:
        resp = requests.get(patient_url, auth=(user, pwd), timeout=30)
        if resp.status_code != 200: return False
        page_content = resp.text
    except:
        return False

    hea_files = set(re.findall(r'href="(\d+_\d+\.hea)"', page_content))
    
    if not hea_files:
        return False

    found_something = False
    
    for hea_name in hea_files:
        dat_name = hea_name.replace('.hea', '.dat')
        local_hea = os.path.join(local_save_dir, hea_name)
        local_dat = os.path.join(local_save_dir, dat_name)

        # Se esistono già entrambi, saltiamo
        if os.path.exists(local_hea) and os.path.exists(local_dat):
            # Controllo opzionale: se il file dat è 0 byte (errore precedente), lo riscarichiamo
            if os.path.getsize(local_dat) > 0:
                continue

        hea_url = urljoin(patient_url, hea_name)
        
        # Scarica Header (Testo piccolo, meno probabile fallisca, ma usiamo retry per sicurezza)
        # Usiamo una chiamata diretta per leggere il contenuto prima di salvare
        try:
            h_resp = requests.get(hea_url, auth=(user, pwd), timeout=30)
            if h_resp.status_code != 200: continue
            header_text = h_resp.text
        except:
            continue

        has_pleth = 'PLETH' in header_text
        has_ii = re.search(r'\bII\b', header_text) is not None

        if has_pleth and has_ii:
            print(f"   [FOUND] {patient_rel_path}/{hea_name} ha ECG+PPG!")
            
            os.makedirs(local_save_dir, exist_ok=True)
            
            # 1. Salva .hea
            with open(local_hea, 'w') as f:
                f.write(header_text)
            
            # 2. Scarica .dat con RETRY ROBUSTO
            dat_url = urljoin(patient_url, dat_name)
            print(f"      -> Scaricamento {dat_name}...", end='', flush=True)
            
            if download_file_with_retry(dat_url, local_dat, user, pwd):
                print(" OK.")
                found_something = True
            else:
                print(" FALLITO.")
    
    return found_something

def main():
    print("--- MIMIC-III Robust Downloader (Auto-Retry) ---")
    user, pwd = load_credentials(CREDENTIALS_FILE)
    
    all_patients = get_patient_list(user, pwd)
    
    downloaded_count = 0
    print(f"\n2. Scansione profonda pazienti...")
    
    start_processing = True
    if RESUME_FROM_PATIENT:
        print(f"-> Modalità RESUME attiva. Salto tutto fino a: {RESUME_FROM_PATIENT}")
        start_processing = False

    if NUM_PATIENTS_TO_DOWNLOAD is not None:
        print(f"   (Limite attivo: stop dopo {NUM_PATIENTS_TO_DOWNLOAD} nuovi download)")

    for p_path in all_patients:
        # Gestione Resume
        if not start_processing:
            if p_path == RESUME_FROM_PATIENT:
                print(f"-> Trovato punto di ripresa: {p_path}. Ricomincio a processare.")
                start_processing = True
            else:
                continue 

        if NUM_PATIENTS_TO_DOWNLOAD is not None and downloaded_count >= NUM_PATIENTS_TO_DOWNLOAD:
            print("\nRaggiunto limite numerico di download impostato.")
            break
            
        print(f"🔍 Scan: {p_path}", end='\r')
        
        try:
            if scan_and_download_patient(p_path, user, pwd):
                downloaded_count += 1
                print(f"✅ SCARICATO/AGGIORNATO: {p_path}")
        except KeyboardInterrupt:
            print("\n\nInterrotto dall'utente. Puoi riprendere impostando:")
            print(f'RESUME_FROM_PATIENT = "{p_path}"')
            sys.exit(0)
        except Exception as e:
            print(f"\n[WARN] Errore imprevisto su {p_path}: {e}. Passo al prossimo.")
            continue
            
    print(f"\n\nFinito. File salvati in {DOWNLOAD_DIR}")

if __name__ == "__main__":
    main()