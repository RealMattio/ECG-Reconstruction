import requests
import os
import re
import json
import sys
import time
import shutil
import pandas as pd
from urllib.parse import urljoin
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
import argparse

# --- CONFIGURAZIONE ---
DOWNLOAD_DIR = "./mimic3wdb-matched_healthy_data"
CREDENTIALS_FILE = "physionet_creds.json"
ALLOWED_PATIENTS_CSV = "./mimic_clinical_data/allowed_patients.csv"

# Costanti operative
MIN_DURATION_SEC = 180.0     # Minimo 3 minuti (180 secondi)
MAX_PATIENT_HOURS = 36.0     # Massimo 36 ore per paziente per bilanciare il dataset
MAX_PATIENT_SEC = MAX_PATIENT_HOURS * 3600.0 

MIN_DISK_SPACE_GB = 4.0      # Fermati se lo spazio su disco scende sotto i 4 GB
DISK_PARTITION = "/home"     # Partizione da monitorare

BASE_URL = "https://physionet.org/files/mimic3wdb-matched/1.0/"
# ----------------------

parser = argparse.ArgumentParser(description="Downloader MIMIC-III Clinically Safe & Duration Filtered")
parser.add_argument('--resume_from', type=str, default=None, help="Paziente da cui riprendere (es. 'p00/p002345').")
parser.add_argument('--limit', type=int, default=None, help="Stop dopo N nuovi pazienti scaricati.")

# Argomenti per la parallelizzazione Slurm
parser.add_argument('--worker_id', type=int, default=0, help="ID di questo worker (es. 0, 1, 2...)")
parser.add_argument('--total_workers', type=int, default=1, help="Numero totale di worker in esecuzione")

args = parser.parse_args()

RESUME_FROM_PATIENT = args.resume_from
NUM_PATIENTS_TO_DOWNLOAD = args.limit

def check_disk_space():
    """Controlla se lo spazio sulla partizione /home è superiore al limite richiesto."""
    try:
        target_dir = DISK_PARTITION if os.path.exists(DISK_PARTITION) else "."
        total, used, free = shutil.disk_usage(target_dir)
        free_gb = free / (1024 ** 3)
        if free_gb < MIN_DISK_SPACE_GB:
            print(f"\n[ALLARME] Spazio su disco insufficiente ({free_gb:.2f} GB liberi). Soglia: {MIN_DISK_SPACE_GB} GB.")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Impossibile verificare spazio su disco: {e}")
        return True

def load_credentials(filepath):
    try:
        with open(filepath, 'r') as f:
            creds = json.load(f)
            return creds.get('username'), creds.get('password')
    except Exception as e:
        print(f"[ERRORE] Credenziali: {e}")
        sys.exit(1)

def load_allowed_patients(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERRORE] File {csv_path} non trovato! Esegui prima gli script di filtraggio clinico.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    allowed_set = set(df['SUBJECT_ID'].astype(int))
    print(f"-> Caricati {len(allowed_set)} pazienti AMMESSI dalla valutazione clinica.")
    return allowed_set

def get_patient_list(user, pwd, allowed_ids_set):
    print("1. Recupero lista pazienti globale da PhysioNet...")
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
                    subj_str = parts[1][1:] 
                    try:
                        subj_id = int(subj_str)
                        if subj_id in allowed_ids_set:
                            patient_dirs.add(f"{parts[0]}/{parts[1]}") 
                    except ValueError:
                        pass
        
        sorted_patients = sorted(list(patient_dirs))
        print(f"-> Trovati {len(sorted_patients)} pazienti che soddisfano i criteri clinici.")
        return sorted_patients
    except Exception as e:
        print(f"[FATAL] Errore connessione iniziale: {e}")
        sys.exit(1)

def download_file_with_retry(url, local_path, user, pwd, max_retries=5):
    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, auth=(user, pwd), stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: 
                            f.write(chunk)
            return True 
        except (ChunkedEncodingError, ConnectionError, ReadTimeout, Exception) as e:
            attempt += 1
            wait_time = attempt * 2 
            print(f"\n      ⚠️ Errore download. Riprovo {attempt}/{max_retries} tra {wait_time}s...", end='', flush=True)
            time.sleep(wait_time)
            
    print(f"\n      ❌ Fallito download di {os.path.basename(local_path)} dopo {max_retries} tentativi.")
    if os.path.exists(local_path):
        os.remove(local_path)
    return False

def check_duration_from_hea(header_text):
    lines = header_text.strip().split('\n')
    if not lines: return 0.0
    
    parts = lines[0].split()
    if len(parts) >= 4:
        try:
            fs = float(parts[2])
            num_samples = int(parts[3])
            return num_samples / fs
        except ValueError:
            return 0.0
    return 0.0

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
    if not hea_files: return False

    found_something = False
    patient_total_downloaded_sec = 0.0  
    
    for hea_name in hea_files:
        
        # --- CONTROLLO LIMITE ORE PER PAZIENTE ---
        if patient_total_downloaded_sec >= MAX_PATIENT_SEC:
            print(f"\n  [LIMITE RAGGIUNTO] Scaricate {patient_total_downloaded_sec/3600:.1f} ore per questo paziente. Passo al prossimo.")
            break
        # -----------------------------------------
        
        dat_name = hea_name.replace('.hea', '.dat')
        local_hea = os.path.join(local_save_dir, hea_name)
        local_dat = os.path.join(local_save_dir, dat_name)

        # Se esiste già, aggiorniamo il contatore e skippiamo il download
        if os.path.exists(local_hea) and os.path.exists(local_dat):
            if os.path.getsize(local_dat) > 0:
                with open(local_hea, 'r') as f:
                    duration = check_duration_from_hea(f.read())
                    patient_total_downloaded_sec += duration
                continue

        hea_url = urljoin(patient_url, hea_name)
        
        try:
            h_resp = requests.get(hea_url, auth=(user, pwd), timeout=30)
            if h_resp.status_code != 200: continue
            header_text = h_resp.text
        except:
            continue

        has_pleth = 'PLETH' in header_text
        has_ii = re.search(r'\bII\b', header_text) is not None

        if not (has_pleth and has_ii):
            continue
            
        duration = check_duration_from_hea(header_text)
        if duration < MIN_DURATION_SEC:
            continue

        if not check_disk_space():
            print(f"\n[STOP] Salvataggio interrotto per proteggere il disco.")
            print(f"Per riprendere successivamente lancia lo script con: --resume_from {patient_rel_path}")
            sys.exit(0)

        print(f"  [FOUND] {patient_rel_path}/{hea_name} idoneo! (ECG+PPG, {duration/60:.1f} min)")
        os.makedirs(local_save_dir, exist_ok=True)
        
        with open(local_hea, 'w') as f:
            f.write(header_text)
        
        dat_url = urljoin(patient_url, dat_name)
        print(f"      -> Scaricamento {dat_name}...", end='', flush=True)
        
        if download_file_with_retry(dat_url, local_dat, user, pwd):
            print(" OK.")
            found_something = True
            patient_total_downloaded_sec += duration
        else:
            print(" FALLITO.")
    
    return found_something

def main():
    print("--- MIMIC-III: Clinically Safe & Balanced Downloader (Max 36h/Paziente) ---")
    user, pwd = load_credentials(CREDENTIALS_FILE)
    
    allowed_ids_set = load_allowed_patients(ALLOWED_PATIENTS_CSV)
    all_patients = get_patient_list(user, pwd, allowed_ids_set)
    
    # --- SHARDING MATEMATICO ---
    if args.total_workers > 1:
        print(f"\n[PARALLELIZZAZIONE] Esecuzione Worker {args.worker_id} di {args.total_workers}")
        all_patients = [p for i, p in enumerate(all_patients) if i % args.total_workers == args.worker_id]
        print(f"-> Questo worker elaborerà {len(all_patients)} pazienti in esclusiva.")
    # ---------------------------

    downloaded_count = 0
    print(f"\n2. Scansione profonda pazienti idonei...")
    
    start_processing = True
    if RESUME_FROM_PATIENT:
        print(f"-> Modalità RESUME attiva. Salto tutto fino a: {RESUME_FROM_PATIENT}")
        start_processing = False

    if NUM_PATIENTS_TO_DOWNLOAD is not None:
        print(f"   (Limite attivo: stop dopo {NUM_PATIENTS_TO_DOWNLOAD} nuovi download)")

    for p_path in all_patients:
        if not start_processing:
            # Sostituito == con >= in modo che ogni worker riprenda dal SUO primo paziente utile
            if p_path >= RESUME_FROM_PATIENT:
                print(f"-> [W{args.worker_id}] Trovato punto di ripresa idoneo: {p_path}. Ricomincio a processare.")
                start_processing = True
            else:
                continue

        if NUM_PATIENTS_TO_DOWNLOAD is not None and downloaded_count >= NUM_PATIENTS_TO_DOWNLOAD:
            print("\nRaggiunto limite numerico di download impostato.")
            break
            
        print(f"[W{args.worker_id}] 🔍 Scan: {p_path}", end='\r')
        
        try:
            if scan_and_download_patient(p_path, user, pwd):
                downloaded_count += 1
                print(f"[W{args.worker_id}] ✅ PAZIENTE COMPLETATO: {p_path} (Totale nuovi pazienti worker: {downloaded_count})")
        except KeyboardInterrupt:
            print("\n\nInterrotto dall'utente. Puoi riprendere lanciando:")
            print(f'python download_healthy_waveforms.py --worker_id {args.worker_id} --total_workers {args.total_workers} --resume_from "{p_path}"')
            sys.exit(0)
        except SystemExit:
            sys.exit(0)
        except Exception as e:
            print(f"\n[WARN] Errore imprevisto su {p_path}: {e}. Passo al prossimo.")
            continue
            
    print(f"\n\n[W{args.worker_id}] Finito. File salvati in {DOWNLOAD_DIR}")

if __name__ == "__main__":
    main()