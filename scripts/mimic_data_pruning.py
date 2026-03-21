import wfdb
import os
import shutil
import datetime

# --- CONFIGURAZIONE ---
ROOT_DIR = "./mimic3wdb-matched_raw_data"        # Cartella dati attuale
PRUNED_DIR = "./mimic3wdb-matched_pruned_excess" # Cartella dove spostare l'eccesso

# Limite massimo di durata per paziente
MAX_DAYS_PER_PATIENT = 3.0 
MAX_SECONDS = MAX_DAYS_PER_PATIENT * 24 * 3600

# Se True, simula solo lo spostamento
DRY_RUN = False
# ----------------------

def get_all_patient_folders(root_dir):
    patient_dirs = []
    if not os.path.exists(root_dir): return []
    for group in sorted(os.listdir(root_dir)):
        g_path = os.path.join(root_dir, group)
        if os.path.isdir(g_path) and group.startswith('p'):
            for pat in sorted(os.listdir(g_path)):
                p_path = os.path.join(g_path, pat)
                if os.path.isdir(p_path):
                    patient_dirs.append(f"{group}/{pat}")
    return patient_dirs

def move_record_files(base_path, dest_dir):
    """Sposta .hea e .dat nella cartella di destinazione."""
    hea_src = base_path + ".hea"
    dat_src = base_path + ".dat"
    
    file_name = os.path.basename(base_path)
    dest_hea = os.path.join(dest_dir, file_name + ".hea")
    dest_dat = os.path.join(dest_dir, file_name + ".dat")
    
    moved = []
    
    try:
        if os.path.exists(hea_src):
            if not DRY_RUN: shutil.move(hea_src, dest_hea)
            moved.append(".hea")
            
        if os.path.exists(dat_src):
            if not DRY_RUN: shutil.move(dat_src, dest_dat)
            moved.append(".dat")
            
        return moved
    except Exception as e:
        print(f"      ❌ Errore spostamento: {e}")
        return []

def main():
    print(f"--- DATASET PRUNING TOOL (Limit: {MAX_DAYS_PER_PATIENT} days) ---")
    print(f"Source: {ROOT_DIR}")
    print(f"Destination (Excess): {PRUNED_DIR}")
    print(f"Mode: {'DRY RUN (Simulation)' if DRY_RUN else 'ACTIVE (Moving Files)'}")
    
    all_patients = get_all_patient_folders(ROOT_DIR)
    
    total_moved = 0
    patients_pruned = 0
    
    for rel_path in all_patients:
        full_src_dir = os.path.join(ROOT_DIR, rel_path)
        
        # Prepara cartella destinazione
        full_dest_dir = os.path.join(PRUNED_DIR, rel_path)
        
        # Trova file
        hea_files = sorted([f for f in os.listdir(full_src_dir) 
                            if f.endswith('.hea') and not f.startswith('p')])
        
        if not hea_files: continue

        current_duration = 0.0
        pruned_for_this_patient = False
        
        # Calcoliamo prima le durate per decidere cosa tenere
        # Strategia: Teniamo i primi file fino a riempire il budget temporale
        
        print(f"\nAnalisi {rel_path}...")
        
        for f in hea_files:
            rec_name = f.replace('.hea', '')
            rec_path = os.path.join(full_src_dir, rec_name)
            
            try:
                # Leggiamo solo header per velocità
                header = wfdb.rdheader(rec_path)
                duration = header.sig_len / header.fs
                
                if current_duration + duration <= MAX_SECONDS:
                    # MANTENIAMO
                    current_duration += duration
                    # print(f"   KEEP: {rec_name} ({duration/3600:.1f}h)")
                else:
                    # SPOSTIAMO
                    if not pruned_for_this_patient:
                        if not DRY_RUN: os.makedirs(full_dest_dir, exist_ok=True)
                        pruned_for_this_patient = True
                        patients_pruned += 1
                        print(f"   ⚠️ Raggiunto limite ({current_duration/3600/24:.1f} giorni). Inizio spostamento...")
                    
                    moved = move_record_files(rec_path, full_dest_dir)
                    if moved:
                        print(f"      -> MOVED: {rec_name} ({duration/3600:.1f}h)")
                        total_moved += 1
                        
            except Exception as e:
                print(f"   [ERR] Lettura {rec_name}: {e}")

    print("\n" + "="*30)
    print("PRUNING COMPLETATO")
    print(f"Pazienti Ridotti: {patients_pruned}")
    print(f"File Spostati: {total_moved}")
    print(f"Dati in eccesso salvati in: {PRUNED_DIR}")

if __name__ == "__main__":
    main()