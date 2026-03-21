import wfdb
import os
import datetime
import numpy as np

# --- CONFIGURAZIONE ---
ROOT_DIR = "./mimic3wdb-matched_raw_data"
# ----------------------

def format_duration(seconds):
    """Converte secondi in stringa leggibile HH:MM:SS"""
    return str(datetime.timedelta(seconds=int(seconds)))

def get_all_patient_folders(root_dir):
    patient_dirs = []
    if not os.path.exists(root_dir): return []
    
    # Scorre gruppi p00, p01...
    for group in sorted(os.listdir(root_dir)):
        g_path = os.path.join(root_dir, group)
        if os.path.isdir(g_path) and group.startswith('p'):
            # Scorre pazienti p000xxx...
            for pat in sorted(os.listdir(g_path)):
                p_path = os.path.join(g_path, pat)
                if os.path.isdir(p_path):
                    patient_dirs.append(p_path)
    return patient_dirs

def analyze_dataset_stats():
    print(f"--- REPORT STATISTICO DATASET MIMIC-III ---")
    print(f"Sorgente: {ROOT_DIR}\n")
    
    patient_folders = get_all_patient_folders(ROOT_DIR)
    
    grand_total_files = 0
    grand_total_seconds = 0
    patients_with_data = 0
    
    print(f"{'PAZIENTE':<15} | {'N. FILE':<8} | {'DURATA TOTALE':<15} | {'DETTAGLI (File > Durata)'}")
    print("-" * 100)

    for pat_folder in patient_folders:
        patient_id = os.path.basename(pat_folder)
        
        # Trova i file header (escludendo i layout 'pXXXXXX.hea')
        hea_files = sorted([f for f in os.listdir(pat_folder) 
                            if f.endswith('.hea') and not f.startswith('p')])
        
        if not hea_files:
            continue # Salta pazienti vuoti (cancellati completamente)

        patient_files_count = 0
        patient_total_sec = 0
        file_details = []

        for f in hea_files:
            rec_name = f.replace('.hea', '')
            rec_path = os.path.join(pat_folder, rec_name)
            
            try:
                # Leggiamo solo l'header per velocità
                header = wfdb.rdheader(rec_path)
                duration_sec = header.sig_len / header.fs
                
                patient_total_sec += duration_sec
                patient_files_count += 1
                
                # Aggiungi dettaglio (es. "3238451_0001: 10m")
                dur_fmt = f"{int(duration_sec)}s"
                if duration_sec > 3600:
                    dur_fmt = f"{duration_sec/3600:.1f}h"
                elif duration_sec > 60:
                    dur_fmt = f"{duration_sec/60:.0f}m"
                    
                file_details.append(f"{rec_name} ({dur_fmt})")
                
            except Exception as e:
                print(f"[WARN] Errore lettura {rec_name}: {e}")

        if patient_files_count > 0:
            patients_with_data += 1
            grand_total_files += patient_files_count
            grand_total_seconds += patient_total_sec
            
            # Formattazione per la tabella
            details_str = ", ".join(file_details)
            # Tronca i dettagli se troppo lunghi per la stampa a schermo
            if len(details_str) > 50:
                details_str = details_str[:47] + "..."
                
            print(f"{patient_id:<15} | {patient_files_count:<8} | {format_duration(patient_total_sec):<15} | {details_str}")

    print("-" * 100)
    print("\n=== RIEPILOGO FINALE ===")
    print(f"Pazienti Validi:      {patients_with_data} (su {len(patient_folders)} cartelle totali)")
    print(f"Totale File/Segmenti: {grand_total_files}")
    print(f"Tempo Totale Dati:    {format_duration(grand_total_seconds)} (approx {grand_total_seconds/3600:.1f} ore)")
    
    if patients_with_data > 0:
        avg_per_patient = grand_total_seconds / patients_with_data
        print(f"Media per Paziente:   {format_duration(avg_per_patient)}")

if __name__ == "__main__":
    analyze_dataset_stats()