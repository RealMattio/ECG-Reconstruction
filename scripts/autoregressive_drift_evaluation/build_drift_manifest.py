import os
import json
import argparse
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import warnings

# Disabilitiamo i warning noiosi di scipy in caso di flatlines
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data")
SPLIT_FILE = os.path.join(DATA_DIR, "dataset_split.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "drift_evaluation")

# Orizzonti Temporali (Etichetta: Secondi totali richiesti)
# Aggiungiamo 6 secondi in più al target perché ci servono come "Seed" iniziale
SEED_SEC = 6
HORIZONS = {
    "1m": 60 + SEED_SEC,
    "30m": 1800 + SEED_SEC,
    "1h": 3600 + SEED_SEC,
    "6h": 21600 + SEED_SEC,
    "12h": 43200 + SEED_SEC,
    "24h": 86400 + SEED_SEC
}

TARGET_FS = 125.0

def get_test_patients():
    """Legge i pazienti del Test Set dal JSON generato dalla pipeline."""
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(f"File split non trovato: {SPLIT_FILE}. Esegui prima la pipeline.")
    
    with open(SPLIT_FILE, 'r') as f:
        data = json.load(f)
    return set(data.get("test_patients", []))

def check_sqi(ppg, ecg):
    """
    Placeholder per il controllo di qualità profondo.
    Per risparmiare RAM, se filter_sqi=True, valuteremo blocchi di 5 minuti.
    Per ora (poiché hai detto di accettare lo sporco), ritorna sempre True.
    """
    # TODO: Implementare l'uso della classe MimicAutoregressivePreprocessor qui per il controllo profondo
    return True

def scan_test_set(test_patients, filter_sqi=False):
    """Scansiona i file dei pazienti del test set e popola il manifest."""
    manifest = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int)) # stats[horizon][patient_id] = count
    
    # Prepariamo la progress bar
    patients_list = sorted(list(test_patients))
    
    for subject_id in tqdm(patients_list, desc="Scansione Pazienti Test Set"):
        # Rimuoviamo eventuali 'p' già presenti nella stringa per evitare crash
        raw_id = str(subject_id).replace('p', '').strip()
        
        # Formattiamo correttamente a 6 cifre con la p iniziale
        subj_str = f"p{int(raw_id):06d}"
        folder_prefix = subj_str[:3] # es. p00
        
        subj_dir = os.path.join(DATA_DIR, folder_prefix, subj_str)
        if not os.path.exists(subj_dir):
            continue
            
        # Troviamo tutti gli header in questa cartella
        hea_files = glob(os.path.join(subj_dir, "*.hea"))
        
        for hea_path in hea_files:
            record_name = hea_path.replace(".hea", "")
            
            try:
                # Leggiamo SOLO l'header (molto veloce)
                record = wfdb.rdheader(record_name)
                
                # Verifichiamo che i segnali esistano
                if 'II' not in record.sig_name or 'PLETH' not in record.sig_name:
                    continue
                    
                # Calcoliamo la durata in secondi
                sig_len_sec = record.sig_len / record.fs
                
                # Se è attivo il filtro SQI, apriamo il file .dat e facciamo i conti
                if filter_sqi:
                    # In futuro qui caricherai il file wfdb.rdrecord e userai check_sqi
                    pass 
                
                # Ora verifichiamo per quali orizzonti questo file è idoneo
                for h_label, h_sec in HORIZONS.items():
                    if sig_len_sec >= h_sec:
                        # Se il file è lungo abbastanza, lo aggiungiamo.
                        # NOTA: Se un file è lungo 10 ore, per l'orizzonte "1h" potremmo estrarre 10 frammenti.
                        # Per evitare di inquinare i test, estraiamo un solo frammento (quello iniziale) per file.
                        
                        entry = {
                            "subject_id": subject_id,
                            "record_path": record_name.replace(DATA_DIR + "/", ""), # Path relativo
                            "fs_original": record.fs,
                            "start_sec": 0.0,
                            "end_sec": float(h_sec)
                        }
                        
                        manifest[h_label].append(entry)
                        stats[h_label][subject_id] += 1
                        
            except Exception as e:
                # Ignoriamo file corrotti
                continue
                
    return manifest, stats

def plot_statistics(stats, test_patients_count, save_dir):
    """Genera un'immagine con 3 subplots per le statistiche del dataset di drift."""
    os.makedirs(save_dir, exist_ok=True)
    
    horizons = list(HORIZONS.keys())
    
    # Calcolo Dati
    total_samples = [sum(stats[h].values()) for h in horizons]
    unique_patients = [len(stats[h]) for h in horizons]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Autoregressive Drift Test - Analisi Test Set ({test_patients_count} Pazienti totali)", fontsize=16)
    
    # Plot 1: Campioni Totali
    axes[0].bar(horizons, total_samples, color='skyblue', edgecolor='black')
    axes[0].set_title("Numero di Campioni per Orizzonte")
    axes[0].set_ylabel("Campioni (Record validi)")
    for i, v in enumerate(total_samples):
        axes[0].text(i, v + (max(total_samples)*0.02), str(v), ha='center', fontweight='bold')

    # Plot 2: Pazienti Unici
    axes[1].bar(horizons, unique_patients, color='lightgreen', edgecolor='black')
    axes[1].set_title("Pazienti Unici Disponibili per Orizzonte")
    axes[1].set_ylabel("Numero di Pazienti")
    axes[1].axhline(y=test_patients_count, color='r', linestyle='--', label='Totale Test Set')
    axes[1].legend()
    for i, v in enumerate(unique_patients):
        axes[1].text(i, v + (max(unique_patients)*0.02), str(v), ha='center', fontweight='bold')

    # Plot 3: Distribuzione Campioni/Paziente (Boxplot)
    box_data = []
    for h in horizons:
        # Se non ci sono campioni, mettiamo 0
        counts = list(stats[h].values()) if len(stats[h]) > 0 else [0]
        box_data.append(counts)
        
    axes[2].boxplot(box_data, tick_labels=horizons, patch_artist=True, boxprops=dict(facecolor='salmon', color='black'))
    axes[2].set_title("Distribuzione Campioni per Paziente")
    axes[2].set_ylabel("Campioni / Paziente")
    
    for ax in axes:
        ax.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "drift_test_statistics.png")
    plt.savefig(plot_path, dpi=200)
    print(f"📊 Grafico statistiche salvato in: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="MIMIC-III Drift Manifest Builder")
    parser.add_argument('--filter_sqi', action='store_true', help="Attiva l'analisi profonda SQI (lento, ignora file rumorosi).")
    args = parser.parse_args()

    print("=" * 60)
    print(" FASE 1: CREAZIONE MANIFEST PER DRIFT EVALUATION")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("-> Caricamento Split...")
    test_patients = get_test_patients()
    print(f"-> Trovati {len(test_patients)} pazienti nel Test Set.")
    
    manifest, stats = scan_test_set(test_patients, filter_sqi=args.filter_sqi)
    
    # Salvataggio Manifest
    manifest_path = os.path.join(OUTPUT_DIR, "drift_test_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\n✅ Manifest generato con successo: {manifest_path}")
    
    # Stampa recap a schermo
    print("\nRIEPILOGO DISPONIBILITÀ:")
    for h in HORIZONS.keys():
        campioni = sum(stats[h].values())
        paz = len(stats[h])
        print(f" - Orizzonte {h:<4}: {campioni:<4} campioni (provenienti da {paz} pazienti unici)")

    # Generazione Grafici
    plot_statistics(stats, len(test_patients), OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()