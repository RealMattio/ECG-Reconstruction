import os
import torch
import numpy as np
from scipy.stats import skew
from tqdm import tqdm

def find_inverted_ppg(root_dir):
    """
    Scorre il dataset, analizza la skewness dei segnali PPG e identifica i file invertiti.
    """
    inverted_files = []
    total_files = 0
    error_files = []

    # Raccogliamo tutti i percorsi dei file .pt
    all_pt_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pt"):
                all_pt_files.append(os.path.join(root, file))

    print(f"🔍 Trovati {len(all_pt_files)} file da analizzare.")
    print("🚀 Inizio analisi della polarità...")

    for file_path in tqdm(all_pt_files, desc="Analisi"):
        try:
            # Caricamento su CPU per evitare l'uso della memoria GPU
            data = torch.load(file_path, map_location='cpu')
            
            if 'ppg' not in data:
                continue
            
            # Conversione in numpy e flattening (assicuriamoci che sia un vettore 1D)
            ppg = data['ppg']
            if torch.is_tensor(ppg):
                ppg = ppg.numpy()
            ppg = ppg.flatten().astype(np.float32)

            # Controllo segnale nullo o piatto (std troppo bassa)
            if np.std(ppg) < 1e-6:
                continue

            # CALCOLO DELLA SKEWNESS
            # PPG corretta -> Picchi verso l'alto (stretti) -> Skewness > 0
            # PPG invertita -> Picchi verso il basso (stretti) -> Skewness < 0
            s_val = skew(ppg)

            if s_val < 0:
                inverted_files.append((file_path, s_val))

        except Exception as e:
            error_files.append((file_path, str(e)))

    # --- REPORT FINALE ---
    print("\n" + "="*60)
    print("📊 REPORT ANALISI POLARITÀ")
    print("="*60)
    print(f"• File analizzati:   {len(all_pt_files)}")
    print(f"• PPG Invertite:     {len(inverted_files)}")
    print(f"• Errori lettura:    {len(error_files)}")
    print("-" * 60)

    if inverted_files:
        print("\n📂 ELENCO FILE CON PPG INVERTITA:")
        # Ordiniamo per valore di skewness (le più negative sono le più "sicuramente" invertite)
        inverted_files.sort(key=lambda x: x[1])
        for path, s in inverted_files:
            print(f"  [Skew: {s:6.3f}]  {path}")
    else:
        print("\n✅ Nessun file invertito trovato!")

    if error_files:
        print("\n⚠️ ERRORI RISCONTRATI:")
        for path, err in error_files:
            print(f"  {path}: {err}")

if __name__ == "__main__":
    # Imposta qui il percorso del tuo dataset
    # Se lo script è nella stessa cartella del dataset, usa "."
    DATASET_ROOT = "." 
    
    find_inverted_ppg(DATASET_ROOT)