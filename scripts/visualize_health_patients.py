import os
import json
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

def plot_selected_windows(items, cache_dir, title_main, save_path, num_plots=4):
    """
    Estrae 'num_plots' campioni casuali, li carica e crea un grafico multiplo.
    """
    num_plots = min(num_plots, len(items))
    if num_plots == 0:
        print(f"Nessun elemento da plottare per {title_main}")
        return

    # Campionamento casuale
    sampled_items = random.sample(items, num_plots)

    # Creazione della griglia di subplot
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots))
    if num_plots == 1:
        axs = [axs] # Per iterare in modo sicuro se c'è un solo plot

    fig.suptitle(title_main, fontsize=16, fontweight='bold', y=0.98)

    fs = 125 # Frequenza di campionamento attesa

    for i, entry in enumerate(sampled_items):
        subj_id = entry['subject_id']
        file_name = entry['file_path']
        
        # Gestione robusta del path
        rel_path = os.path.join(subj_id, os.path.basename(file_name))
        full_path = os.path.join(cache_dir, rel_path)

        try:
            # Caricamento tensori
            data = torch.load(full_path, map_location='cpu')
            ecg = data['ecg'].numpy()
            ppg = data['ppg'].numpy() 
            
            time_axis = np.arange(len(ecg)) / fs

            # Normalizzazione Z-score rapida solo per scopi visivi (per plottarli insieme)
            ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
            ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)

            # Plottiamo l'ECG (nero) e la PPG di contesto (blu trasparente)
            axs[i].plot(time_axis, ecg_norm, color='black', linewidth=1.5, label='ECG')
            axs[i].plot(time_axis, ppg_norm, color='blue', alpha=0.4, linewidth=1.0, label='PPG')
            
            # Formattazione Titolo come richiesto
            title_str = f"Paziente: {subj_id} | Finestra: {os.path.basename(file_name)}"
            axs[i].set_title(title_str, fontsize=12, color='darkred' if 'ESCLUSE' in title_main else 'darkgreen')
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc='upper right')
            axs[i].set_ylabel("Ampiezza (Z-score)")
            
            if i == num_plots - 1:
                axs[i].set_xlabel("Tempo (secondi)")
                
        except Exception as e:
            axs[i].set_title(f"ERRORE nel caricare {file_name}: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Salvato plot: {save_path}")

def main():
    # Definisci i percorsi
    CACHE_DIR = "/home/mmerone/mattia/ecg_generation/mimic3_pinn_cache"
    ORIGINAL_MANIFEST = os.path.join(CACHE_DIR, 'dataset_manifest.json')
    HEALTHY_MANIFEST = os.path.join(CACHE_DIR, 'dataset_manifest_healthy.json')

    print("🔍 Lettura dei manifest in corso...")
    
    with open(ORIGINAL_MANIFEST, 'r') as f:
        orig_data = json.load(f)
    with open(HEALTHY_MANIFEST, 'r') as f:
        healthy_data = json.load(f)

    # Identifichiamo quali sono stati esclusi usando il file_path come chiave
    healthy_paths = set(item['file_path'] for item in healthy_data)
    excluded_data = [item for item in orig_data if item['file_path'] not in healthy_paths]

    print(f"✅ Trovate {len(healthy_data)} finestre mantenute.")
    print(f"❌ Trovate {len(excluded_data)} finestre escluse.")

    # Generiamo l'immagine per i mantenuti
    plot_selected_windows(
        items=healthy_data, 
        cache_dir=CACHE_DIR, 
        title_main="Finestre MANTENUTE (Morfologia ECG Sana)", 
        save_path=os.path.join(CACHE_DIR, "plot_finestre_mantenute.png"), 
        num_plots=4
    )

    # Generiamo l'immagine per gli esclusi
    plot_selected_windows(
        items=excluded_data, 
        cache_dir=CACHE_DIR, 
        title_main="Finestre ESCLUSE (Patologia o Rumore)", 
        save_path=os.path.join(CACHE_DIR, "plot_finestre_escluse.png"), 
        num_plots=4
    )

if __name__ == "__main__":
    main()