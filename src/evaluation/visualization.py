import matplotlib.pyplot as plt
import os
import json
import random
import torch
import numpy as np

def save_training_plots(history, save_dir):
    """Genera e salva i grafici delle perdite rilevando le chiavi dinamicamente."""
    os.makedirs(save_dir, exist_ok=True)
    if not history['train']: return

    epochs = range(1, len(history['train']) + 1)
    available_keys = history['train'][0].keys()
    
    for key in available_keys:
        if key == 'total': continue # Evitiamo duplicati se 'loss' è già presente
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [h[key] for h in history['train']], 'b-', label=f'Train {key.upper()}')
        plt.plot(epochs, [h[key] for h in history['val']], 'r-', label=f'Val {key.upper()}')
        plt.title(f'Andamento: {key.upper()}')
        plt.xlabel('Epoche'); plt.ylabel(key.upper()); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"plot_{key}.png"))
        plt.close()

def save_history_to_json(history, save_dir):
    """Salva la history in formato JSON."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=4)

def _get_input_labels(input_tuple):
    """Helper per mappare dinamicamente i nomi dei segnali in base alla lunghezza."""
    if len(input_tuple) == 4:
        return ["PPG", "EDA", "ACC", "Prev ECG"]
    return ["PPG", "EDA", "ACC"]

def plot_random_sample(windowed_train_data, save_path, window_size=8):
    """Salva un campione casuale gestendo dinamicamente il numero di input."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = random.choice(windowed_train_data)
    inputs = sample['input']
    target = sample['target']
    labels = _get_input_labels(inputs)
    
    num_plots = len(labels) + 1 # Input + Target
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=False)
    fig.suptitle(f"Sample Check - Subject: {sample['subject']}", fontsize=16)

    for i, label in enumerate(labels):
        data = inputs[i]
        ax = axes[i]
        fs = len(data) / window_size
        
        if label == "ACC":
            for j, axis_label in enumerate(['X', 'Y', 'Z']):
                ax.plot(data[:, j], label=axis_label, alpha=0.7)
            ax.legend(loc='upper right')
        else:
            ax.plot(data, color='blue' if 'PPG' in label else 'green')
            
        ax.set_title(f"{label} | Punti: {len(data)} | Fs: {fs:.1f} Hz")
        ax.grid(True, alpha=0.3)

    # Plot Target (ECG)
    axes[-1].plot(target, color='red')
    axes[-1].set_title(f"Target: ECG | Punti: {len(target)} | Fs: {len(target)/window_size:.1f} Hz")
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path); plt.close()

def plot_inference_comparison(model, val_data, device, save_path):
    """
    Esegue inferenza su un campione casuale e confronta l'output con la ground truth.
    Gestisce il riordino corretto degli argomenti per il modello (PPG, ACC, EDA, PrevECG).
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Selezione casuale dal validation set
    sample = random.choice(val_data)
    # L'ordine nel preprocessor è: (ppg, eda, acc, prev_ecg)
    ppg_raw, eda_raw, acc_raw, prev_ecg_raw = sample['input']
    ecg_gt = sample['target']
    
    # 2. Preparazione tensori con le dimensioni corrette per PyTorch (B, C, L)
    # PPG: (Length) -> (1, 1, 512)
    ppg = torch.tensor(ppg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    # EDA: (Length) -> (1, 1, 32)
    eda = torch.tensor(eda_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    # ACC: (Length, 3) -> (1, 3, 256)
    acc = torch.tensor(acc_raw).float().transpose(0, 1).unsqueeze(0).to(device)
    # Prev ECG: (Length) -> (1, 1, 2048)
    prev_ecg = torch.tensor(prev_ecg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Inferenza - ATTENZIONE ALL'ORDINE: ppg, acc, eda, prev_ecg
    with torch.no_grad():
        ecg_gen = model(ppg, acc, eda, prev_ecg).cpu().squeeze().numpy()
    
    # 4. Plotting
    fig, axes = plt.subplots(5, 1, figsize=(12, 20))
    fig.suptitle(f"Inference Check - Subject: {sample['subject']}", fontsize=16)
    
    # Visualizzazione Input
    axes[0].plot(ppg_raw, color='blue'); axes[0].set_title("Input: PPG")
    axes[1].plot(acc_raw); axes[1].set_title("Input: ACC (3-axis)")
    axes[2].plot(eda_raw, color='green'); axes[2].set_title("Input: EDA")
    axes[3].plot(prev_ecg_raw, color='purple', alpha=0.6); axes[3].set_title("Condition: Prev ECG")
    
    # Confronto Finale
    axes[4].plot(ecg_gt, label='Ground Truth (Real)', color='black', alpha=0.4, linestyle='--')
    axes[4].plot(ecg_gen, label='Generated (Model)', color='red', alpha=0.8)
    axes[4].set_title("Output Comparison: Generated vs Real ECG")
    axes[4].legend()
    
    for ax in axes: ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Grafico di inferenza salvato in: {save_path}")

def plot_ppg_ecg_comparison(model, val_data, device, save_path, epoch):
    """
    Versione corretta: allinea temporalmente PPG ed ECG usando i secondi invece degli indici.
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Selezione casuale
    sample = random.choice(val_data)
    ppg_raw = sample['input'][0] 
    ecg_gt = sample['target']
    
    # Parametri temporali
    duration = 7.0 # secondi
    t_ppg = np.linspace(0, duration, len(ppg_raw))
    t_ecg = np.linspace(0, duration, len(ecg_gt))
    
    # 2. Preparazione tensore (B, C, L)
    ppg_tensor = torch.tensor(ppg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Inferenza
    with torch.no_grad():
        ecg_gen = model(ppg_tensor).cpu().squeeze().numpy()
    
    # 4. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Epoch {epoch} | Subject: {sample['subject']} | Temporal Alignment (7s)", fontsize=14)
    
    # Subplot 1: Input PPG (64 Hz)
    axes[0].plot(t_ppg, ppg_raw, color='blue', label='Input PPG (64Hz)')
    axes[0].set_title("Input: Photoplethysmogram (PPG)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    
    # Subplot 2: Ground Truth ECG (256 Hz)
    axes[1].plot(t_ecg, ecg_gt, color='black', alpha=0.6, label='Real ECG (256Hz)')
    axes[1].set_title("Target: Real ECG (Ground Truth)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc='upper right')
    
    # Subplot 3: Comparison (Overlay)
    axes[2].plot(t_ecg, ecg_gt, color='black', alpha=0.3, label='Real', linestyle='--')
    axes[2].plot(t_ecg, ecg_gen, color='red', alpha=0.8, label='Generated')
    axes[2].set_title("Overlay Comparison (Signals Synced in Time)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc='upper right')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration) # Forza il limite a 7 secondi
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_inference_plot(model, val_loader, device, save_path, title="Random Validation Window Inference"):
    """
    Esegue l'inferenza su una finestra casuale del validation set e salva il grafico di confronto.

    Args:
        model (torch.nn.Module): Il modello addestrato (es. HA-CNN-BILSTM).
        val_loader (DataLoader): Il DataLoader contenente i dati di validazione.
        device (torch.device): Il dispositivo su cui eseguire l'inferenza (cuda/cpu).
        save_path (str): Il percorso completo dove salvare l'immagine (es. 'results/final_plot.png').
        title (str, optional): Il titolo del grafico. Default: "Random Validation Window Inference".
    """
    model.eval() # Imposta il modello in modalità valutazione (disattiva dropout, ecc.)
    
    # Assicuriamoci che la directory di destinazione esista
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # 1. Estrazione Casuale
        # Per garantire la casualità, carichiamo tutti i batch (se la memoria lo consente)
        # o iteriamo fino a un punto casuale. Dato che sono segmenti beat-by-beat,
        # il validation set non dovrebbe essere enorme.
        all_batches = list(val_loader)
        if not all_batches:
            print("Errore: Validation loader vuoto.")
            return

        # Scelta di un batch casuale
        random_batch_idx = random.randint(0, len(all_batches) - 1)
        ppg_batch, ecg_batch = all_batches[random_batch_idx]

        # Scelta di una finestra casuale all'interno del batch
        # ppg_batch shape: (Batch_Size, 1, Seq_Len)
        random_window_idx = random.randint(0, ppg_batch.shape[0] - 1)
        
        # Estrazione e preparazione tensori per l'input (aggiunta dim batch=1)
        ppg_win_tensor = ppg_batch[random_window_idx].unsqueeze(0).to(device)
        ecg_real_tensor = ecg_batch[random_window_idx].unsqueeze(0).to(device)

        # 2. Inferenza
        with torch.no_grad():
            ecg_gen_tensor = model(ppg_win_tensor)

        # 3. Conversione in Numpy per il plotting
        # .squeeze() rimuove le dimensioni batch e channel (1, 1, 120) -> (120,)
        ppg_np = ppg_win_tensor.squeeze().cpu().numpy()
        ecg_real_np = ecg_real_tensor.squeeze().cpu().numpy()
        ecg_gen_np = ecg_gen_tensor.squeeze().cpu().numpy()
        
        # Asse X (campioni)
        time_axis = np.arange(len(ecg_real_np))

        # 4. Creazione del Grafico
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Subplot 1: Input PPG
        axes[0].plot(time_axis, ppg_np, color='blue', label='Input PPG (Normalized)', linewidth=1.5)
        axes[0].set_title("Input Photoplethysmogram (PPG)")
        axes[0].set_ylabel("Normalized Amplitude")
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[0].legend()

        # Subplot 2: Confronto ECG Reale vs Generato
        # Ground Truth in grigio tratteggiato, Generato in rosso solido
        axes[1].plot(time_axis, ecg_real_np, color='black', linestyle='--', alpha=0.6, label='Ground Truth (Real ECG)', linewidth=2)
        axes[1].plot(time_axis, ecg_gen_np, color='red', label='Generated ECG (HA-CNN-BILSTM)', linewidth=2)
        
        axes[1].set_title(title)
        axes[1].set_xlabel("Time (Samples)")
        axes[1].set_ylabel("Normalized Amplitude")
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        # 5. Salvataggio
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Chiude la figura per liberare memoria
        print(f"Grafico di inferenza salvato correttamente in: {save_path}")

    except Exception as e:
        print(f"Errore durante la generazione del grafico di inferenza: {e}")
    finally:
        # Opzionale: riporta il modello in train mode se necessario, 
        # ma solitamente questa funzione si chiama a fine addestramento.
        # model.train() 
        pass

def generate_long_window_smooth(model, ppg_signal, window_len=120, overlap_pct=0.5):
    """Genera un ECG lungo usando una sliding window e facendo la media degli overlap."""
    step = int(window_len * (1 - overlap_pct))
    output_ecg = np.zeros(len(ppg_signal))
    count_map = np.zeros(len(ppg_signal)) # Per fare la media
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ppg_signal) - window_len, step):
            seg = ppg_signal[i : i + window_len]
            seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(seg_t).cpu().numpy().flatten()
            
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    # Evita divisioni per zero e fai la media
    count_map[count_map == 0] = 1
    return output_ecg / count_map