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

def save_inference_plot(model, val_loader, device, save_path, title="Validation Inference"):
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        all_batches = list(val_loader)
        ppg_batch, ecg_batch = all_batches[random.randint(0, len(all_batches) - 1)]
        idx = random.randint(0, ppg_batch.shape[0] - 1)
        
        ppg_win = ppg_batch[idx].unsqueeze(0).to(device)
        ecg_real = ecg_batch[idx].unsqueeze(0).to(device)

        with torch.no_grad():
            ecg_gen = model(ppg_win)

        # Conversione per plotting
        ppg_np = ppg_win.squeeze().cpu().numpy()
        ecg_real_np = ecg_real.squeeze().cpu().numpy()
        ecg_gen_np = ecg_gen.squeeze().cpu().numpy()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # --- Subplot 1: Input PPG ---
        # Gestione dinamica: WST ha 2 dimensioni (Canali, Tempo), Raw ne ha 1
        if ppg_np.ndim == 2:
            # Se è WST (8x30), mostriamo i canali sovrapposti
            time_ppg = np.arange(ppg_np.shape[1])
            for i in range(ppg_np.shape[0]):
                axes[0].plot(time_ppg, ppg_np[i], alpha=0.6)
            axes[0].set_title(f"Input PPG WST Features ({ppg_np.shape[0]} channels)")
        else:
            axes[0].plot(np.arange(len(ppg_np)), ppg_np, color='blue')
            axes[0].set_title("Input PPG (Time Domain)")

        # --- Subplot 2: ECG Comparison ---
        time_ecg = np.arange(len(ecg_real_np))
        axes[1].plot(time_ecg, ecg_real_np, color='black', linestyle='--', alpha=0.5, label='Real')
        axes[1].plot(time_ecg, ecg_gen_np, color='red', label='Generated')
        axes[1].set_title(title)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Grafico salvato in: {save_path}")

    except Exception as e:
        print(f"Errore durante la generazione del grafico: {e}")

def generate_long_window_smooth(model, ppg_signal, preprocessor, device, configs):
    """
    Genera un ECG lungo gestendo dinamicamente overlap e WST.
    Risolve il problema dei campioni nulli alla fine coprendo l'intero segnale.
    """
    window_len = configs['beat_len']
    overlap_pct = configs.get('overlap_pct', 0.5)
    step = int(window_len * (1 - overlap_pct))
    
    total_samples = len(ppg_signal)
    output_ecg = np.zeros(total_samples)
    count_map = np.zeros(total_samples)
    
    model.eval()
    with torch.no_grad():
        start_indices = list(range(0, total_samples - window_len, step))
        if total_samples > window_len and start_indices[-1] + window_len < total_samples:
            start_indices.append(total_samples - window_len)
            
        for i in start_indices:
            seg = ppg_signal[i : i + window_len]
            
            if configs.get('apply_wst', False):
                # Trasformazione WST tramite preprocessor
                seg_input = preprocessor.extract_wst_features(np.expand_dims(seg, axis=0))
                seg_t = torch.tensor(seg_input).float().to(device)
            else:
                seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(seg_t).cpu().numpy().flatten()
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    count_map[count_map == 0] = 1
    return output_ecg / count_map

def save_extended_reports(model, subject_ids, raw_data, preprocessor, device, configs, set_name):
    """Genera grafici a finestra lunga per un set specifico (Train/Val/Test)."""
    save_dir = configs['model_save_path']
    fs = configs['target_fs']
    
    # 1. Selezione casuale di un soggetto
    s_id = random.choice(subject_ids)
    subj = raw_data['subjects_data'][s_id]
    
    # --- LOGICA DI PRE-PROCESSING ALLINEATA ---
    # Filtraggio differenziato (Soluzione 2 implementata precedentemente)
    ppg_f = preprocessor.apply_bandpass_filter(subj['PPG'], lowcut=0.5, highcut=8.0)
    
    if configs.get('apply_ecg_filter', True):
        ecg_f = preprocessor.apply_bandpass_filter(subj['ECG'], lowcut=0.5, highcut=30.0)
    else:
        ecg_f = preprocessor.apply_bandpass_filter(subj['ECG'], lowcut=0.5, highcut=8.0)

    # Normalizzazione modulare (Soluzione 1 implementata precedentemente)
    if configs.get('normalize_01', False):
        ppg_norm = preprocessor.normalize_min_max(ppg_f)
        ecg_norm = preprocessor.normalize_min_max(ecg_f)
    else:
        ppg_norm = preprocessor.normalize_signal(ppg_f)
        ecg_norm = preprocessor.normalize_signal(ecg_f)
    # ------------------------------------------
    
    # Finestra di circa 7.2s (900 campioni a 125Hz)
    total_len = min(900, len(ppg_norm))
    start = random.randint(0, len(ppg_norm) - total_len)
    long_ppg = ppg_norm[start : start + total_len]
    long_ecg_real = ecg_norm[start : start + total_len]
    
    # Generazione ECG sintetico
    long_ecg_gen = generate_long_window_smooth(model, long_ppg, preprocessor, device, configs)
    
    time_axis = np.arange(len(long_ppg)) / fs
    plt.figure(figsize=(15, 8))
    
    # Plotting (Invariato)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, long_ppg, color='blue', label='Input PPG')
    plt.title(f"[{set_name.upper()}] PPG Continuo (0.5-8Hz) - Soggetto {s_id}")
    plt.ylabel("Ampiezza " + ("[0,1]" if configs.get('normalize_01') else "Z-Score"))
    plt.grid(alpha=0.3); plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, long_ecg_real, color='black', alpha=0.3, label='ECG Reale (0.5-30Hz)')
    plt.plot(time_axis, long_ecg_gen, color='red', label='ECG Ricostruito (Smooth Stitching)')
    plt.title(f"[{set_name.upper()}] Ricostruzione ECG - Visualizzazione Temporale")
    plt.xlabel("Tempo (s)"); plt.ylabel("Ampiezza")
    plt.legend(); plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{set_name}_long_reconstruction.png"))
    plt.close()

def plot_training_history_metrics(history, save_dir):
    """Grafico a tre pannelli per le metriche di training."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Pannello 1: Total Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val', linestyle='--')
    axes[0].set_title('Total Loss'); axes[0].legend()

    # Pannello 2: Weighted RMSE
    axes[1].plot(epochs, history['train_rmse'], label='Train', color='green')
    axes[1].plot(epochs, history['val_rmse'], label='Val', color='lightgreen', linestyle='--')
    axes[1].set_title('Weighted RMSE'); axes[1].legend()

    # Pannello 3: Pearson Loss
    axes[2].plot(epochs, history['train_pearson'], label='Train', color='red')
    axes[2].plot(epochs, history['val_pearson'], label='Val', color='salmon', linestyle='--')
    axes[2].set_title('Pearson Loss (1 - Corr)'); axes[2].legend()

    for ax in axes: ax.set_xlabel('Epochs'); ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_metrics_curves.png')
    plt.savefig(path); plt.close()
    print(f"Grafico metriche salvato in: {path}")