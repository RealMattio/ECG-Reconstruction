import matplotlib.pyplot as plt
import os
import json
import random
import torch
import numpy as np


def _minmax_norm(signal: np.ndarray) -> np.ndarray:
    """Normalizza un segnale numpy nell'intervallo [0, 1] (per finestra)."""
    mn, mx = signal.min(), signal.max()
    return (signal - mn) / (mx - mn + 1e-8)

def save_training_plots(history, save_dir):
    """Genera e salva i grafici delle perdite rilevando le chiavi dinamicamente."""
    os.makedirs(save_dir, exist_ok=True)
    if not history.get('train'): return

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
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sample = random.choice(val_data)
    ppg_raw, eda_raw, acc_raw, prev_ecg_raw = sample['input']
    ecg_gt = sample['target']
    
    ppg = torch.tensor(ppg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    eda = torch.tensor(eda_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    acc = torch.tensor(acc_raw).float().transpose(0, 1).unsqueeze(0).to(device)
    prev_ecg = torch.tensor(prev_ecg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ecg_gen = model(ppg, acc, eda, prev_ecg).cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 20))
    fig.suptitle(f"Inference Check - Subject: {sample['subject']}", fontsize=16)
    
    axes[0].plot(ppg_raw, color='blue'); axes[0].set_title("Input: PPG")
    axes[1].plot(acc_raw); axes[1].set_title("Input: ACC (3-axis)")
    axes[2].plot(eda_raw, color='green'); axes[2].set_title("Input: EDA")
    axes[3].plot(prev_ecg_raw, color='purple', alpha=0.6); axes[3].set_title("Condition: Prev ECG")
    
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
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sample = random.choice(val_data)
    ppg_raw = sample['input'][0] 
    ecg_gt = sample['target']
    
    duration = 7.0 
    t_ppg = np.linspace(0, duration, len(ppg_raw))
    t_ecg = np.linspace(0, duration, len(ecg_gt))
    
    ppg_tensor = torch.tensor(ppg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ecg_gen = model(ppg_tensor).cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Epoch {epoch} | Subject: {sample['subject']} | Temporal Alignment (7s)", fontsize=14)
    
    axes[0].plot(t_ppg, ppg_raw, color='blue', label='Input PPG (64Hz)')
    axes[0].set_title("Input: Photoplethysmogram (PPG)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t_ecg, ecg_gt, color='black', alpha=0.6, label='Real ECG (256Hz)')
    axes[1].set_title("Target: Real ECG (Ground Truth)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc='upper right')
    
    axes[2].plot(t_ecg, ecg_gt, color='black', alpha=0.3, label='Real', linestyle='--')
    axes[2].plot(t_ecg, ecg_gen, color='red', alpha=0.8, label='Generated')
    axes[2].set_title("Overlay Comparison (Signals Synced in Time)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc='upper right')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration) 
    
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

        ppg_np = ppg_win.squeeze().cpu().numpy()
        ecg_real_np = ecg_real.squeeze().cpu().numpy()
        ecg_gen_np = ecg_gen.squeeze().cpu().numpy()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        if ppg_np.ndim == 2:
            time_ppg = np.arange(ppg_np.shape[1])
            for i in range(ppg_np.shape[0]):
                axes[0].plot(time_ppg, ppg_np[i], alpha=0.6)
            axes[0].set_title(f"Input PPG WST Features ({ppg_np.shape[0]} channels)")
        else:
            axes[0].plot(np.arange(len(ppg_np)), ppg_np, color='blue')
            axes[0].set_title("Input PPG (Time Domain)")

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
                seg_input = preprocessor.extract_wst_features(np.expand_dims(seg, axis=0))
                seg_t = torch.tensor(seg_input).float().to(device)
            else:
                seg_t = torch.tensor(seg).float().unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(seg_t).cpu().numpy().flatten()
            output_ecg[i : i + window_len] += pred
            count_map[i : i + window_len] += 1
            
    count_map[count_map == 0] = 1
    return output_ecg / count_map

def generate_autoregressive_recon(model, ppg_long, ecg_seed, device, configs):
    """
    Genera ECG in modo autoregressivo partendo da un seed.
    Ogni finestra viene normalizzata [0,1] prima dell'inferenza, coerentemente
    con la normalizzazione applicata durante il training in MimicSmartDataset.__getitem__.
    """
    model.eval()
    fs = configs['target_fs']
    x_sec = configs.get('x_sec', 7)
    gen_sec = configs.get('gen_sec', 1)

    win_samples  = int(fs * x_sec)
    gen_samples  = int(fs * gen_sec)
    seed_samples = win_samples - gen_samples

    # Il seed ECG viene normalizzato [0,1] una volta sola prima di partire
    generated_ecg = _minmax_norm(ecg_seed).tolist()

    with torch.no_grad():
        cursor = seed_samples

        while cursor + gen_samples <= len(ppg_long):
            start_win = cursor - seed_samples
            end_win   = cursor + gen_samples

            # PPG: correzione polarità + normalizzazione [0,1] (coerente con __getitem__)
            curr_ppg = ppg_long[start_win : end_win].copy()
            centered = curr_ppg - curr_ppg.mean()
            if centered.max() < -centered.min():
                curr_ppg = -curr_ppg
            curr_ppg_norm = _minmax_norm(curr_ppg)

            # ECG passato: già in [0,1] (seed normalizzato o output del modello)
            # NON ri-normalizzare: stretching su valori piatti distorce le ampiezze relative
            ecg_real_part = np.array(generated_ecg[start_win : cursor])

            last_val      = ecg_real_part[-1]
            padding       = np.full((gen_samples,), last_val)
            curr_ecg_past = np.concatenate([ecg_real_part, padding])

            ppg_t    = torch.tensor(curr_ppg_norm).float()
            ppg_diff = torch.zeros_like(ppg_t)
            ppg_diff[1:] = ppg_t[1:] - ppg_t[:-1]
            ecg_p_t  = torch.tensor(curr_ecg_past).float()

            input_stack = torch.stack([ppg_t, ppg_diff, ecg_p_t], dim=0).unsqueeze(0).to(device)
            pred_sec    = model(input_stack).cpu().squeeze().numpy()

            if pred_sec.ndim == 0:
                generated_ecg.append(pred_sec.item())
            else:
                generated_ecg.extend(pred_sec.tolist())

            cursor += gen_samples

    return np.array(generated_ecg)

def save_extended_reports(model, subject_ids, raw_data, preprocessor, device, configs, set_name):
    save_dir = configs['model_save_path']
    fs = configs['target_fs']
    
    x_sec = configs.get('x_sec', 7)
    gen_sec = configs.get('gen_sec', 1)
    
    s_id = random.choice(subject_ids)
    subj = raw_data['subjects_data'][s_id]
    
    ppg_f = preprocessor.apply_bandpass_filter(subj['PPG'], 0.5, 8.0)
    ecg_f = preprocessor.apply_bandpass_filter(subj['ECG'], 0.5, 30.0)
    
    if configs.get('normalize_01', False):
        ppg_n, ecg_n = preprocessor.normalize_min_max(ppg_f), preprocessor.normalize_min_max(ecg_f)
    else:
        ppg_n, ecg_n = preprocessor.normalize_signal(ppg_f), preprocessor.normalize_signal(ecg_f)

    total_sec = 15
    total_samples = min(fs * total_sec, len(ppg_n))
    
    start_idx = random.randint(0, len(ppg_n) - total_samples)
    
    long_ppg_gt = ppg_n[start_idx : start_idx + total_samples]
    long_ecg_gt = ecg_n[start_idx : start_idx + total_samples]
    
    seed_samples = int(fs * x_sec)
    ecg_seed = long_ecg_gt[:seed_samples]
    
    long_ecg_gen = generate_autoregressive_recon(model, long_ppg_gt, ecg_seed, device, configs, preprocessor)
    
    plot_len = min(len(long_ecg_gt), len(long_ecg_gen))
    time_axis = np.arange(plot_len) / fs
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, long_ppg_gt[:plot_len], color='blue', label='Input PPG (Full Context)')
    plt.axvline(x=x_sec, color='green', linestyle='--', alpha=0.5, label='Generation Start')
    plt.title(f"[{set_name.upper()}] PPG Input - Subject {s_id}")
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, long_ecg_gt[:plot_len], color='black', alpha=0.3, label='Real ECG (Ground Truth)')
    
    plt.plot(time_axis[:seed_samples], long_ecg_gen[:seed_samples], color='green', alpha=0.6, label='Seed (Real Data)')
    plt.plot(time_axis[seed_samples:], long_ecg_gen[seed_samples:], color='red', label='Autoregressive Prediction')
    
    plt.axvline(x=x_sec, color='green', linestyle='--')
    plt.title(f"[{set_name.upper()}] AR Reconstruction (Window: {x_sec}s, Gen: {gen_sec}s)")
    plt.xlabel("Time (s)"); plt.legend(); plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{set_name}_autoregressive_recon.png"))
    plt.close()


def plot_training_history_metrics(history, save_dir):
    """Grafico a tre pannelli per le metriche di training. Sicuro anche senza validation set."""
    if 'train_loss' not in history or len(history['train_loss']) == 0:
        return
        
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Pannello 1: Total Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(epochs, history['val_loss'], label='Val', linestyle='--')
    axes[0].set_title('Total Loss'); axes[0].legend()

    # Rilevamento dinamico della chiave per l'ampiezza
    amp_train_key = 'train_amp_loss' if 'train_amp_loss' in history else 'train_rmse'
    amp_val_key = 'val_amp_loss' if 'val_amp_loss' in history else 'val_rmse'

    # Pannello 2: Amplitude Loss (MAE / Huber / RMSE)
    if amp_train_key in history:
        axes[1].plot(epochs, history[amp_train_key], label='Train', color='green')
    if amp_val_key in history and len(history[amp_val_key]) > 0:
        axes[1].plot(epochs, history[amp_val_key], label='Val', color='lightgreen', linestyle='--')
    axes[1].set_title('Amplitude Loss (MAE/HUBER/RMSE)'); axes[1].legend()

    # Pannello 3: Pearson Loss
    if 'train_pearson' in history:
        axes[2].plot(epochs, history['train_pearson'], label='Train', color='red')
    if 'val_pearson' in history and len(history['val_pearson']) > 0:
        axes[2].plot(epochs, history['val_pearson'], label='Val', color='salmon', linestyle='--')
    axes[2].set_title('Pearson Loss (1 - Corr)'); axes[2].legend()

    for ax in axes: ax.set_xlabel('Epochs'); ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_metrics_curves.png')
    plt.savefig(path); plt.close()
    print(f"Grafico metriche salvato in: {path}")


def plot_validation_snapshot(model, val_loader, device, save_dir, epoch, step=None, prefix='val'):
    """Genera un plot di validazione estraendo un sample casuale dal primo batch."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        X, Y, theta, omega, file_info = batch
        
        X = X.to(device)
        Y = Y.to(device)
        preds = model(X)
        
        X_cpu = X.cpu().numpy()
        Y_cpu = Y.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        # FIX: Scegliamo un indice casuale nel batch invece di prendere sempre lo 0!
        idx = random.randint(0, X_cpu.shape[0] - 1)
        pat_string = file_info[idx] if isinstance(file_info, (list, tuple)) else str(file_info)

        ppg_signal = X_cpu[idx, 0, :]
        ecg_past = X_cpu[idx, 2, :]
        target_ecg = Y_cpu[idx, 0, :]
        pred_ecg = preds_cpu[idx, 0, :]
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        
        title = f"{prefix.upper()} Snapshot - Epoch {epoch}"
        if step: title += f" Step {step}"
        fig.suptitle(f"{title}\n{pat_string}", fontsize=14, fontweight='bold', color='darkblue')
        
        axs[0].plot(ppg_signal, color='green')
        axs[0].set_title('Input: PPG Signal (Clean & Fixed)')
        axs[0].grid(True, alpha=0.3)
        
        axs[1].plot(ecg_past, color='black')
        axs[1].set_title('Input: ECG Context')
        axs[1].grid(True, alpha=0.3)
        
        axs[2].plot(target_ecg, label='True ECG Target', color='black', linewidth=2)
        axs[2].plot(pred_ecg, label='Predicted ECG', color='red', linestyle='dashed', linewidth=2)
        axs[2].set_title('Output: Predicted vs True')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{prefix}_epoch_{epoch}.png" if step is None else f"{prefix}_epoch_{epoch}_step_{step}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

def plot_autoregressive_epoch(model, val_dataset, preprocessor, device, configs, epoch, save_dir):
    try:
        model.eval()
        fs = configs['target_fs']
        x_sec = configs.get('x_sec', 7)
        gen_sec = configs.get('gen_sec', 1)
        
        if not hasattr(val_dataset, 'file_cache') or len(val_dataset.file_cache) == 0:
            return

        cache_item = random.choice(val_dataset.file_cache)
        full_ppg = cache_item['ppg'].float().numpy()
        full_ecg = cache_item['ecg'].float().numpy()
        
        num_generations = 5 
        
        seed_samples = int((x_sec - gen_sec) * fs) 
        gen_samples = int(gen_sec * fs)            
        total_needed = seed_samples + (num_generations * gen_samples)
        
        if len(full_ppg) < total_needed:
            return

        start_idx = random.randint(0, len(full_ppg) - total_needed)
        long_ppg_gt = full_ppg[start_idx : start_idx + total_needed]
        long_ecg_gt = full_ecg[start_idx : start_idx + total_needed]

        ecg_seed = long_ecg_gt[:seed_samples]

        long_ecg_gen = generate_autoregressive_recon(
            model, long_ppg_gt, ecg_seed, device, configs
        )

        plot_len  = min(len(long_ecg_gt), len(long_ecg_gen))
        time_axis = np.arange(plot_len) / fs

        long_ppg_gt_norm = _minmax_norm(long_ppg_gt[:plot_len])
        long_ecg_gt_norm = _minmax_norm(long_ecg_gt[:plot_len])

        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, long_ppg_gt_norm, color='blue', label='Input PPG [0-1]')
        plt.axvline(x=(seed_samples/fs), color='green', linestyle='--', label='Start Generation')
        plt.title(f"Epoch {epoch} - Autoregressive Input (PPG normalizzato)")
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        plt.plot(time_axis, long_ecg_gt_norm, color='black', alpha=0.3, label='Real ECG [0-1]')
        plt.plot(time_axis[:seed_samples], long_ecg_gen[:seed_samples], color='green', label='Initial Seed (Real)')
        plt.plot(time_axis[seed_samples:plot_len], long_ecg_gen[seed_samples:plot_len], color='red', label='Autoregressive Generated')
        plt.axvline(x=(seed_samples/fs), color='green', linestyle='--')
        plt.title("Reconstruction Comparison [scala 0-1]")
        plt.legend(loc='upper right')
        
        os.makedirs(save_dir, exist_ok=True)
        # FIX: Gestione sicura per stringhe (es. "TEST_1") o interi
        epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch_str}_autoregressive.png"))
        plt.close()
        
    except Exception as e:
        print(f"[PLOT ERROR] Autoregressive plot failed: {e}")