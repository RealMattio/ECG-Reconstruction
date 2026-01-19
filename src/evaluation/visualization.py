import matplotlib.pyplot as plt
import os
import json
import random
import torch
import numpy as np

def save_training_plots(history, save_dir):
    """
    Genera e salva i grafici delle perdite. 
    Adattata per gestire sia Approccio 1 (multi-loss) che Approccio 2 (single-loss).
    """
    os.makedirs(save_dir, exist_ok=True)
    if not history['train']:
        return

    epochs = range(1, len(history['train']) + 1)
    
    # Identifica dinamicamente le chiavi presenti nel primo record della history
    available_keys = history['train'][0].keys()
    
    for key in available_keys:
        plt.figure(figsize=(10, 6))
        
        train_loss = [h[key] for h in history['train']]
        val_loss = [h[key] for h in history['val']]
        
        plt.plot(epochs, train_loss, 'b-', label=f'Train {key.upper()}')
        plt.plot(epochs, val_loss, 'r-', label=f'Val {key.upper()}')
        
        plt.title(f'Andamento Loss: {key.upper()}')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f"loss_plot_{key}.png"))
        plt.close()
    
    print(f"✓ Grafici salvati per le chiavi: {list(available_keys)}")

def save_history_to_json(history, save_dir):
    """Salva la history in formato JSON."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "training_history.json")
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✓ History salvata in: {file_path}")

def plot_random_sample(windowed_train_data, save_path, window_size=8):
    """
    Seleziona un campione casuale dal training set e salva un grafico dei segnali.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Selezione campione casuale
    sample = random.choice(windowed_train_data)
    ppg, eda, acc = sample['input']
    ecg = sample['target']
    sub_id = sample['subject']

    # Creazione figura con 4 subplot
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False)
    fig.suptitle(f"Campione Casuale - Soggetto: {sub_id} (Finestra: {window_size}s)", fontsize=16)

    signals = [
        ("PPG", ppg, "blue"),
        ("EDA", eda, "green"),
        ("ACC", acc, "orange"), # ACC ha 3 assi
        ("ECG", ecg, "red")
    ]

    for i, (name, data, color) in enumerate(signals):
        ax = axes[i]
        
        # Calcolo frequenza e punti
        num_points = len(data)
        calc_fs = num_points / window_size
        
        if name == "ACC":
            # Plot dei 3 assi dell'accelerometro
            ax.plot(data[:, 0], label="X", alpha=0.7)
            ax.plot(data[:, 1], label="Y", alpha=0.7)
            ax.plot(data[:, 2], label="Z", alpha=0.7)
            ax.legend(loc='upper right')
        else:
            ax.plot(data, color=color)
        
        ax.set_title(f"{name} | Punti: {num_points} | Fs Calcolata: {calc_fs:.2f} Hz")
        ax.set_ylabel("Ampiezza (Norm)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Grafico del campione casuale salvato in: {save_path}")

def plot_inference_comparison(model, val_data, device, save_path):
    """
    Esegue inferenza su un campione casuale e confronta l'output con la ground truth.
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Selezione casuale dal validation set
    sample = random.choice(val_data)
    ppg_raw, eda_raw, acc_raw = sample['input']
    ecg_gt = sample['target']
    
    # 2. Preparazione tensori per il modello
    ppg = torch.tensor(ppg_raw).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 512)
    eda = torch.tensor(eda_raw).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 32)
    acc = torch.tensor(acc_raw).float().transpose(0, 1).unsqueeze(0).to(device) # (1, 3, 256)
    
    # 3. Inferenza
    with torch.no_grad():
        ecg_gen = model(ppg, acc, eda).cpu().squeeze().numpy()
    
    # 4. Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 18))
    fig.suptitle(f"Inference Check - Subject: {sample['subject']}", fontsize=16)
    
    # Subplot 1: PPG
    axes[0].plot(ppg_raw, color='blue')
    axes[0].set_title("Input: PPG")
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: EDA
    axes[1].plot(eda_raw, color='green')
    axes[1].set_title("Input: EDA")
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: ACC
    axes[2].plot(acc_raw[:, 0], label='X')
    axes[2].plot(acc_raw[:, 1], label='Y')
    axes[2].plot(acc_raw[:, 2], label='Z')
    axes[2].set_title("Input: ACC (3-axis)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Subplot 4: ECG Comparison (Generated vs Ground Truth)
    axes[3].plot(ecg_gt, label='Ground Truth (Real)', color='black', alpha=0.5, linestyle='--')
    axes[3].plot(ecg_gen, label='Generated (Model)', color='red', alpha=0.8)
    axes[3].set_title("Output Comparison: Generated vs Real ECG")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Grafico di inferenza salvato in: {save_path}")