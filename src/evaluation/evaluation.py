import torch
import numpy as np
import os
import json
import scipy.stats
from scipy.signal import find_peaks
import pandas as pd

def calculate_snr(y_true, y_pred):
    """
    Calcola il Signal-to-Noise Ratio (dB).
    SNR = 10 * log10(Power_Signal / Power_Noise)
    """
    # Evitiamo divisioni per zero
    noise = y_true - y_pred
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10: return 100.0 # Ricostruzione perfetta
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_cosine_similarity(y_true, y_pred):
    """
    Calcola la similarità coseno media.
    Valori vicini a 1 indicano forma identica.
    """
    if y_true.ndim > 1:
        # Calcolo per riga (batch)
        sims = []
        for i in range(len(y_true)):
            dot_prod = np.dot(y_true[i], y_pred[i])
            norm_t = np.linalg.norm(y_true[i]) + 1e-8
            norm_p = np.linalg.norm(y_pred[i]) + 1e-8
            sims.append(dot_prod / (norm_t * norm_p))
        return float(np.mean(sims))
    else:
        dot_prod = np.dot(y_true, y_pred)
        norm_t = np.linalg.norm(y_true) + 1e-8
        norm_p = np.linalg.norm(y_pred) + 1e-8
        return float(dot_prod / (norm_t * norm_p))

def calculate_bpm(signal, fs=125):
    """Stima i BPM trovando i picchi."""
    thresh = np.max(signal) * 0.5 
    distance = int(fs * 0.4) # Minimo 240ms tra battiti (max 250 BPM)
    
    peaks, _ = find_peaks(signal, height=thresh, distance=distance)
    
    if len(peaks) < 2:
        return 0.0 # Impossibile calcolare BPM
        
    diffs = np.diff(peaks)
    mean_diff = np.mean(diffs) # in campioni
    bpm = 60 * (fs / mean_diff)
    return bpm

def compute_batch_metrics(y_true_np, y_pred_np, fs):
    """Calcola tutte le metriche per un batch numpy."""
    
    # 1. Metriche Matematiche
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    mae = calculate_mae(y_true_np, y_pred_np)
    snr = calculate_snr(y_true_np, y_pred_np)
    cosine = calculate_cosine_similarity(y_true_np, y_pred_np)
    
    pearsons = []
    bpms_true = []
    bpms_pred = []
    
    for i in range(len(y_true_np)):
        t_flat = y_true_np[i].flatten()
        p_flat = y_pred_np[i].flatten()
        
        # FIX SICUREZZA PEARSON: Evita NaN se il modello genera una linea piatta
        if np.std(t_flat) < 1e-6 or np.std(p_flat) < 1e-6:
            pearsons.append(0.0)
        else:
            p_val, _ = scipy.stats.pearsonr(t_flat, p_flat)
            pearsons.append(p_val)
        
        # BPM
        b_true = calculate_bpm(t_flat, fs)
        b_pred = calculate_bpm(p_flat, fs)
        if b_true > 0 and b_pred > 0:
            bpms_true.append(b_true)
            bpms_pred.append(b_pred)

    avg_pearson = np.mean(pearsons)
    
    # BPM Error (Absolute diff)
    if len(bpms_true) > 0:
        bpm_error = np.mean(np.abs(np.array(bpms_true) - np.array(bpms_pred)))
    else:
        bpm_error = None # Non calcolabile (es. segnale piatto o finestra senza picchi)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'snr': float(snr),
        'pearson': float(avg_pearson),
        'cosine_sim': float(cosine),
        'bpm_mae': float(bpm_error) if bpm_error is not None else None
    }

def evaluate_test_set_performance(model, test_loader, device, save_dir, configs):
    """
    Valuta l'intero Test Set e salva le metriche in performances.json
    """
    model.eval()
    all_metrics = {
        'rmse': [], 'mae': [], 'snr': [], 
        'pearson': [], 'cosine_sim': [], 'bpm_mae': []
    }
    
    fs = configs['target_fs']
    gen_len = configs.get('target_len', 125) 
    
    print("\n[EVALUATION] Calcolo metriche estese sul Test Set...")
    
    with torch.no_grad():
        for batch in test_loader:
            # Estrazione sicura dei primi due elementi ignorando gli altri (theta, omega, ecc.)
            if isinstance(batch, list) or isinstance(batch, tuple):
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
            else:
                raise ValueError("Formato batch non riconosciuto.")

            # Inferenza
            outputs = model(inputs)
            
            # FIX SQUEEZE: Forza la shape a [Batch, Time] per prevenire crash sull'ultimo batch (se size=1)
            y_pred = outputs.detach().cpu().reshape(outputs.shape[0], -1).numpy()
            y_true = targets.detach().cpu().reshape(targets.shape[0], -1).numpy()
            
            # Calcolo metriche batch
            batch_res = compute_batch_metrics(y_true, y_pred, fs)
            
            # Aggregazione
            for k, v in batch_res.items():
                if v is not None and not np.isnan(v):
                    all_metrics[k].append(v)
    
    # Calcolo Medie Finali (ignorando i NaN eventualmente rimasti)
    final_results = {k: float(np.nanmean(v)) if len(v) > 0 else 0.0 for k, v in all_metrics.items()}
    
    # Aggiunta Deviazione Standard
    final_std = {f"{k}_std": float(np.nanstd(v)) if len(v) > 0 else 0.0 for k, v in all_metrics.items()}
    
    # Struttura JSON finale
    output_json = {
        "model_type": configs.get("model_type", "Unknown"),
        "generation_params": {
            "fs": fs,
            "num_predicted_points": int(gen_len), 
            "input_window_sec": configs.get("x_sec", 7),
            "generation_sec": configs.get("gen_sec", 1)
        },
        "performance_metrics": final_results,
        "performance_std": final_std
    }
    
    # Salvataggio
    json_path = os.path.join(save_dir, "performances.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=4)
        
    print(f"[SUCCESS] Metriche salvate in: {json_path}")
    print(f"Metrics: RMSE={final_results['rmse']:.4f}, Pearson={final_results['pearson']:.4f}, SNR={final_results['snr']:.2f} dB")
    
    return final_results

def save_training_history(history, save_dir):
    """Salva la history dell'addestramento in CSV."""
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    path = os.path.join(save_dir, 'training_history.csv')
    df.to_csv(path)
    print(f"History salvata in: {path}")