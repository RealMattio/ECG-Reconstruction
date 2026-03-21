import os
import torch
import numpy as np
import warnings
import json
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, pearsonr
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# 1. FUNZIONI DI QUALITÀ (SQI) E FISICA
# ==========================================

def check_polarity_by_slope(ppg_signal):
    diff = np.diff(ppg_signal)
    max_slope_up = np.max(diff)
    max_slope_down = np.min(diff)
    if abs(max_slope_down) > abs(max_slope_up):
        return True # Invertito
    return False # Dritto

def get_spectral_sqi(ppg_signal, fs=125):
    freqs, psd = welch(ppg_signal, fs, nperseg=len(ppg_signal)//2)
    mask_hr = (freqs >= 0.5) & (freqs <= 3.0)
    if not np.any(mask_hr): return 0.0
    idx_hr = np.argmax(psd[mask_hr])
    peak_freq = freqs[mask_hr][idx_hr]
    mask_peak = (freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2)
    power_peak = np.sum(psd[mask_peak])
    mask_total = (freqs >= 0.5) & (freqs <= 10.0)
    power_total = np.sum(psd[mask_total])
    if power_total == 0: return 0.0
    return power_peak / power_total

def get_ppg_quality_score(ppg_signal, fs=125, spectral_thr=0.5):
    if np.std(ppg_signal) < 1e-6: return 0.0, False
    is_inverted = check_polarity_by_slope(ppg_signal)
    sig_fixed = ppg_signal * -1 if is_inverted else ppg_signal
    
    spec_score = get_spectral_sqi(sig_fixed, fs)
    if spec_score < spectral_thr: return 0.0, False
    
    sig_norm = (sig_fixed - np.mean(sig_fixed)) / (np.std(sig_fixed) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs * 0.35), prominence=0.4)
    if len(peaks) < 4: return 0.0, False
    
    beats = []
    for i in range(1, len(peaks)):
        beat = sig_norm[peaks[i-1]:peaks[i]]
        if fs * 0.3 < len(beat) < fs * 2.0:
            beat_interp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(beat)), beat)
            beats.append(beat_interp)
    if len(beats) < 3: return 0.0, False
    
    template = np.mean(beats, axis=0)
    score = np.mean([pearsonr(template, b)[0] for b in beats])
    return score, is_inverted

def get_ecg_quality_score(ecg_signal, fs=125, kurtosis_thr=5.0):
    if np.std(ecg_signal) < 1e-4: return 0.0
    k_val = kurtosis(ecg_signal, fisher=False)
    if k_val < kurtosis_thr: return 0.0
    
    sig_norm = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs*0.4), height=1.5)
    if len(peaks) < 3: return 0.0
    
    beats = []
    pre, post = int(fs*0.2), int(fs*0.4)
    for p in peaks:
        if p - pre >= 0 and p + post < len(sig_norm):
            beats.append(sig_norm[p - pre : p + post])
    if len(beats) < 3: return 0.0
    
    template = np.mean(beats, axis=0)
    return np.mean([pearsonr(template, b)[0] for b in beats])

def extract_physics_signals(ecg_np, fs=125):
    """
    Data una finestra ECG, ne calcola la derivata theta e omega.
    """
    sig_norm = (ecg_np - np.mean(ecg_np)) / (np.std(ecg_np) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs*0.4), height=1.0)
    
    N = len(ecg_np)
    omega = np.full(N, 2.0 * np.pi, dtype=np.float32) 
    theta = np.zeros(N, dtype=np.float32)
    
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            T_sec = (p2 - p1) / fs
            w = 2.0 * np.pi / T_sec
            omega[p1:p2] = w
            theta[p1:p2] = np.linspace(0, 2*np.pi, p2 - p1, endpoint=False)
        
        p_first = peaks[0]
        if p_first > 0:
            omega[:p_first] = omega[p_first]
            t_arr = np.arange(-p_first, 0) / fs
            theta[:p_first] = (t_arr * omega[p_first]) % (2 * np.pi)
            
        p_last = peaks[-1]
        if p_last < N:
            omega[p_last:] = omega[p_last - 1]
            t_arr = np.arange(0, N - p_last) / fs
            theta[p_last:] = (t_arr * omega[p_last - 1]) % (2 * np.pi)
    else:
        t_arr = np.arange(N) / fs
        theta = (t_arr * omega) % (2 * np.pi)
        
    return theta, omega

# ==========================================
# 2. LOGICA DI MERGING E CREAZIONE DATASET
# ==========================================

def merge_intervals(intervals):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for next_start, next_end in intervals[1:]:
        curr_start, curr_end = merged[-1]
        if next_start <= curr_end:
            merged[-1] = (curr_start, max(curr_end, next_end))
        else:
            merged.append((next_start, next_end))
    return merged

def process_and_save_patient_file(data_np, file_name, pat_id, save_dir, manifest, 
                                  win_sec=4, step_sec=1, min_size_sec=7, fs=125, ppg_thr=0.9, ecg_thr=0.9):
    total_samples = len(data_np['ppg'])
    win_samples = int(win_sec * fs)
    step_samples = int(step_sec * fs)
    min_samples = int(min_size_sec * fs)
    
    ppg_raw_intervals = []
    ecg_raw_intervals = []
    # Salviamo l'informazione se quel tratto di PPG andava girato
    inversions_map = np.zeros(total_samples, dtype=bool) 
    
    # 1. Scansione Sliding Window
    for start in range(0, total_samples - win_samples, step_samples):
        end = start + win_samples
        
        ppg_score, is_inv = get_ppg_quality_score(data_np['ppg'][start:end], fs)
        if ppg_score > ppg_thr:
            ppg_raw_intervals.append((start, end))
            if is_inv:
                inversions_map[start:end] = True
                
        if get_ecg_quality_score(data_np['ecg'][start:end], fs) > ecg_thr:
            ecg_raw_intervals.append((start, end))
            
    # 2. Merging
    ppg_merged = merge_intervals(ppg_raw_intervals)
    ecg_merged = merge_intervals(ecg_raw_intervals)
    
    mask_ppg = np.zeros(total_samples, dtype=bool)
    for s, e in ppg_merged: mask_ppg[s:e] = True
    mask_ecg = np.zeros(total_samples, dtype=bool)
    for s, e in ecg_merged: mask_ecg[s:e] = True
    
    combined_mask = mask_ppg & mask_ecg
    
    if not np.any(combined_mask):
        return 0 # Nessun intervallo valido
        
    diff = np.diff(combined_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if combined_mask[0]: starts = np.insert(starts, 0, 0)
    if combined_mask[-1]: ends = np.append(ends, total_samples)
    
    raw_combined = list(zip(starts, ends))
    valid_intervals = [(s, e) for s, e in raw_combined if (e - s) >= min_samples]
    
    # 3. Estrazione e Salvataggio dei Segmenti
    segments_saved = 0
    for idx, (start, end) in enumerate(valid_intervals):
        seg_ppg = data_np['ppg'][start:end].copy()
        seg_ecg = data_np['ecg'][start:end].copy()
        
        # Raddrizzamento PPG se necessario (basato sulla maggioranza della maschera in quell'intervallo)
        if np.mean(inversions_map[start:end]) > 0.5:
            seg_ppg = seg_ppg * -1.0
            
        # Calcolo Fisica PINN su questo intervallo
        seg_theta, seg_omega = extract_physics_signals(seg_ecg, fs)
        
        # Salvataggio come Tensori PyTorch
        out_data = {
            'ppg': torch.tensor(seg_ppg, dtype=torch.float32),
            'ecg': torch.tensor(seg_ecg, dtype=torch.float32),
            'theta': torch.tensor(seg_theta, dtype=torch.float32),
            'omega': torch.tensor(seg_omega, dtype=torch.float32)
        }
        
        # Nome file univoco per il frammento
        base_name = file_name.replace(".pt", "")
        new_file_name = f"{base_name}_seg{idx}.pt"
        
        # Creazione cartella paziente se non esiste
        pat_out_dir = os.path.join(save_dir, pat_id)
        os.makedirs(pat_out_dir, exist_ok=True)
        
        full_out_path = os.path.join(pat_out_dir, new_file_name)
        torch.save(out_data, full_out_path)
        
        # Aggiornamento Manifest
        manifest.append({
            'subject_id': pat_id,
            'file_path': new_file_name, # Solo nome file, il caricatore gestirà le cartelle
            'length_samples': end - start,
            'duration_sec': (end - start) / fs
        })
        segments_saved += 1
        
    return segments_saved

# ==========================================
# 3. MOTORE PRINCIPALE
# ==========================================

def build_clean_pinn_dataset(source_dir, dest_dir, win_sec=4, step_sec=1, min_size_sec=7, ppg_thr=0.9, ecg_thr=0.9, fs=125):
    os.makedirs(dest_dir, exist_ok=True)
    
    patient_dirs = sorted([d for d in os.listdir(source_dir) if d.startswith('p') and os.path.isdir(os.path.join(source_dir, d))])
    print(f"🚀 Inizio Creazione Dataset PINN: Trovati {len(patient_dirs)} pazienti.\n")
    
    manifest_data = []
    total_segments = 0
    
    for patient in tqdm(patient_dirs, desc="Generazione Nuovo Dataset"):
        pat_dir = os.path.join(source_dir, patient)
        pt_files = [f for f in os.listdir(pat_dir) if f.endswith('.pt')]
        
        for file in pt_files:
            file_path = os.path.join(pat_dir, file)
            try:
                data = torch.load(file_path, map_location='cpu')
                data_np = {
                    'ppg': data['ppg'].numpy().flatten().astype(np.float32),
                    'ecg': data['ecg'].numpy().flatten().astype(np.float32)
                }
                
                # Questa funzione calcola tutto e salva i file fisicamente
                saved = process_and_save_patient_file(
                    data_np, file, patient, dest_dir, manifest_data,
                    win_sec, step_sec, min_size_sec, fs, ppg_thr, ecg_thr
                )
                total_segments += saved
                
            except Exception as e:
                pass
                
    # Salvataggio del Manifest Finale
    manifest_path = os.path.join(dest_dir, 'dataset_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=4)
        
    print("\n" + "="*60)
    print("✅ CREAZIONE DATASET COMPLETATA")
    print("="*60)
    print(f"🔹 Totale Segmenti Puliti Salvati: {total_segments}")
    print(f"🔹 Destinazione: {dest_dir}")
    print(f"🔹 Manifest aggiornato salvato con successo.")

if __name__ == "__main__":
    SOURCE_DATASET = "/home/mmerone/mattia/ecg_generation/mimic3_smart_cache" 
    DEST_DATASET = "/home/mmerone/mattia/ecg_generation/mimic3_pinn_cache" 
    
    build_clean_pinn_dataset(
        source_dir=SOURCE_DATASET, 
        dest_dir=DEST_DATASET,
        win_sec=4, 
        step_sec=1, 
        min_size_sec=7, 
        ppg_thr=0.9, 
        ecg_thr=0.9, 
        fs=125
    )