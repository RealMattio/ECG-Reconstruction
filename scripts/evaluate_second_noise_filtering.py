import os
import torch
import numpy as np
import warnings
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, pearsonr
from tqdm import tqdm

# Ignoriamo i warning di scipy per calcoli su finestre saltuariamente piatte o caotiche
warnings.filterwarnings('ignore')

# ==========================================
# 1. FUNZIONI DI QUALITÀ (SQI)
# ==========================================

def check_polarity_by_slope(ppg_signal):
    diff = np.diff(ppg_signal)
    max_slope_up = np.max(diff)
    max_slope_down = np.min(diff)
    if abs(max_slope_down) > abs(max_slope_up):
        return ppg_signal * -1
    return ppg_signal

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
    if np.std(ppg_signal) < 1e-6: return 0.0
    sig_fixed = check_polarity_by_slope(ppg_signal)
    spec_score = get_spectral_sqi(sig_fixed, fs)
    if spec_score < spectral_thr: return 0.0
    sig_norm = (sig_fixed - np.mean(sig_fixed)) / (np.std(sig_fixed) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs * 0.35), prominence=0.4)
    if len(peaks) < 4: return 0.0
    beats = []
    for i in range(1, len(peaks)):
        beat = sig_norm[peaks[i-1]:peaks[i]]
        if fs * 0.3 < len(beat) < fs * 2.0:
            beat_interp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(beat)), beat)
            beats.append(beat_interp)
    if len(beats) < 3: return 0.0
    template = np.mean(beats, axis=0)
    return np.mean([pearsonr(template, b)[0] for b in beats])

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

# ==========================================
# 2. LOGICA DI MERGING E INTERSEZIONE
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

def cross_check_analysis(data, win_sec=4, step_sec=1, min_size_sec=7, fs=125, ppg_thr=0.9, ecg_thr=0.9):
    total_samples = len(data['ppg'])
    win_samples = int(win_sec * fs)
    step_samples = int(step_sec * fs)
    min_samples = int(min_size_sec * fs)
    
    ppg_raw_intervals = []
    ecg_raw_intervals = []
    
    for start in range(0, total_samples - win_samples, step_samples):
        end = start + win_samples
        if get_ppg_quality_score(data['ppg'][start:end], fs) > ppg_thr:
            ppg_raw_intervals.append((start, end))
        if get_ecg_quality_score(data['ecg'][start:end], fs) > ecg_thr:
            ecg_raw_intervals.append((start, end))
            
    ppg_merged = merge_intervals(ppg_raw_intervals)
    ecg_merged = merge_intervals(ecg_raw_intervals)
    
    mask_ppg = np.zeros(total_samples, dtype=bool)
    for s, e in ppg_merged: mask_ppg[s:e] = True
    mask_ecg = np.zeros(total_samples, dtype=bool)
    for s, e in ecg_merged: mask_ecg[s:e] = True
    
    combined_mask = mask_ppg & mask_ecg
    
    combined_intervals = []
    if np.any(combined_mask):
        diff = np.diff(combined_mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if combined_mask[0]: starts = np.insert(starts, 0, 0)
        if combined_mask[-1]: ends = np.append(ends, total_samples)
        
        raw_combined = list(zip(starts, ends))
        combined_intervals = [(s, e) for s, e in raw_combined if (e - s) >= min_samples]
        
    return total_samples, combined_intervals

# ==========================================
# 3. UTILITY DI FORMATTAZIONE E REPORT
# ==========================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

# ==========================================
# 4. MOTORE DI SCANSIONE DATASET
# ==========================================

def evaluate_dataset(root_dir, win_sec=4, step_sec=1, min_size_sec=7, ppg_thr=0.9, ecg_thr=0.9, fs=125):
    patient_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith('p') and os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"🚀 Inizio valutazione dataset: Trovati {len(patient_dirs)} pazienti.\n")
    
    global_original_sec = 0
    global_valid_sec = 0
    global_valid_segments = 0
    
    patient_stats = {}
    
    for patient in tqdm(patient_dirs, desc="Elaborazione Pazienti"):
        pat_dir = os.path.join(root_dir, patient)
        pt_files = [f for f in os.listdir(pat_dir) if f.endswith('.pt')]
        
        pat_orig_sec = 0
        pat_valid_sec = 0
        pat_segments = 0
        
        for file in pt_files:
            file_path = os.path.join(pat_dir, file)
            try:
                data = torch.load(file_path, map_location='cpu')
                # Converte i tensori in array numpy e assicura il tipo corretto
                data_np = {
                    'ppg': data['ppg'].numpy().flatten().astype(np.float32),
                    'ecg': data['ecg'].numpy().flatten().astype(np.float32)
                }
                
                total_samples, valid_intervals = cross_check_analysis(
                    data_np, win_sec, step_sec, min_size_sec, fs, ppg_thr, ecg_thr
                )
                
                pat_orig_sec += total_samples / fs
                for s, e in valid_intervals:
                    pat_valid_sec += (e - s) / fs
                pat_segments += len(valid_intervals)
                
            except Exception as e:
                # Se un file è corrotto, lo saltiamo e proseguiamo
                pass
        
        # Salviamo le statistiche del paziente
        retention = (pat_valid_sec / pat_orig_sec * 100) if pat_orig_sec > 0 else 0
        patient_stats[patient] = {
            'orig_sec': pat_orig_sec,
            'valid_sec': pat_valid_sec,
            'segments': pat_segments,
            'retention': retention
        }
        
        global_original_sec += pat_orig_sec
        global_valid_sec += pat_valid_sec
        global_valid_segments += pat_segments
        
    # ==========================================
    # 5. STAMPA DEL REPORT FINALE
    # ==========================================
    print("\n" + "="*80)
    print("📋 REPORT DETTAGLIATO PER PAZIENTE")
    print("="*80)
    print(f"{'Paziente':<12} | {'Durata Iniziale':<15} | {'Durata Valida':<15} | {'Ritenzione':<12} | {'N° Segmenti':<10}")
    print("-" * 80)
    
    # Stampa i pazienti in ordine di percentuale di ritenzione (dal migliore al peggiore)
    sorted_pats = sorted(patient_stats.items(), key=lambda item: item[1]['retention'], reverse=True)
    for pat, stats in sorted_pats:
        orig_str = format_time(stats['orig_sec'])
        val_str = format_time(stats['valid_sec'])
        ret_str = f"{stats['retention']:.1f}%"
        print(f"{pat:<12} | {orig_str:<15} | {val_str:<15} | {ret_str:<12} | {stats['segments']:<10}")
        
    print("\n" + "="*80)
    print("🌍 SOMMARIO GLOBALE DATASET (SIMULAZIONE PREPROCESSING)")
    print("="*80)
    
    glob_retention = (global_valid_sec / global_original_sec * 100) if global_original_sec > 0 else 0
    print(f"🔹 Ore Totali Originali:    {format_time(global_original_sec)}")
    print(f"🔹 Ore Totali Conservate:   {format_time(global_valid_sec)} ✅")
    print(f"🔹 Dati Scartati (Rumore):  {format_time(global_original_sec - global_valid_sec)} ❌")
    print(f"🔹 Tasso di Ritenzione:     {glob_retention:.2f}%")
    print(f"🔹 N° di Segmenti Utili:    {global_valid_segments} (Intervalli puliti continui >= {min_size_sec}s)")
    print("="*80)

if __name__ == "__main__":
    # ⚠️ Modifica questo percorso con la posizione esatta del tuo dataset ⚠️
    # Es: DATASET_ROOT = "/mnt/d/Datasets/mimic3_smart_cache"
    DATASET_ROOT = "/home/mmerone/mattia/ecg_generation/mimic3_smart_cache" 
    
    # Parametri scelti nelle conversazioni precedenti
    evaluate_dataset(
        root_dir=DATASET_ROOT, 
        win_sec=4, 
        step_sec=1, 
        min_size_sec=7, 
        ppg_thr=0.9, 
        ecg_thr=0.9, 
        fs=125
    )