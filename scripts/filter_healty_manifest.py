import os
import json
import torch
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

def analyze_ecg_morphology(ecg_signal, fs=125):
    """
    Analizza la morfologia di un segmento ECG per determinare se appartiene
    a un cuore sano (no T invertite, no ST elevation/depression).
    Ritorna True se sano, False se patologico/anomalo.
    """
    # 1. Normalizzazione Z-score per avere soglie universali
    std = np.std(ecg_signal)
    if std < 1e-5: return False # Segnale piatto
    
    ecg_norm = (ecg_signal - np.mean(ecg_signal)) / std
    
    # 2. Rilevamento Picchi R
    peaks, _ = find_peaks(ecg_norm, distance=int(fs*0.4), height=1.0)
    if len(peaks) < 3: return False # Troppi pochi battiti per giudicare
    
    # 3. Estrazione dei singoli battiti
    # Per fs=125, prendiamo -40 campioni (320ms prima) e +60 campioni (480ms dopo)
    pre_samples = int(fs * 0.32)  # ~40
    post_samples = int(fs * 0.48) # ~60
    
    beats = []
    for p in peaks:
        if p - pre_samples >= 0 and p + post_samples < len(ecg_norm):
            beats.append(ecg_norm[p - pre_samples : p + post_samples])
            
    if len(beats) < 3: return False
    
    # 4. Calcolo del BATTITO MEDIANO (robusto contro artefatti singoli)
    median_beat = np.median(beats, axis=0)
    
    # 5. Definizione delle Regioni d'Interesse (ROI)
    # L'indice del picco R nel median_beat è esattamente 'pre_samples' (es. indice 40)
    idx_R = pre_samples
    
    # Baseline: segmento prima dell'onda P (es. primi 15 campioni)
    baseline_val = np.mean(median_beat[0:15])
    
    # Tratto ST: Subito dopo il QRS (es. da +10 a +25 campioni dal picco R)
    # 80ms - 200ms dopo la R
    st_start = idx_R + int(fs * 0.08)
    st_end = idx_R + int(fs * 0.20)
    st_segment = median_beat[st_start:st_end]
    st_mean = np.mean(st_segment)
    
    # Onda T: La "collina" dopo il tratto ST (es. da +25 a +50 campioni dalla R)
    # 200ms - 400ms dopo la R
    t_start = idx_R + int(fs * 0.20)
    t_end = idx_R + int(fs * 0.40)
    t_wave = median_beat[t_start:t_end]
    t_max = np.max(t_wave)
    
    # --- CRITERI DI ESCLUSIONE CARDIOLOGICA ---
    
    # Criterio A: Onda T invertita o assente
    # La T deve essere positiva rispetto alla baseline
    if (t_max - baseline_val) < 0.15:
        return False
        
    # Criterio B: Sottoslivellamento o Sovraslivellamento ST (Ischemia / Infarto)
    # Una deviazione > 0.4 deviazioni standard dalla baseline è sospetta
    if abs(st_mean - baseline_val) > 0.4:
        return False

    return True


def create_healthy_manifest(cache_dir):
    original_manifest_path = os.path.join(cache_dir, 'dataset_manifest.json')
    new_manifest_path = os.path.join(cache_dir, 'dataset_manifest_healthy.json')
    
    print(f"📥 Caricamento manifest originale da: {original_manifest_path}")
    with open(original_manifest_path, 'r') as f:
        original_manifest = json.load(f)
        
    healthy_manifest = []
    dropped_count = 0
    
    # --- STRUTTURE DATI PER STATISTICHE PAZIENTI ---
    all_patients = set()
    healthy_patients = set()
    
    print("🔍 Inizio screening morfologico (Sanificazione ECG)...")
    for entry in tqdm(original_manifest, desc="Analisi File"):
        rel_path = os.path.join(entry['subject_id'], os.path.basename(entry['file_path']))
        full_path = os.path.join(cache_dir, rel_path)
        
        subj_id = entry['subject_id']
        all_patients.add(subj_id)
        
        try:
            # Carichiamo solo per testare la morfologia
            data = torch.load(full_path, map_location='cpu')
            ecg_signal = data['ecg'].numpy()
            
            # Applichiamo il filtro Cardiologico
            is_healthy = analyze_ecg_morphology(ecg_signal, fs=125)
            
            if is_healthy:
                healthy_manifest.append(entry)
                healthy_patients.add(subj_id)
            else:
                dropped_count += 1
                
        except Exception as e:
            dropped_count += 1
            pass
            
    # Salvataggio del nuovo manifest
    with open(new_manifest_path, 'w') as f:
        json.dump(healthy_manifest, f, indent=4)
        
    # --- CALCOLO STATISTICHE ---
    total_segments = len(original_manifest)
    total_patients = len(all_patients)
    num_healthy_patients = len(healthy_patients)
    excluded_patients = all_patients - healthy_patients
    num_excluded_patients = len(excluded_patients)
    
    perc_dropped_seg = (dropped_count / total_segments * 100) if total_segments > 0 else 0
    perc_kept_seg = (len(healthy_manifest) / total_segments * 100) if total_segments > 0 else 0
    
    perc_kept_pat = (num_healthy_patients / total_patients * 100) if total_patients > 0 else 0
    perc_exc_pat = (num_excluded_patients / total_patients * 100) if total_patients > 0 else 0
        
    print("\n" + "="*60)
    print("✅ SANIFICAZIONE COMPLETATA - REPORT STATISTICO")
    print("="*60)
    print("📊 STATISTICHE FINESTRE TEMPORALI:")
    print(f"   🔹 Totale segmenti originali: {total_segments}")
    print(f"   🔹 Segmenti scartati (Patologici/Rumorosi): {dropped_count} ({perc_dropped_seg:.1f}%)")
    print(f"   🔹 Segmenti Conservati (Sani): {len(healthy_manifest)} ({perc_kept_seg:.1f}%)")
    print("-" * 60)
    print("👥 STATISTICHE PAZIENTI:")
    print(f"   🔹 Totale pazienti originali: {total_patients}")
    print(f"   🔹 Pazienti con almeno un segmento sano conservato: {num_healthy_patients} ({perc_kept_pat:.1f}%)")
    print(f"   🔹 Pazienti totalmente esclusi (es. infarto grave ovunque): {num_excluded_patients} ({perc_exc_pat:.1f}%)")
    print("="*60)
    print(f"📁 Nuovo manifest salvato in: {new_manifest_path}")
    print("="*60)


if __name__ == "__main__":
    # Inserisci qui il path della tua cartella mimic3_pinn_cache
    CACHE_DIR = "/home/mmerone/mattia/ecg_generation/mimic3_pinn_cache" 
    create_healthy_manifest(CACHE_DIR)