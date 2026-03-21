import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import gc
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your specific modules
from src.data_loader.mimic3wdb_data_loader import MimicDataLoader
from src.preprocessing.mimic_autoregressive_preprocessor import MimicAutoregressivePreprocessor

def main():
    parser = argparse.ArgumentParser(description="MIMIC-III Smart Offline Preprocessing (Space-Efficient)")
    parser.add_argument('--output_dir', type=str, default='../mimic3_smart_cache', help="Output directory for processed data")
    args = parser.parse_args()

    # --- CONFIGURATION (Must match your training config) ---
    configs = {
        'target_fs': 125,
        'x_sec': 7,
        'gen_sec': 1,
        'stride_sec': 1,
        'normalize_01': True, # We will normalize ON THE FLY during training to allow different scaling techniques
    }

    raw_path = os.path.join(PROJECT_ROOT, '../mimic3wdb-matched_raw_data')
    out_path = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_path, exist_ok=True)

    print(f"--- START SMART PREPROCESSING ---")
    print(f"Source: {raw_path}")
    print(f"Dest:   {out_path}")
    
    # 1. Initialize Loader
    loader = MimicDataLoader(data_dir=raw_path, target_fs=configs['target_fs'])
    raw_index = loader.load_data(lazy=True)
    
    # 2. Initialize YOUR Preprocessor (to access its methods)
    # We use this to ensure filtering and validation logic is identical
    preprocessor = MimicAutoregressivePreprocessor(
        fs=configs['target_fs'],
        window_sec=configs['x_sec'],
        gen_sec=configs['gen_sec'],
        stride_sec=configs['stride_sec']
    )

    # Temp loader for signal loading
    temp_loader = MimicDataLoader(data_dir="", target_fs=configs['target_fs'])
    temp_loader.subjects_index = raw_index['subjects_data']

    # Manifest to track all valid data
    # Format: List of dicts { 'file_path': 'path/to/p00/file.pt', 'valid_indices': [0, 125, 250...] }
    dataset_manifest = [] 
    
    total_windows_found = 0
    total_files_saved = 0

    all_keys = list(raw_index['subjects_data'].keys())

    for key in tqdm(all_keys, desc="Processing Patients"):
        meta = raw_index['subjects_data'][key]
        subject_id = meta['subject_id']

        # A. Load Signal
        try:
            sig = temp_loader.load_signal(key, meta)
            ppg_raw = sig['PPG']
            ecg_raw = sig['ECG']
        except Exception as e:
            # print(f"[WARN] Corrupt file {key}: {e}")
            continue

        if len(ppg_raw) < preprocessor.total_samples:
            continue

        # B. Filter (Identical to your class)
        # We filter the WHOLE signal once. This is efficient.
        ppg_filt = preprocessor.apply_bandpass_filter(ppg_raw, 0.5, 5.0)
        ecg_filt = preprocessor.apply_bandpass_filter(ecg_raw, 0.5, 40.0)

        # C. Scan for Valid Windows (Without duplicating data)
        valid_start_indices = []
        
        total_len = len(ppg_filt)
        X = preprocessor.total_samples  # Window size (e.g. 7s * 125 = 875)
        n = preprocessor.gen_samples    # Gen size (e.g. 1s * 125 = 125)
        stride = preprocessor.stride

        # Loop logic copied from your segment_autoregressive
        # range(n, ...) because we need history for ECG context
        for i in range(n, total_len - X + 1, stride):
            # Create views (no copy yet)
            ppg_win = ppg_filt[i : i + X]
            target_sec = ecg_filt[i + X - n : i + X]
            ecg_past = ecg_filt[i - n : i + X - n]
            
            # --- VALIDATION 1: STD DEV ---
            if np.std(ppg_win) < 1e-4 or np.std(target_sec) < 1e-4:
                continue

            # --- VALIDATION 2: SQA (Kurtosis/Skewness) ---
            # Replicate your concatenation logic for checking
            full_ecg_check = np.concatenate([ecg_past, target_sec])
            
            is_valid = preprocessor.validate_segment_sqi(full_ecg_check, ppg_win)
            
            if is_valid:
                valid_start_indices.append(i)

        # D. Save Efficiently
        if len(valid_start_indices) > 0:
            # Create patient folder
            pat_dir = os.path.join(out_path, subject_id)
            os.makedirs(pat_dir, exist_ok=True)
            dest_file = os.path.join(pat_dir, f"{key}.pt")

            # FIX: Usiamo .copy() per eliminare gli 'strides' negativi che fanno crashare PyTorch
            # Questo riordina i byte in memoria in modo contiguo (C-contiguous)
            ppg_cont = ppg_filt.copy()
            ecg_cont = ecg_filt.copy()

            data_payload = {
                'ppg': torch.tensor(ppg_cont, dtype=torch.float16),
                'ecg': torch.tensor(ecg_cont, dtype=torch.float16),
                'valid_indices': torch.tensor(valid_start_indices, dtype=torch.long),
                'fs': configs['target_fs']
            }
            
            torch.save(data_payload, dest_file)
            
            # Add to manifest
            dataset_manifest.append({
                'file_path': os.path.abspath(dest_file), 
                'subject_id': subject_id,
                'num_windows': len(valid_start_indices)
            })
            
            total_files_saved += 1
            total_windows_found += len(valid_start_indices)

        # Cleanup
        del ppg_raw, ecg_raw, ppg_filt, ecg_filt, sig
        # Variabili temporanee del fix
        if 'ppg_cont' in locals(): del ppg_cont
        if 'ecg_cont' in locals(): del ecg_cont

    # Save Global Manifest
    manifest_path = os.path.join(out_path, "dataset_manifest.json")
    # Convert to JSON-serializable format just in case
    with open(manifest_path, 'w') as f:
        json.dump(dataset_manifest, f, indent=4)

    print("\n" + "="*50)
    print("SMART PREPROCESSING COMPLETE")
    print(f"Files Saved: {total_files_saved}")
    print(f"Total Valid Windows: {total_windows_found}")
    print(f"Manifest Saved: {manifest_path}")
    print(f"Data Location: {out_path}")

if __name__ == "__main__":
    main()