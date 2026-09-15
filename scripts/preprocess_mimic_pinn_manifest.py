"""
Preprocessing MIMIC-III zero-copia per PINN (Ottimizzato per SLURM HPC).

Questo script:
  1. Legge i file WFDB direttamente da --input_dir
  2. Applica filtri e controlli di qualità clinica
  3. Salva SOLO il manifest JSON (dataset_manifest.json)
  4. Esegue salvataggi intermedi (checkpointing) atomici ogni N iterazioni
     per permettere il ripristino sicuro dopo i timeout di SLURM.
"""

import os
import sys
import json
import argparse
import errno

import wfdb
import numpy as np
from scipy.signal import butter, sosfilt, resample, find_peaks, welch
from scipy.stats import kurtosis, pearsonr
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────
# FUNZIONI DI UTILITÀ E QUALITÀ
# ─────────────────────────────────────────────────────────────

def _save_manifest_atomically(data, filepath):
    """
    Salvataggio atomico con gestione degli errori I/O (es. disco pieno).
    """
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except OSError as e:
        if e.errno == errno.ENOSPC: # Codice di errore 28: No space left on device
            print(f"\n[ERRORE CRITICO HPC] Spazio su disco esaurito!")
            print(f"Impossibile scrivere il file temporaneo: {tmp_path}")
            print("L'ultimo checkpoint valido (dataset_manifest.json) è stato preservato.")
            print("Interruzione forzata dell'elaborazione. Libera spazio o cambia --output_dir e usa il restore.")
            # Esce con codice 1 così SLURM sa che il job è fallito e non fa il resubmit infinito
            sys.exit(1)
        else:
            raise e


def _bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfilt(sos, signal).astype(np.float32)


def _check_polarity(ppg_signal):
    p10 = float(np.percentile(ppg_signal, 10))
    p50 = float(np.percentile(ppg_signal, 50))
    p90 = float(np.percentile(ppg_signal, 90))
    return (p50 - p10) > (p90 - p50)


def _spectral_sqi(ppg_signal, fs):
    freqs, psd = welch(ppg_signal, fs, nperseg=max(len(ppg_signal) // 2, 2))
    mask_hr = (freqs >= 0.5) & (freqs <= 3.0)
    if not np.any(mask_hr):
        return 0.0
    peak_freq = freqs[mask_hr][np.argmax(psd[mask_hr])]
    power_peak = np.sum(psd[(freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2)])
    power_total = np.sum(psd[(freqs >= 0.5) & (freqs <= 10.0)])
    return 0.0 if power_total == 0 else float(power_peak / power_total)


def _ppg_quality(ppg_signal, fs, spectral_thr=0.5):
    if np.std(ppg_signal) < 1e-6:
        return 0.0, False
    is_inverted = _check_polarity(ppg_signal)
    sig = ppg_signal * -1 if is_inverted else ppg_signal
    if _spectral_sqi(sig, fs) < spectral_thr:
        return 0.0, False
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs * 0.35), prominence=0.4)
    if len(peaks) < 4:
        return 0.0, False
    beats = []
    for i in range(1, len(peaks)):
        beat = sig_norm[peaks[i - 1]:peaks[i]]
        if fs * 0.3 < len(beat) < fs * 2.0:
            beat_interp = np.interp(
                np.linspace(0, 1, 100), np.linspace(0, 1, len(beat)), beat
            )
            beats.append(beat_interp)
    if len(beats) < 3:
        return 0.0, False
    template = np.mean(beats, axis=0)
    score = float(np.mean([pearsonr(template, b)[0] for b in beats]))
    return score, is_inverted


def _ecg_quality(ecg_signal, fs, kurtosis_thr=5.0):
    if np.std(ecg_signal) < 1e-4:
        return 0.0
    if kurtosis(ecg_signal, fisher=False) < kurtosis_thr:
        return 0.0
    sig_norm = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    peaks, _ = find_peaks(sig_norm, distance=int(fs * 0.4), height=1.5)
    if len(peaks) < 3:
        return 0.0
    pre, post = int(fs * 0.2), int(fs * 0.4)
    beats = [
        sig_norm[p - pre:p + post]
        for p in peaks
        if p - pre >= 0 and p + post < len(sig_norm)
    ]
    if len(beats) < 3:
        return 0.0
    template = np.mean(beats, axis=0)
    return float(np.mean([pearsonr(template, b)[0] for b in beats]))


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(iv) for iv in merged]


def _find_records(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory non trovata: {data_dir}")
    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.hea') and not f.startswith('p'):
                base = f[:-4]
                if (base + '.dat') in files:
                    records.append(os.path.join(root, base))
    return sorted(records)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea dataset_manifest.json PINN ottimizzato per cluster HPC (SLURM)."
    )
    parser.add_argument('--input_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, '..', 'mimic3wdb-matched_healthy_data'),
                        help="Cartella radice con i file WFDB")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Dove salvare dataset_manifest.json (default: uguale a --input_dir)")
    parser.add_argument('--target_fs', type=int, default=125)
    parser.add_argument('--win_sec', type=float, default=4.0)
    parser.add_argument('--step_sec', type=float, default=1.0)
    parser.add_argument('--min_size_sec', type=float, default=7.0)
    parser.add_argument('--ppg_thr', type=float, default=0.9)
    parser.add_argument('--ecg_thr', type=float, default=0.9)
    
    # Parametri per l'HPC (Restore e Checkpointing)
    parser.add_argument('--restore', action='store_true',
                        help="Se attivato, riprende l'elaborazione da un manifest preesistente.")
    parser.add_argument('--resume_idx', type=int, default=0,
                        help="Ultimo numero stampato dalla barra tqdm da cui riprendere.")
    parser.add_argument('--prev_manifest_path', type=str, default=None,
                        help="Path del file dataset_manifest.json da cui caricare i dati precedenti.")
    parser.add_argument('--save_every', type=int, default=50,
                        help="Frequenza di aggiornamento del manifest su disco (default: ogni 50 record).")
    
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'dataset_manifest.json')
    
    fs = args.target_fs
    win_samples = int(args.win_sec * fs)
    step_samples = int(args.step_sec * fs)
    min_samples = int(args.min_size_sec * fs)

    print("=" * 60)
    print("PREPROCESSING PINN — HPC READY (Zero-copia + Checkpointing)")
    print("=" * 60)
    print(f"Input WFDB : {input_dir}")
    print(f"Output     : {manifest_path}")
    print(f"Soglie: PPG>={args.ppg_thr} | ECG>={args.ecg_thr} | Autosave ogni {args.save_every} iter.")
    print("=" * 60)

    records = _find_records(input_dir)
    total_records_count = len(records)
    print(f"Record WFDB totali trovati: {total_records_count}\n")

    manifest = []
    total_segments = 0
    skipped_no_channels = 0
    skipped_too_short = 0
    skipped_no_valid = 0

    # ── Gestione Restore da iterazione precedente ──
    if args.restore:
        load_path = args.prev_manifest_path if args.prev_manifest_path else manifest_path
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"ERRORE CRITICO: Manifest non trovato al percorso: {load_path}")
        
        with open(load_path, 'r') as f:
            manifest = json.load(f)
            
        # Trova l'indice REALE basato sull'ultimo file salvato fisicamente nel JSON
        if len(manifest) > 0:
            last_saved_path = manifest[-1]['wfdb_path']
            try:
                # Cerca la posizione dell'ultimo file salvato nella lista globale
                last_idx = records.index(last_saved_path)
                args.resume_idx = last_idx + 1
            except ValueError:
                args.resume_idx = 0
        else:
            args.resume_idx = 0
            
        total_segments = sum(item.get('num_segments', 0) for item in manifest)
        
        # Taglia i record per riprendere dal punto esatto
        records = records[args.resume_idx:]
        
        print(f"[*] Fase di Restore attivata.")
        if len(manifest) > 0:
            print(f"[*] Ultimo record valido in JSON: {os.path.basename(last_saved_path)}")
        print(f"[*] Indice reale di ripresa nella coda: {args.resume_idx}")
        print(f"[*] {len(manifest)} record pregressi caricati ({total_segments} segmenti pronti).")
        print(f"[*] Ripresa elaborazione per i rimanenti {len(records)} record.\n")
    # ───────────────────────────────────────────────
    
    # Utilizziamo enumerate per tracciare le iterazioni effettive della sessione corrente
    for current_i, rec_path in enumerate(tqdm(records, desc="Analisi record", initial=args.resume_idx, total=total_records_count)):
        try:
            header = wfdb.rdheader(rec_path)
            if 'II' not in header.sig_name or 'PLETH' not in header.sig_name:
                skipped_no_channels += 1
                continue

            subject_id = os.path.basename(os.path.dirname(rec_path))

            record = wfdb.rdrecord(rec_path)
            idx_ecg = record.sig_name.index('II')
            idx_ppg = record.sig_name.index('PLETH')
            ecg_raw = np.nan_to_num(record.p_signal[:, idx_ecg]).astype(np.float32)
            ppg_raw = np.nan_to_num(record.p_signal[:, idx_ppg]).astype(np.float32)

            orig_fs = record.fs
            if orig_fs != fs:
                n = int(len(ecg_raw) / orig_fs * fs)
                ecg_raw = resample(ecg_raw, n).astype(np.float32)
                ppg_raw = resample(ppg_raw, n).astype(np.float32)

            n_samples = min(len(ecg_raw), len(ppg_raw))
            ecg_raw = ecg_raw[:n_samples]
            ppg_raw = ppg_raw[:n_samples]

            if n_samples < min_samples:
                skipped_too_short += 1
                continue

            ppg_filt = _bandpass(ppg_raw, 0.5, 5.0, fs)
            ecg_filt = _bandpass(ecg_raw, 0.5, 40.0, fs)

            ppg_intervals = []
            ecg_intervals = []
            inversions_map = np.zeros(n_samples, dtype=bool)

            for start in range(0, n_samples - win_samples, step_samples):
                end = start + win_samples
                ppg_score, is_inv = _ppg_quality(ppg_filt[start:end], fs)
                if ppg_score >= args.ppg_thr:
                    ppg_intervals.append((start, end))
                    if is_inv:
                        inversions_map[start:end] = True
                if _ecg_quality(ecg_filt[start:end], fs) >= args.ecg_thr:
                    ecg_intervals.append((start, end))

            ppg_merged = _merge_intervals(ppg_intervals)
            ecg_merged = _merge_intervals(ecg_intervals)

            mask = np.zeros(n_samples, dtype=bool)
            for s, e in ppg_merged:
                mask[s:e] = True
            ecg_mask = np.zeros(n_samples, dtype=bool)
            for s, e in ecg_merged:
                ecg_mask[s:e] = True
            combined = mask & ecg_mask

            if not np.any(combined):
                skipped_no_valid += 1
                continue

            d = np.diff(combined.astype(np.int8))
            seg_starts = np.where(d == 1)[0] + 1
            seg_ends = np.where(d == -1)[0] + 1
            if combined[0]:
                seg_starts = np.insert(seg_starts, 0, 0)
            if combined[-1]:
                seg_ends = np.append(seg_ends, n_samples)

            segments = []
            for s, e in zip(seg_starts, seg_ends):
                if (e - s) < min_samples:
                    continue
                ppg_inverted = bool(np.mean(inversions_map[s:e]) > 0.5)
                segments.append({
                    'start': int(s),
                    'end': int(e),
                    'ppg_inverted': ppg_inverted
                })
                total_segments += 1

            if not segments:
                skipped_no_valid += 1
                continue

            manifest.append({
                'subject_id': subject_id,
                'wfdb_path': rec_path,
                'segments': segments,
                'num_segments': len(segments)
            })

        except Exception as e:
            continue
        
        # ── Checkpointing Periodico ──
        if (current_i + 1) % args.save_every == 0:
            _save_manifest_atomically(manifest, manifest_path)

    # ── Salvataggio Finale ──
    _save_manifest_atomically(manifest, manifest_path)
    manifest_kb = os.path.getsize(manifest_path) / 1024

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETATO O INTERROTTO VOLONTARIAMENTE")
    print("=" * 60)
    print(f"Record processati con successo : {len(manifest)}")
    print(f"Segmenti puliti totali         : {total_segments}")
    print(f"Scartati (no canali II/PLETH)  : {skipped_no_channels} (in questa run)")
    print(f"Scartati (troppo corti)        : {skipped_too_short} (in questa run)")
    print(f"Scartati (nessuna finestra ok) : {skipped_no_valid} (in questa run)")
    print(f"Manifest finale salvato in     : {manifest_path}")
    print(f"Spazio occupato                : {manifest_kb:.1f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()