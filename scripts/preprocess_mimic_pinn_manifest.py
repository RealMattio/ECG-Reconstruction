"""
Preprocessing MIMIC-III zero-copia per PINN.

Sostituisce il vecchio flusso a 3 step (preprocess_mimic_smart → create_pinn_dataset →
filter_healty_manifest), ognuno dei quali creava copie fisiche dei segnali su disco.


Questo script:
  1. Legge i file WFDB direttamente da --input_dir
  2. Applica il filtro bandpass (PPG 0.5-5 Hz, ECG 0.5-40 Hz)
  3. Esegue i controlli di qualità di create_pinn_dataset.py
       - PPG: SQI spettrale (Welch), morfologia (Pearson sui battiti), polarità
       - ECG: kurtosis ≥ 5, morfologia (Pearson sui complessi QRS)
  4. Salva SOLO il manifest JSON (nessuna copia dei segnali)

Il manifest risultante viene letto da MimicSmartDataset (pipeline PINN) che carica
i segnali WFDB in RAM al momento dell'addestramento senza mai scrivere nulla su disco.

Output:
  <output_dir>/dataset_manifest.json
"""

import os
import sys
import json
import argparse

import wfdb
import numpy as np
from scipy.signal import butter, sosfilt, resample, find_peaks, welch
from scipy.stats import kurtosis, pearsonr
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────
# FUNZIONI DI QUALITÀ  (riprese da create_pinn_dataset.py)
# ─────────────────────────────────────────────────────────────

def _bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfilt(sos, signal).astype(np.float32)


def _check_polarity(ppg_signal):
    """
    True se il PPG è invertito.
    Criterio: asimmetria di ampiezza sul segnale centrato.
    In un PPG corretto il picco sistolico è la deviazione positiva dominante,
    quindi max(centrato) > |min(centrato)|. Se vale il contrario, è invertito.
    """
    centered = ppg_signal - np.mean(ppg_signal)
    return float(centered.max()) < float(-centered.min())


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
    """
    Ritorna (score, is_inverted).
    score = 0.0 se il segmento non supera i controlli di qualità.
    """
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
    """Ritorna uno score di qualità ECG ∈ [0, 1]."""
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


# ─────────────────────────────────────────────────────────────
# SCAN WFDB
# ─────────────────────────────────────────────────────────────

def _find_records(data_dir):
    """Restituisce tutti i path base di record WFDB validi (coppia .hea + .dat)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Directory non trovata: {data_dir}\n"
            "Controlla il parametro --input_dir o il path di default."
        )
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
        description="Crea dataset_manifest.json PINN senza copiare i segnali su disco."
    )
    parser.add_argument(
        '--input_dir', type=str,
        default=os.path.join(PROJECT_ROOT, '..', 'mimic3wdb-matched_healthy_data'),
        help="Cartella radice con i file WFDB (default: ../mimic3wdb-matched_healthy_data)"
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Dove salvare dataset_manifest.json (default: uguale a --input_dir)"
    )
    parser.add_argument('--target_fs', type=int, default=125,
                        help="Frequenza di campionamento target (default: 125 Hz)")
    parser.add_argument('--win_sec', type=float, default=4.0,
                        help="Finestra di valutazione qualità in secondi (default: 4)")
    parser.add_argument('--step_sec', type=float, default=1.0,
                        help="Passo dello sliding window di qualità (default: 1)")
    parser.add_argument('--min_size_sec', type=float, default=7.0,
                        help="Lunghezza minima segmento valido in secondi (default: 7)")
    parser.add_argument('--ppg_thr', type=float, default=0.9,
                        help="Soglia Pearson PPG (default: 0.9)")
    parser.add_argument('--ecg_thr', type=float, default=0.9,
                        help="Soglia Pearson ECG (default: 0.9)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    fs = args.target_fs
    win_samples = int(args.win_sec * fs)
    step_samples = int(args.step_sec * fs)
    min_samples = int(args.min_size_sec * fs)

    print("=" * 60)
    print("PREPROCESSING PINN — MANIFEST ONLY (zero copie su disco)")
    print("=" * 60)
    print(f"Input WFDB : {input_dir}")
    print(f"Output     : {output_dir}/dataset_manifest.json")
    print(f"fs={fs} Hz | finestra qualità={args.win_sec}s | "
          f"min segmento={args.min_size_sec}s")
    print(f"Soglie: PPG Pearson≥{args.ppg_thr} | ECG Pearson≥{args.ecg_thr}")
    print("=" * 60)

    records = _find_records(input_dir)
    print(f"Record WFDB trovati: {len(records)}\n")

    manifest = []
    total_segments = 0
    skipped_no_channels = 0
    skipped_too_short = 0
    skipped_no_valid = 0

    for rec_path in tqdm(records, desc="Analisi record"):
        try:
            # ── Controllo header (veloce, senza caricare i segnali) ──
            header = wfdb.rdheader(rec_path)
            if 'II' not in header.sig_name or 'PLETH' not in header.sig_name:
                skipped_no_channels += 1
                continue

            subject_id = os.path.basename(os.path.dirname(rec_path))

            # ── Caricamento segnali ──
            record = wfdb.rdrecord(rec_path)
            idx_ecg = record.sig_name.index('II')
            idx_ppg = record.sig_name.index('PLETH')
            ecg_raw = np.nan_to_num(record.p_signal[:, idx_ecg]).astype(np.float32)
            ppg_raw = np.nan_to_num(record.p_signal[:, idx_ppg]).astype(np.float32)

            # ── Ricampionamento ──
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

            # ── Filtro bandpass ──
            ppg_filt = _bandpass(ppg_raw, 0.5, 5.0, fs)
            ecg_filt = _bandpass(ecg_raw, 0.5, 40.0, fs)

            # ── Sliding window quality scan ──
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

            # ── Intersezione intervalli PPG ∩ ECG ──
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

            # ── Estrazione segmenti contigui ──
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
                'wfdb_path': rec_path,          # path assoluto al record WFDB originale
                'segments': segments,
                'num_segments': len(segments)
            })

        except Exception as e:
            continue

    # ── Salvataggio manifest ──
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'dataset_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    manifest_kb = os.path.getsize(manifest_path) / 1024

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETATO")
    print("=" * 60)
    print(f"Record processati con successo : {len(manifest)}")
    print(f"Segmenti puliti totali         : {total_segments}")
    print(f"Scartati (no canali II/PLETH)  : {skipped_no_channels}")
    print(f"Scartati (troppo corti)        : {skipped_too_short}")
    print(f"Scartati (nessuna finestra ok) : {skipped_no_valid}")
    print(f"Manifest salvato in            : {manifest_path}")
    print(f"Spazio occupato                : {manifest_kb:.1f} KB  (zero copie segnali)")
    print("=" * 60)


if __name__ == "__main__":
    main()
