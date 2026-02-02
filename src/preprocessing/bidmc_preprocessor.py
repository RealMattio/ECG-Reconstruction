import numpy as np
import torch
import pywt
import random
from scipy.signal import butter, filtfilt, find_peaks
from kymatio.torch import Scattering1D

class BidmcPreprocessor:
    def __init__(self, fs=125, beat_len=120):
        self.fs = fs  # Frequenza campionamento MIMIC II [cite: 313]
        self.beat_len = beat_len  # 120 campioni 
        
        # Inizializziamo la WST per ottenere circa 19 coefficienti [cite: 318, 530]
        # J=2 (scale), Q=8 (risoluzione) sono parametri tipici per segnali fisiologici
        self.scattering = Scattering1D(J=2, shape=(beat_len,), Q=8)

    def split_subjects(self, available_keys, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Divide i soggetti reali caricati dal loader in Train, Val e Test set.
        """
        random.seed(seed)
        keys = sorted(list(available_keys)) # Assicura ordine prima dello shuffle
        random.shuffle(keys)

        n_total = len(keys)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_keys = sorted(keys[:n_train])
        val_keys = sorted(keys[n_train : n_train + n_val])
        test_keys = sorted(keys[n_train + n_val :])

        print("\n" + "="*40)
        print(f"SOGGETTI SPLIT (Seed: {seed})")
        print(f"Train ({len(train_keys)}): {train_keys}")
        print(f"Val   ({len(val_keys)}): {val_keys}")
        print(f"Test  ({len(test_keys)}): {test_keys}")
        print("="*40 + "\n")

        return train_keys, val_keys, test_keys

    def apply_bandpass_filter(self, signal, lowcut=0.5, highcut=8.0, order=4):
        """Filtro 0.5-8Hz come da specifiche HA-CNN-BILSTM[cite: 326]."""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandpass')
        return filtfilt(b, a, signal)

    def normalize_signal(self, signal):
        """Z-score normalization."""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def detect_r_peaks(self, ecg_signal):
        """Rilevamento picchi R per segmentazione beat-by-beat[cite: 86]."""
        distance = int(self.fs * 0.6) 
        peaks, _ = find_peaks(ecg_signal, distance=distance, height=np.mean(ecg_signal))
        return peaks

    def extract_wst_features(self, ppg_beats):
        """
        Trasforma i battiti PPG in 19 canali tramite Wavelet Scattering.
        """
        ppg_tensor = torch.from_numpy(ppg_beats).float()
        # Scattering calcola i coefficienti di ordine 0, 1 e 2
        wst_features = self.scattering(ppg_tensor) 
        # Risultato: (Batch, Channels=19, Time_reduced)
        return wst_features.numpy()

    def apply_dwt_ecg(self, ecg_beats):
        """
        Applica DWT all'ECG come label di addestramento[cite: 154, 337].
        """
        dwt_beats = []
        for beat in ecg_beats:
            # Wavelet 'db4' suggerita per segnali ECG [cite: 336, 337]
            coeffs = pywt.wavedec(beat, 'db4', level=2)
            # Concateniamo i coefficienti per creare un vettore di label
            dwt_beats.append(np.concatenate(coeffs))
        return np.array(dwt_beats)

    def segment_into_beats(self, ppg_signal, ecg_signal, r_peaks):
        """Segmentazione centrata sui picchi R (No overlap)[cite: 86, 333]."""
        ppg_beats, ecg_beats = [], []
        half_len = self.beat_len // 2
        for peak in r_peaks:
            start, end = peak - half_len, peak + half_len
            if start >= 0 and end <= len(ppg_signal):
                ppg_beats.append(ppg_signal[start:end])
                ecg_beats.append(ecg_signal[start:end])
        return np.array(ppg_beats), np.array(ecg_beats)

    def segment_with_overlap(self, ppg_signal, ecg_signal, overlap_pct=0.5):
        """Segmentazione con sliding window per continuità."""
        ppg_segments, ecg_segments = [], []
        step = int(self.beat_len * (1 - overlap_pct))
        for i in range(0, len(ppg_signal) - self.beat_len, step):
            p_seg, e_seg = ppg_signal[i:i+self.beat_len], ecg_signal[i:i+self.beat_len]
            if np.std(p_seg) > 1e-3:
                ppg_segments.append(p_seg)
                ecg_segments.append(e_seg)
        return np.array(ppg_segments), np.array(ecg_segments)

    def process_subject(self, ppg_raw, ecg_raw, configs):
        # (Il codice esistente della funzione process_subject rimane lo stesso)
        ppg_filtered = self.apply_bandpass_filter(ppg_raw)
        ppg_norm = self.normalize_signal(ppg_filtered)
        ecg_norm = self.normalize_signal(ecg_raw)

        overlap_val = configs.get('overlap_pct', 0.1)
        if configs.get('overlap_windows', False):
            ppg_beats, ecg_beats = self.segment_with_overlap(ppg_norm, ecg_norm, overlap_pct=overlap_val)
        else:
            peaks = self.detect_r_peaks(ecg_norm)
            ppg_beats, ecg_beats = self.segment_into_beats(ppg_norm, ecg_norm, peaks)

        if configs.get('apply_wst', False):
            ppg_beats = self.extract_wst_features(ppg_beats)
            
        if configs.get('apply_dwt', False):
            ecg_beats = self.apply_dwt_ecg(ecg_beats)

        return ppg_beats, ecg_beats