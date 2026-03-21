import numpy as np
import torch
import pywt
import random
from scipy.signal import butter, filtfilt, find_peaks
from kymatio.torch import Scattering1D

class BidmcPreprocessor:
    def __init__(self, fs=125, beat_len=120):
        self.fs = fs  # Frequenza campionamento MIMIC II
        self.beat_len = beat_len  # 120 campioni 
        
        # Inizializziamo la WST per ottenere circa 19 coefficienti
        self.scattering = Scattering1D(J=2, shape=(beat_len,), Q=8)

    def split_subjects(self, available_keys, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Divide i soggetti reali caricati dal loader in Train, Val e Test set.
        """
        random.seed(seed)
        keys = sorted(list(available_keys))
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

    def apply_bandpass_filter(self, signal, lowcut, highcut, order=4):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        if high >= 1.0: high = 0.99 
        b, a = butter(order, [low, high], btype='bandpass')
        return filtfilt(b, a, signal)

    def normalize_signal(self, signal):
        """Z-score normalization."""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def normalize_min_max(self, signal):
        """Normalizza il segnale nell'intervallo [0, 1] LOCALE."""
        s_min = np.min(signal)
        s_max = np.max(signal)
        # Protezione contro divisione per zero se il segnale è piatto
        if s_max - s_min < 1e-8:
            return signal - s_min # Ritorna array di zeri
        return (signal - s_min) / (s_max - s_min)

    def detect_r_peaks(self, ecg_signal):
        """Rilevamento picchi R per segmentazione beat-by-beat."""
        distance = int(self.fs * 0.6) 
        # Nota: Lavorando su segnale non normalizzato globalmente, usiamo la media locale o zero
        peaks, _ = find_peaks(ecg_signal, distance=distance, height=np.mean(ecg_signal))
        return peaks

    def extract_wst_features(self, ppg_beats):
        ppg_tensor = torch.from_numpy(ppg_beats).float()
        wst_features = self.scattering(ppg_tensor) 
        return wst_features.numpy()

    def apply_dwt_ecg(self, ecg_beats):
        dwt_beats = []
        for beat in ecg_beats:
            coeffs = pywt.wavedec(beat, 'db4', level=2)
            dwt_beats.append(np.concatenate(coeffs))
        return np.array(dwt_beats)

    def segment_into_beats(self, ppg_signal, ecg_signal, r_peaks, normalize=False):
        """Segmentazione centrata sui picchi R con normalizzazione locale."""
        ppg_beats, ecg_beats = [], []
        half_len = self.beat_len // 2
        
        for peak in r_peaks:
            start, end = peak - half_len, peak + half_len
            if start >= 0 and end <= len(ppg_signal):
                p_seg = ppg_signal[start:end]
                e_seg = ecg_signal[start:end]
                
                # --- MODIFICA: Normalizzazione Locale ---
                if normalize:
                    p_seg = self.normalize_min_max(p_seg)
                    e_seg = self.normalize_min_max(e_seg)
                
                ppg_beats.append(p_seg)
                ecg_beats.append(e_seg)
                
        return np.array(ppg_beats), np.array(ecg_beats)

    def segment_with_overlap(self, ppg_signal, ecg_signal, overlap_pct=0.5, normalize=False):
        """Segmentazione sliding window con normalizzazione locale finestra per finestra."""
        ppg_segments, ecg_segments = [], []
        step = int(self.beat_len * (1 - overlap_pct))
        
        for i in range(0, len(ppg_signal) - self.beat_len, step):
            p_seg = ppg_signal[i : i + self.beat_len]
            e_seg = ecg_signal[i : i + self.beat_len]
            
            # Controllo validità segnale (evita finestre vuote/piatte)
            if np.std(p_seg) > 1e-3:
                # --- MODIFICA: Normalizzazione Locale ---
                # Normalizziamo SOLO questa finestra specifica.
                # Se c'è uno spike altrove nel segnale, questa finestra non ne risente.
                if normalize:
                    p_seg = self.normalize_min_max(p_seg)
                    e_seg = self.normalize_min_max(e_seg)

                ppg_segments.append(p_seg)
                ecg_segments.append(e_seg)
                
        return np.array(ppg_segments), np.array(ecg_segments)

    def process_subject(self, ppg_raw, ecg_raw, configs, is_training=False):
        """
        Esegue processing con logica ROBUSTA agli outlier.
        """
        
        # 1. FILTRAGGIO
        ppg_filtered = self.apply_bandpass_filter(ppg_raw, lowcut=configs.get('ppg_bandpass', [0.5, 8.0])[0], highcut=configs.get('ppg_bandpass', [0.5, 8.0])[1])
        
        if configs.get('apply_ecg_filter', True):
            ecg_filtered = self.apply_bandpass_filter(ecg_raw, lowcut=configs.get('ecg_bandpass', [0.5, 25.0])[0], highcut=configs.get('ecg_bandpass', [0.5, 25.0])[1])
        else:
            ecg_filtered = self.apply_bandpass_filter(ecg_raw, lowcut=0.5, highcut=8.0)

        # --- MODIFICA CRITICA: RIMOSSA NORMALIZZAZIONE GLOBALE QUI ---
        # Prima normalizzavamo tutto il segnale qui. Ora passiamo i segnali filtrati (ma grezzi in ampiezza)
        # alle funzioni di segmentazione, che normalizzeranno pezzo per pezzo.
        
        # Leggiamo il flag di normalizzazione
        do_normalize = configs.get('normalize_01', False)

        # 3. SEGMENTAZIONE (+ NORMALIZZAZIONE LOCALE)
        overlap_val = configs.get('overlap_pct', 0.1)
        
        if configs.get('overlap_windows', False):
            # Passiamo il flag do_normalize alla funzione
            ppg_beats, ecg_beats = self.segment_with_overlap(ppg_filtered, ecg_filtered, overlap_pct=overlap_val, normalize=do_normalize)
        else:
            peaks = self.detect_r_peaks(ecg_filtered)
            ppg_beats, ecg_beats = self.segment_into_beats(ppg_filtered, ecg_filtered, peaks, normalize=do_normalize)

        # Se non ci sono battiti
        if len(ppg_beats) == 0:
            return np.array([]), np.array([])

        # 4. DATA AUGMENTATION
        if is_training and configs.get('apply_augmentation', False):
            augmented_ppg = []
            augmented_ecg = []
            for i in range(len(ppg_beats)):
                p_aug, e_aug = self.apply_augmentation(ppg_beats[i], ecg_beats[i], configs)
                augmented_ppg.append(p_aug)
                augmented_ecg.append(e_aug)
            ppg_beats = np.array(augmented_ppg)
            ecg_beats = np.array(augmented_ecg)

        # 5. TRASFORMAZIONI
        if configs.get('apply_wst', False):
            ppg_beats = self.extract_wst_features(ppg_beats)
            
        if configs.get('apply_dwt', False):
            ecg_beats = self.apply_dwt_ecg(ecg_beats)

        return ppg_beats, ecg_beats
    
    def apply_augmentation(self, ppg_segment, ecg_segment, configs):
        aug_ppg = ppg_segment.copy()
        aug_ecg = ecg_segment.copy()

        # Augmentation (Invariata)
        if configs.get('aug_random_gain', False):
            factor = np.random.uniform(0.8, 1.2)
            aug_ppg *= factor
        
        if configs.get('aug_time_stretch', False):
            stretch_factor = np.random.uniform(0.9, 1.1)
            orig_len = len(ppg_segment)
            new_len = int(orig_len * stretch_factor)
            indices = np.linspace(0, orig_len - 1, new_len)
            aug_ppg = np.interp(np.linspace(0, orig_len - 1, orig_len), indices, np.interp(indices, np.arange(orig_len), aug_ppg))
            aug_ecg = np.interp(np.linspace(0, orig_len - 1, orig_len), indices, np.interp(indices, np.arange(orig_len), aug_ecg))

        return aug_ppg, aug_ecg