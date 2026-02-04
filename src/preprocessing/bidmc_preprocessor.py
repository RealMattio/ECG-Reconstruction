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

    def apply_bandpass_filter(self, signal, lowcut, highcut, order=4):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        # Clip high per sicurezza se vicino alla frequenza di Nyquist
        if high >= 1.0: high = 0.99 
        b, a = butter(order, [low, high], btype='bandpass')
        return filtfilt(b, a, signal)

    def normalize_signal(self, signal):
        """Z-score normalization."""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def normalize_min_max(self, signal):
        """Normalizza il segnale nell'intervallo [0, 1]."""
        s_min = np.min(signal)
        s_max = np.max(signal)
        return (signal - s_min) / (s_max - s_min + 1e-8)

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

    def process_subject(self, ppg_raw, ecg_raw, configs, is_training=False):
        """
        Esegue l'intera pipeline di processing: 
        Filtro -> Normalizzazione -> Segmentazione -> (Augmentation) -> (WST/DWT)
        """
        
        # 1. FILTRAGGIO DIFFERENZIATO (Soluzione 2)
        # Il PPG viene filtrato 0.5-8Hz (standard per ossimetria)
        ppg_filtered = self.apply_bandpass_filter(ppg_raw, lowcut=0.5, highcut=8.0)
        
        # L'ECG viene filtrato 0.5-30Hz (per mantenere i picchi R nitidi)
        if configs.get('apply_ecg_filter', True):
            ecg_filtered = self.apply_bandpass_filter(ecg_raw, lowcut=0.5, highcut=30.0)
        else:
            # Se disattivato, usa un filtro base o il segnale grezzo
            ecg_filtered = self.apply_bandpass_filter(ecg_raw, lowcut=0.5, highcut=8.0)

        # 2. NORMALIZZAZIONE MODULARE (Soluzione 1)
        if configs.get('normalize_01', False):
            # Intervallo [0, 1] richiesto dai nuovi paper
            ppg_norm = self.normalize_min_max(ppg_filtered)
            ecg_norm = self.normalize_min_max(ecg_filtered)
        else:
            # Vecchia Z-score normalization (media 0, std 1)
            ppg_norm = self.normalize_signal(ppg_filtered)
            ecg_norm = self.normalize_signal(ecg_filtered)

        # 3. SEGMENTAZIONE (Invariata)
        overlap_val = configs.get('overlap_pct', 0.1)
        if configs.get('overlap_windows', False):
            ppg_beats, ecg_beats = self.segment_with_overlap(ppg_norm, ecg_norm, overlap_pct=overlap_val)
        else:
            peaks = self.detect_r_peaks(ecg_norm)
            ppg_beats, ecg_beats = self.segment_into_beats(ppg_norm, ecg_norm, peaks)

        # Se non ci sono battiti estratti (es. segnale troppo corto o piatto), esci subito
        if len(ppg_beats) == 0:
            return np.array([]), np.array([])

        # 4. DATA AUGMENTATION (Soluzione 3)
        # Si applica solo in fase di training e se attivata nei configs
        if is_training and configs.get('apply_augmentation', False):
            augmented_ppg = []
            augmented_ecg = []
            for i in range(len(ppg_beats)):
                p_aug, e_aug = self.apply_augmentation(ppg_beats[i], ecg_beats[i], configs)
                augmented_ppg.append(p_aug)
                augmented_ecg.append(e_aug)
            ppg_beats = np.array(augmented_ppg)
            ecg_beats = np.array(augmented_ecg)

        # 5. TRASFORMAZIONI OPZIONALI (WST / DWT)
        if configs.get('apply_wst', False):
            ppg_beats = self.extract_wst_features(ppg_beats)
            
        if configs.get('apply_dwt', False):
            ecg_beats = self.apply_dwt_ecg(ecg_beats)

        return ppg_beats, ecg_beats
    
    def apply_augmentation(self, ppg_segment, ecg_segment, configs):
        """Applica augmentation in modo sincronizzato a input e target."""
        aug_ppg = ppg_segment.copy()
        aug_ecg = ecg_segment.copy()

        # 1. Random Gain (Ampiezza)
        if configs.get('aug_random_gain', False):
            factor = np.random.uniform(0.8, 1.2)
            aug_ppg *= factor
            # L'ECG non lo scaliamo o lo scaliamo meno per mantenere il target stabile
        
        # 2. Time Stretching (Frequenza cardiaca)
        if configs.get('aug_time_stretch', False):
            # Stretching leggero tra 90% e 110%
            stretch_factor = np.random.uniform(0.9, 1.1)
            orig_len = len(ppg_segment)
            new_len = int(orig_len * stretch_factor)
            
            # Resampling
            indices = np.linspace(0, orig_len - 1, new_len)
            aug_ppg = np.interp(np.linspace(0, orig_len - 1, orig_len), indices, 
                                np.interp(indices, np.arange(orig_len), aug_ppg))
            aug_ecg = np.interp(np.linspace(0, orig_len - 1, orig_len), indices, 
                                np.interp(indices, np.arange(orig_len), aug_ecg))

        return aug_ppg, aug_ecg