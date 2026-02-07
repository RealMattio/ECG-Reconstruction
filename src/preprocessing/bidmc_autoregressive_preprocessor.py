import numpy as np
import random
import torch
from scipy.signal import butter, filtfilt
from kymatio.torch import Scattering1D  # <--- Import Fondamentale

class BidmcAutoregressivePreprocessor:
    def __init__(self, fs=125, window_sec=7, gen_sec=1, stride_sec=None):
        """
        Args:
            fs (int): Frequenza campionamento.
            window_sec (int): Finestra totale Input (X).
            gen_sec (int): Finestra generazione (n).
            stride_sec (float): Step scorrimento.
        """
        self.fs = fs
        self.window_sec = window_sec
        self.gen_sec = gen_sec
        
        self.total_samples = int(fs * window_sec)
        self.gen_samples = int(fs * gen_sec)
        
        # Logica Stride
        if stride_sec is None:
            self.stride = self.gen_samples
        else:
            self.stride = int(fs * stride_sec)

        # --- SETUP WAVELET SCATTERING ---
        # J=2, Q=8 sono standard per biosignals. 
        # L'output avrà lunghezza temporale ridotta di un fattore 2^J = 4
        self.scattering = Scattering1D(J=2, shape=(self.total_samples,), Q=8)

        if self.gen_samples >= self.total_samples:
            raise ValueError("n (gen_sec) deve essere minore di X (window_sec).")

    def split_subjects(self, available_keys, train_ratio=0.8, val_ratio=0.1, seed=42):
        random.seed(seed)
        keys = sorted(list(available_keys))
        random.shuffle(keys)
        n_train = int(len(keys) * train_ratio)
        n_val = int(len(keys) * val_ratio)
        return sorted(keys[:n_train]), sorted(keys[n_train:n_train+n_val]), sorted(keys[n_train+n_val:])

    def apply_bandpass_filter(self, signal, lowcut, highcut, order=4):
        nyquist = 0.5 * self.fs
        b, a = butter(order, [lowcut/nyquist, (highcut if highcut < nyquist else nyquist-0.1)/nyquist], btype='bandpass')
        return filtfilt(b, a, signal)

    def normalize_min_max(self, signal):
        s_min, s_max = np.min(signal), np.max(signal)
        return (signal - s_min) / (s_max - s_min + 1e-8)

    def normalize_signal(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def apply_augmentation(self, ppg_seg, ecg_target, configs):
        aug_ppg = ppg_seg.copy()
        if configs.get('aug_random_gain', False):
            aug_ppg *= np.random.uniform(0.8, 1.2)
        return aug_ppg, ecg_target

    def apply_context_corruption(self, ecg_past, configs):
        noise = np.random.normal(0, configs.get('aug_context_noise', 0.05), ecg_past.shape)
        corrupted = ecg_past + noise
        if configs.get('aug_context_mask', False) and random.random() < 0.3:
            mask_len = self.fs * 1
            start = random.randint(0, len(ecg_past) - mask_len)
            corrupted[start : start + mask_len] = 0.0
        return corrupted

    def extract_wst_features(self, batch_signals):
        """
        Applica WST a un batch di segnali (N, Time).
        Return: (N, Channels, Time_Reduced)
        """
        # Convertiamo in tensore PyTorch se è numpy
        if isinstance(batch_signals, np.ndarray):
            tensor_signals = torch.from_numpy(batch_signals).float()
        else:
            tensor_signals = batch_signals.float()
            
        # Kymatio si aspetta (Batch, Time) -> Output (Batch, C, Time_Red)
        with torch.no_grad():
            wst_out = self.scattering(tensor_signals)
            
        # Importante: Gestione Nan/Inf che a volte escono dalla WST
        wst_out = torch.nan_to_num(wst_out, nan=0.0)
        return wst_out.numpy()

    def segment_autoregressive(self, ppg_signal, ecg_signal):
        ppg_list, ecg_past_list, target_list = [], [], []
        total_len = len(ppg_signal)
        X, n = self.total_samples, self.gen_samples
        
        for i in range(n, total_len - X + 1, self.stride):
            ppg_win = ppg_signal[i : i + X]
            ecg_past = ecg_signal[i - n : i + X - n]
            target_sec = ecg_signal[i + X - n : i + X]
            
            if np.std(ppg_win) > 1e-3:
                ppg_list.append(ppg_win)
                ecg_past_list.append(ecg_past)
                target_list.append(target_sec)

        return np.array(ppg_list), np.array(ecg_past_list), np.array(target_list)

    def process_subject(self, ppg_raw, ecg_raw, configs, is_training=False):
        # 1. Filtri & Norm
        ppg_f = self.apply_bandpass_filter(ppg_raw, 0.5, 8.0)
        ecg_f = self.apply_bandpass_filter(ecg_raw, 0.5, 30.0 if configs.get('apply_ecg_filter', True) else 8.0)
        
        if configs.get('normalize_01', False):
            ppg_n, ecg_n = self.normalize_min_max(ppg_f), self.normalize_min_max(ecg_f)
        else:
            ppg_n, ecg_n = self.normalize_signal(ppg_f), self.normalize_signal(ecg_f)

        # 2. Segmentazione
        beats = self.segment_autoregressive(ppg_n, ecg_n)
        if len(beats[0]) == 0: return None
        ppg_beats, ecg_past_beats, target_beats = beats

        # 3. Augmentation
        if is_training and configs.get('apply_augmentation', False):
            # ... (logica augmentation come prima) ...
            aug_ppg, aug_target, aug_past = [], [], []
            for i in range(len(ppg_beats)):
                p, t = self.apply_augmentation(ppg_beats[i], target_beats[i], configs)
                e_pst = ecg_past_beats[i].copy()
                if configs.get('apply_context_augmentation', True):
                    e_pst = self.apply_context_corruption(e_pst, configs)
                aug_ppg.append(p); aug_target.append(t); aug_past.append(e_pst)
            ppg_beats, target_beats, ecg_past_beats = np.array(aug_ppg), np.array(aug_target), np.array(aug_past)

        # 4. WAVELET SCATTERING (Nuovo Step)
        if configs.get('apply_wst', False):
            # Applichiamo WST a PPG e ECG Past
            # Input: (N, 875) -> Output: (N, 19, 218) (circa)
            ppg_beats = self.extract_wst_features(ppg_beats)
            ecg_past_beats = self.extract_wst_features(ecg_past_beats)
            
            # Nota: NON applichiamo WST al target. Il target deve rimanere il segnale grezzo da ricostruire.

        return {
            'ppg': ppg_beats,
            'ecg_past': ecg_past_beats,
            'target': target_beats
        }