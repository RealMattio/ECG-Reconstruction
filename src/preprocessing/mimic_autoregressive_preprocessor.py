import numpy as np
import torch
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew  # <--- Import necessario
from kymatio.torch import Scattering1D
import gc 

class MimicAutoregressivePreprocessor:
    def __init__(self, fs=125, window_sec=7, gen_sec=1, stride_sec=None):
        self.fs = fs
        self.window_sec = window_sec
        self.gen_sec = gen_sec
        self.total_samples = int(fs * window_sec)
        self.gen_samples = int(fs * gen_sec)
        if stride_sec is None:
            self.stride = self.gen_samples
        else:
            self.stride = int(fs * stride_sec)
        self.scattering = None

    def _init_wst(self):
        if self.scattering is None:
            self.scattering = Scattering1D(J=2, shape=(self.total_samples,), Q=8)

    def apply_bandpass_filter(self, signal, lowcut, highcut, order=4):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        if high >= 1.0: high = 0.99
        if low <= 0.0: low = 0.001
        b, a = butter(order, [low, high], btype='bandpass')
        if np.isnan(signal).any():
            clean_sig = np.nan_to_num(signal, nan=np.nanmean(signal))
            return filtfilt(b, a, clean_sig)
        return filtfilt(b, a, signal)

    def normalize_local_min_max(self, signal):
        s_min, s_max = np.min(signal), np.max(signal)
        if s_max - s_min < 1e-6: 
            return np.zeros_like(signal)
        return (signal - s_min) / (s_max - s_min)

    # ... Metodi augmentation ...
    def apply_augmentation(self, ppg_seg, ecg_target, configs):
        aug_ppg = ppg_seg.copy()
        if configs.get('aug_random_gain', False):
            aug_ppg *= np.random.uniform(0.8, 1.2)
        if configs.get('aug_additive_noise', False):
            noise = np.random.normal(0, 0.01, aug_ppg.shape)
            aug_ppg += noise
        return aug_ppg, ecg_target

    def apply_context_corruption(self, ecg_past, configs):
        noise = np.random.normal(0, configs.get('aug_context_noise', 0.05), ecg_past.shape)
        return ecg_past + noise

    def extract_wst_features(self, batch_signals):
        self._init_wst()
        if isinstance(batch_signals, np.ndarray):
            tensor_signals = torch.from_numpy(batch_signals).float()
        else:
            tensor_signals = batch_signals.float()
        with torch.no_grad():
            wst_out = self.scattering(tensor_signals)
        return torch.nan_to_num(wst_out, nan=0.0).numpy()

    # --- METODO DI VALIDAZIONE SQI ---
    def validate_segment_sqi(self, full_ecg_window, ppg_win):
        """
        Valida il segmento usando metriche statistiche (Kurtosi, Skewness).
        Ritorna True se valido, False se rumore/artefatto.
        """
        # 1. Kurtosi ECG (Deve essere > 5.0 per avere picchi QRS chiari)
        # Il rumore gaussiano ha kurtosi ~3.0
        k_ecg = kurtosis(full_ecg_window, fisher=False) 
        if k_ecg < 5.0:
            return False

        # 2. Skewness PPG (Deve essere > 0.2, onda asimmetrica)
        # Il rumore di fondo tende a essere simmetrico (skew ~0)
        s_ppg = skew(ppg_win)
        if s_ppg < 0.2:
            return False

        return True

    # --- SEGMENTAZIONE CON VALIDAZIONE ---
    def segment_autoregressive(self, ppg_signal, ecg_signal, do_normalize):
        ppg_list, ecg_past_list, target_list = [], [], []
        total_len = len(ppg_signal)
        X, n = self.total_samples, self.gen_samples
        
        for i in range(n, total_len - X + 1, self.stride):
            # Creiamo viste temporanee
            ppg_win = ppg_signal[i : i + X]
            target_sec = ecg_signal[i + X - n : i + X]
            ecg_past = ecg_signal[i - n : i + X - n] # Recuperiamo anche il passato per il check completo
            
            # --- VALIDAZIONE 1: CONTROLLI DI BASE (Flatline) ---
            if np.std(ppg_win) < 1e-4 or np.std(target_sec) < 1e-4:
                del ppg_win, target_sec, ecg_past
                continue 

            # --- VALIDAZIONE 2: SQA AVANZATA (Kurtosi & Skewness) ---
            # Uniamo passato e futuro per valutare la qualità dell'ECG su tutta la finestra temporale rilevante
            # Usiamo np.concatenate che crea una copia temporanea
            full_ecg_check = np.concatenate([ecg_past, target_sec])
            
            is_valid = self.validate_segment_sqi(full_ecg_check, ppg_win)
            
            # Puliamo subito l'array temporaneo di check
            del full_ecg_check

            if not is_valid:
                # Se non passa i controlli di qualità, scartiamo e liberiamo
                del ppg_win, target_sec, ecg_past
                continue

            # --- SALVATAGGIO ---
            # Normalizzazione
            if do_normalize:
                ppg_win = self.normalize_local_min_max(ppg_win)
                ecg_past = self.normalize_local_min_max(ecg_past)
                target_sec = self.normalize_local_min_max(target_sec)
            
            # Usiamo .copy() per staccare dalla memoria del segnale padre
            ppg_list.append(ppg_win.copy() if not do_normalize else ppg_win)
            ecg_past_list.append(ecg_past.copy() if not do_normalize else ecg_past)
            target_list.append(target_sec.copy() if not do_normalize else target_sec)

        return np.array(ppg_list), np.array(ecg_past_list), np.array(target_list)

    def process_subject(self, ppg_raw, ecg_raw, configs, is_training=False):
        # 1. Filtri (Sovrascriviamo per liberare memoria dei 'raw')
        ppg_raw = self.apply_bandpass_filter(ppg_raw, 0.5, 5.0)
        ecg_raw = self.apply_bandpass_filter(ecg_raw, 0.5, 40.0)

        # 2. Segmentazione
        do_norm = configs.get('normalize_01', True)
        
        beats = self.segment_autoregressive(ppg_raw, ecg_raw, do_norm)
        
        # ORA possiamo liberare immediatamente i segnali interi filtrati!
        del ppg_raw, ecg_raw
        
        if len(beats[0]) == 0: 
            return None
        
        ppg_beats, ecg_past_beats, target_beats = beats

        # 3. Augmentation (Solo Training)
        if is_training and configs.get('apply_augmentation', False):
            aug_ppg, aug_target, aug_past = [], [], []
            
            for i in range(len(ppg_beats)):
                p, t = self.apply_augmentation(ppg_beats[i], target_beats[i], configs)
                e_pst = ecg_past_beats[i].copy()
                if configs.get('apply_context_augmentation', True):
                    e_pst = self.apply_context_corruption(e_pst, configs)
                
                aug_ppg.append(p); aug_target.append(t); aug_past.append(e_pst)
            
            ppg_beats = np.array(aug_ppg)
            target_beats = np.array(aug_target)
            ecg_past_beats = np.array(aug_past)
            
            del aug_ppg, aug_target, aug_past

        # 4. WST
        if configs.get('apply_wst', False):
            ppg_beats = self.extract_wst_features(ppg_beats)
            
        return {
            'ppg': ppg_beats,
            'ecg_past': ecg_past_beats,
            'target': target_beats
        }