import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class BidmcPreprocessor:
    def __init__(self, fs=125, beat_len=120):
        self.fs = fs # Frequenza campionamento MIMIC II [cite: 313]
        self.beat_len = beat_len # 120 campioni [cite: 318]

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

    def segment_into_beats(self, ppg_signal, ecg_signal, r_peaks):
        """Segmentazione centrata sui picchi R (Fase 1: No overlap)."""
        ppg_beats, ecg_beats = [], []
        half_len = self.beat_len // 2
        for peak in r_peaks:
            start, end = peak - half_len, peak + half_len
            if start >= 0 and end <= len(ppg_signal):
                ppg_beats.append(ppg_signal[start:end])
                ecg_beats.append(ecg_signal[start:end])
        return np.array(ppg_beats), np.array(ecg_beats)

    def segment_with_overlap(self, ppg_signal, ecg_signal, overlap_pct=0.5):
        """Segmentazione con sliding window per continuità (Fase 2: Overlap)."""
        ppg_segments, ecg_segments = [], []
        step = int(self.beat_len * (1 - overlap_pct))
        for i in range(0, len(ppg_signal) - self.beat_len, step):
            p_seg, e_seg = ppg_signal[i:i+self.beat_len], ecg_signal[i:i+self.beat_len]
            if np.std(p_seg) > 1e-3: # Rimuove segmenti silenti
                ppg_segments.append(p_seg)
                ecg_segments.append(e_seg)
        return np.array(ppg_segments), np.array(ecg_segments)

    def process_subject(self, ppg_raw, ecg_raw, configs):
        """
        Orchestra il preprocessing in base alle configs.
        """
        # Preprocessing base
        ppg_filtered = self.apply_bandpass_filter(ppg_raw)
        ppg_norm = self.normalize_signal(ppg_filtered)
        ecg_norm = self.normalize_signal(ecg_raw)

        # Scelta della strategia di segmentazione
        if configs.get('overlap_windows', False):
            # Modalità Overlap: finestra scorrevole
            print("INFO: Segmentazione con Overlap attivata.")
            return self.segment_with_overlap(ppg_norm, ecg_norm, overlap_pct=0.5)
        else:
            # Modalità Standard: beat-by-beat centrato su R [cite: 86]
            print("INFO: Segmentazione Beat-by-Beat standard.")
            peaks = self.detect_r_peaks(ecg_norm)
            return self.segment_into_beats(ppg_norm, ecg_norm, peaks)