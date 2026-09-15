"""
Segmentazione SQI a finestre per la valutazione del drift autoregressivo
(NUOVO paradigma, vedi run_windowed_drift_test.py).

Differenza rispetto a signal_quality.py (approccio "splicing"):
  - signal_quality.splice_clean_signal RIMUOVE i tratti rumorosi e INCOLLA i
    tratti puliti rimasti in un unico segnale. Ogni giunzione introduce un
    gradino artificiale (discontinuita' di ampiezza e di fase cardiaca) che
    non e' mai stato visto dal modello: genera rumore nell'input e altera
    la frequenza cardiaca.
  - Qui invece NON si incolla nulla. Il segnale viene scandito in finestre di
    `win_sec` secondi e si conservano SOLO i tratti in cui piu' finestre
    consecutive superano l'SQI, ciascuno come segmento INDIPENDENTE. Si
    ottengono cosi' tanti input di lunghezza diversa (>= min_seg_sec), ognuno
    interamente contiguo nel tempo originale, senza alcuna giunzione.

Criterio SQI (allineato a scripts/preprocess_mimic_pinn_manifest.py, la stessa
logica usata per costruire il training set):
  - PPG: deve essere ESTREMAMENTE buona su OGNI finestra del segmento (soglia
    piu' severa che in addestramento — vedi ppg_thr, default 0.925 vs 0.9 del
    preprocessing). E' l'input principale del modello lungo tutto l'orizzonte.
  - ECG: valutato SOLO sui primi `seed_sec` secondi del segmento (i 6s di seed
    reale dati in pasto al modello). Oltre il seed l'ECG puo' anche essere
    rumoroso: e' il ground truth con cui si misura il drift, non un input.
"""
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, pearsonr

# --- Soglie di default (coerenti con il preprocessing PINN) ---
DEFAULT_WIN_SEC = 4.0        # finestra di valutazione SQI
DEFAULT_STEP_SEC = 1.0       # passo di scorrimento (finestre 4s sovrapposte)
DEFAULT_PPG_THR = 0.925      # PPG molto piu' severa che in training (0.9)
DEFAULT_ECG_THR = 0.9        # come in training, ma solo sul seed
DEFAULT_SPECTRAL_THR = 0.5   # gate spettrale interno al giudizio PPG
DEFAULT_KURTOSIS_THR = 5.0   # gate di kurtosi interno al giudizio ECG
DEFAULT_SEED_SEC = 6.0       # seed reale = primi 6s (deve combaciare con la generazione)
DEFAULT_MIN_SEG_SEC = 7.0    # segmento minimo utile (6s seed + >=1s da generare)


# ─────────────────────────────────────────────────────────────
# GIUDIZI DI QUALITA' (identici a preprocess_mimic_pinn_manifest.py)
# I segnali in ingresso sono gia' filtrati a monte (band-pass).
# ─────────────────────────────────────────────────────────────

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


def ppg_quality(ppg_signal, fs, spectral_thr=DEFAULT_SPECTRAL_THR):
    """Ritorna (score in [0,1], is_inverted). score 0.0 = scarto immediato."""
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


def ecg_quality(ecg_signal, fs, kurtosis_thr=DEFAULT_KURTOSIS_THR):
    """Ritorna uno score in [0,1]. 0.0 = scarto immediato."""
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
# SEGMENTAZIONE
# ─────────────────────────────────────────────────────────────

def find_clean_segments(
    ecg_filt: np.ndarray,
    ppg_filt: np.ndarray,
    fs: int,
    win_sec: float = DEFAULT_WIN_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    ppg_thr: float = DEFAULT_PPG_THR,
    ecg_thr: float = DEFAULT_ECG_THR,
    seed_sec: float = DEFAULT_SEED_SEC,
    min_seg_sec: float = DEFAULT_MIN_SEG_SEC,
):
    """Individua i segmenti CONTIGUI in cui la PPG supera l'SQI su tutte le
    finestre e l'ECG del seed iniziale e' pulito. NON incolla nulla: ogni
    segmento e' un intervallo continuo del segnale originale.

    Passi:
      1. Scandisce finestre di `win_sec` (passo `step_sec`); marca quelle in cui
         ppg_quality >= ppg_thr.
      2. Fonde le finestre PPG-pulite adiacenti in intervalli contigui.
      3. Tiene solo gli intervalli lunghi >= min_seg_sec il cui seed iniziale
         (primi seed_sec) supera ecg_quality >= ecg_thr.

    Ritorna (segments, stats):
      segments: lista di dict {start, end, ppg_inverted, seed_ecg_score}
                (start/end in campioni; end esclusivo).
      stats:    dict con conteggi utili al logging.
    """
    n = int(min(len(ecg_filt), len(ppg_filt)))
    win_samples = int(round(win_sec * fs))
    step_samples = int(round(step_sec * fs))
    seed_samples = int(round(seed_sec * fs))
    min_samples = int(round(min_seg_sec * fs))

    stats = {
        "n_ppg_windows_ok": 0,
        "n_ppg_intervals": 0,
        "n_seg_too_short": 0,
        "n_seg_bad_ecg_seed": 0,
        "n_segments": 0,
    }

    if n < win_samples:
        return [], stats

    # --- 1. Finestre PPG che superano l'SQI ---
    ppg_intervals = []
    inversions = {}  # (start,end) -> is_inverted
    for start in range(0, n - win_samples + 1, step_samples):
        end = start + win_samples
        score, is_inv = ppg_quality(ppg_filt[start:end], fs)
        if score >= ppg_thr:
            ppg_intervals.append((start, end))
            inversions[(start, end)] = is_inv
            stats["n_ppg_windows_ok"] += 1

    if not ppg_intervals:
        return [], stats

    merged = _merge_intervals(ppg_intervals)
    stats["n_ppg_intervals"] = len(merged)

    # --- 2/3. Filtro lunghezza minima + qualita' ECG del seed ---
    segments = []
    for s, e in merged:
        if (e - s) < min_samples:
            stats["n_seg_too_short"] += 1
            continue

        seed_score = ecg_quality(ecg_filt[s:s + seed_samples], fs)
        if seed_score < ecg_thr:
            stats["n_seg_bad_ecg_seed"] += 1
            continue

        # Polarita' PPG dominante sulle finestre che coprono questo segmento
        inv_flags = [inversions[iv] for iv in inversions if iv[0] >= s and iv[1] <= e]
        ppg_inverted = bool(np.mean(inv_flags) > 0.5) if inv_flags else False

        segments.append({
            "start": int(s),
            "end": int(e),
            "ppg_inverted": ppg_inverted,
            "seed_ecg_score": float(seed_score),
        })

    stats["n_segments"] = len(segments)
    return segments, stats
