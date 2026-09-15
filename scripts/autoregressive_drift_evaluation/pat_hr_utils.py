"""
Utilita' per Pulse Arrival Time (PAT) ed Heart Rate (HR) sui segmenti generati
dal test di drift a finestre (run_windowed_drift_test.py).

Definizioni operative (vedi discussione col committente):
  - R-peak ECG: rilevati con neurokit2 (piu' robusto quando l'ECG generato si
    appiattisce a orizzonti lunghi), con fallback a scipy.
  - Picco PPG: picco sistolico, rilevato con scipy (la PPG e' SQI-pulita per
    costruzione, quindi il rilevamento e' affidabile e veloce).
  - PAT (per battito) = tempo dal picco R al PICCO PPG che lo segue.
      * PAT_real usa gli R dell'ECG reale, PAT_gen quelli dell'ECG generato,
        accoppiati sullo STESSO picco PPG (stesso ciclo cardiaco).
      * Si tengono solo i battiti con PAT in range fisiologico [50, 500] ms.
  - HR istantaneo = 60 / RR (dagli intervalli tra R-peak consecutivi).
  - Bias temporale = differenza media (ms) tra dove il picco R si trova
    realmente e dove e' stato generato; equivale a mean(PAT_real - PAT_gen).
"""
import warnings
import numpy as np
from scipy.signal import find_peaks

DEFAULT_MIN_PAT_MS = 50.0
DEFAULT_MAX_PAT_MS = 550.0
HR_MIN_BPM = 30.0
HR_MAX_BPM = 220.0


def _znorm(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / (s + 1e-8)


def detect_r_peaks(ecg, fs):
    """Indici (campioni) dei picchi R. neurokit2 con fallback scipy."""
    ecg = np.nan_to_num(np.asarray(ecg, dtype=np.float64))
    if ecg.std() < 1e-6:
        return np.array([], dtype=int)
    try:
        import neurokit2 as nk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_peaks(_znorm(ecg), sampling_rate=fs)
        pk = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        pk = pk[(pk >= 0) & (pk < len(ecg))]
        if len(pk) >= 2:
            return pk
    except Exception:
        pass
    pk, _ = find_peaks(_znorm(ecg), distance=int(fs * 0.35), height=1.0)
    return pk.astype(int)


def _ppg_peaks_pol(ppg, fs, pol):
    p = _znorm(ppg) * pol
    pk, _ = find_peaks(p, distance=int(fs * 0.45), prominence=0.25)
    return pk.astype(int)


def _count_valid_preceding(ppg_peaks, r_ref, fs, min_pat_ms, max_pat_ms):
    r_ref = np.asarray(r_ref)
    mn = min_pat_ms / 1000.0 * fs
    mx = max_pat_ms / 1000.0 * fs
    c = 0
    for p in ppg_peaks:
        if np.any((r_ref <= p - mn) & (r_ref >= p - mx)):
            c += 1
    return c


def detect_ppg_peaks(ppg, fs, r_ref=None,
                     min_pat_ms=DEFAULT_MIN_PAT_MS, max_pat_ms=DEFAULT_MAX_PAT_MS):
    """Indici (campioni) dei picchi sistolici PPG. La polarita' della PLETH e'
    mista nel dataset (spesso invertita): se vengono forniti gli R di
    riferimento (r_ref, tipicamente quelli dell'ECG reale), la polarita' viene
    scelta PER-SEGMENTO come quella che massimizza i picchi PPG preceduti da un
    R in range fisiologico. Senza r_ref si ripiega sull'asimmetria (skewness).
    """
    ppg = np.nan_to_num(np.asarray(ppg, dtype=np.float64))
    if ppg.std() < 1e-6:
        return np.array([], dtype=int)

    if r_ref is not None and len(r_ref) >= 2:
        best_pk, best_c = None, -1
        for pol in (1, -1):
            pk = _ppg_peaks_pol(ppg, fs, pol)
            c = _count_valid_preceding(pk, r_ref, fs, min_pat_ms, max_pat_ms)
            if c > best_c:
                best_c, best_pk = c, pk
        return best_pk if best_pk is not None else np.array([], dtype=int)

    from scipy.stats import skew
    pol = -1 if skew(_znorm(ppg)) < 0 else 1
    return _ppg_peaks_pol(ppg, fs, pol)


def extract_pat(ecg_true, ecg_gen, ppg, fs,
                min_pat_ms=DEFAULT_MIN_PAT_MS, max_pat_ms=DEFAULT_MAX_PAT_MS):
    """Pipeline completa per un segmento: rileva R (reale/generato) e picchi
    PPG (polarita' calibrata sugli R reali), poi accoppia i PAT sullo stesso
    picco PPG. Ritorna (r_real, r_gen, beat_times_sec, pat_real_ms, pat_gen_ms)."""
    r_real = detect_r_peaks(ecg_true, fs)
    r_gen = detect_r_peaks(ecg_gen, fs)
    ppg_peaks = detect_ppg_peaks(ppg, fs, r_ref=r_real,
                                 min_pat_ms=min_pat_ms, max_pat_ms=max_pat_ms)
    bt, pr, pg = compute_pat_pairs(r_real, r_gen, ppg_peaks, fs, min_pat_ms, max_pat_ms)
    return r_real, r_gen, bt, pr, pg


def compute_pat_pairs(r_real, r_gen, ppg_peaks, fs,
                      min_pat_ms=DEFAULT_MIN_PAT_MS, max_pat_ms=DEFAULT_MAX_PAT_MS):
    """Per ogni picco PPG cerca il picco R (reale e generato) che lo precede
    entro [min_pat_ms, max_pat_ms]. Ritorna (beat_times_sec, pat_real_ms,
    pat_gen_ms) con SOLO i battiti in cui entrambi gli R sono presenti e in
    range (stesso ciclo cardiaco -> accoppiamento naturale via picco PPG)."""
    r_real = np.asarray(r_real)
    r_gen = np.asarray(r_gen)
    min_s = min_pat_ms / 1000.0 * fs
    max_s = max_pat_ms / 1000.0 * fs

    bt, pr, pg = [], [], []
    for p in ppg_peaks:
        cr = r_real[(r_real <= p - min_s) & (r_real >= p - max_s)]
        cg = r_gen[(r_gen <= p - min_s) & (r_gen >= p - max_s)]
        if len(cr) == 0 or len(cg) == 0:
            continue
        tr = cr.max()   # R reale piu' vicino che precede il picco PPG
        tg = cg.max()   # R generato piu' vicino
        bt.append(p / fs)
        pr.append((p - tr) / fs * 1000.0)
        pg.append((p - tg) / fs * 1000.0)
    return np.array(bt), np.array(pr), np.array(pg)


def _hr_curve(r_peaks, fs):
    """(t_mid_sec, hr_bpm) dell'HR istantaneo dagli intervalli RR, filtrato
    al range fisiologico. None se troppo pochi picchi."""
    r = np.asarray(r_peaks)
    if len(r) < 2:
        return None
    t = r / fs
    rr = np.diff(t)
    hr = 60.0 / np.where(rr > 1e-6, rr, np.nan)
    t_mid = (t[:-1] + t[1:]) / 2.0
    ok = np.isfinite(hr) & (hr >= HR_MIN_BPM) & (hr <= HR_MAX_BPM)
    if ok.sum() < 1:
        return None
    return t_mid[ok], hr[ok]


def hr_error_per_second(r_real, r_gen, fs, n_windows, seed_sec=6, gen_sec=1):
    """|HR_gen - HR_real| valutato al centro di ogni secondo generato
    (indice j -> centro a seed_sec + (j+0.5)*gen_sec). Lista lunga n_windows;
    NaN dove l'HR non e' stimabile (verra' escluso in aggregazione)."""
    cr = _hr_curve(r_real, fs)
    cg = _hr_curve(r_gen, fs)
    out = []
    for j in range(n_windows):
        center = seed_sec + (j + 0.5) * gen_sec
        if cr is None or cg is None:
            out.append(float("nan"))
            continue
        hr_r = float(np.interp(center, cr[0], cr[1]))
        hr_g = float(np.interp(center, cg[0], cg[1]))
        out.append(abs(hr_g - hr_r))
    return out


def pat_bias_ms(pat_real_ms, pat_gen_ms):
    """Bias (ms) da AGGIUNGERE a PAT_gen per rimuovere lo scostamento
    sistematico: mean(PAT_real - PAT_gen). Positivo = R generati in ritardo
    (PAT_gen sistematicamente piu' corto del reale)."""
    d = np.asarray(pat_real_ms) - np.asarray(pat_gen_ms)
    return float(np.mean(d)) if len(d) else 0.0
