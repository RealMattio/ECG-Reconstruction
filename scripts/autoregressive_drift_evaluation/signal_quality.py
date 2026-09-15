"""
Maschera di qualita' del segnale per la valutazione del drift autoregressivo.

Il modello (src/preprocessing/mimic_autoregressive_preprocessor.py,
MimicAutoregressivePreprocessor.validate_segment_sqi) e' allenato e valutato
OVUNQUE nel progetto solo su finestre che superano un controllo SQI:
kurtosi ECG >= 5.0 (un ECG pulito ha picchi QRS netti -> kurtosi alta;
rumore gaussiano/artefatti hanno kurtosi ~3) e skewness PPG >= 0.2 (un
polso pletismografico pulito e' asimmetrico; rumore di fondo e' ~simmetrico).

La pipeline di drift evaluation (build_drift_manifest.py +
run_autoregressive_drift_test.py) NON applicava questo controllo: prende il
segnale WFDB grezzo e continuo dal secondo 0 in poi, quindi puo' includere
ore di disconnessioni/artefatti da movimento/saturazione mai viste dal
modello in nessun'altra valutazione. Questo modulo replica lo stesso
criterio per costruire, a posteriori, una maschera booleana lungo l'intera
generazione: True = il ground truth in quel punto e' abbastanza pulito da
essere un riferimento valido per misurare il drift del modello.
"""
import numpy as np

DEFAULT_KURTOSIS_THR = 5.0
DEFAULT_SKEW_THR = 0.2
DEFAULT_WINDOW_SEC = 7.0  # = x_sec del modello (stesso contesto usato in training/generazione)


def compute_quality_mask(
    ecg_full: np.ndarray,
    ppg_full: np.ndarray,
    fs: int,
    window_sec: float = DEFAULT_WINDOW_SEC,
    kurtosis_thr: float = DEFAULT_KURTOSIS_THR,
    skew_thr: float = DEFAULT_SKEW_THR,
) -> np.ndarray:
    """Ritorna una maschera booleana a risoluzione di campione (stessa
    lunghezza di ecg_full/ppg_full). mask[t] = True se il secondo a cui
    appartiene il campione t e' "pulito": kurtosi(ECG) >= kurtosis_thr e
    skewness(PPG) >= skew_thr sulla finestra di window_sec secondi che
    termina in quel punto (stesso contesto temporale x_sec usato dal
    modello). I primi window_sec secondi (senza abbastanza contesto
    pregresso) sono marcati non validi per costruzione.

    Implementazione O(N) via momenti rolling su cumsum (mean/var/skew/
    kurtosis per ogni finestra derivati da sum(x^k) senza mai materializzare
    una matrice (n_finestre x campioni_finestra)): su un record di 24h porta
    il calcolo da ~1 minuto (loop) a <1 secondo, con un footprint di memoria
    di poche centinaia di MB anche per le run piu' lunghe — necessario per
    poterlo applicare a tutti i ~790 record del test set anche su macchine
    con poca RAM disponibile.
    """
    total_len = min(len(ecg_full), len(ppg_full))
    win_samples = int(round(window_sec * fs))
    step_samples = fs  # un giudizio di qualita' per ogni secondo, come i generation-step del modello

    mask = np.zeros(total_len, dtype=bool)
    if total_len < win_samples:
        return mask

    def _rolling_moments(x, W, step):
        c1 = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
        c2 = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
        c3 = np.concatenate(([0.0], np.cumsum(x ** 3, dtype=np.float64)))
        c4 = np.concatenate(([0.0], np.cumsum(x ** 4, dtype=np.float64)))
        ends = np.arange(W, len(x) + 1, step)
        starts = ends - W
        s1 = c1[ends] - c1[starts]
        s2 = c2[ends] - c2[starts]
        s3 = c3[ends] - c3[starts]
        s4 = c4[ends] - c4[starts]
        mean = s1 / W
        m2 = s2 / W - mean ** 2
        m3 = s3 / W - 3 * mean * s2 / W + 2 * mean ** 3
        m4 = s4 / W - 4 * mean * s3 / W + 6 * (mean ** 2) * s2 / W - 3 * mean ** 4
        return m2, m3, m4

    ecg = ecg_full[:total_len].astype(np.float64, copy=False)
    ppg = ppg_full[:total_len].astype(np.float64, copy=False)

    ecg_m2, _, ecg_m4 = _rolling_moments(ecg, win_samples, step_samples)
    ppg_m2, ppg_m3, _ = _rolling_moments(ppg, win_samples, step_samples)

    # varianze numericamente negative per errori di floating point su
    # finestre quasi costanti: clip a 0 prima di usarle come denominatore
    ecg_m2 = np.maximum(ecg_m2, 0.0)
    ppg_m2 = np.maximum(ppg_m2, 0.0)
    std_ok = (np.sqrt(ecg_m2) >= 1e-4) & (np.sqrt(ppg_m2) >= 1e-4)

    with np.errstate(divide='ignore', invalid='ignore'):
        k_ecg = np.where(ecg_m2 > 0, ecg_m4 / (ecg_m2 ** 2), 0.0)
        s_ppg = np.where(ppg_m2 > 0, ppg_m3 / (ppg_m2 ** 1.5), 0.0)

    is_valid = std_ok & (k_ecg >= kurtosis_thr) & (s_ppg >= skew_thr)

    valid_repeated = np.repeat(is_valid, step_samples)
    fill_start = win_samples - step_samples
    fill_end = min(fill_start + len(valid_repeated), total_len)
    mask[fill_start:fill_end] = valid_repeated[: fill_end - fill_start]

    return mask


def quality_fraction(mask: np.ndarray, start: int = 0, end: int | None = None) -> float:
    """Frazione di campioni validi in mask[start:end] (1.0 = tutto pulito)."""
    segment = mask[start:end]
    if len(segment) == 0:
        return 0.0
    return float(segment.mean())


DEFAULT_SEED_SEC = 6.0  # deve combaciare con il seed usato in generazione (fast_autoregressive_inference)


def splice_clean_signal(
    ecg_full: np.ndarray,
    ppg_full: np.ndarray,
    fs: int,
    window_sec: float = DEFAULT_WINDOW_SEC,
    kurtosis_thr: float = DEFAULT_KURTOSIS_THR,
    skew_thr: float = DEFAULT_SKEW_THR,
    seed_sec: float = DEFAULT_SEED_SEC,
):
    """Rimuove interamente i tratti che non superano il controllo SQI e
    concatena i tratti puliti rimasti, nell'ordine originale, producendo un
    nuovo segnale ECG/PPG piu' corto ma interamente "pulito" secondo il
    criterio SQI. A differenza di compute_quality_mask (che si limita a
    PESARE/escludere le finestre rumorose in fase di valutazione), qui il
    segnale rumoroso viene tolto a monte, prima della generazione: la
    ricostruzione autoregressiva verra' quindi eseguita solo su input
    realmente puliti.

    NOTA: i primi `seed_sec` (6s) del segnale non hanno contesto sufficiente
    per un giudizio SQI (serve una finestra di window_sec=7s di storia) e
    coincidono comunque con il seed reale usato dalla generazione (mai
    passato al modello come proprio output, quindi non e' necessario
    giudicarne la qualita' con lo stesso criterio): vengono trattati come
    validi per costruzione, cosi' un record perfettamente pulito raggiunge
    esattamente la stessa durata piena che raggiungeva prima dello splicing
    (altrimenti l'orizzonte piu' corto, "1m" = 66s, sarebbe irraggiungibile
    anche per un segnale perfetto). Il resto del segnale (da seed_sec in poi)
    usa i giudizi SQI reali calcolati da compute_quality_mask.

    NOTA 2: ogni punto di giunzione fra due tratti puliti non adiacenti nel
    segnale originale introduce una discontinuita' artificiale (il campione
    N e il campione N+1 del segnale risultante possono provenire da istanti
    temporali anche molto lontani nella registrazione originale). E' un
    limite intrinseco di questo approccio, non eliminabile.

    Ritorna (ecg_clean, ppg_clean, stats) dove stats e' un dict con
    n_kept_samples, n_total_samples, kept_fraction, n_splices.
    """
    mask = compute_quality_mask(ecg_full, ppg_full, fs, window_sec, kurtosis_thr, skew_thr)
    total_len = min(len(ecg_full), len(ppg_full))
    seed_samples = int(round(seed_sec * fs))

    mask = mask[:total_len].copy()
    mask[:seed_samples] = True  # prefisso non giudicabile == seed, trattato come valido

    ecg_clean = ecg_full[:total_len][mask]
    ppg_clean = ppg_full[:total_len][mask]

    # Conta i punti di giunzione: transizioni False->True nella maschera
    # (ogni tratto pulito che INIZIA e' una nuova giunzione, tranne il
    # primissimo che parte gia' valido per via del seed forzato a True).
    transitions = np.diff(mask.astype(np.int8))
    n_splices = max(0, int(np.sum(transitions == 1)) - 1)

    stats = {
        "n_kept_samples": int(len(ecg_clean)),
        "n_total_samples": int(total_len),
        "kept_fraction": float(mask.mean()) if total_len > 0 else 0.0,
        "n_splices": n_splices,
    }

    return ecg_clean, ppg_clean, stats
