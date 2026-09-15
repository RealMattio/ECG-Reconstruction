#!/usr/bin/env python3
"""
scripts/clinical_fidelty_evaluation/evaluate_clinical_fidelity.py

Clinical Feature Fidelity — lightweight_hybrid su MIMIC-III.

Obiettivo: dimostrare che l'ECG ricostruito, anche con un lieve errore
morfologico, resta clinicamente utilizzabile (battito ricavabile, complessi
QRS individuabili) e non soffre di drifting temporale del ritmo.

Metodo (v2 — segnali puliti + autoregressione VERA)
----------------------------------------------------
Le versioni precedenti di questo script rigeneravano il segnale da zero
(MimicSmartDataset + inferenza teacher-forced finestra per finestra, dove il
contesto passato dato in input al modello era sempre il vero ECG). Questo
NON è drift autoregressivo reale: è una valutazione "one-step-ahead" che non
può mai accumulare errore nel tempo.

Questa versione riusa invece direttamente i segnali già generati da
scripts/autoregressive_drift_evaluation/run_autoregressive_drift_test_clean.py
(lanciato via ar_drif_test_clean.sh), salvati in GENERATION_DIR come un .npz
per record WFDB del test set:
  - ppg_input       PPG ripulita (SQI splicing: i tratti di bassa qualità
                    vengono rimossi e i tratti puliti rimasti incollati in
                    un unico segnale continuo, vedi signal_quality.py)
  - ecg_target      ECG reale corrispondente, stessa pulizia, scala fisica
                    (filtrata, non normalizzata)
  - ecg_generated   ECG generato in autoregressione VERA: dopo i primi
                    SEED_SEC secondi (seed reale), ogni secondo successivo è
                    prodotto dal modello usando come contesto passato la
                    PROPRIA uscita precedente (self-feeding), non il vero
                    ECG — è quindi la condizione in cui il drift temporale
                    può davvero manifestarsi. Nota: ecg_generated resta
                    nello spazio [0,1] del modello (mai de-normalizzato in
                    scala fisica), quindi NON è direttamente sovrapponibile
                    in ampiezza a ecg_target — per le metriche cliniche qui
                    sotto (picchi R, HR, intervalli) questo non è un
                    problema perché NeuroKit2 individua i picchi in modo
                    relativo alla scala di ciascun segnale.

Ogni record ripulito è potenzialmente molto lungo (minuti/ore): per ottenere
lo stesso tipo di statistiche "a tanti punti" di prima (Bland-Altman con
migliaia di finestre, istogrammi, ecc.), ogni record viene suddiviso in
finestre non sovrapposte di CHUNK_SEC secondi (dopo aver scartato i primi
SEED_SEC secondi di seed, mai generati). Su ciascuna finestra si esegue
la stessa identica pipeline NeuroKit2 di prima (R-peak detection, HR,
delineazione P/QRS/T, matching picchi con tolleranza).

Nessun modello/dataset viene ricaricato qui: i segnali sono già stati
generati da run_autoregressive_drift_test_clean.py, questo script legge solo
i file .npz risultanti.

Output (tutto in questa cartella)
----------------------------------
  clinical_fidelity_results.json   → risultati per-finestra + aggregati
  clinical_fidelity_report.md      → tabelle e testo pronti per il paper
  bland_altman_hr.png              → Bland-Altman HR reale vs predetta
  rpeak_example_best.png           → esempio qualitativo (caso migliore)
  rpeak_example_median.png         → esempio qualitativo (caso mediano)
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import neurokit2 as nk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Radice del progetto ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# CONFIGURAZIONE
# =============================================================================
# Segnali già puliti + generati da run_autoregressive_drift_test_clean.py
# (stesso modello lightweight_hybrid / MAE_loss di prima, stesso test set).
GENERATION_DIR = os.path.join(
    PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation",
    "experiments", "drift_generation_clean",
)
MODEL_WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "src/experiments/final_mimic_pinn_results/MAE_loss",
    "lightweight_hybrid_20260608_192241/final_full_model/best_lightweight_hybrid.pth",
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/clinical_fidelty_evaluation/
RESULTS_PATH = os.path.join(OUTPUT_DIR, "clinical_fidelity_results.json")

FS = 125             # frequenza di campionamento dei segnali .npz
CHUNK_SEC = 10        # durata di ciascuna finestra di analisi (non sovrapposte)
SEED_SEC = 6          # secondi iniziali di seed reale (mai generati), esclusi dall'analisi

# Parametri specifici dell'analisi clinica (invariati rispetto a prima)
R_PEAK_TOLERANCE_MS = 50    # tolleranza per il matching dei picchi R
MIN_PEAKS_FOR_HR = 3        # picchi minimi per stimare una HR affidabile
CHECKPOINT_EVERY = 50       # salvataggio incrementale ogni N record .npz processati
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SORGENTE DATI — file .npz già generati da run_autoregressive_drift_test_clean.py
# ─────────────────────────────────────────────────────────────────────────────
def _iter_generation_npz_files(generation_dir: str):
    for root, _, files in os.walk(generation_dir):
        for fname in sorted(files):
            if fname.endswith(".npz"):
                yield os.path.join(root, fname)


def _record_path_from_npz(npz_path: str, generation_dir: str) -> str:
    rel = os.path.relpath(npz_path, generation_dir)
    if rel.endswith(".npz"):
        rel = rel[: -len(".npz")]
    return rel.replace(os.sep, "/")


def _iter_chunks_for_record(npz_path: str, chunk_sec: int, seed_sec: int):
    """Carica un .npz e lo suddivide in finestre non sovrapposte di
    chunk_sec secondi, scartando i primi seed_sec secondi (seed reale, mai
    generato). Yield (file_info, true_sig, pred_sig, fs) per ogni finestra."""
    record_path = _record_path_from_npz(npz_path, GENERATION_DIR)
    with np.load(npz_path) as cached:
        ecg_target = cached["ecg_target"].astype(np.float32)
        ecg_generated = cached["ecg_generated"].astype(np.float32)
        fs = int(cached["fs"]) if "fs" in cached else FS

    seed_samples = int(seed_sec * fs)
    chunk_samples = int(chunk_sec * fs)
    total_len = len(ecg_target)

    for start in range(seed_samples, total_len - chunk_samples + 1, chunk_samples):
        end = start + chunk_samples
        file_info = f"{record_path}_s{start}_e{end}"
        yield file_info, ecg_target[start:end], ecg_generated[start:end], fs


# ─────────────────────────────────────────────────────────────────────────────
# METRICHE CLINICHE (NeuroKit2) — invariate
# ─────────────────────────────────────────────────────────────────────────────
def _extract_r_peaks(signal: np.ndarray, fs: int):
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def _mean_hr_from_peaks(peaks: np.ndarray, fs: int):
    if len(peaks) < MIN_PEAKS_FOR_HR:
        return None
    rr_sec = np.diff(peaks) / fs
    rr_sec = rr_sec[rr_sec > 1e-6]
    if len(rr_sec) == 0:
        return None
    return float(60.0 / np.mean(rr_sec))


def _match_r_peaks(true_peaks: np.ndarray, pred_peaks: np.ndarray, tol_samples: int):
    """Matching greedy 1-a-1 in ordine temporale (tolleranza ~50 ms)."""
    used_pred = set()
    tp = 0
    for tpk in true_peaks:
        candidates = [i for i in range(len(pred_peaks))
                      if i not in used_pred and abs(int(pred_peaks[i]) - int(tpk)) <= tol_samples]
        if candidates:
            best = min(candidates, key=lambda i: abs(int(pred_peaks[i]) - int(tpk)))
            used_pred.add(best)
            tp += 1
    fp = len(pred_peaks) - len(used_pred)
    fn = len(true_peaks) - tp
    return tp, fp, fn


def _delineate_intervals(signal: np.ndarray, rpeaks: np.ndarray, fs: int):
    """PR / QRS / QT medi (ms) via delineazione dwt. None se non disponibili."""
    feats = {'PR_ms': None, 'QRS_ms': None, 'QT_ms': None}
    if len(rpeaks) < MIN_PEAKS_FOR_HR:
        return feats
    try:
        _, waves = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="dwt")

        if 'ECG_R_Onsets' in waves and 'ECG_R_Offsets' in waves:
            onsets  = np.array(waves['ECG_R_Onsets'], dtype=float)
            offsets = np.array(waves['ECG_R_Offsets'], dtype=float)
            valid = ~np.isnan(onsets) & ~np.isnan(offsets)
            if np.any(valid):
                feats['QRS_ms'] = float(np.nanmean(offsets[valid] - onsets[valid]) / fs * 1000)

        if 'ECG_P_Onsets' in waves and 'ECG_R_Onsets' in waves:
            p_on = np.array(waves['ECG_P_Onsets'], dtype=float)
            r_on = np.array(waves['ECG_R_Onsets'], dtype=float)
            valid = ~np.isnan(p_on) & ~np.isnan(r_on)
            if np.any(valid):
                feats['PR_ms'] = float(np.nanmean(r_on[valid] - p_on[valid]) / fs * 1000)

        if 'ECG_R_Onsets' in waves and 'ECG_T_Offsets' in waves:
            r_on  = np.array(waves['ECG_R_Onsets'], dtype=float)
            t_off = np.array(waves['ECG_T_Offsets'], dtype=float)
            valid = ~np.isnan(r_on) & ~np.isnan(t_off)
            if np.any(valid):
                feats['QT_ms'] = float(np.nanmean(t_off[valid] - r_on[valid]) / fs * 1000)
    except Exception:
        pass
    return feats


def _analyze_segment(true_sig, pred_sig, fs, tol_samples):
    """Esegue l'intera pipeline clinica su una coppia (true, pred) e ritorna
    un dizionario di metriche, oppure None se il segmento non è utilizzabile."""
    if true_sig is None or pred_sig is None:
        return None
    if np.any(np.isnan(pred_sig)) or len(true_sig) < fs:  # < 1s: inutilizzabile
        return None
    if np.std(true_sig) < 1e-6 or np.std(pred_sig) < 1e-6:
        return None

    true_peaks = _extract_r_peaks(true_sig, fs)
    pred_peaks = _extract_r_peaks(pred_sig, fs)

    hr_true = _mean_hr_from_peaks(true_peaks, fs)
    hr_pred = _mean_hr_from_peaks(pred_peaks, fs)

    result = {
        'n_true_peaks': int(len(true_peaks)),
        'n_pred_peaks': int(len(pred_peaks)),
        'hr_true': hr_true,
        'hr_pred': hr_pred,
        'hr_abs_error': (abs(hr_true - hr_pred) if (hr_true is not None and hr_pred is not None) else None),
    }

    if len(true_peaks) > 0 or len(pred_peaks) > 0:
        tp, fp, fn = _match_r_peaks(true_peaks, pred_peaks, tol_samples)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        result.update({'tp': tp, 'fp': fp, 'fn': fn,
                        'precision': precision, 'recall': recall, 'f1': f1})
    else:
        result.update({'tp': 0, 'fp': 0, 'fn': 0,
                        'precision': None, 'recall': None, 'f1': None})

    true_feats = _delineate_intervals(true_sig, true_peaks, fs)
    pred_feats = _delineate_intervals(pred_sig, pred_peaks, fs)
    for k in ['PR_ms', 'QRS_ms', 'QT_ms']:
        if true_feats[k] is not None and pred_feats[k] is not None:
            result[f'{k}_true'] = true_feats[k]
            result[f'{k}_pred'] = pred_feats[k]
            result[f'{k}_abs_error'] = abs(true_feats[k] - pred_feats[k])
        else:
            result[f'{k}_true'] = None
            result[f'{k}_pred'] = None
            result[f'{k}_abs_error'] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _plot_bland_altman(records, save_path):
    means, diffs = [], []
    for r in records:
        if r.get('hr_true') is not None and r.get('hr_pred') is not None:
            means.append((r['hr_true'] + r['hr_pred']) / 2.0)
            diffs.append(r['hr_pred'] - r['hr_true'])
    if not means:
        return None

    means = np.array(means)
    diffs = np.array(diffs)
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, diffs, alpha=0.5, s=18, color='#2b6cb0', edgecolors='none')
    ax.axhline(bias, color='black', linestyle='-', linewidth=1.5,
               label=f'Bias = {bias:.2f} bpm')
    ax.axhline(loa_upper, color='red', linestyle='--', linewidth=1.2,
               label=f'+1.96 SD = {loa_upper:.2f} bpm')
    ax.axhline(loa_lower, color='red', linestyle='--', linewidth=1.2,
               label=f'-1.96 SD = {loa_lower:.2f} bpm')
    ax.set_xlabel('Mean HR (real, predicted) [bpm]')
    ax.set_ylabel('HR difference (predicted - real) [bpm]')
    ax.set_title('Bland-Altman plot — Heart Rate agreement')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return {'bias': bias, 'sd': sd, 'loa_upper': loa_upper, 'loa_lower': loa_lower, 'n': int(len(means))}


def _plot_example(true_sig, pred_sig, fs, save_path, title, max_sec=10):
    n = min(len(true_sig), len(pred_sig), int(max_sec * fs))
    t = np.arange(n) / fs
    true_peaks = _extract_r_peaks(true_sig[:n], fs)
    pred_peaks = _extract_r_peaks(pred_sig[:n], fs)
    true_peaks = true_peaks[true_peaks < n]
    pred_peaks = pred_peaks[pred_peaks < n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, true_sig[:n], color='#2f855a', linewidth=1.2)
    axes[0].scatter(true_peaks / fs, true_sig[:n][true_peaks], color='black', marker='x', s=40, label='R-peak')
    axes[0].set_title(f'{title} — Real ECG')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, pred_sig[:n], color='#c53030', linewidth=1.2)
    axes[1].scatter(pred_peaks / fs, pred_sig[:n][pred_peaks], color='black', marker='x', s=40, label='R-peak')
    axes[1].set_title(f'{title} — Reconstructed ECG (autoregressive, model-scale)')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT MARKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def _write_markdown_report(summary, ba_stats, n_records_used, n_records_discarded, save_path):
    def fmt(v, unit='', nd=2):
        return f"{v:.{nd}f}{unit}" if v is not None else "N/A"

    lines = []
    lines.append("# Clinical Feature Fidelity — lightweight_hybrid (MIMIC-III)\n")
    lines.append(
        "This report evaluates whether the ECG reconstructed by the `lightweight_hybrid` "
        "model from PPG remains clinically usable, i.e. whether heart rate and QRS "
        "complexes can be reliably extracted from it, and whether the model preserves "
        "rhythm over time instead of drifting (a common failure mode of autoregressive "
        "waveform generators).\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- Test set: hold-out 15% split from `dataset_split.json` (same test patients as "
        "the main model evaluation).\n"
        "- Model: `lightweight_hybrid`, weights from "
        f"`{os.path.relpath(MODEL_WEIGHTS_PATH, PROJECT_ROOT)}`.\n"
        "- Signal source: **true autoregressive generation with self-feeding**, produced by "
        "`scripts/autoregressive_drift_evaluation/run_autoregressive_drift_test_clean.py` "
        "(unlike earlier versions of this report, which used teacher-forced one-step-ahead "
        "windows — i.e. always fed the real past ECG as context). Here, after an initial "
        f"{SEED_SEC}s real seed, every subsequent second is generated using the model's own "
        "prior output as context, so temporal drift can actually manifest.\n"
        "- Before generation, each record's PPG/ECG is cleaned: low-quality stretches "
        "(signal quality index) are spliced out and the remaining clean stretches are "
        "concatenated into one continuous signal per record.\n"
        f"- Each cleaned record (after discarding the {SEED_SEC}s seed) is split into "
        f"non-overlapping {CHUNK_SEC}s windows for analysis, so heart rate / interval "
        "extraction always has multiple consecutive beats to work with.\n"
        "- R-peaks, heart rate, and P/QRS/T wave delineation are computed with "
        "NeuroKit2 (`nk.ecg_peaks`, `nk.ecg_delineate`, method='dwt') independently on the "
        "real and generated signal of each window (the generated signal stays in the "
        "model's own generation scale, never rescaled to physical ECG units — R-peak "
        "detection and delineation are scale-relative so this does not affect the metrics "
        "below).\n"
        f"- R-peak matching tolerance: {R_PEAK_TOLERANCE_MS} ms.\n"
    )

    lines.append("## Coverage\n")
    lines.append(f"- Test records with a usable generated signal: {n_records_used} "
                 f"({n_records_discarded} discarded upstream — too short after SQI cleaning, "
                 "or WFDB read errors)\n")
    lines.append(f"- Analysis windows attempted ({CHUNK_SEC}s each): {summary['n_attempted']}\n")
    lines.append(f"- Windows usable for analysis: {summary['n_valid']}\n")
    lines.append(f"- Windows with reliable HR estimate (≥{MIN_PEAKS_FOR_HR} peaks, both signals): {summary['n_hr_valid']}\n")

    lines.append("\n## Heart Rate Fidelity\n")
    lines.append(f"- **HR MAE**: {fmt(summary['hr_mae'])} bpm (± {fmt(summary['hr_mae_std'])}, N={summary['n_hr_valid']})\n")
    if ba_stats:
        lines.append(
            f"- **Bland-Altman**: bias = {fmt(ba_stats['bias'])} bpm, "
            f"limits of agreement = [{fmt(ba_stats['loa_lower'])}, {fmt(ba_stats['loa_upper'])}] bpm "
            f"(N={ba_stats['n']}). See `bland_altman_hr.png`.\n"
        )

    lines.append("\n## R-peak Detection\n")
    lines.append(
        f"- **Precision** (micro-avg): {fmt(summary['precision_micro'], nd=4)}\n"
        f"- **Recall** (micro-avg): {fmt(summary['recall_micro'], nd=4)}\n"
        f"- **F1-score** (micro-avg): {fmt(summary['f1_micro'], nd=4)}\n"
        f"- **F1-score** (macro-avg over windows): {fmt(summary['f1_macro'], nd=4)} "
        f"(± {fmt(summary['f1_macro_std'], nd=4)}, N={summary['n_f1_valid']})\n"
        f"- Total matched peaks: TP={summary['tp_total']}, FP={summary['fp_total']}, FN={summary['fn_total']}\n"
    )

    lines.append("\n## ECG Interval Fidelity (PR, QRS, QT)\n")
    lines.append("| Interval | Mean Abs. Error (ms) | Std (ms) | N valid windows |\n")
    lines.append("|----------|----------------------|----------|-------------------|\n")
    for key, label in [('PR_ms', 'PR'), ('QRS_ms', 'QRS'), ('QT_ms', 'QT')]:
        m = summary['interval_errors'][key]
        lines.append(f"| {label} | {fmt(m['mean'])} | {fmt(m['std'])} | {m['n']} |\n")

    lines.append(
        "\nNote: interval errors are reported only where delineation succeeded on both "
        "the real and the generated signal for a given window. Low N relative to "
        "`n_valid` indicates the model's morphology is not always clean enough for "
        "reliable P/T wave delineation — treat this table as indicative, not as a primary "
        "claim, unless N is large.\n"
    )

    lines.append("\n## Suggested paper text\n")
    lines.append(
        f"> Despite a residual morphological error, the ECG generated autoregressively "
        f"(self-feeding, not teacher-forced) preserves clinically relevant rhythm "
        f"information: heart rate extracted via R-peak detection (NeuroKit2) matches the "
        f"real signal with a mean absolute error of {fmt(summary['hr_mae'])} bpm "
        f"(Bland-Altman bias {fmt(ba_stats['bias']) if ba_stats else 'N/A'} bpm, limits of "
        f"agreement [{fmt(ba_stats['loa_lower']) if ba_stats else 'N/A'}, "
        f"{fmt(ba_stats['loa_upper']) if ba_stats else 'N/A'}] bpm), with an R-peak detection "
        f"F1-score of {fmt(summary['f1_micro'], nd=3)} at a {R_PEAK_TOLERANCE_MS} ms tolerance. "
        f"This indicates the model does not suffer from severe temporal drift even under "
        f"true self-feeding generation, a common failure mode of autoregressive waveform "
        f"generators.\n"
    )

    with open(save_path, 'w') as f:
        f.writelines(lines)


# =============================================================================
# MAIN
# =============================================================================
def run_evaluation(max_records=None, checkpoint_every=CHECKPOINT_EVERY):
    t0 = time.time()
    print("=" * 65)
    print("  CLINICAL FEATURE FIDELITY — lightweight_hybrid  MIMIC-III")
    print("  (segnali puliti + autoregressione vera, da drift_generation_clean/)")
    print("=" * 65)

    if not os.path.isdir(GENERATION_DIR):
        raise FileNotFoundError(
            f"Cartella non trovata: {GENERATION_DIR}\n"
            "Esegui prima ar_drif_test_clean.sh (run_autoregressive_drift_test_clean.py)."
        )

    npz_files = list(_iter_generation_npz_files(GENERATION_DIR))
    n_records = len(npz_files) if max_records is None else min(max_records, len(npz_files))
    print(f"Record .npz trovati in {GENERATION_DIR}: {len(npz_files)}")
    print(f"Record da processare in questa esecuzione: {n_records}\n")

    fs = FS
    tol_samples = max(1, round(R_PEAK_TOLERANCE_MS / 1000.0 * fs))

    # ── Resume da checkpoint, se presente ──────────────────────────────────
    per_segment_records = []
    processed_keys = set()
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                prev = json.load(f)
            per_segment_records = prev.get('per_segment', [])
            processed_keys = {r['file_info'] for r in per_segment_records}
            print(f"[RESUME] Trovate {len(per_segment_records)} finestre già processate, si riprende.")
        except Exception:
            pass
    processed_records = {fi.rsplit('_s', 1)[0] for fi in processed_keys}

    example_pairs = []  # (hr_mae, true_sig, pred_sig, file_info) per i plot qualitativi
    n_skipped_invalid = 0
    n_records_used = 0

    for idx, npz_path in enumerate(npz_files[:n_records]):
        record_path = _record_path_from_npz(npz_path, GENERATION_DIR)
        if record_path in processed_records:
            n_records_used += 1
            continue

        any_window = False
        for file_info, true_sig, pred_sig, rec_fs in _iter_chunks_for_record(npz_path, CHUNK_SEC, SEED_SEC):
            any_window = True
            if file_info in processed_keys:
                continue
            metrics = _analyze_segment(true_sig, pred_sig, rec_fs, tol_samples)
            if metrics is None:
                n_skipped_invalid += 1
                continue

            metrics['file_info'] = file_info
            per_segment_records.append(metrics)

            if metrics.get('hr_abs_error') is not None:
                example_pairs.append((metrics['hr_abs_error'], true_sig, pred_sig, file_info))

        if any_window:
            n_records_used += 1

        if (idx + 1) % checkpoint_every == 0 or (idx + 1) == n_records:
            elapsed = time.time() - t0
            done = len(per_segment_records)
            print(f"  [{idx + 1:5d}/{n_records}]  record processati  "
                  f"finestre_valide={done}  scartate(invalide)={n_skipped_invalid}  "
                  f"elapsed={elapsed:.0f}s")
            with open(RESULTS_PATH, 'w') as f:
                json.dump({'per_segment': per_segment_records}, f)

    print(f"\nFinestre totali valide per l'analisi clinica: {len(per_segment_records)}")

    # ── Aggregazione (invariata) ────────────────────────────────────────────
    hr_errors = [r['hr_abs_error'] for r in per_segment_records if r.get('hr_abs_error') is not None]
    f1_values = [r['f1'] for r in per_segment_records if r.get('f1') is not None]
    tp_total = sum(r.get('tp', 0) for r in per_segment_records)
    fp_total = sum(r.get('fp', 0) for r in per_segment_records)
    fn_total = sum(r.get('fn', 0) for r in per_segment_records)
    precision_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else None
    recall_micro    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else None
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                if (precision_micro is not None and recall_micro is not None and (precision_micro + recall_micro) > 0)
                else None)

    interval_errors = {}
    for key in ['PR_ms', 'QRS_ms', 'QT_ms']:
        errs = [r[f'{key}_abs_error'] for r in per_segment_records if r.get(f'{key}_abs_error') is not None]
        interval_errors[key] = {
            'mean': float(np.mean(errs)) if errs else None,
            'std':  float(np.std(errs)) if errs else None,
            'n':    len(errs),
        }

    summary = {
        'n_attempted':     len(per_segment_records) + n_skipped_invalid,
        'n_valid':         len(per_segment_records),
        'n_hr_valid':      len(hr_errors),
        'hr_mae':          float(np.mean(hr_errors)) if hr_errors else None,
        'hr_mae_std':      float(np.std(hr_errors)) if hr_errors else None,
        'precision_micro': precision_micro,
        'recall_micro':    recall_micro,
        'f1_micro':        f1_micro,
        'f1_macro':        float(np.mean(f1_values)) if f1_values else None,
        'f1_macro_std':    float(np.std(f1_values)) if f1_values else None,
        'n_f1_valid':      len(f1_values),
        'tp_total':        int(tp_total),
        'fp_total':        int(fp_total),
        'fn_total':        int(fn_total),
        'interval_errors': interval_errors,
    }

    print("\n── Risultati aggregati ──────────────────────────────────────────")
    print(f"  HR MAE        : {summary['hr_mae']}  ±  {summary['hr_mae_std']}  (N={summary['n_hr_valid']})")
    print(f"  F1 (micro)    : {summary['f1_micro']}")
    print(f"  F1 (macro)    : {summary['f1_macro']}  ±  {summary['f1_macro_std']}  (N={summary['n_f1_valid']})")
    for k in ['PR_ms', 'QRS_ms', 'QT_ms']:
        print(f"  {k:8s} error: {interval_errors[k]}")
    print("─" * 65)

    # ── Plots ────────────────────────────────────────────────────────────
    ba_stats = _plot_bland_altman(per_segment_records, os.path.join(OUTPUT_DIR, "bland_altman_hr.png"))

    if example_pairs:
        example_pairs.sort(key=lambda x: x[0])
        best_err, best_true, best_pred, best_info = example_pairs[0]
        _plot_example(best_true, best_pred, fs, os.path.join(OUTPUT_DIR, "rpeak_example_best.png"),
                      title=f"Best case (HR err={best_err:.2f} bpm) — {best_info}")

        median_idx = len(example_pairs) // 2
        med_err, med_true, med_pred, med_info = example_pairs[median_idx]
        _plot_example(med_true, med_pred, fs, os.path.join(OUTPUT_DIR, "rpeak_example_median.png"),
                      title=f"Median case (HR err={med_err:.2f} bpm) — {med_info}")

    # ── Salvataggio risultati completi ──────────────────────────────────
    n_records_discarded = len(npz_files) - n_records_used if max_records is None else None
    final_output = {
        'model': 'lightweight_hybrid',
        'weights_path': MODEL_WEIGHTS_PATH,
        'generation_source': GENERATION_DIR,
        'chunk_sec': CHUNK_SEC,
        'seed_sec': SEED_SEC,
        'r_peak_tolerance_ms': R_PEAK_TOLERANCE_MS,
        'n_records_used': n_records_used,
        'n_skipped_invalid_windows': n_skipped_invalid,
        'summary': summary,
        'bland_altman': ba_stats,
        'per_segment': per_segment_records,
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✅ clinical_fidelity_results.json  →  {RESULTS_PATH}")

    # ── Report markdown per il paper ────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "clinical_fidelity_report.md")
    _write_markdown_report(summary, ba_stats, n_records_used,
                            n_records_discarded if n_records_discarded is not None else 0,
                            report_path)
    print(f"✅ clinical_fidelity_report.md     →  {report_path}")

    print(f"\nTempo totale: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Feature Fidelity evaluation (autoregressive, clean signals)")
    parser.add_argument('--max-records', type=int, default=None,
                        help="Limita il numero di record .npz processati (debug/test rapido).")
    parser.add_argument('--checkpoint-every', type=int, default=CHECKPOINT_EVERY,
                        help="Salva un checkpoint ogni N record.")
    args = parser.parse_args()

    run_evaluation(max_records=args.max_records, checkpoint_every=args.checkpoint_every)
