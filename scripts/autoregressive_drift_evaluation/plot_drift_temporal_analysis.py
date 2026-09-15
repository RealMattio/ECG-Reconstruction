"""
Sezione 4.4.2 - Autoregressive Drifting and Long-Term Stability.

Riusa i segnali GIA' generati e salvati da run_autoregressive_drift_test.py
(scripts/autoregressive_drift_evaluation/experiments/drift_generation/<record_path>.npz,
un solo file per record, generato fino al suo orizzonte massimo disponibile)
per studiare come l'errore autoregressivo si accumula nel tempo:

1. Per ogni record disponibile si usa l'intera generazione salvata
   (fino a 24h) e la si divide in finestre temporali (default 2s).
2. Per ogni finestra si calcola Pearson r e RMSE (su segnali normalizzati
   [0,1] una sola volta sull'intero segnale, cosi' la normalizzazione non
   nasconde eventuale perdita di ampiezza nel tempo).
3. Si producono:
   - drift_curve_aggregate.png: line plot Pearson r / RMSE vs tempo (scala
     log), aggregato su tutti i pazienti disponibili, con banda di
     std e conteggio pazienti disponibili per bin (potenza statistica).
   - drift_visual_<subject>.png: per alcuni pazienti rappresentativi, il
     grafico "drifting visivo" con sfondo sfumato sul punto in cui inizia
     il drift, piu' gli snippet di ECG reale vs generato prima/dopo.
   - drift_temporal_summary.json: statistiche numeriche + bozza testuale
     riassuntiva da riusare nella sezione "Analisi Testuale" del paper.

I risultati intermedi (metriche per finestra) sono salvati in una cache
CSV cosi' da poter rigenerare i grafici senza rileggere tutti i .npz
(alcuni pesano >100MB).
"""
import os
import sys
import json
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_quality import compute_quality_mask  # noqa: E402

# --- CONFIGURAZIONI E PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_temporal_analysis")
CACHE_PATH = os.path.join(OUTPUT_DIR, "windowed_drift_metrics_cache.csv.gz")
CACHE_META_PATH = os.path.join(OUTPUT_DIR, "windowed_drift_metrics_cache_meta.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "drift_temporal_summary.json")

FS = 125
SEED_SEC = 6  # secondi di ECG reale usati come contesto iniziale (vedi run_autoregressive_drift_test.py)

# Bump quando cambia lo schema delle righe della cache (es. nuove colonne):
# forza il ricalcolo invece di provare a concatenare schemi incompatibili.
CACHE_SCHEMA_VERSION = 2

# Una finestra e' considerata "affidabile" (ground truth non rumoroso) se
# almeno questa frazione dei suoi campioni supera il controllo SQI applicato
# ovunque nel progetto per costruire i dataset di training/valutazione
# (vedi signal_quality.py). Le finestre sotto soglia vengono escluse dalla
# curva di drift aggregata per non scambiare "target rumoroso" per "drift
# del modello".
QUALITY_WINDOW_THRESHOLD = 0.8

R_THRESH_STRICT = 0.90
R_THRESH_SOFT = 0.70
ONSET_PERSISTENCE_WINDOWS = 3  # numero di finestre consecutive sotto soglia per confermare l'onset


def _minmax_norm(signal: np.ndarray) -> np.ndarray:
    mn, mx = signal.min(), signal.max()
    return (signal - mn) / (mx - mn + 1e-8)


HORIZON_SEC = {"1m": 60, "30m": 1800, "1h": 3600, "6h": 21600, "12h": 43200, "24h": 86400}


def _horizon_label_for_duration(duration_sec: float) -> str:
    """Etichetta informativa (solo per reportistica) basata sulla durata generata."""
    best_label = "1m"
    for label, sec in HORIZON_SEC.items():
        if duration_sec >= sec + SEED_SEC - 1:  # piccola tolleranza
            best_label = label
    return best_label


def discover_best_records():
    """Ogni record e' salvato in UN SOLO file .npz (fino al suo orizzonte
    massimo disponibile): run_autoregressive_drift_test.py genera ciascun
    record una sola volta, quindi qui basta elencare i file presenti."""
    records = []
    for npz_path in glob(os.path.join(GENERATION_DIR, "**", "*.npz"), recursive=True):
        rel = os.path.relpath(npz_path, GENERATION_DIR)
        parts = rel.split(os.sep)
        subject_id = parts[-2] if len(parts) >= 2 else parts[0]
        record_id = os.path.splitext(parts[-1])[0]
        try:
            with np.load(npz_path) as data:
                duration_sec = len(data["ecg_generated"]) / FS
        except Exception:
            continue
        records.append({
            "horizon": _horizon_label_for_duration(duration_sec),
            "npz_path": npz_path,
            "subject_id": subject_id,
            "record_id": record_id,
            "record_rel": rel,
            "duration_sec": duration_sec,
        })
    return records


def windowed_pearson_rmse(true_norm: np.ndarray, gen_norm: np.ndarray, win_samples: int):
    """Calcola Pearson r e RMSE per finestre non sovrapposte, in modo vettorializzato."""
    n_windows = len(true_norm) // win_samples
    if n_windows == 0:
        return np.array([]), np.array([])

    T = true_norm[: n_windows * win_samples].reshape(n_windows, win_samples)
    G = gen_norm[: n_windows * win_samples].reshape(n_windows, win_samples)

    Tc = T - T.mean(axis=1, keepdims=True)
    Gc = G - G.mean(axis=1, keepdims=True)
    num = (Tc * Gc).sum(axis=1)
    den = np.sqrt((Tc ** 2).sum(axis=1) * (Gc ** 2).sum(axis=1)) + 1e-8
    pearson = num / den
    rmse = np.sqrt(((T - G) ** 2).mean(axis=1))
    return pearson, rmse


def _windowed_quality_fraction(quality_mask: np.ndarray, win_samples: int) -> np.ndarray:
    """Frazione di campioni 'puliti' (SQI) in ciascuna finestra non
    sovrapposta, allineata esattamente alle finestre di windowed_pearson_rmse."""
    n_windows = len(quality_mask) // win_samples
    if n_windows == 0:
        return np.array([])
    M = quality_mask[: n_windows * win_samples].reshape(n_windows, win_samples)
    return M.mean(axis=1)


def _compute_windows_for_records(records, window_sec: float) -> pd.DataFrame:
    win_samples = int(round(window_sec * FS))
    rows = []
    for rec in tqdm(records, desc="Calcolo metriche per finestra"):
        try:
            data = np.load(rec["npz_path"])
            true_full = data["ecg_target"].astype(np.float64)
            gen_full = data["ecg_generated"].astype(np.float64)
            ppg_full = data["ppg_input"].astype(np.float64)
        except Exception as e:
            tqdm.write(f"Errore lettura {rec['npz_path']}: {e}")
            continue

        if np.isnan(gen_full).any() or np.isinf(gen_full).any():
            tqdm.write(f"Skip {rec['record_rel']}: collasso numerico (NaN/Inf) nella generazione.")
            continue

        true_norm = _minmax_norm(true_full)
        gen_norm = _minmax_norm(gen_full)

        pearson, rmse = windowed_pearson_rmse(true_norm, gen_norm, win_samples)
        times = (np.arange(len(pearson)) * window_sec) + (window_sec / 2.0)

        quality_mask = compute_quality_mask(true_full, ppg_full, FS)
        quality_frac = _windowed_quality_fraction(quality_mask, win_samples)
        # windowed_pearson_rmse e _windowed_quality_fraction usano lo stesso
        # win_samples sullo stesso segnale: stesso numero di finestre per costruzione
        n = min(len(pearson), len(quality_frac))

        for t, p, r, q in zip(times[:n], pearson[:n], rmse[:n], quality_frac[:n]):
            rows.append({
                "subject_id": rec["subject_id"],
                "record_id": rec["record_id"],
                "record_rel": rec["record_rel"],
                "best_horizon": rec["horizon"],
                "time_sec": float(t),
                "pearson": float(p),
                "rmse": float(r),
                "quality_valid_frac": float(q),
            })
    return pd.DataFrame(rows)


def build_cache(window_sec: float, force: bool = False) -> pd.DataFrame:
    """Cache incrementale: la generazione dei segnali (run_autoregressive_drift_test.py)
    procede in background su Slurm e aggiunge record via via, quindi ad ogni
    esecuzione ricalcoliamo solo i record NUOVI rispetto a quelli gia' in cache
    (a parita' di window_sec), invece di rileggere tutti i .npz da capo."""
    records = discover_best_records()
    print(f"-> Trovati {len(records)} record unici (generazione piu' lunga disponibile per ciascuno).")

    cached_meta = {}
    if os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH) as f:
            cached_meta = json.load(f)

    schema_ok = cached_meta.get("schema_version") == CACHE_SCHEMA_VERSION
    existing_df = None
    if os.path.exists(CACHE_PATH) and not force and schema_ok and np.isclose(cached_meta.get("window_sec", -1), window_sec):
        existing_df = pd.read_csv(CACHE_PATH)
        cached_rels = set(existing_df["record_rel"].unique())
        new_records = [r for r in records if r["record_rel"] not in cached_rels]
        print(f"-> Cache esistente compatibile (window_sec={window_sec}s): "
              f"{len(cached_rels)} record gia' calcolati, {len(new_records)} nuovi da aggiungere.")
    else:
        if os.path.exists(CACHE_PATH):
            print(f"   (cache assente/incompatibile con window_sec={window_sec}s o schema obsoleto, ricalcolo tutto da zero)")
        new_records = records

    new_df = _compute_windows_for_records(new_records, window_sec) if new_records else pd.DataFrame()

    if existing_df is not None and not new_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    elif existing_df is not None:
        df = existing_df
    else:
        df = new_df

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    with open(CACHE_META_PATH, "w") as f:
        json.dump({"window_sec": window_sec, "schema_version": CACHE_SCHEMA_VERSION}, f)
    print(f"-> Cache salvata in: {CACHE_PATH} ({len(df)} finestre totali, {df['record_rel'].nunique() if not df.empty else 0} record).")
    return df


def _time_axis_formatter(x, _pos=None):
    if x < 60:
        return f"{x:.0f}s"
    if x < 3600:
        return f"{x / 60:.0f}m"
    if x < 86400:
        return f"{x / 3600:.0f}h"
    return f"{x / 86400:.1f}d"


def detect_onset_time(times: np.ndarray, pearson: np.ndarray, threshold: float, persistence: int):
    """Prima finestra dopo la quale r resta sotto soglia per `persistence` finestre consecutive."""
    below = pearson < threshold
    for i in range(len(below) - persistence + 1):
        if below[i:i + persistence].all():
            return float(times[i])
    return None


def _aggregate_binned(df_subset: pd.DataFrame, bin_edges: np.ndarray) -> pd.DataFrame:
    """Bin per tempo (log), media per record dentro ogni bin (cosi' un
    record con tante finestre non pesa piu' degli altri), poi media/std tra
    record. Usata sia per la curva 'raw' sia per quella 'quality-filtered'."""
    df_subset = df_subset.copy()
    df_subset["bin"] = pd.cut(df_subset["time_sec"], bins=bin_edges, include_lowest=True)

    per_record_bin = df_subset.groupby(["record_rel", "bin"], observed=True).agg(
        pearson=("pearson", "mean"), rmse=("rmse", "mean")
    ).reset_index()

    agg = per_record_bin.groupby("bin", observed=True).agg(
        pearson_mean=("pearson", "mean"),
        pearson_std=("pearson", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        n_records=("pearson", "count"),
    ).reset_index()
    agg["bin_center"] = agg["bin"].apply(
        lambda b: float((b.left + b.right) / 2) if b.left > 0 else float(b.right / 2)
    ).astype(float)
    agg = agg.sort_values("bin_center")
    return agg[agg["n_records"] > 0]


def plot_aggregate_curve(df: pd.DataFrame, window_sec: float):
    print("-> Genero il line plot aggregato (drift_curve_aggregate.png)...")

    max_time = df["time_sec"].max()
    n_bins = 28
    bin_edges = np.geomspace(window_sec, max_time, num=n_bins)
    bin_edges = np.concatenate(([0.0], bin_edges))

    df_quality = df[df["quality_valid_frac"] >= QUALITY_WINDOW_THRESHOLD]
    n_excluded = len(df) - len(df_quality)
    print(f"   Finestre escluse per qualita' insufficiente del ground truth (SQI < "
          f"{QUALITY_WINDOW_THRESHOLD:.0%}): {n_excluded}/{len(df)} ({n_excluded / max(len(df), 1):.1%})")

    agg_raw = _aggregate_binned(df, bin_edges)
    agg_q = _aggregate_binned(df_quality, bin_edges)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.3]}
    )
    fig.suptitle(
        "Autoregressive Drift: degrado delle metriche nel tempo di generazione",
        fontsize=15, fontweight="bold"
    )

    x_raw = agg_raw["bin_center"].to_numpy(dtype=float)
    x_q = agg_q["bin_center"].to_numpy(dtype=float)

    ax1.plot(x_raw, agg_raw["pearson_mean"], color="gray", linewidth=1.2, linestyle="--",
              alpha=0.7, label="Raw (tutte le finestre)")
    ax1.plot(x_q, agg_q["pearson_mean"], color="crimson", marker="o", markersize=3,
              label="Solo ground truth pulito (SQI)")
    ax1.fill_between(
        x_q, agg_q["pearson_mean"] - agg_q["pearson_std"], agg_q["pearson_mean"] + agg_q["pearson_std"],
        color="crimson", alpha=0.15, label="± std tra pazienti (SQI)"
    )
    ax1.axhline(R_THRESH_STRICT, color="green", linestyle="--", linewidth=1, label=f"r = {R_THRESH_STRICT}")
    ax1.axhline(R_THRESH_SOFT, color="orange", linestyle="--", linewidth=1, label=f"r = {R_THRESH_SOFT}")
    ax1.axvline(SEED_SEC, color="black", linestyle=":", linewidth=1.2, label=f"Fine contesto reale ({SEED_SEC}s)")
    ax1.set_ylabel("Pearson r")
    ax1.set_ylim(-0.6, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Correlazione (finestra vs finestra)", fontsize=11)

    ax2.plot(x_raw, agg_raw["rmse_mean"], color="gray", linewidth=1.2, linestyle="--",
              alpha=0.7, label="Raw (tutte le finestre)")
    ax2.plot(x_q, agg_q["rmse_mean"], color="steelblue", marker="o", markersize=3,
              label="Solo ground truth pulito (SQI)")
    ax2.fill_between(
        x_q, agg_q["rmse_mean"] - agg_q["rmse_std"], agg_q["rmse_mean"] + agg_q["rmse_std"],
        color="steelblue", alpha=0.15, label="± std tra pazienti (SQI)"
    )
    ax2.axvline(SEED_SEC, color="black", linestyle=":", linewidth=1.2)
    ax2.set_ylabel("RMSE (segnale normalizzato)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("Errore quadratico (finestra vs finestra)", fontsize=11)

    ax3.bar(x_raw, agg_raw["n_records"], width=x_raw * 0.35, color="gray", alpha=0.4, label="Raw")
    ax3.bar(x_q, agg_q["n_records"], width=x_q * 0.2, color="crimson", alpha=0.6, label="Solo SQI-pulito")
    ax3.set_ylabel("# pazienti\ndisponibili")
    ax3.set_xlabel("Tempo di generazione (scala log)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_time_axis_formatter))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "drift_curve_aggregate.png")
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"   Salvato: {save_path}")

    return agg_q, agg_raw, n_excluded


def _find_valid_snippet_start(quality_mask: np.ndarray, target_sample: int, span_samples: int,
                               search_radius_sec: float = 900.0, fs: int = FS,
                               min_valid_frac: float = 0.8) -> int:
    """Cerca, in un raggio di search_radius_sec attorno a target_sample, lo
    snippet di lunghezza span_samples con la maggior frazione di campioni
    SQI-validi (cosi' lo snippet 'tardivo' mostrato nel plot non cada per
    puro caso in un buco di segnale rumoroso). Fallback su target_sample se
    non si trova nulla di sufficientemente pulito."""
    radius_samples = int(search_radius_sec * fs)
    lo = max(0, target_sample - radius_samples)
    hi = min(len(quality_mask) - span_samples, target_sample + radius_samples)
    if hi <= lo:
        return max(0, min(target_sample, len(quality_mask) - span_samples))

    step = max(1, fs)  # valuta ogni secondo, non ogni campione
    candidates = np.arange(lo, hi, step)
    best_start, best_frac = target_sample, -1.0
    for c in candidates:
        frac = quality_mask[c:c + span_samples].mean()
        if frac > best_frac:
            best_frac, best_start = frac, c
        if best_frac >= min_valid_frac:
            break
    return int(best_start)


def plot_visual_drift(record: dict, window_sec: float, df_record: pd.DataFrame, late_snippet_frac: float = 0.5):
    subject_id = record["subject_id"]
    npz_path = record["npz_path"]
    print(f"-> Genero il plot di drifting visivo per {subject_id} ({record['record_rel']})...")

    data = np.load(npz_path)
    true_full = data["ecg_target"].astype(np.float64)
    gen_full = data["ecg_generated"].astype(np.float64)
    ppg_full = data["ppg_input"].astype(np.float64)
    quality_mask = compute_quality_mask(true_full, ppg_full, FS)

    times = df_record["time_sec"].values
    pearson = df_record["pearson"].values
    quality_ok = df_record["quality_valid_frac"].values >= QUALITY_WINDOW_THRESHOLD

    # L'onset va stimato SOLO su finestre con ground truth affidabile, altrimenti
    # un tratto di segnale rumoroso puo' simulare un "drift" che non e' del modello.
    if quality_ok.any():
        onset = detect_onset_time(times[quality_ok], pearson[quality_ok], R_THRESH_SOFT, ONSET_PERSISTENCE_WINDOWS)
    else:
        onset = None
    if onset is None:
        onset = SEED_SEC

    total_sec = len(true_full) / FS

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.35, 1.5], hspace=0.55, wspace=0.25)

    # --- Riga 1: curva Pearson r su tutta la durata, con sfondo sfumato ---
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.axvspan(times.min(), onset, color="green", alpha=0.08, label="Regime stabile")
    ax_curve.axvspan(onset, times.max(), color="red", alpha=0.08, label="Regime di drift")
    ax_curve.plot(times[quality_ok], pearson[quality_ok], color="crimson", linewidth=0.9, alpha=0.9,
                  label="Pearson r (ground truth pulito)")
    if (~quality_ok).any():
        ax_curve.plot(times[~quality_ok], pearson[~quality_ok], color="gray", marker=".", markersize=2,
                      linewidth=0, alpha=0.5, label="Pearson r (ground truth rumoroso, non affidabile)")
    ax_curve.axvline(onset, color="black", linestyle="--", linewidth=1.3)
    ax_curve.text(
        onset, 1.05, f"  Onset del drift ≈ {onset:.0f}s", rotation=0,
        va="bottom", ha="left", fontsize=10, fontweight="bold"
    )
    ax_curve.axhline(R_THRESH_STRICT, color="gray", linestyle=":", linewidth=1)
    ax_curve.set_xscale("log")
    ax_curve.xaxis.set_major_formatter(FuncFormatter(_time_axis_formatter))
    ax_curve.set_ylim(-0.6, 1.15)
    ax_curve.set_ylabel("Pearson r (per finestra)")
    ax_curve.set_title(
        f"Paziente {subject_id} — generazione autoregressiva di {_time_axis_formatter(total_sec)} "
        f"(finestre da {window_sec:g}s)",
        fontsize=12, fontweight="bold"
    )
    ax_curve.legend(loc="lower right", fontsize=8)
    ax_curve.grid(True, alpha=0.3)

    # --- Riga 2: striscia di qualita' SQI del ground truth lungo tutta la durata ---
    ax_qual = fig.add_subplot(gs[1, :], sharex=ax_curve)
    strip_time = np.arange(len(quality_mask)) / FS
    # Sottocampioniamo la maschera per il plot (altrimenti milioni di punti per le run lunghe)
    stride = max(1, len(quality_mask) // 5000)
    ax_qual.fill_between(
        strip_time[::stride], 0, 1,
        where=quality_mask[::stride], color="seagreen", alpha=0.6, step="mid",
        label="Ground truth pulito (SQI)"
    )
    ax_qual.fill_between(
        strip_time[::stride], 0, 1,
        where=~quality_mask[::stride], color="dimgray", alpha=0.6, step="mid",
        label="Ground truth rumoroso"
    )
    ax_qual.set_yticks([])
    ax_qual.set_xscale("log")
    ax_qual.xaxis.set_major_formatter(FuncFormatter(_time_axis_formatter))
    ax_qual.set_xlabel("Tempo di generazione (scala log)")
    ax_qual.set_ylabel("Qualita'\nSQI", fontsize=8)
    ax_qual.legend(loc="upper right", fontsize=6, ncol=2)

    # --- Riga 3a: zoom sul punto esatto di drift (waveform reale) ---
    # Normalizzazione LOCALE (solo su questo snippet), non quella globale
    # usata per la curva Pearson/RMSE: il seed viene scritto in ecg_generated
    # gia' normalizzato sulla propria finestra di 6s, quindi confrontarlo con
    # un ECG reale normalizzato sull'intera durata (ore/giorni) crea un
    # disallineamento di scala puramente artificiale, anche se i valori del
    # seed sono letteralmente identici al segnale reale.
    zoom_sec = 20
    zoom_samples = int(zoom_sec * FS)
    zoom_quality_frac = float(quality_mask[:zoom_samples].mean())
    zoom_true_norm = _minmax_norm(true_full[:zoom_samples])
    zoom_gen_norm = _minmax_norm(gen_full[:zoom_samples])
    ax_zoom = fig.add_subplot(gs[2, 0])
    t_axis = np.arange(zoom_samples) / FS
    ax_zoom.axvspan(0, onset, color="green", alpha=0.10)
    ax_zoom.axvspan(onset, zoom_sec, color="red", alpha=0.10)
    ax_zoom.plot(t_axis, zoom_true_norm, color="black", linestyle="--", alpha=0.6, label="ECG reale")
    ax_zoom.plot(t_axis, zoom_gen_norm, color="red", alpha=0.85, label="ECG generato")
    ax_zoom.axvline(onset, color="black", linestyle="--", linewidth=1.2)
    ax_zoom.axvline(SEED_SEC, color="blue", linestyle=":", linewidth=1, label=f"Fine seed ({SEED_SEC}s)")
    title_suffix = "" if zoom_quality_frac >= QUALITY_WINDOW_THRESHOLD else "  ⚠ ground truth rumoroso qui"
    ax_zoom.set_title(f"Punto esatto di insorgenza del drift{title_suffix}", fontsize=11)
    ax_zoom.set_xlabel("Secondi")
    ax_zoom.set_ylabel("Ampiezza normalizzata\n(scala locale allo snippet)")
    ax_zoom.legend(loc="upper right", fontsize=8)
    ax_zoom.grid(True, alpha=0.3)

    # --- Riga 3b: snippet tardivo (cercato vicino al target su ground truth pulito) ---
    target_sample = int(total_sec * late_snippet_frac * FS)
    late_start = _find_valid_snippet_start(quality_mask, target_sample, zoom_samples)
    late_end = min(late_start + zoom_samples, len(true_full))
    late_start = max(0, late_end - zoom_samples)
    late_quality_frac = float(quality_mask[late_start:late_end].mean())
    t_axis_late = np.arange(late_end - late_start) / FS
    late_true_norm = _minmax_norm(true_full[late_start:late_end])
    late_gen_norm = _minmax_norm(gen_full[late_start:late_end])

    ax_late = fig.add_subplot(gs[2, 1])
    ax_late.axvspan(0, zoom_sec, color="red", alpha=0.10)
    ax_late.plot(t_axis_late, late_true_norm, color="black", linestyle="--", alpha=0.6, label="ECG reale")
    ax_late.plot(t_axis_late, late_gen_norm, color="red", alpha=0.85, label="ECG generato")
    late_center_sec = (late_start + late_end) / 2 / FS
    title_suffix = "" if late_quality_frac >= QUALITY_WINDOW_THRESHOLD else "  ⚠ ground truth rumoroso qui"
    ax_late.set_title(f"Snippet tardivo (t ≈ {_time_axis_formatter(late_center_sec)}){title_suffix}", fontsize=11)
    ax_late.set_xlabel("Secondi (finestra locale)")
    ax_late.legend(loc="upper right", fontsize=8)
    ax_late.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"drift_visual_{subject_id}_{record['record_id']}.png")
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"   Salvato: {save_path}")

    return onset


def write_summary(df: pd.DataFrame, agg_q: pd.DataFrame, agg_raw: pd.DataFrame,
                   n_excluded: int, onsets: dict, window_sec: float):
    df_quality = df[df["quality_valid_frac"] >= QUALITY_WINDOW_THRESHOLD]

    per_record_onsets = []
    for rec_rel, group in df_quality.groupby("record_rel"):
        group = group.sort_values("time_sec")
        onset = detect_onset_time(group["time_sec"].values, group["pearson"].values, R_THRESH_SOFT, ONSET_PERSISTENCE_WINDOWS)
        per_record_onsets.append(onset)
    per_record_onsets = [o for o in per_record_onsets if o is not None]

    def _first_crossing(agg, threshold):
        crossed = agg[agg["pearson_mean"] < threshold]
        if crossed.empty:
            return None
        return float(crossed.iloc[0]["bin_center"])

    t90 = _first_crossing(agg_q, R_THRESH_STRICT)
    t70 = _first_crossing(agg_q, R_THRESH_SOFT)
    t90_raw = _first_crossing(agg_raw, R_THRESH_STRICT)
    t70_raw = _first_crossing(agg_raw, R_THRESH_SOFT)
    excluded_pct = 100 * n_excluded / max(len(df), 1)
    excluded_note = (
        f"{n_excluded}/{len(df)} finestre ({excluded_pct:.1f}%) sono state escluse perche' il "
        "ground truth non superava il controllo di qualita'."
    )

    summary = {
        "window_sec": window_sec,
        "quality_window_threshold": QUALITY_WINDOW_THRESHOLD,
        "n_records_analyzed": int(df["record_rel"].nunique()),
        "n_records_by_horizon": df.drop_duplicates("record_rel")["best_horizon"].value_counts().to_dict(),
        "n_windows_total": int(len(df)),
        "n_windows_excluded_low_quality_ground_truth": int(n_excluded),
        "frac_windows_excluded_low_quality_ground_truth": float(n_excluded / max(len(df), 1)),
        "aggregate_first_crossing_below_r0.90_sec": t90,
        "aggregate_first_crossing_below_r0.70_sec": t70,
        "aggregate_first_crossing_below_r0.90_sec_RAW_unfiltered": t90_raw,
        "aggregate_first_crossing_below_r0.70_sec_RAW_unfiltered": t70_raw,
        "per_record_onset_sec_mean": float(np.mean(per_record_onsets)) if per_record_onsets else None,
        "per_record_onset_sec_median": float(np.median(per_record_onsets)) if per_record_onsets else None,
        "per_record_onset_sec_std": float(np.std(per_record_onsets)) if per_record_onsets else None,
        "visual_drift_onsets": onsets,
        "draft_text_it": (
            "Poiche' una parte del ground truth ECG estratto direttamente dai tracciati WFDB "
            "grezzi e' di qualita' insufficiente (disconnessioni, artefatti da movimento, "
            "saturazione — non filtrata dalla pipeline di generazione, a differenza dei dati "
            "usati per allenare il modello), la curva di drift viene calcolata solo sulle "
            f"finestre che superano lo stesso controllo SQI (kurtosi ECG, skewness PPG) usato "
            f"per costruire i dataset di training: {excluded_note} "
            "Come si evince in Figura X, il modello mantiene una correlazione r > "
            f"{R_THRESH_STRICT} solo durante i primi {SEED_SEC} secondi, corrispondenti "
            "al contesto reale (seed) fornito in input. Non appena la generazione passa "
            "in modalita' puramente autoregressiva, l'errore di drifting inizia a "
            f"dominare: la correlazione (calcolata solo su ground truth affidabile) crolla "
            f"sotto {R_THRESH_SOFT} entro ~{t70:.0f}s" if t70 else ""
        ) + (
            f" e si stabilizza in un regime rumoroso e scorrelato (r prossimo a 0 o "
            "negativo) che persiste per l'intera durata della generazione, senza segni "
            "di recupero, indicando un forte exposure bias dovuto alla ridotta finestra "
            "di contesto reale e al passo di generazione a singolo step."
            if t70 else ""
        ),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> Riepilogo statistico salvato in: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Autoregressive Drift - Analisi temporale (Sezione 4.4.2)")
    parser.add_argument("--window_sec", type=float, default=2.0, help="Durata delle finestre per il calcolo di Pearson r / RMSE (default: 2s).")
    parser.add_argument("--force_recompute", action="store_true", help="Ignora la cache e ricalcola le metriche per finestra da zero.")
    parser.add_argument("--visual_subjects", nargs="*", default=None, help="Subject id da usare per i grafici di drifting visivo (default: i N record con generazione piu' lunga).")
    parser.add_argument("--n_visual", type=int, default=3, help="Numero di pazienti da usare per i grafici di drifting visivo se --visual_subjects non e' specificato.")
    args = parser.parse_args()

    print("=" * 60)
    print(" SEZIONE 4.4.2 - AUTOREGRESSIVE DRIFTING & LONG-TERM STABILITY")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = build_cache(args.window_sec, force=args.force_recompute)
    if df.empty:
        print("[ERRORE] Nessuna metrica calcolata: controlla che experiments/drift_generation contenga dei .npz validi.")
        sys.exit(1)

    agg_q, agg_raw, n_excluded = plot_aggregate_curve(df, args.window_sec)

    records = discover_best_records()

    if args.visual_subjects:
        chosen = [r for r in records if r["subject_id"] in set(args.visual_subjects)]
    else:
        # prendi i record con la generazione piu' lunga in assoluto
        chosen = sorted(records, key=lambda r: r["duration_sec"], reverse=True)[: args.n_visual]

    onsets = {}
    for rec in chosen:
        df_record = df[df["record_rel"] == rec["record_rel"]].sort_values("time_sec")
        if df_record.empty:
            continue
        onset = plot_visual_drift(rec, args.window_sec, df_record)
        onsets[f"{rec['subject_id']}_{rec['record_id']}"] = onset

    write_summary(df, agg_q, agg_raw, n_excluded, onsets, args.window_sec)

    print("\n" + "=" * 60)
    print(f"✅ Analisi temporale completata! Output in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
