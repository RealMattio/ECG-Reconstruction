"""
Grafici del drift a finestre (vedi run_windowed_drift_test.py).

Profilo di drift SECONDO-PER-SECONDO aggregato su tutti i segmenti puliti.

  - Asse X: posizione del secondo GENERATO.
      x = 7s  -> 1o secondo generato (intervallo [6s, 7s])
      x = 8s  -> 2o secondo generato (intervallo [7s, 8s])
      x = k   -> intervallo [k-1, k]
    ... fino alla lunghezza del segmento piu' lungo.
  - Asse Y: a ciascuna X, MEDIA +/- DEVIAZIONE STANDARD dell'errore su quel
    singolo secondo, calcolate su TUTTI i segmenti che raggiungono quel secondo.
    A x=8 contribuiscono tutti i segmenti lunghi >= 8s, a x=9 tutti quelli
    >= 9s, e cosi' via: il numero di segmenti (n) cala al crescere di X.

Tre figure:
  1. drift_windowed_MAE.png   — MAE  medio +/- std per secondo (atteso in salita)
  2. drift_windowed_RMSE.png  — RMSE medio +/- std per secondo (atteso in salita)
  3. drift_windowed_corr.png  — Pearson r + Cosine (medie +/- std, in discesa)

La dimensione dei marker e' proporzionale al numero di segmenti che
contribuiscono a quel secondo (marker piu' grande = stima piu' affidabile);
la fascia ombreggiata e' +/- 1 deviazione standard fra segmenti.
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_windowed")
BASE_GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation_windowed")

# Placeholder, sovrascritti in main() da --model_id prima di ogni uso reale.
OUTPUT_DIR = BASE_OUTPUT_DIR
GENERATION_DIR = BASE_GENERATION_DIR
RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_windowed_results.json")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual_context")

FS = 125
FIRST_X_SEC = 7            # il secondo generato con indice 0 corrisponde a x = 7s
MIN_COUNT = 1             # posizioni con meno di MIN_COUNT segmenti non vengono plottate

# --- Visual context (esempi di generazione, riusa gli .npz gia' prodotti) ---
PLOT_WINDOW_SEC = 10       # ampiezza delle finestre mostrate
# Fine-finestra (in secondi dall'inizio del segmento) a profondita' di
# generazione crescente: 16s = generato +10s, 36s = +30s, ... Si mostrano solo
# quelle che il segmento raggiunge, cosi' si vede il drift progredire.
SHOWCASE_ENDS_SEC = [16, 36, 66, 126, 306, 606, 1206, 2406]
MAX_ROWS = 7               # tetto di righe per figura (leggibilita')
N_EXAMPLES = 2             # quanti segmenti per categoria (best/worst/longest)
MIN_LEN_BW_SEC = 30        # lunghezza minima per i candidati best/worst (serve drift da mostrare)

# Colori (coppia categorica colorblind-safe per il grafico a 2 serie)
ACCENT = "#4E79A7"        # blu — serie singola / Pearson / PPG
ACCENT_2 = "#F28E2B"      # arancio — Cosine
REAL_COL = "#333333"      # ECG reale
GEN_COL = "#B03A2E"       # ECG generato
GRID = "#D9D9D9"
INK = "#333333"


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _subject_id_from_path(record_path: str) -> str:
    # record_path es. 'p05/p053609/3924929_0291' -> 'p053609'
    parts = record_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else parts[-1]


def load_segments():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Risultati non trovati: {RESULTS_PATH}. Esegui prima run_windowed_drift_test.py.")
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)
    segs = data.get("segments", [])
    if not segs:
        raise ValueError("Nessun segmento nei risultati: la valutazione ha prodotto 0 punti.")
    return segs


def aggregate_per_second(segs, metric):
    """Raggruppa i valori per posizione X (secondo generato) su tutti i segmenti.
    Ritorna array ordinati (x, mean, std, count)."""
    buckets = defaultdict(list)
    for s in segs:
        arr = s.get("per_second", {}).get(metric)
        if not arr:
            continue
        for j, v in enumerate(arr):
            if v is not None and np.isfinite(v):
                buckets[FIRST_X_SEC + j].append(v)

    xs = sorted(x for x, vals in buckets.items() if len(vals) >= MIN_COUNT)
    x = np.array(xs, dtype=float)
    mean = np.array([np.mean(buckets[k]) for k in xs])
    std = np.array([np.std(buckets[k]) for k in xs])
    count = np.array([len(buckets[k]) for k in xs])
    return x, mean, std, count


def _marker_sizes(count):
    """Marker piu' grande dove piu' segmenti contribuiscono (scala sqrt)."""
    if len(count) == 0:
        return count
    c = np.sqrt(count.astype(float))
    c = c / c.max()
    return 20 + 90 * c


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK)
    ax.set_xlabel("Lunghezza della finestra = secondo generato [s]", color=INK)


def plot_single(segs, metric, ylabel, title, filename, expected, nonneg=True):
    x, mean, std, count = aggregate_per_second(segs, metric)
    if len(x) == 0:
        print(f"[WARN] Nessun dato per {metric}, salto {filename}.")
        return

    lower = mean - std
    if nonneg:
        lower = np.maximum(lower, 0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, lower, mean + std, color=ACCENT, alpha=0.18, linewidth=0, label="± 1 deviazione standard")
    ax.plot(x, mean, color=ACCENT, linewidth=1.8, zorder=3, label="Media fra segmenti")
    ax.scatter(x, mean, s=_marker_sizes(count), color=ACCENT, edgecolors="white",
               linewidths=0.5, zorder=4)

    _style_axes(ax)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(f"{title}\n(atteso: {expected} con la lunghezza — marker ∝ n. segmenti)",
                 color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="best")
    _annotate_counts(ax, x, mean, count)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✓ Salvato: {out}  (x da {x.min():.0f}s a {x.max():.0f}s, n max={count.max()})")


def plot_correlations(segs):
    xp, mp, sp, cp = aggregate_per_second(segs, "Pearson")
    xc, mc, sc, cc = aggregate_per_second(segs, "Cosine")
    if len(xp) == 0 and len(xc) == 0:
        print("[WARN] Nessun dato per Pearson/Cosine, salto il grafico correlazioni.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    if len(xp):
        ax.fill_between(xp, mp - sp, mp + sp, color=ACCENT, alpha=0.15, linewidth=0)
        ax.plot(xp, mp, color=ACCENT, linewidth=1.8, zorder=3, label="Pearson r")
        ax.scatter(xp, mp, s=_marker_sizes(cp), color=ACCENT, edgecolors="white", linewidths=0.5, zorder=4)
    if len(xc):
        ax.fill_between(xc, mc - sc, mc + sc, color=ACCENT_2, alpha=0.15, linewidth=0)
        ax.plot(xc, mc, color=ACCENT_2, linewidth=1.8, zorder=3, label="Cosine similarity")
        ax.scatter(xc, mc, s=_marker_sizes(cc), color=ACCENT_2, marker="s",
                   edgecolors="white", linewidths=0.5, zorder=4)

    _style_axes(ax)
    ax.set_ylabel("Similarita' generato vs reale", color=INK)
    ax.set_title("Correlazione morfologica per secondo generato\n"
                 "(atteso: in discesa con la lunghezza — fascia = ± 1 std, marker ∝ n. segmenti)",
                 color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="best")
    if len(xp):
        _annotate_counts(ax, xp, mp, cp)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "drift_windowed_corr.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✓ Salvato: {out}")


def _annotate_counts(ax, x, y, count):
    """Etichetta il numero di segmenti al primo e all'ultimo punto, per dare il
    senso di quanti segmenti reggono ciascun orizzonte."""
    if len(x) == 0:
        return
    for idx in (0, len(x) - 1):
        ax.annotate(f"n={count[idx]}", (x[idx], y[idx]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=INK)


# =====================================================================
# VISUAL CONTEXT — esempi di generazione (riusa gli .npz gia' prodotti)
# =====================================================================
def _segment_mean(seg, metric):
    arr = seg.get("per_second", {}).get(metric)
    if not arr:
        return None
    vals = [v for v in arr if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def load_segment_signals(record_path, seg_idx):
    """Ritorna (ppg, true, gen) per un segmento dal suo .npz, o None."""
    npz_path = os.path.join(GENERATION_DIR, record_path + ".npz")
    if not os.path.exists(npz_path):
        return None
    try:
        with np.load(npz_path) as z:
            k_ppg, k_true, k_gen = f"seg{seg_idx}_ppg", f"seg{seg_idx}_true", f"seg{seg_idx}_gen"
            if k_true not in z.files or k_gen not in z.files:
                return None
            ppg = z[k_ppg].astype(np.float32) if k_ppg in z.files else None
            return ppg, z[k_true].astype(np.float32), z[k_gen].astype(np.float32)
    except Exception as e:
        print(f"[WARN] impossibile leggere {npz_path}: {e}")
        return None


def select_showcase_segments(segs):
    """Sceglie i segmenti da illustrare: N migliori/peggiori per MAE e RMSE
    (solo tra quelli abbastanza lunghi da mostrare drift) piu' gli N piu' lunghi
    (analogo dei pazienti '24h' del paradigma precedente)."""
    jobs = []  # (kind_label, folder, seg)
    seen = set()  # (record_path, segment_index) gia' assegnati, per non ripetere

    def _take(cands, kind, folder):
        added = 0
        for seg in cands:
            key = (seg["record_path"], seg["segment_index"])
            if key in seen:
                continue
            jobs.append((kind, folder, seg))
            seen.add(key)
            added += 1
            if added >= N_EXAMPLES:
                break

    for metric in ("MAE", "RMSE"):
        scored = [(s, _segment_mean(s, metric)) for s in segs if s.get("length_sec", 0) >= MIN_LEN_BW_SEC]
        scored = [(s, v) for s, v in scored if v is not None]
        scored.sort(key=lambda sv: sv[1])
        _take([s for s, _ in scored[:N_EXAMPLES]], f"Miglior {metric}", metric)
        _take([s for s, _ in reversed(scored[-N_EXAMPLES:])], f"Peggior {metric}", metric)

    longest = sorted(segs, key=lambda s: s.get("length_sec", 0), reverse=True)
    _take(longest[:N_EXAMPLES], "Segmento piu' lungo", "Longest")
    return jobs


def _showcase_ends(length_sec):
    ends = [e for e in SHOWCASE_ENDS_SEC if e <= length_sec + 1e-6]
    if not ends:
        ends = [length_sec]  # segmento corto: si mostra l'ultima finestra disponibile
    if len(ends) > MAX_ROWS:  # tiene primo, ultimo e un sottoinsieme uniforme in mezzo
        idx = np.unique(np.linspace(0, len(ends) - 1, MAX_ROWS).round().astype(int))
        ends = [ends[i] for i in idx]
    return ends


def plot_segment_showcase(seg, kind_label, out_dir):
    sig = load_segment_signals(seg["record_path"], seg["segment_index"])
    if sig is None:
        print(f"[WARN] segnali non disponibili per {seg['record_path']} seg{seg['segment_index']}")
        return
    ppg, true, gen = sig
    total_len = min(len(true), len(gen))
    length_sec = total_len / FS

    ends = _showcase_ends(length_sec)
    win = min(int(PLOT_WINDOW_SEC * FS), total_len)
    n_rows = len(ends)

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 2.6 * n_rows), squeeze=False)
    mae = _segment_mean(seg, "MAE")
    subj = seg.get("subject_id", "?")
    fig.suptitle(f"{kind_label} — {subj}  (segmento {length_sec:.0f}s, MAE medio {mae:.3f})",
                 fontsize=15, fontweight="bold", y=0.995)

    for r, end_sec in enumerate(ends):
        end_s = min(int(round(end_sec * FS)), total_len)
        start_s = max(0, end_s - win)
        t = np.arange(start_s, end_s) / FS

        ppg_w = _minmax_norm(ppg[start_s:end_s]) if ppg is not None else None
        true_w = _minmax_norm(true[start_s:end_s])
        g = gen[start_s:end_s]
        exploded = not np.all(np.isfinite(g))
        gen_w = np.zeros_like(g) if exploded else _minmax_norm(g)

        ax_p, ax_e = axes[r, 0], axes[r, 1]
        if ppg_w is not None:
            ax_p.plot(t, ppg_w, color=ACCENT, linewidth=1.0)
        ax_p.set_title(f"PPG input — gen +{max(0, end_sec - FIRST_X_SEC + 1):.0f}s (fino a {end_sec:.0f}s)", fontsize=10)
        ax_p.set_ylabel("Norm [0,1]")

        ax_e.plot(t, true_w, color=REAL_COL, alpha=0.45, linestyle="--", linewidth=1.0, label="ECG reale")
        ax_e.plot(t, gen_w, color=GEN_COL, alpha=0.9, linewidth=1.2, label="ECG generato")
        if exploded:
            ax_e.text(0.5, 0.5, "DIVERGENZA NUMERICA (NaN/Inf)", transform=ax_e.transAxes,
                      color=GEN_COL, fontsize=11, fontweight="bold", ha="center", va="center",
                      bbox=dict(facecolor="white", alpha=0.8))
        # linea al confine seed(6s)/generazione, se cade nella finestra
        seed_boundary = FIRST_X_SEC - 1  # 6s
        if start_s / FS <= seed_boundary <= end_s / FS:
            for ax in (ax_p, ax_e):
                ax.axvline(seed_boundary, color="#888888", linewidth=0.8, linestyle=":")
        ax_e.set_title(f"Ricostruzione ECG — fino a {end_sec:.0f}s", fontsize=10)
        ax_e.legend(loc="upper right", fontsize=8)

        for ax in (ax_p, ax_e):
            ax.grid(True, color=GRID, alpha=0.4)
            if r == n_rows - 1:
                ax.set_xlabel("Secondi dall'inizio del segmento")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{kind_label.replace(' ', '_').replace(chr(39), '')}_{subj}_seg{seg['segment_index']}.png"
    out = os.path.join(out_dir, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✓ Visual context: {out}")


def plot_full_segment_strip(record_path, seg_idx, out_dir, filename=None):
    """Plot dell'INTERO segmento come striscia continua: tanti subplot impilati,
    ciascuno lungo PLOT_WINDOW_SEC (10s) e continuazione del precedente, cosi'
    da coprire tutto il segnale generato vs reale (stile stampa ECG).
    La normalizzazione min-max e' GLOBALE sul segmento (una volta), cosi' le
    righe sono confrontabili fra loro e il drift d'ampiezza resta visibile."""
    sig = load_segment_signals(record_path, seg_idx)
    if sig is None:
        print(f"[WARN] segnali non disponibili per {record_path} seg{seg_idx}")
        return
    ppg, true, gen = sig
    total_len = int(min(len(true), len(gen)))
    length_sec = total_len / FS

    true_g = _minmax_norm(true[:total_len])
    exploded_all = not np.all(np.isfinite(gen[:total_len]))
    gen_g = np.zeros(total_len, dtype=np.float32) if exploded_all else _minmax_norm(gen[:total_len])

    win = int(PLOT_WINDOW_SEC * FS)
    n_rows = int(np.ceil(total_len / win))

    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 2.1 * n_rows), squeeze=False)
    subj = _subject_id_from_path(record_path)
    fig.suptitle(f"Segnale completo — {subj} seg{seg_idx}  ({length_sec:.0f}s, record {os.path.basename(record_path)})\n"
                 f"ECG reale vs generato, righe consecutive da {PLOT_WINDOW_SEC}s",
                 fontsize=14, fontweight="bold", y=0.997)

    for r in range(n_rows):
        start_s = r * win
        end_s = min(start_s + win, total_len)
        t = np.arange(start_s, end_s) / FS
        ax = axes[r, 0]

        ax.plot(t, true_g[start_s:end_s], color=REAL_COL, alpha=0.45, linestyle="--",
                linewidth=1.0, label="ECG reale")
        ax.plot(t, gen_g[start_s:end_s], color=GEN_COL, alpha=0.9, linewidth=1.2, label="ECG generato")

        # confine seed(6s)/generazione, se cade in questa riga
        seed_boundary = FIRST_X_SEC - 1  # 6s
        if start_s / FS <= seed_boundary <= end_s / FS:
            ax.axvline(seed_boundary, color="#888888", linewidth=0.8, linestyle=":")
            ax.text(seed_boundary, 1.02, "fine seed", fontsize=7, color="#888888", ha="center")

        ax.set_xlim(start_s / FS, (start_s + win) / FS)   # tutte le righe larghe 10s
        ax.set_ylim(-0.05, 1.10)
        ax.set_ylabel("Norm [0,1]")
        ax.grid(True, color=GRID, alpha=0.4)
        if r == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        if r == n_rows - 1:
            ax.set_xlabel("Secondi dall'inizio del segmento")

    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    os.makedirs(out_dir, exist_ok=True)
    fname = filename or f"FullStrip_{subj}_seg{seg_idx}.png"
    out = os.path.join(out_dir, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✓ Striscia completa: {out}")
    return out


def _resolve_record_for_segment(subject_or_record, seg_idx):
    """Se viene passato solo 'pXXXXXX' (subject), trova il record_path il cui
    .npz contiene quel seg_idx; se viene passato un record_path completo lo usa."""
    # gia' un record_path completo?
    if os.path.exists(os.path.join(GENERATION_DIR, subject_or_record + ".npz")):
        return subject_or_record
    subj = subject_or_record
    matches = []
    for npz in glob.glob(os.path.join(GENERATION_DIR, "**", subj, "*.npz"), recursive=True):
        try:
            with np.load(npz) as z:
                if f"seg{seg_idx}_true" in z.files:
                    rec = os.path.relpath(npz, GENERATION_DIR)[:-4]
                    n = len(z[f"seg{seg_idx}_true"])
                    matches.append((rec, n))
        except Exception:
            continue
    if not matches:
        return None
    # se ambiguo, prende il segmento piu' lungo (di solito quello di interesse)
    matches.sort(key=lambda x: x[1], reverse=True)
    if len(matches) > 1:
        print(f"[INFO] seg{seg_idx} presente in piu' record per {subj}: "
              f"{[m[0] for m in matches]} -> uso il piu' lungo ({matches[0][0]}, {matches[0][1]/FS:.0f}s)")
    return matches[0][0]


def run_visual_context(segs):
    print("\n--- VISUAL CONTEXT (esempi di generazione) ---")
    jobs = select_showcase_segments(segs)
    if not jobs:
        print("[WARN] nessun segmento idoneo per il visual context.")
        return
    for kind_label, folder, seg in jobs:
        plot_segment_showcase(seg, kind_label, os.path.join(VISUAL_DIR, folder))
    print(f"-> Esempi salvati in: {VISUAL_DIR}")


def main():
    global OUTPUT_DIR, GENERATION_DIR, RESULTS_PATH, VISUAL_DIR

    parser = argparse.ArgumentParser(description="Grafici drift a finestre + visual context.")
    parser.add_argument("--model_id", required=True,
                        help="Sotto-cartella del modello (es. lightweight_hybrid_20260716_145739), "
                             "la stessa usata da run_windowed_drift_test.py --model_weights_path.")
    parser.add_argument("--strip", metavar="SUBJECT_OR_RECORD:SEG",
                        help="Genera SOLO la striscia completa (righe da 10s) di un segmento e salva in "
                             "visual_context. Es: --strip p053609:99  oppure  --strip p05/p053609/3924929_0291:99")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, args.model_id)
    GENERATION_DIR = os.path.join(BASE_GENERATION_DIR, args.model_id)
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "drift_windowed_results.json")
    VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual_context")

    if args.strip:
        try:
            ref, seg_s = args.strip.rsplit(":", 1)
            seg_idx = int(seg_s)
        except ValueError:
            print("[ERRORE] formato --strip non valido. Usa SUBJECT_OR_RECORD:SEG (es. p053609:99)")
            sys.exit(1)
        record_path = _resolve_record_for_segment(ref, seg_idx)
        if record_path is None:
            print(f"[ERRORE] nessun .npz con seg{seg_idx} per '{ref}' in {GENERATION_DIR}")
            sys.exit(1)
        plot_full_segment_strip(record_path, seg_idx, VISUAL_DIR)
        return

    print("=" * 60)
    print(" GRAFICI DRIFT A FINESTRE (errore per secondo generato)")
    print("=" * 60)
    segs = load_segments()
    lengths = [s["length_sec"] for s in segs]
    print(f"-> Segmenti totali: {len(segs)} | lunghezza min {min(lengths):.0f}s, "
          f"max {max(lengths):.0f}s, mediana {np.median(lengths):.0f}s")

    plot_single(segs, "MAE", "MAE sul secondo generato", "MAE per secondo generato",
                "drift_windowed_MAE.png", "in salita", nonneg=True)
    plot_single(segs, "RMSE", "RMSE sul secondo generato", "RMSE per secondo generato",
                "drift_windowed_RMSE.png", "in salita", nonneg=True)
    # 4a curva: errore Heart Rate (presente se il JSON e' stato ricalcolato con HR)
    if any("HR" in s.get("per_second", {}) for s in segs):
        plot_single(segs, "HR", "Errore Heart Rate [bpm]", "Errore Heart Rate per secondo generato",
                    "drift_windowed_HR.png", "in salita", nonneg=True)
    else:
        print("[INFO] Nessun dato HR nel JSON: rigenera con recompute_per_second_from_npz.py per il 4o grafico.")
    plot_correlations(segs)

    run_visual_context(segs)

    print("\n" + "=" * 60)
    print(f"✅ Finito. Grafici in: {OUTPUT_DIR}")
    print(f"   Visual context in: {VISUAL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
