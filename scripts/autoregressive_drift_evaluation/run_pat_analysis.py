"""
Analisi del Pulse Arrival Time (PAT) sui segmenti generati dal test di drift a
finestre (run_windowed_drift_test.py). Il razionale dell'articolo e' il
monitoraggio continuo del PAT (tempo tra picco R dell'ECG e picco sistolico
della PPG): qui si valuta quanto bene il PAT ricostruito dall'ECG GENERATO
segue quello reale, e con quale bias sistematico.

Riusa le generazioni gia' salvate negli .npz (nessuna GPU).

--- ANALISI 1 (tutti i pazienti del test set) ---
Per ogni segmento si calcolano PAT_real (dagli R reali) e PAT_gen (dagli R
generati), accoppiati sullo stesso picco PPG. Si applica la formula del MAE/
RMSE/MSE TRA i valori di PAT (non tra i campioni del segnale). Per ogni
orizzonte temporale h il valore e' CUMULATIVO: mediato su tutti i battiti
generati dall'inizio fino a h. Si aggrega poi (media +/- std) su tutti i
segmenti che raggiungono h. -> 3 grafici (MAE_pat, RMSE_pat, MSE_pat).

--- ANALISI 2 (bias) ---
Si stima un bias temporale (ms) su un "validation set" di 14 pazienti (~20%
dei 71 del test): la differenza media tra dove il picco R si trova realmente e
dove e' stato generato, ossia mean(PAT_real - PAT_gen). Poi si RIPETE l'analisi
1 sui 57 pazienti rimanenti (held-out) correggendo ogni PAT_gen con il bias.
Per ogni metrica si produce una figura con due subplot: senza correzione e con
correzione del bias.

Uso:
  python scripts/autoregressive_drift_evaluation/run_pat_analysis.py
  # opzioni: --recompute (rifa' i PAT dagli .npz), --min_count N, --val_size 14
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

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
import pat_hr_utils as PU  # noqa: E402

BASE_GENERATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_generation_windowed")
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation_windowed")

# Placeholder, sovrascritti in main() da --model_id prima di ogni uso reale.
GENERATION_DIR = BASE_GENERATION_DIR
OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "pat_analysis")
PAT_CACHE = os.path.join(OUT_DIR, "pat_pairs.json")

FS = 125
SEED_SEC = 6
FIRST_X_SEC = 7
VAL_SEED = 42

# Stile (coerente con plot_windowed_drift.py)
ACCENT = "#4E79A7"
ACCENT_2 = "#B03A2E"
GRID = "#D9D9D9"
INK = "#333333"

METRICS = {
    "MAE":  {"label": "MAE del PAT [ms]",  "title": "MAE del Pulse Arrival Time"},
    "RMSE": {"label": "RMSE del PAT [ms]", "title": "RMSE del Pulse Arrival Time"},
    "MSE":  {"label": "MSE del PAT [ms²]", "title": "MSE del Pulse Arrival Time"},
    # errore percentuale (relativo all'ampiezza del PAT reale): 50 ms possono
    # sembrare tanti in assoluto ma poco rispetto a un PAT reale di ~400 ms
    "MAPE": {"label": "Errore % del PAT [%]", "title": "Errore percentuale del PAT (MAPE)"},
}


def _subject_id_from_path(record_path):
    parts = record_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else parts[-1]


# =====================================================================
# 1. Estrazione PAT per segmento (con cache)
# =====================================================================
def build_pat_pairs(gen_dir):
    """Per ogni segmento: PAT_real/PAT_gen dei battiti nella REGIONE GENERATA
    (beat_time > SEED_SEC). Ritorna lista di dict."""
    npz_files = sorted(glob.glob(os.path.join(gen_dir, "**", "*.npz"), recursive=True))
    print(f"-> {len(npz_files)} file .npz da analizzare (estrazione PAT, ~3 min)...")
    seg_pats = []
    for n, npz_path in enumerate(npz_files):
        rec_path = os.path.relpath(npz_path, gen_dir)[:-4]
        subj = _subject_id_from_path(rec_path)
        try:
            with np.load(npz_path) as z:
                seg_ids = sorted(set(int(k[3:].split("_")[0]) for k in z.files if k.startswith("seg")))
                for i in seg_ids:
                    tk, gk, pk = f"seg{i}_true", f"seg{i}_gen", f"seg{i}_ppg"
                    if tk not in z.files or gk not in z.files or pk not in z.files:
                        continue
                    true = z[tk].astype(np.float32)
                    gen = z[gk].astype(np.float32)
                    ppg = z[pk].astype(np.float32)
                    length_sec = min(len(true), len(gen)) / FS

                    _, _, bt, pr, pg = PU.extract_pat(true, gen, ppg, FS)
                    if len(bt) == 0:
                        continue
                    mask = bt > SEED_SEC  # solo battiti nella regione generata
                    if not np.any(mask):
                        continue
                    seg_pats.append({
                        "subject_id": subj,
                        "record_path": rec_path,
                        "segment_index": i,
                        "length_sec": float(length_sec),
                        "beat_times": bt[mask].tolist(),
                        "pat_real": pr[mask].tolist(),
                        "pat_gen": pg[mask].tolist(),
                    })
        except Exception as e:
            print(f"[WARN] {npz_path}: {e}")
        if (n + 1) % 100 == 0:
            print(f"   ...{n + 1}/{len(npz_files)} record", flush=True)
    print(f"-> Segmenti con PAT validi: {len(seg_pats)}")
    return seg_pats


def load_or_build(gen_dir, force=False):
    if not force and os.path.exists(PAT_CACHE):
        with open(PAT_CACHE, "r") as f:
            seg_pats = json.load(f)["segments"]
        print(f"-> PAT caricati dalla cache: {len(seg_pats)} segmenti ({PAT_CACHE})")
        return seg_pats
    seg_pats = build_pat_pairs(gen_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = PAT_CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"segments": seg_pats}, f)
    os.replace(tmp, PAT_CACHE)
    print(f"-> Cache PAT salvata: {PAT_CACHE}")
    return seg_pats


# =====================================================================
# 2. Curva cumulativa per orizzonte (con prefix-sum, vettorizzata)
# =====================================================================
def cumulative_curves(seg_pats, bias_ms=0.0, min_count=10):
    """Per ciascun orizzonte h (secondi interi da FIRST_X_SEC), il valore
    cumulativo della metrica su tutti i battiti generati di un segmento fino a
    h; poi media/std/conteggio fra i segmenti. Ritorna {metric: (x,mean,std,count)}.
    bias_ms viene AGGIUNTO a PAT_gen (correzione del bias)."""
    buckets = {m: defaultdict(list) for m in METRICS}

    for s in seg_pats:
        bt = np.asarray(s["beat_times"], dtype=float)
        if len(bt) == 0:
            continue
        pat_real = np.asarray(s["pat_real"], dtype=float)
        d = (np.asarray(s["pat_gen"], dtype=float) + bias_ms) - pat_real
        order = np.argsort(bt)
        bt, d, pat_real = bt[order], d[order], pat_real[order]

        abs_cum = np.concatenate([[0.0], np.cumsum(np.abs(d))])
        sq_cum = np.concatenate([[0.0], np.cumsum(d ** 2)])
        # errore percentuale per battito, |d| / |PAT_real| * 100
        pct = np.abs(d) / np.maximum(np.abs(pat_real), 1e-6) * 100.0
        pct_cum = np.concatenate([[0.0], np.cumsum(pct)])

        h_max = int(np.floor(min(s["length_sec"], bt[-1])))
        h_min = max(FIRST_X_SEC, int(np.ceil(bt[0])))
        if h_max < h_min:
            continue
        hors = np.arange(h_min, h_max + 1)
        k = np.searchsorted(bt, hors, side="right")  # n. battiti <= h
        valid = k >= 1
        hors, k = hors[valid], k[valid]
        mae = abs_cum[k] / k
        mse = sq_cum[k] / k
        rmse = np.sqrt(mse)
        mape = pct_cum[k] / k
        for h, a, r, m, p in zip(hors, mae, rmse, mse, mape):
            buckets["MAE"][int(h)].append(a)
            buckets["RMSE"][int(h)].append(r)
            buckets["MSE"][int(h)].append(m)
            buckets["MAPE"][int(h)].append(p)

    out = {}
    for m in METRICS:
        xs = sorted(x for x, v in buckets[m].items() if len(v) >= min_count)
        x = np.array(xs, dtype=float)
        mean = np.array([np.mean(buckets[m][k]) for k in xs])
        std = np.array([np.std(buckets[m][k]) for k in xs])
        count = np.array([len(buckets[m][k]) for k in xs])
        out[m] = (x, mean, std, count)
    return out


# =====================================================================
# 3. Plotting
# =====================================================================
def _style_main(ax, is_bottom=False):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK)
    if not is_bottom:
        plt.setp(ax.get_xticklabels(), visible=False)


def _draw_curve(ax, x, mean, std, color, label, nonneg=True):
    """Curva media come LINEA (niente marker) con fascia +/- 1 std."""
    if len(x) == 0:
        return
    lower = np.maximum(mean - std, 0.0) if nonneg else mean - std
    ax.fill_between(x, lower, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)


def _draw_dist(axd, x, count):
    """Pannello piccolo: distribuzione dei punti lungo l'asse X (quanti
    segmenti reggono ciascun orizzonte). Scala log per rendere visibile la coda."""
    axd.grid(True, color=GRID, linewidth=0.8, alpha=0.5, axis="x")
    axd.set_axisbelow(True)
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axd.spines[sp].set_color(GRID)
    if len(x):
        axd.fill_between(x, 1, count, color="#9AA7B4", alpha=0.55, linewidth=0, step="mid")
        axd.plot(x, count, color="#6B7885", linewidth=0.8)
    axd.set_yscale("log")
    axd.set_ylabel("n. segm.", fontsize=8, color=INK)
    axd.tick_params(colors=INK, labelsize=8)
    axd.set_xlabel("Orizzonte di generazione [s]", color=INK)


def plot_analysis1(curves, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for m, info in METRICS.items():
        x, mean, std, count = curves[m]
        fig = plt.figure(figsize=(11, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.07)
        ax = fig.add_subplot(gs[0])
        axd = fig.add_subplot(gs[1], sharex=ax)

        _draw_curve(ax, x, mean, std, ACCENT, "Media fra segmenti")
        _style_main(ax)
        ax.set_ylabel(info["label"], color=INK)
        ax.set_title(f"{info['title']} vs orizzonte (cumulativo, tutti i {N_SUBJECTS_ALL} pazienti)\n"
                     "(atteso: in salita con la lunghezza)", color=INK, fontweight="bold")
        ax.legend(frameon=False, loc="best")
        _draw_dist(axd, x, count)

        fig.tight_layout()
        out = os.path.join(out_dir, f"PAT_{m}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"✓ {out}")


def plot_analysis2(curves_uncorr, curves_corr, bias_ms, n_val, n_held, out_dir):
    """Figure impilate verticalmente: SENZA correzione (sopra), CON correzione
    (in mezzo) e UNA sola distribuzione dei punti in basso (identica per le due,
    stessi segmenti held-out)."""
    os.makedirs(out_dir, exist_ok=True)
    for m, info in METRICS.items():
        xu, mu, su, cu = curves_uncorr[m]
        xc, mc, sc, _ = curves_corr[m]

        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1], hspace=0.28)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
        axd = fig.add_subplot(gs[2], sharex=ax1)

        _draw_curve(ax1, xu, mu, su, ACCENT, "SENZA correzione")
        _style_main(ax1)
        ax1.set_title("SENZA correzione bias", color=INK, fontweight="bold", fontsize=11)
        ax1.set_ylabel(info["label"], color=INK)

        _draw_curve(ax2, xc, mc, sc, ACCENT_2, "CON correzione")
        _style_main(ax2)
        ax2.set_title(f"CON correzione bias ({bias_ms:+.0f} ms)", color=INK, fontweight="bold", fontsize=11)
        ax2.set_ylabel(info["label"], color=INK)

        _draw_dist(axd, xu, cu)  # stessa distribuzione per entrambi

        fig.suptitle(f"{info['title']} — effetto della correzione del bias\n"
                     f"held-out: {n_held} pazienti (bias stimato su {n_val} di validation)",
                     fontsize=14, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(out_dir, f"PAT_{m}_bias.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"✓ {out}")


# =====================================================================
# MAIN
# =====================================================================
N_SUBJECTS_ALL = 0  # riempito in main (per i titoli)


def main():
    global N_SUBJECTS_ALL, GENERATION_DIR, OUT_DIR, PAT_CACHE
    parser = argparse.ArgumentParser(description="Analisi PAT (drift a finestre).")
    parser.add_argument("--model_id", required=True,
                        help="Sotto-cartella del modello (es. lightweight_hybrid_20260716_145739), "
                             "la stessa usata da run_windowed_drift_test.py --model_weights_path.")
    parser.add_argument("--recompute", action="store_true", help="Rifa' l'estrazione PAT dagli .npz.")
    parser.add_argument("--min_count", type=int, default=10, help="Min. segmenti per orizzonte nei grafici.")
    parser.add_argument("--val_size", type=int, default=14, help="Numero pazienti nel validation set per il bias.")
    args = parser.parse_args()

    GENERATION_DIR = os.path.join(BASE_GENERATION_DIR, args.model_id)
    OUT_DIR = os.path.join(BASE_OUTPUT_DIR, args.model_id, "pat_analysis")
    PAT_CACHE = os.path.join(OUT_DIR, "pat_pairs.json")

    print("=" * 60)
    print(" ANALISI PULSE ARRIVAL TIME (PAT) — drift a finestre")
    print(f" Modello: {args.model_id}")
    print("=" * 60)

    seg_pats = load_or_build(GENERATION_DIR, force=args.recompute)
    if not seg_pats:
        print("[ERRORE] Nessun PAT estratto.")
        sys.exit(1)

    subjects = sorted({s["subject_id"] for s in seg_pats})
    N_SUBJECTS_ALL = len(subjects)
    print(f"-> Pazienti con PAT: {N_SUBJECTS_ALL} | segmenti: {len(seg_pats)}")

    # --- ANALISI 1 (tutti) ---
    print("\n--- ANALISI 1: PAT vs orizzonte (tutti i pazienti) ---")
    curves_all = cumulative_curves(seg_pats, bias_ms=0.0, min_count=args.min_count)
    plot_analysis1(curves_all, OUT_DIR)

    # --- ANALISI 2 (bias) ---
    print("\n--- ANALISI 2: correzione del bias ---")
    rng = np.random.default_rng(VAL_SEED)
    val_size = min(args.val_size, len(subjects))
    val_subjects = set(rng.choice(subjects, size=val_size, replace=False).tolist())
    held_subjects = [s for s in subjects if s not in val_subjects]

    val_segs = [s for s in seg_pats if s["subject_id"] in val_subjects]
    held_segs = [s for s in seg_pats if s["subject_id"] not in val_subjects]

    # bias = mean(PAT_real - PAT_gen) sui battiti generati del validation set
    d_all = []
    for s in val_segs:
        d_all.append(np.asarray(s["pat_real"]) - np.asarray(s["pat_gen"]))
    d_all = np.concatenate(d_all) if d_all else np.array([])
    bias_ms = float(np.mean(d_all)) if len(d_all) else 0.0
    print(f"-> Validation: {len(val_subjects)} pazienti, {len(d_all)} battiti "
          f"-> bias = mean(PAT_real - PAT_gen) = {bias_ms:+.1f} ms")
    print(f"-> Held-out: {len(held_subjects)} pazienti, {len(held_segs)} segmenti")

    curves_unc = cumulative_curves(held_segs, bias_ms=0.0, min_count=args.min_count)
    curves_cor = cumulative_curves(held_segs, bias_ms=bias_ms, min_count=args.min_count)
    plot_analysis2(curves_unc, curves_cor, bias_ms, len(val_subjects), len(held_subjects), OUT_DIR)

    # riepilogo bias su file
    with open(os.path.join(OUT_DIR, "pat_bias_summary.json"), "w") as f:
        json.dump({
            "bias_ms": bias_ms,
            "val_subjects": sorted(val_subjects),
            "n_val_beats": int(len(d_all)),
            "held_out_subjects": held_subjects,
            "val_seed": VAL_SEED,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Finito. Grafici PAT in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
