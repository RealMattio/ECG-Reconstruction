#!/usr/bin/env python3
"""
Clinical Feature Fidelity — WINDOWED paradigm (no splicing), full test set.

Why this script exists
----------------------
evaluate_clinical_fidelity.py reads from `drift_generation_clean/`, i.e. the
OLD splicing paradigm in which low-SQI stretches were cut out and the clean
remainder concatenated. That generation set covers only 44 of the 71 held-out
test subjects, and every splice junction injects an amplitude step and a
cardiac-phase discontinuity the model never saw in training — precisely the
artefact that corrupts rhythm metrics.

This version reads instead from `drift_generation_windowed/`, produced by
run_windowed_drift_test.py: maximal runs of consecutive windows passing the SQI
gate are kept INTACT as independent contiguous segments, nothing is ever
concatenated. That generation set covers all 71 test subjects.

Analysis unit
-------------
Only the GENERATED portion of each segment is analysed (everything after the
6 s real seed, which the model never produced). Each generated portion is split
into non-overlapping CHUNK_SEC windows; chunks never cross a segment boundary,
so the no-splicing property is preserved end to end.

Inclusion criterion (per request)
---------------------------------
A chunk is retained only if NeuroKit2 detects at least MIN_R_PEAKS R peaks in
the GENERATED signal *and* at least MIN_R_PEAKS in the real reference. Chunks
failing either test are discarded and counted separately, so the exclusion rate
is auditable rather than silent. Rationale: heart rate, R-peak matching and
P/QRS/T delineation are all undefined or degenerate on a tract containing fewer
than three beats, and including them would contaminate the statistics with
detector failures rather than model error.

Both counts are reported, because they mean different things:
  - dropped for <3 peaks in the GENERATED signal  → model/morphology failure
  - dropped for <3 peaks in the REAL signal       → reference/detector failure

Usage
-----
    conda activate ecg_gen_env
    python scripts/clinical_fidelty_evaluation/evaluate_clinical_fidelity_windowed.py --model_id <model_id>
    # options: --chunk-sec, --min-r-peaks, --n-workers, --max-records

Outputs (written to <this folder>/<model_id>/)
------------------------------------------------
    clinical_fidelity_windowed_results.json
    clinical_fidelity_windowed_summary.json
"""

import os
import sys
import json
import glob
import time
import argparse
import multiprocessing as mp

import numpy as np
import neurokit2 as nk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_GENERATION_DIR = os.path.join(
    PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation",
    "experiments", "drift_generation_windowed",
)
SPLIT_PATH = os.path.join(PROJECT_ROOT, "mimic3wdb-matched_healthy_data", "dataset_split.json")
BASE_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set by main() once --model_id is known; module-level so the mp.Pool workers
# (fork start method on Linux, default for mp.Pool) inherit them via COW.
GENERATION_DIR = None
OUTPUT_DIR = None
RESULTS_PATH = None
SUMMARY_PATH = None


def resolve_paths(model_id):
    generation_dir = os.path.join(BASE_GENERATION_DIR, model_id)
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_id)
    return {
        "GENERATION_DIR": generation_dir,
        "OUTPUT_DIR": output_dir,
        "RESULTS_PATH": os.path.join(output_dir, "clinical_fidelity_windowed_results.json"),
        "SUMMARY_PATH": os.path.join(output_dir, "clinical_fidelity_windowed_summary.json"),
    }


def set_paths(paths):
    global GENERATION_DIR, OUTPUT_DIR, RESULTS_PATH, SUMMARY_PATH
    GENERATION_DIR = paths["GENERATION_DIR"]
    OUTPUT_DIR = paths["OUTPUT_DIR"]
    RESULTS_PATH = paths["RESULTS_PATH"]
    SUMMARY_PATH = paths["SUMMARY_PATH"]

FS = 125
SEED_SEC = 6                 # real seed, never generated — excluded from analysis
CHUNK_SEC = 10               # analysis window inside a segment's generated portion
MIN_R_PEAKS = 3              # inclusion criterion (see docstring)
R_PEAK_TOLERANCE_MS = 50


# ─────────────────────────────────────────────────────────────
# NeuroKit2 primitives (identical to evaluate_clinical_fidelity.py)
# ─────────────────────────────────────────────────────────────
def _extract_r_peaks(signal, fs):
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def _mean_hr_from_peaks(peaks, fs):
    if len(peaks) < MIN_R_PEAKS:
        return None
    rr = np.diff(peaks) / fs
    rr = rr[rr > 1e-6]
    return float(60.0 / np.mean(rr)) if len(rr) else None


def _match_r_peaks(true_peaks, pred_peaks, tol_samples):
    used, tp = set(), 0
    for tpk in true_peaks:
        cand = [i for i in range(len(pred_peaks))
                if i not in used and abs(int(pred_peaks[i]) - int(tpk)) <= tol_samples]
        if cand:
            best = min(cand, key=lambda i: abs(int(pred_peaks[i]) - int(tpk)))
            used.add(best)
            tp += 1
    return tp, len(pred_peaks) - len(used), len(true_peaks) - tp


def _delineate_intervals(signal, rpeaks, fs):
    feats = {"PR_ms": None, "QRS_ms": None, "QT_ms": None}
    if len(rpeaks) < MIN_R_PEAKS:
        return feats
    try:
        _, w = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="dwt")

        def _mean_diff(a_key, b_key):
            if a_key not in w or b_key not in w:
                return None
            a = np.array(w[a_key], dtype=float)
            b = np.array(w[b_key], dtype=float)
            valid = ~np.isnan(a) & ~np.isnan(b)
            if not np.any(valid):
                return None
            return float(np.nanmean(b[valid] - a[valid]) / fs * 1000)

        feats["QRS_ms"] = _mean_diff("ECG_R_Onsets", "ECG_R_Offsets")
        feats["PR_ms"] = _mean_diff("ECG_P_Onsets", "ECG_R_Onsets")
        feats["QT_ms"] = _mean_diff("ECG_R_Onsets", "ECG_T_Offsets")
    except Exception:
        pass
    return feats


def _analyze_chunk(true_sig, pred_sig, fs, tol_samples, min_peaks):
    """Returns (metrics | None, drop_reason | None)."""
    if np.any(~np.isfinite(pred_sig)) or np.any(~np.isfinite(true_sig)):
        return None, "non_finite"
    if np.std(true_sig) < 1e-6 or np.std(pred_sig) < 1e-6:
        return None, "flat_signal"

    true_peaks = _extract_r_peaks(true_sig, fs)
    pred_peaks = _extract_r_peaks(pred_sig, fs)

    # ── Inclusion criterion ────────────────────────────────────────────
    if len(pred_peaks) < min_peaks:
        return None, "lt_min_peaks_generated"
    if len(true_peaks) < min_peaks:
        return None, "lt_min_peaks_real"

    hr_true = _mean_hr_from_peaks(true_peaks, fs)
    hr_pred = _mean_hr_from_peaks(pred_peaks, fs)
    tp, fp, fn = _match_r_peaks(true_peaks, pred_peaks, tol_samples)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    res = {
        "n_true_peaks": int(len(true_peaks)),
        "n_pred_peaks": int(len(pred_peaks)),
        "hr_true": hr_true,
        "hr_pred": hr_pred,
        "hr_abs_error": abs(hr_true - hr_pred) if (hr_true and hr_pred) else None,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }

    tf = _delineate_intervals(true_sig, true_peaks, fs)
    pf = _delineate_intervals(pred_sig, pred_peaks, fs)
    for k in ("PR_ms", "QRS_ms", "QT_ms"):
        if tf[k] is not None and pf[k] is not None:
            res[f"{k}_true"], res[f"{k}_pred"] = tf[k], pf[k]
            res[f"{k}_abs_error"] = abs(tf[k] - pf[k])
        else:
            res[f"{k}_true"] = res[f"{k}_pred"] = res[f"{k}_abs_error"] = None
    return res, None


# ─────────────────────────────────────────────────────────────
# Per-record worker
# ─────────────────────────────────────────────────────────────
def process_record(args):
    npz_path, chunk_sec, min_peaks, tol_ms = args
    rec_path = os.path.relpath(npz_path, GENERATION_DIR)[:-4].replace(os.sep, "/")
    subject_id = rec_path.split("/")[-2] if "/" in rec_path else rec_path

    out, drops = [], {}
    try:
        with np.load(npz_path) as z:
            seg_ids = sorted({int(k[3:].split("_")[0]) for k in z.files if k.startswith("seg")})
            for i in seg_ids:
                tk, gk = f"seg{i}_true", f"seg{i}_gen"
                if tk not in z.files or gk not in z.files:
                    continue
                true_full = z[tk].astype(np.float64)
                gen_full = z[gk].astype(np.float64)

                seed_n = int(SEED_SEC * FS)
                n = min(len(true_full), len(gen_full))
                if n <= seed_n:
                    drops["segment_all_seed"] = drops.get("segment_all_seed", 0) + 1
                    continue

                # Generated portion only — never includes the real seed.
                true_gen = true_full[seed_n:n]
                gen_gen = gen_full[seed_n:n]

                chunk_n = int(chunk_sec * FS)
                tol_samples = max(1, round(tol_ms / 1000.0 * FS))

                # Chunks never cross a segment boundary → no splicing.
                for start in range(0, len(true_gen) - chunk_n + 1, chunk_n):
                    end = start + chunk_n
                    m, reason = _analyze_chunk(
                        true_gen[start:end], gen_gen[start:end], FS, tol_samples, min_peaks
                    )
                    if m is None:
                        drops[reason] = drops.get(reason, 0) + 1
                        continue
                    m["subject_id"] = subject_id
                    m["record_path"] = rec_path
                    m["segment_index"] = i
                    # Seconds of blind generation at the START of this chunk.
                    m["gen_offset_sec"] = float(start) / FS
                    out.append(m)
    except Exception as e:
        return rec_path, subject_id, [], {"record_error": 1}, str(e)
    return rec_path, subject_id, out, drops, None


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────
def aggregate(records, drops, subjects_seen, cfg, elapsed):
    def arr(key):
        return np.array([r[key] for r in records if r.get(key) is not None], dtype=float)

    hr_err = arr("hr_abs_error")
    hr_t, hr_p = arr("hr_true"), arr("hr_pred")
    diff = hr_p - hr_t if len(hr_t) == len(hr_p) else np.array([])

    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else None

    intervals = {}
    for k in ("PR_ms", "QRS_ms", "QT_ms"):
        e = arr(f"{k}_abs_error")
        intervals[k] = {
            "mean": float(e.mean()) if len(e) else None,
            "std": float(e.std()) if len(e) else None,
            "median": float(np.median(e)) if len(e) else None,
            "n": int(len(e)),
        }

    summary = {
        "config": cfg,
        "generation_source": GENERATION_DIR,
        "n_subjects": len(subjects_seen),
        "n_chunks_retained": len(records),
        "n_chunks_dropped": {k: int(v) for k, v in sorted(drops.items())},
        "n_chunks_dropped_total": int(sum(drops.values())),
        "retention_rate": (len(records) / (len(records) + sum(drops.values()))
                           if (len(records) + sum(drops.values())) else None),
        "hr": {
            "mae": float(hr_err.mean()) if len(hr_err) else None,
            "mae_std": float(hr_err.std()) if len(hr_err) else None,
            "mae_median": float(np.median(hr_err)) if len(hr_err) else None,
            "n": int(len(hr_err)),
        },
        "bland_altman": {
            "bias": float(diff.mean()) if len(diff) else None,
            "sd": float(diff.std()) if len(diff) else None,
            "loa_lower": float(diff.mean() - 1.96 * diff.std()) if len(diff) else None,
            "loa_upper": float(diff.mean() + 1.96 * diff.std()) if len(diff) else None,
            "within_5_bpm_pct": float((np.abs(diff) <= 5).mean() * 100) if len(diff) else None,
            "within_10_bpm_pct": float((np.abs(diff) <= 10).mean() * 100) if len(diff) else None,
            "n": int(len(diff)),
        },
        "r_peaks": {
            "precision_micro": prec, "recall_micro": rec, "f1_micro": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "f1_macro": float(arr("f1").mean()) if len(arr("f1")) else None,
            "f1_macro_std": float(arr("f1").std()) if len(arr("f1")) else None,
        },
        "intervals": intervals,
        "elapsed_seconds": round(elapsed, 1),
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="Clinical fidelity on windowed (no-splicing) segments")
    p.add_argument("--model_id", required=True,
                   help="Model folder name under drift_generation_windowed/, e.g. "
                        "lightweight_hybrid_20260608_192241")
    p.add_argument("--chunk-sec", type=float, default=CHUNK_SEC)
    p.add_argument("--min-r-peaks", type=int, default=MIN_R_PEAKS)
    p.add_argument("--tolerance-ms", type=float, default=R_PEAK_TOLERANCE_MS)
    p.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    set_paths(resolve_paths(args.model_id))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(GENERATION_DIR):
        sys.exit(f"[ERROR] Generation dir not found: {GENERATION_DIR}\n"
                 "Run run_windowed_drift_test.py first.")

    npz_files = sorted(glob.glob(os.path.join(GENERATION_DIR, "**", "*.npz"), recursive=True))
    if args.max_records:
        npz_files = npz_files[: args.max_records]

    print("=" * 66)
    print(" CLINICAL FIDELITY — WINDOWED SEGMENTS (no splicing)")
    print("=" * 66)
    print(f" Source        : {GENERATION_DIR}")
    print(f" Records       : {len(npz_files)}")
    print(f" Chunk         : {args.chunk_sec}s inside the generated portion (seed {SEED_SEC}s excluded)")
    print(f" Inclusion     : >= {args.min_r_peaks} R peaks in BOTH generated and real")
    print(f" R-peak tol.   : {args.tolerance_ms} ms")
    print(f" Workers       : {args.n_workers}")
    print("=" * 66, flush=True)

    tasks = [(f, args.chunk_sec, args.min_r_peaks, args.tolerance_ms) for f in npz_files]
    records, drops, subjects_seen, errors = [], {}, set(), []
    t0 = time.time()

    with mp.Pool(args.n_workers) as pool:
        for n, (rec_path, subj, out, d, err) in enumerate(
            pool.imap_unordered(process_record, tasks, chunksize=4), 1
        ):
            records.extend(out)
            for k, v in d.items():
                drops[k] = drops.get(k, 0) + v
            if out:
                subjects_seen.add(subj)
            if err:
                errors.append({"record": rec_path, "error": err})
            if n % 50 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  [{n:5d}/{len(tasks)}] chunks={len(records):7d} "
                      f"dropped={sum(drops.values()):7d} subj={len(subjects_seen):3d} "
                      f"elapsed={el:6.0f}s", flush=True)

    cfg = {
        "chunk_sec": args.chunk_sec,
        "seed_sec": SEED_SEC,
        "min_r_peaks": args.min_r_peaks,
        "r_peak_tolerance_ms": args.tolerance_ms,
        "fs": FS,
    }
    summary = aggregate(records, drops, subjects_seen, cfg, time.time() - t0)
    summary["n_records"] = len(npz_files)
    summary["errors"] = errors[:50]

    # Cross-check against the official hold-out split.
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            sp = json.load(f)
        test_set, cv_set = set(sp["test_patients"]), set(sp["cv_patients"])
        summary["split_check"] = {
            "n_test_patients_in_split": len(test_set),
            "subjects_analysed": len(subjects_seen),
            "all_subjects_in_test_split": subjects_seen.issubset(test_set),
            "leaked_from_cv": sorted(subjects_seen & cv_set),
            "test_subjects_absent": sorted(test_set - subjects_seen),
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "per_chunk": records}, f)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    s = summary
    print("\n" + "=" * 66)
    print(" RESULTS")
    print("=" * 66)
    print(f" Subjects analysed : {s['n_subjects']}")
    print(f" Chunks retained   : {s['n_chunks_retained']:,} "
          f"(retention {s['retention_rate']*100:.1f}%)")
    print(f" Chunks dropped    : {s['n_chunks_dropped_total']:,}")
    for k, v in s["n_chunks_dropped"].items():
        print(f"     {k:26s}: {v:,}")
    print(f"\n HR MAE            : {s['hr']['mae']:.2f} ± {s['hr']['mae_std']:.2f} bpm "
          f"(median {s['hr']['mae_median']:.2f}, N={s['hr']['n']:,})")
    ba = s["bland_altman"]
    print(f" Bland-Altman      : bias {ba['bias']:+.2f} bpm, "
          f"LoA [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]")
    print(f"                     within ±5 bpm {ba['within_5_bpm_pct']:.1f}% | "
          f"±10 bpm {ba['within_10_bpm_pct']:.1f}%")
    rp = s["r_peaks"]
    print(f" R-peak (micro)    : P={rp['precision_micro']:.4f} R={rp['recall_micro']:.4f} "
          f"F1={rp['f1_micro']:.4f}")
    print(f"                     TP={rp['tp']:,} FP={rp['fp']:,} FN={rp['fn']:,}")
    for k, lab in [("PR_ms", "PR"), ("QRS_ms", "QRS"), ("QT_ms", "QT")]:
        m = s["intervals"][k]
        print(f" {lab:4s} error        : {m['mean']:.2f} ± {m['std']:.2f} ms "
              f"(median {m['median']:.2f}, N={m['n']:,})")
    if "split_check" in s:
        sc = s["split_check"]
        print(f"\n Split check       : all in test split = {sc['all_subjects_in_test_split']}, "
              f"leaked from CV = {len(sc['leaked_from_cv'])}")
    print("=" * 66)
    print(f" → {RESULTS_PATH}")
    print(f" → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
