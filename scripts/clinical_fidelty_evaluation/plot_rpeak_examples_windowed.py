#!/usr/bin/env python3
"""
R-peak qualitative examples — WINDOWED paradigm, consistent with the
autoregressive drift pipeline.

Why this script exists
-----------------------
The original rpeak_example_best.png / rpeak_example_median.png (in
evaluate_clinical_fidelity.py) were rendered from drift_generation_clean/,
the superseded SPLICING paradigm. This version instead:
  - reads chunks from clinical_fidelity_windowed_results.json (per_chunk),
    itself derived from drift_generation_windowed/<model_id>/ — the exact
    same no-splicing autoregressive generations used for the drift/stability
    analysis (run_windowed_drift_test.py) and for the PAT analysis
    (run_pat_analysis.py, pat_hr_utils.py).
  - detects R-peaks with pat_hr_utils.detect_r_peaks, the same
    neurokit2(+scipy fallback) routine used throughout the rest of the
    autoregressive-drift/PAT pipeline, instead of a separately maintained
    detector — so peak markers here are directly comparable to the PAT/HR
    numbers reported elsewhere for the same model.

Selection
---------
Chunks are ranked by hr_abs_error (ascending): the best case is the chunk
with the lowest HR error, the median case is the middle-ranked chunk —
the same convention used by the original evaluate_clinical_fidelity.py.

Usage
-----
    conda activate ecg_gen_env
    python scripts/clinical_fidelty_evaluation/plot_rpeak_examples_windowed.py --model_id <model_id>

Outputs (written to <this folder>/<model_id>/)
------------------------------------------------
    rpeak_example_best.png / rpeak_example_median.png
    rpeak_examples_selected.json   (selected chunk ids + metrics)
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.autoregressive_drift_evaluation.pat_hr_utils import detect_r_peaks  # noqa: E402

BASE_GENERATION_DIR = os.path.join(
    PROJECT_ROOT, "scripts", "autoregressive_drift_evaluation",
    "experiments", "drift_generation_windowed",
)
CLINICAL_FIDELITY_DIR = os.path.dirname(os.path.abspath(__file__))

FS = 125
SEED_SEC = 6

C_TRUE = "#2f855a"
C_PRED = "#c53030"
GRID = "#D9D9D9"
INK = "#333333"


def resolve_paths(model_id):
    out_dir = os.path.join(CLINICAL_FIDELITY_DIR, model_id)
    return {
        "GENERATION_DIR": os.path.join(BASE_GENERATION_DIR, model_id),
        "RESULTS_PATH": os.path.join(out_dir, "clinical_fidelity_windowed_results.json"),
        "OUT_DIR": out_dir,
    }


def load_chunk_signal(generation_dir, chunk, chunk_sec, fs):
    """Re-extract the exact (true, gen) sample pair analysed for this chunk."""
    npz_path = os.path.join(generation_dir, chunk["record_path"] + ".npz")
    with np.load(npz_path) as z:
        i = chunk["segment_index"]
        true_full = z[f"seg{i}_true"].astype(np.float64)
        gen_full = z[f"seg{i}_gen"].astype(np.float64)

    seed_n = int(SEED_SEC * fs)
    start = seed_n + int(round(chunk["gen_offset_sec"] * fs))
    end = start + int(round(chunk_sec * fs))
    n = min(len(true_full), len(gen_full), end)
    return true_full[start:n], gen_full[start:n]


def select_examples(model_id):
    """Returns (best, median) dicts, each with true/gen signals + R-peak indices
    + the originating per_chunk metadata, ranked by hr_abs_error (ascending)."""
    paths = resolve_paths(model_id)
    if not os.path.exists(paths["RESULTS_PATH"]):
        sys.exit(f"[ERROR] Results not found: {paths['RESULTS_PATH']}\n"
                 f"Run evaluate_clinical_fidelity_windowed.py --model_id {model_id} first.")
    with open(paths["RESULTS_PATH"]) as f:
        data = json.load(f)
    chunk_sec = data["summary"]["config"]["chunk_sec"]

    per_chunk = [c for c in data["per_chunk"] if c.get("hr_abs_error") is not None]
    if not per_chunk:
        sys.exit("[ERROR] No chunks with a valid hr_abs_error to select examples from.")
    ranked = sorted(per_chunk, key=lambda c: c["hr_abs_error"])

    best_chunk = ranked[0]
    median_chunk = ranked[len(ranked) // 2]

    examples = {}
    for name, chunk in (("best", best_chunk), ("median", median_chunk)):
        true_sig, gen_sig = load_chunk_signal(paths["GENERATION_DIR"], chunk, chunk_sec, FS)
        examples[name] = {
            "chunk": chunk,
            "true": true_sig,
            "gen": gen_sig,
            "true_peaks": detect_r_peaks(true_sig, FS),
            "gen_peaks": detect_r_peaks(gen_sig, FS),
        }
    return examples, paths, chunk_sec


def plot_example(ex, fs, save_path, title):
    true_sig, gen_sig = ex["true"], ex["gen"]
    n = min(len(true_sig), len(gen_sig))
    t = np.arange(n) / fs
    tp = ex["true_peaks"][ex["true_peaks"] < n]
    gp = ex["gen_peaks"][ex["gen_peaks"] < n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, true_sig[:n], color=C_TRUE, linewidth=1.2)
    axes[0].scatter(tp / fs, true_sig[:n][tp], color="black", marker="x", s=40, label="R-peak")
    axes[0].set_title(f"{title} — Real ECG", color=INK)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3, color=GRID)

    axes[1].plot(t, gen_sig[:n], color=C_PRED, linewidth=1.2)
    axes[1].scatter(gp / fs, gen_sig[:n][gp], color="black", marker="x", s=40, label="R-peak")
    axes[1].set_title(f"{title} — Reconstructed ECG (autoregressive, model-scale)", color=INK)
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, color=GRID)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="R-peak qualitative examples (windowed, no-splicing)")
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    examples, paths, chunk_sec = select_examples(args.model_id)
    os.makedirs(paths["OUT_DIR"], exist_ok=True)

    best_err = examples["best"]["chunk"]["hr_abs_error"]
    med_err = examples["median"]["chunk"]["hr_abs_error"]
    best_info = f"{examples['best']['chunk']['subject_id']} / {examples['best']['chunk']['record_path']}"
    med_info = f"{examples['median']['chunk']['subject_id']} / {examples['median']['chunk']['record_path']}"

    plot_example(examples["best"], FS, os.path.join(paths["OUT_DIR"], "rpeak_example_best.png"),
                 title=f"Best case (HR err={best_err:.2f} bpm) — {best_info}")
    plot_example(examples["median"], FS, os.path.join(paths["OUT_DIR"], "rpeak_example_median.png"),
                 title=f"Median case (HR err={med_err:.2f} bpm) — {med_info}")

    with open(os.path.join(paths["OUT_DIR"], "rpeak_examples_selected.json"), "w") as f:
        json.dump({
            "model_id": args.model_id,
            "generation_source": paths["GENERATION_DIR"],
            "chunk_sec": chunk_sec,
            "best": examples["best"]["chunk"],
            "median": examples["median"]["chunk"],
        }, f, indent=2)

    print(f"BEST   : HR err={best_err:.2f} bpm  {best_info}")
    print(f"MEDIAN : HR err={med_err:.2f} bpm  {med_info}")
    print(f"-> {paths['OUT_DIR']}")


if __name__ == "__main__":
    main()
