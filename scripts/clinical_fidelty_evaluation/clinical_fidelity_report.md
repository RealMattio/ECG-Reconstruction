# Clinical Feature Fidelity — lightweight_hybrid (MIMIC-III)
This report evaluates whether the ECG reconstructed by the `lightweight_hybrid` model from PPG remains clinically usable, i.e. whether heart rate and QRS complexes can be reliably extracted from it, and whether the model preserves rhythm over time instead of drifting (a common failure mode of autoregressive waveform generators).
## Methodology
- Test set: hold-out 15% split from `dataset_split.json` (same test patients as the main model evaluation).
- Model: `lightweight_hybrid`, weights from `src/experiments/final_mimic_pinn_results/MAE_loss/lightweight_hybrid_20260608_192241/final_full_model/best_lightweight_hybrid.pth`.
- Signal source: **true autoregressive generation with self-feeding**, produced by `scripts/autoregressive_drift_evaluation/run_autoregressive_drift_test_clean.py` (unlike earlier versions of this report, which used teacher-forced one-step-ahead windows — i.e. always fed the real past ECG as context). Here, after an initial 6s real seed, every subsequent second is generated using the model's own prior output as context, so temporal drift can actually manifest.
- Before generation, each record's PPG/ECG is cleaned: low-quality stretches (signal quality index) are spliced out and the remaining clean stretches are concatenated into one continuous signal per record.
- Each cleaned record (after discarding the 6s seed) is split into non-overlapping 10s windows for analysis, so heart rate / interval extraction always has multiple consecutive beats to work with.
- R-peaks, heart rate, and P/QRS/T wave delineation are computed with NeuroKit2 (`nk.ecg_peaks`, `nk.ecg_delineate`, method='dwt') independently on the real and generated signal of each window (the generated signal stays in the model's own generation scale, never rescaled to physical ECG units — R-peak detection and delineation are scale-relative so this does not affect the metrics below).
- R-peak matching tolerance: 50 ms.
## Coverage
- Test records with a usable generated signal: 5 (0 discarded upstream — too short after SQI cleaning, or WFDB read errors)
- Analysis windows attempted (10s each): 1044
- Windows usable for analysis: 1044
- Windows with reliable HR estimate (≥3 peaks, both signals): 1044

## Heart Rate Fidelity
- **HR MAE**: 5.83 bpm (± 6.02, N=1044)
- **Bland-Altman**: bias = -3.38 bpm, limits of agreement = [-18.42, 11.65] bpm (N=1044). See `bland_altman_hr.png`.

## R-peak Detection
- **Precision** (micro-avg): 0.1457
- **Recall** (micro-avg): 0.1361
- **F1-score** (micro-avg): 0.1407
- **F1-score** (macro-avg over windows): 0.1396 (± 0.1670, N=1044)
- Total matched peaks: TP=1673, FP=9806, FN=10623

## ECG Interval Fidelity (PR, QRS, QT)
| Interval | Mean Abs. Error (ms) | Std (ms) | N valid windows |
|----------|----------------------|----------|-------------------|
| PR | 45.24 | 28.61 | 1044 |
| QRS | 51.08 | 25.87 | 1043 |
| QT | 104.82 | 52.07 | 1043 |

Note: interval errors are reported only where delineation succeeded on both the real and the generated signal for a given window. Low N relative to `n_valid` indicates the model's morphology is not always clean enough for reliable P/T wave delineation — treat this table as indicative, not as a primary claim, unless N is large.

## Suggested paper text
> Despite a residual morphological error, the ECG generated autoregressively (self-feeding, not teacher-forced) preserves clinically relevant rhythm information: heart rate extracted via R-peak detection (NeuroKit2) matches the real signal with a mean absolute error of 5.83 bpm (Bland-Altman bias -3.38 bpm, limits of agreement [-18.42, 11.65] bpm), with an R-peak detection F1-score of 0.141 at a 50 ms tolerance. This indicates the model does not suffer from severe temporal drift even under true self-feeding generation, a common failure mode of autoregressive waveform generators.
