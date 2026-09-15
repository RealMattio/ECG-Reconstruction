# Computational complexity analysis

Characterises the deployed PPG→ECG generator (`LightweightHybrid`, MAE loss, run
`lightweight_hybrid_20260608_192241`) in terms of **trainable parameters**,
**FLOPs per inference window** and **RAM requirements**, for the complexity
section of the paper.

## Run

```bash
conda activate ecg_gen_env
python scripts/computational_complexity/analyze_computational_complexity.py
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--ckpt` | the `best_lightweight_hybrid.pth` of run `20260608_192241` | checkpoint to analyse |
| `--device` | `auto` | `cpu` / `cuda`; drives the latency and memory probes |
| `--batch_sizes` | `1 8 32 64 128 256` | batch sizes swept for the RAM figure |
| `--runs` | `50` | timed repetitions per batch size |
| `--outdir` | `results/` | where JSON, tables and figures land |
| `--paper_md` | `paper_utils/computational_complexity.md` | where the paragraph is written |
| `--no-copy-to-paper` | off | skip copying the figure into `paper_utils/script_for_paper/` |

Latency, real-time factor and measured peak memory are **hardware-dependent** —
re-run on the target machine before quoting them. Parameter counts and FLOPs are
hardware-independent.

## Outputs

Written to `results/`:

| File | Contents |
|---|---|
| `complexity_results.json` | every raw number, per layer and per batch size |
| `parameter_table.md` | per-layer and per-block parameter/FLOP tables |
| `parameter_table.tex` | paper-ready `\begin{table}` (per block) |
| `parameter_table.csv` | same data for spreadsheets |
| `fig_computational_complexity.png` / `.pdf` | (a) FLOPs per stage, (b) RAM vs batch size |

The figure is also copied to `paper_utils/script_for_paper/`, and the write-up
paragraph is generated at `paper_utils/computational_complexity.md`.

## Configuration under analysis

Mirrors `src/main_onlyPPG_PINN_mimic3wdb.py` exactly:

```
input   X = [PPG, ΔPPG, ECG_past]  →  (3, 875)    7 s @ 125 Hz
output  Y = ECG                    →  (1, 125)    1 s @ 125 Hz
WST front-end: Scattering1D(J=2, Q=8, shape=875)  →  8 coeffs × 219 frames per channel
```

The checkpoint additionally carries `scattering.tensor*` entries — the constant
kymatio wavelet filters, saved because the transform was a module attribute at
training time. They are counted separately as *fixed* coefficients, never as
trainable parameters. The script fails loudly if any learned tensor fails to load.

## FLOP accounting convention

1 MAC = 2 FLOPs. Element-wise operations (bias add, non-linearity, pooling
comparison, linear interpolation) cost 1 FLOP per element. BatchNorm is counted at
2 FLOPs/element; folding it into the preceding convolution at deployment removes
that term (~0.03 MFLOPs, 0.1% of the total).

The scattering transform is not a `nn.Module` here (kymatio runs it functionally
inside `forward`), so hooks cannot see it. Its cost is derived analytically from
the operations in `kymatio/scattering1d/core/scattering1d.py`:

- complex FFT / IFFT of length `N` → `5·N·log2(N)` FLOPs (radix-2 Cooley–Tukey;
  kymatio's `rfft` materialises a full complex spectrum, so no real-FFT halving)
- `cdgmm` (complex tensor × real filter) → 2 FLOPs per bin
- `subsample_fourier(x, 2**k)` → `2·(2**k − 1)` FLOPs per output bin
- `modulus` → 4 FLOPs per sample

## Memory measurement

Peak transient memory is **not** sampled from process RSS (the PyTorch allocator
reuses arenas, which makes RSS deltas non-monotonic and unreliable at this model
size). Instead:

- **CUDA** — `torch.cuda.max_memory_allocated()` around one forward pass.
- **CPU** — the profiler's allocation events are replayed in chronological order
  and the running maximum of the live-allocation total is taken.

The learned layers are also measured in isolation (fed pre-computed WST features)
so the FFT working space of the front-end can be reported separately.

## Related

`scripts/count_model_complexity.py` is the older, coarser exploratory script that
compares all six candidate architectures at once with `torchinfo`. This folder
supersedes it for the *deployed* model: it loads the actual trained checkpoint,
accounts for the WST front-end, and emits paper-ready artefacts.
