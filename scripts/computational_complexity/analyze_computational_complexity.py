"""
analyze_computational_complexity.py
===================================
Computational-complexity characterisation of the deployed PPG->ECG model
(LightweightHybrid, MAE loss, run `lightweight_hybrid_20260608_192241`).

It reproduces exactly the inference configuration of the training pipeline
(`src/main_onlyPPG_PINN_mimic3wdb.py` -> `src/mimic_generation_PINN/pipeline.py`):

    input   X = [PPG, dPPG, ECG_past]  ->  (3, 875)    7 s @ 125 Hz
    output  Y = ECG                    ->  (1, 125)    1 s @ 125 Hz
    WST front-end: Scattering1D(J=2, Q=8, shape=875)   -> 8 coeffs x 219 frames

and produces the three artefacts requested for the paper:

  1. TABLE   number of trainable parameters (per block and total)
  2. FIGURE  FLOPs per inference window + RAM requirements
  3. PARAGRAPH  a short markdown write-up in `paper_utils/`

--------------------------------------------------------------------------
FLOP accounting convention (stated explicitly so the numbers are auditable)
--------------------------------------------------------------------------
1 MAC = 2 FLOPs (one multiply + one add). Element-wise operations (bias add,
non-linearity, pooling comparison, interpolation) are counted at 1 FLOP per
element, following the convention used by torchinfo/thop-style counters.

The Wavelet Scattering Transform is *not* a torch module here (kymatio runs it
functionally inside `LightweightHybrid.forward`), so its cost is derived
analytically from the operations in `kymatio/scattering1d/core/scattering1d.py`:

    - complex FFT / IFFT of length N        : 5 * N * log2(N)   FLOPs
      (standard radix-2 Cooley-Tukey estimate; kymatio's `rfft` materialises a
       full complex spectrum, so no real-FFT halving is applied)
    - `cdgmm` (complex tensor x real filter): 2 FLOPs per bin
    - `subsample_fourier(x, 2**k)`          : 2 * (2**k - 1) FLOPs per output bin
    - `modulus` (sqrt(re^2 + im^2))         : 4 FLOPs per sample

Usage
-----
    conda activate ecg_gen_env
    python scripts/computational_complexity/analyze_computational_complexity.py
    # options:
    #   --device cpu|cuda|auto      device used for latency / memory probes
    #   --batch_sizes 1 8 32 ...    batch sizes swept for the RAM figure
    #   --runs 50                   timed repetitions per batch size
    #   --outdir <path>             where figures/tables/json are written
    #   --paper_md <path>           where the markdown paragraph is written
    #   --no-copy-to-paper          do not copy the figure into script_for_paper/
"""
import os
import sys
import gc
import json
import time
import math
import shutil
import argparse
import platform
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import psutil
from torch.profiler import profile, ProfilerActivity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mimic_generation_PINN.models.lightweight_hybrid import LightweightHybrid  # noqa: E402
from src.mimic_generation_PINN.models import lightweight_hybrid as lh_module        # noqa: E402

# --------------------------------------------------------------------------
# Configuration mirrored 1:1 from the training entry point
# --------------------------------------------------------------------------
DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "src", "experiments", "final_mimic_pinn_results", "MAE_loss",
    "lightweight_hybrid_20260608_192241", "final_full_model",
    "best_lightweight_hybrid.pth",
)
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_PAPER_MD = os.path.join(PROJECT_ROOT, "paper_utils", "computational_complexity.md")
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, "paper_utils", "script_for_paper")

TARGET_FS = 125     # Hz
X_SEC = 7           # seconds of context
GEN_SEC = 1         # seconds generated per forward pass
IN_CH = 3           # [PPG, dPPG, ECG_past]
SEQ_LEN = TARGET_FS * X_SEC     # 875
TARGET_LEN = TARGET_FS * GEN_SEC  # 125

MODEL_CONFIGS = {
    "input_channels": IN_CH,
    "actual_seq_len": SEQ_LEN,
    "target_len": TARGET_LEN,
    "apply_wst": True,      # configs['apply_wst'] = True in main_onlyPPG_PINN_mimic3wdb.py
    "normalize_01": False,  # configs['normalize_01'] = False
}

BYTES_FP32 = 4
MB = 1024 ** 2

# Paper figure palette (same tokens as paper_utils/script_for_paper/*.py)
ACCENT = "#4E79A7"
ACCENT_2 = "#F28E2B"
INK = "#333333"
GRID = "#D9D9D9"
MUTED = "#6B7885"


# ==========================================================================
# 1. MODEL LOADING
# ==========================================================================
def build_model(ckpt_path, device):
    """Instantiate LightweightHybrid with the training config and load weights.

    The checkpoint additionally carries `scattering.tensor*` buffers, because at
    training time the Scattering1D object was an attribute of the module. The
    current implementation keeps the transform in a per-device global cache, so
    those keys are expected to be unused: they are constant wavelet filters, not
    learned parameters. We therefore load non-strictly and assert that every
    *model* tensor was matched.
    """
    model = LightweightHybrid(MODEL_CONFIGS, seq_len=TARGET_LEN)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = OrderedDict((k.replace("module.", "", 1) if k.startswith("module.") else k, v)
                        for k, v in state.items())

    wst_filter_elems = sum(v.numel() for k, v in state.items() if k.startswith("scattering."))
    incompatible = model.load_state_dict(state, strict=False)

    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("scattering.")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model definition.\n"
            f"  missing:    {incompatible.missing_keys}\n"
            f"  unexpected: {unexpected}"
        )

    model.to(device).eval()
    return model, wst_filter_elems


# ==========================================================================
# 2. PARAMETER COUNTING
# ==========================================================================
BLOCK_LABELS = {
    "cnn_encoder": "CNN encoder",
    "lstm": "BiLSTM",
    "upsample": "Upsampling",
    "decoder": "Decoder head",
}


def cpu_model_name():
    """Human-readable CPU name (platform.processor() only returns the ISA on Linux)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def count_parameters(model):
    """Return (per-module records, per-block totals, global totals)."""
    per_module = []
    for name, module in model.named_modules():
        if name == "" or len(list(module.children())) > 0:
            continue  # skip container / root
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)
        buffers = sum(b.numel() for b in module.buffers(recurse=False))
        per_module.append({
            "name": name,
            "type": module.__class__.__name__,
            "trainable": trainable,
            "frozen": frozen,
            "buffers": buffers,
        })

    blocks = OrderedDict()
    for rec in per_module:
        block = rec["name"].split(".")[0]
        blocks.setdefault(block, {"trainable": 0, "buffers": 0})
        blocks[block]["trainable"] += rec["trainable"]
        blocks[block]["buffers"] += rec["buffers"]

    totals = {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "buffers": sum(b.numel() for b in model.buffers()),
    }
    return per_module, blocks, totals


# ==========================================================================
# 3. FLOP COUNTING
# ==========================================================================
def _conv1d_cost(m, out):
    l_out = out.shape[-1]
    macs = m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * l_out
    flops = 2 * macs
    if m.bias is not None:
        flops += m.out_channels * l_out
    return macs, flops


def _lstm_cost(m, inp):
    """(B, T, I) -> matmul MACs + element-wise gate arithmetic."""
    _, t_steps, in_size = inp.shape
    h = m.hidden_size
    n_dir = 2 if m.bidirectional else 1
    macs = 0
    flops = 0
    for layer in range(m.num_layers):
        i_size = in_size if layer == 0 else h * n_dir
        step_macs = 4 * h * (i_size + h)          # W_ih @ x  +  W_hh @ h_prev
        macs += step_macs * t_steps * n_dir
        flops += 2 * step_macs * t_steps * n_dir
        # element-wise per step per direction:
        #   bias add 2*4H | 4 gate non-linearities 4H | c = f*c + i*g -> 3H | h = o*tanh(c) -> 2H
        ew = (8 * h if m.bias else 0) + 4 * h + 3 * h + 2 * h
        flops += ew * t_steps * n_dir
    return macs, flops


def profile_network(model, device):
    """Hook every leaf module and record output shape, params, MACs and FLOPs."""
    records = []
    handles = []

    def make_hook(name, module):
        def hook(mod, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            macs, flops = 0, 0
            if isinstance(mod, nn.Conv1d):
                macs, flops = _conv1d_cost(mod, out)
            elif isinstance(mod, nn.LSTM):
                macs, flops = _lstm_cost(mod, inputs[0])
            elif isinstance(mod, nn.BatchNorm1d):
                # eval mode: y = gamma_hat * x + beta_hat -> 2 FLOPs/element
                # (foldable into the preceding conv at deployment time)
                flops = 2 * out.numel()
            elif isinstance(mod, (nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                flops = out.numel()
            elif isinstance(mod, nn.MaxPool1d):
                flops = out.numel() * (mod.kernel_size - 1)
            elif isinstance(mod, nn.Upsample):
                flops = 3 * out.numel()   # linear interp: 2 mul + 1 add
            elif isinstance(mod, nn.Dropout):
                flops = 0                 # identity at inference
            records.append({
                "name": name,
                "type": mod.__class__.__name__,
                "out_shape": tuple(out.shape),
                "trainable": sum(p.numel() for p in mod.parameters(recurse=False)
                                 if p.requires_grad),
                "act_bytes": out.numel() * out.element_size(),
                "macs": int(macs),
                "flops": int(flops),
            })
        return hook

    for name, module in model.named_modules():
        if name == "" or len(list(module.children())) > 0:
            continue
        handles.append(module.register_forward_hook(make_hook(name, module)))

    dummy = torch.rand(1, IN_CH, SEQ_LEN, device=device)
    with torch.no_grad():
        out = model(dummy)
    for h in handles:
        h.remove()

    return records, tuple(out.shape)


def wst_flops(scattering, n_channels):
    """Analytic FLOP count of kymatio's Scattering1D forward pass.

    Follows `kymatio/scattering1d/core/scattering1d.py` operation by operation.
    """
    def fft(n):
        return 5 * n * math.log2(n)

    def cdgmm(n_bins):
        return 2 * n_bins                       # complex x real filter

    def subsample(n_in, k):
        n_out = n_in // (2 ** k)
        return 2 * ((2 ** k) - 1) * n_out, n_out

    n_pad = int(getattr(scattering, "_N_padded", len(scattering.phi_f["levels"][0])))
    log2_T = int(scattering.phi_f["j"])
    oversampling = int(getattr(scattering, "oversampling", 0))
    max_order = int(getattr(scattering, "max_order", 2))
    average = bool(getattr(scattering, "average", True))

    breakdown = OrderedDict()

    # ---- input FFT ----
    breakdown["fft_input"] = fft(n_pad)

    # ---- order 0 (low-pass path) ----
    order0 = 0.0
    if average:
        k0 = max(log2_T - oversampling, 0)
        order0 += cdgmm(n_pad)
        c, n_out = subsample(n_pad, k0)
        order0 += c + fft(n_out)
    breakdown["order0_lowpass"] = order0

    # ---- order 1 (and, if present, order 2) ----
    order1 = 0.0
    order2 = 0.0
    for n1, psi1 in enumerate(scattering.psi1_f):
        j1 = psi1["j"]
        sub1 = min(j1, log2_T) if average else j1
        k1 = max(sub1 - oversampling, 0)

        order1 += cdgmm(n_pad)
        c, n1_bins = subsample(n_pad, k1)
        order1 += c
        order1 += fft(n1_bins)          # ifft
        order1 += 4 * n1_bins           # modulus
        if average or max_order > 1:
            order1 += fft(n1_bins)      # rfft of the modulus
        if average:
            k1_J = max(log2_T - k1 - oversampling, 0)
            order1 += cdgmm(n1_bins)
            c, n_out = subsample(n1_bins, k1_J)
            order1 += c + fft(n_out)

        if max_order == 2:
            for psi2 in scattering.psi2_f:
                j2 = psi2["j"]
                if j2 <= j1:
                    continue
                sub2 = min(j2, log2_T) if average else j2
                k2 = max(sub2 - k1 - oversampling, 0)
                order2 += cdgmm(n1_bins)
                c, n2_bins = subsample(n1_bins, k2)
                order2 += c
                order2 += fft(n2_bins) + 4 * n2_bins
                if average:
                    order2 += fft(n2_bins)
                    k2_T = max(log2_T - k2 - k1 - oversampling, 0)
                    order2 += cdgmm(n2_bins)
                    c, n_out = subsample(n2_bins, k2_T)
                    order2 += c + fft(n_out)
    breakdown["order1_wavelets"] = order1
    breakdown["order2_wavelets"] = order2

    per_channel = sum(breakdown.values())
    return {
        "per_channel": per_channel,
        "total": per_channel * n_channels,
        "breakdown_per_channel": {k: float(v) for k, v in breakdown.items()},
        "n_padded": n_pad,
        "n_psi1": len(scattering.psi1_f),
        "n_psi2": len(scattering.psi2_f),
        "log2_T": log2_T,
        "n_channels": n_channels,
    }


def get_active_scattering(device):
    """Return the Scattering1D instance the model actually used (global cache)."""
    cache = lh_module._SCATTERING_CACHE
    if str(device) in cache:
        return cache[str(device)]
    return next(iter(cache.values())) if cache else None


# ==========================================================================
# 4. MEMORY
# ==========================================================================
class _Backbone(nn.Module):
    """The trainable network alone, fed with pre-computed WST features.

    Used to separate the transient memory of the (fixed) scattering front-end
    from that of the learned layers.
    """

    def __init__(self, model):
        super().__init__()
        self.cnn_encoder = model.cnn_encoder
        self.lstm = model.lstm
        self.upsample = model.upsample
        self.decoder = model.decoder

    def forward(self, x):
        features = self.cnn_encoder(x)
        lstm_out, _ = self.lstm(features.transpose(1, 2))
        return self.decoder(self.upsample(lstm_out.transpose(1, 2)))


def measure_peak_memory(fn, device):
    """Peak *transient* memory of one forward pass, in bytes.

    On CUDA this is `max_memory_allocated`. On CPU we replay the allocation
    events recorded by the PyTorch profiler in chronological order and take the
    running maximum of the live-allocation total; unlike sampling the process
    RSS, this is deterministic and immune to allocator arena reuse.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated() - base)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with torch.no_grad():
            fn()
    events = [e for e in prof.events() if getattr(e, "cpu_memory_usage", 0) != 0]
    events.sort(key=lambda e: e.time_range.start)
    live = 0
    peak = 0
    for e in events:
        live += e.cpu_memory_usage
        peak = max(peak, live)
    return int(peak)


def measure_memory_and_latency(model, device, batch_sizes, n_runs):
    """Peak allocation and forward latency for each batch size."""
    use_cuda = device.type == "cuda"
    backbone = _Backbone(model).eval()
    wst_len = 219  # frames produced by Scattering1D(J=2, Q=8, shape=875)
    wst_ch = IN_CH * 8
    results = []

    for bs in batch_sizes:
        dummy = torch.rand(bs, IN_CH, SEQ_LEN, device=device)
        wst_feat = torch.rand(bs, wst_ch, wst_len, device=device)

        # ---- warm-up (builds the kymatio filter bank, fills allocator arenas) ----
        with torch.no_grad():
            for _ in range(3):
                model(dummy)
                backbone(wst_feat)
        if use_cuda:
            torch.cuda.synchronize()
        gc.collect()

        peak_total = measure_peak_memory(lambda: model(dummy), device)
        peak_backbone = measure_peak_memory(lambda: backbone(wst_feat), device)

        # ---- latency ----
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if use_cuda:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        times = np.asarray(times)

        results.append({
            "batch_size": bs,
            "peak_dynamic_bytes": peak_total,
            "peak_backbone_bytes": peak_backbone,
            "latency_ms_mean": float(times.mean()),
            "latency_ms_std": float(times.std()),
            "latency_ms_median": float(np.median(times)),
            "latency_ms_per_window": float(np.median(times) / bs),
        })
        del dummy, wst_feat
        gc.collect()

    return results


def analytic_activation_bytes(net_records, wst_info, batch_size):
    """Upper bound on activation memory: every intermediate tensor of one pass.

    `net_records` was captured at batch size 1, so bytes scale linearly with the
    batch. The WST term counts the complex working buffers kymatio materialises
    (input spectrum + one wavelet path at a time) plus the concatenated output.
    """
    net = sum(r["act_bytes"] for r in net_records) * batch_size
    n_pad = wst_info["n_padded"]
    n_ch = wst_info["n_channels"]
    # per channel: padded real signal + complex input spectrum (2 floats/bin)
    #              + one complex sub-band buffer at the coarsest subsampling
    wst = n_ch * (n_pad + 2 * n_pad + 2 * n_pad) * BYTES_FP32 * batch_size
    io = (IN_CH * SEQ_LEN + TARGET_LEN) * BYTES_FP32 * batch_size
    return net + wst + io


# ==========================================================================
# 5. OUTPUT: TABLES
# ==========================================================================
def write_parameter_tables(outdir, per_module, blocks, totals, net_records,
                           wst_info, wst_filter_elems):
    """Markdown + LaTeX + CSV parameter/complexity tables."""
    flops_by_module = {r["name"]: r for r in net_records}

    rows = []
    for rec in per_module:
        prof = flops_by_module.get(rec["name"], {})
        rows.append({
            "block": BLOCK_LABELS.get(rec["name"].split(".")[0], rec["name"].split(".")[0]),
            "layer": rec["name"],
            "type": rec["type"],
            "out_shape": prof.get("out_shape", ("-",)),
            "params": rec["trainable"],
            "flops": prof.get("flops", 0),
        })

    total_net_flops = sum(r["flops"] for r in net_records)
    total_flops = total_net_flops + wst_info["total"]

    # ---------------- Markdown ----------------
    md = []
    md.append("# Computational complexity — LightweightHybrid (deployed model)\n")
    md.append(f"Input `(3, {SEQ_LEN})` = {X_SEC} s @ {TARGET_FS} Hz  ->  "
              f"output `(1, {TARGET_LEN})` = {GEN_SEC} s @ {TARGET_FS} Hz\n")
    md.append("\n## Table 1 — Trainable parameters and FLOPs per layer\n")
    md.append("| Block | Layer | Type | Output shape | Trainable params | FLOPs |")
    md.append("|---|---|---|---|---:|---:|")
    md.append(f"| WST front-end | `scattering` | Scattering1D (J=2, Q=8) | "
              f"(1, {IN_CH * 8}, 219) | 0 | {wst_info['total']:,.0f} |")
    for r in rows:
        shape = "x".join(str(s) for s in r["out_shape"]) if r["out_shape"] != ("-",) else "-"
        md.append(f"| {r['block']} | `{r['layer']}` | {r['type']} | {shape} | "
                  f"{r['params']:,} | {r['flops']:,} |")
    md.append(f"| **Total** | | | | **{totals['trainable']:,}** | **{total_flops:,.0f}** |")

    md.append("\n## Table 2 — Trainable parameters per block\n")
    md.append("| Block | Trainable params | Share |")
    md.append("|---|---:|---:|")
    for block, vals in blocks.items():
        share = 100.0 * vals["trainable"] / totals["trainable"] if totals["trainable"] else 0.0
        md.append(f"| {BLOCK_LABELS.get(block, block)} | {vals['trainable']:,} | {share:.1f}% |")
    md.append(f"| **Total trainable** | **{totals['trainable']:,}** | **100.0%** |")
    md.append(f"| Non-trainable buffers (BatchNorm statistics) | {totals['buffers']:,} | — |")
    md.append(f"| Fixed WST filter coefficients | {wst_filter_elems:,} | — |")

    md_path = os.path.join(outdir, "parameter_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    # ---------------- LaTeX (per-block, the paper-ready one) ----------------
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Computational footprint of the deployed PPG$\rightarrow$ECG "
               r"generator. Input: $7$~s of context at $125$~Hz "
               r"($3\times875$); output: $1$~s of ECG ($1\times125$). "
               r"FLOPs are reported per inference window ($1$~MAC $=2$~FLOPs).}")
    tex.append(r"\label{tab:complexity}")
    tex.append(r"\begin{tabular}{lrrr}")
    tex.append(r"\toprule")
    tex.append(r"Block & Trainable params & Share (\%) & FLOPs (M) \\")
    tex.append(r"\midrule")
    tex.append(rf"WST front-end (J{{=}}2, Q{{=}}8) & 0 & 0.0 & "
               rf"{wst_info['total'] / 1e6:.2f} \\")
    block_flops = OrderedDict()
    for r in net_records:
        b = r["name"].split(".")[0]
        block_flops[b] = block_flops.get(b, 0) + r["flops"]
    for block, vals in blocks.items():
        share = 100.0 * vals["trainable"] / totals["trainable"] if totals["trainable"] else 0.0
        tex.append(rf"{BLOCK_LABELS.get(block, block)} & {vals['trainable']:,} & "
                   rf"{share:.1f} & {block_flops.get(block, 0) / 1e6:.2f} \\")
    tex.append(r"\midrule")
    tex.append(rf"\textbf{{Total}} & \textbf{{{totals['trainable']:,}}} & "
               rf"\textbf{{100.0}} & \textbf{{{total_flops / 1e6:.2f}}} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex_path = os.path.join(outdir, "parameter_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")

    # ---------------- CSV ----------------
    csv_path = os.path.join(outdir, "parameter_table.csv")
    with open(csv_path, "w") as f:
        f.write("block,layer,type,output_shape,trainable_params,flops\n")
        f.write(f"WST front-end,scattering,Scattering1D,"
                f"1x{IN_CH * 8}x219,0,{wst_info['total']:.0f}\n")
        for r in rows:
            shape = "x".join(str(s) for s in r["out_shape"]) if r["out_shape"] != ("-",) else "-"
            f.write(f"{r['block']},{r['layer']},{r['type']},{shape},"
                    f"{r['params']},{r['flops']}\n")

    return md_path, tex_path, csv_path, block_flops, total_net_flops, total_flops


# ==========================================================================
# 6. OUTPUT: FIGURE
# ==========================================================================
def _style_axis(ax, axis="both"):
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7, axis=axis)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def make_figure(outdir, stage_flops, total_flops, mem_rows, static_mb, copy_to_paper):
    """Two panels: (a) FLOPs per inference window, (b) RAM requirements."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---------------- (a) FLOPs breakdown ----------------
    labels = list(stage_flops.keys())
    values = np.array([stage_flops[k] / 1e6 for k in labels])
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]
    y = np.arange(len(labels))

    ax_a.barh(y, values, height=0.62, color=ACCENT, edgecolor="none")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, color=INK, fontsize=8.5)
    ax_a.set_xlabel("MFLOPs per inference window (1 s of ECG)", color=INK, fontsize=9)
    ax_a.set_xlim(0, values.max() * 1.28)
    for yi, v in zip(y, values):
        ax_a.text(v + values.max() * 0.02, yi,
                  f"{v:.2f} M  ({100 * v / (total_flops / 1e6):.1f}%)",
                  va="center", ha="left", color=INK, fontsize=8)
    _style_axis(ax_a, axis="x")
    ax_a.set_title(f"(a) FLOPs per inference window — total {total_flops / 1e6:.2f} MFLOPs",
                   fontsize=10, color=INK, fontweight="bold", loc="left")

    # ---------------- (b) RAM requirements ----------------
    # Log-log: the constant weight footprint and the 256-window activation peak
    # differ by three orders of magnitude, so a linear axis would hide the former.
    batches = np.array([r["batch_size"] for r in mem_rows], dtype=float)
    total = np.array([r["total_mb"] for r in mem_rows])

    ax_b.plot(batches, total, color=ACCENT, linewidth=2.0, marker="o", markersize=7,
              markeredgecolor="white", markeredgewidth=1.5, zorder=3,
              label="Peak RAM (weights + activations)")
    ax_b.axhline(static_mb, color=ACCENT_2, linewidth=1.8, linestyle="--", zorder=2,
                 label=f"Model weights + WST filters ({static_mb:.2f} MB, constant)")
    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xticks(batches)
    ax_b.set_xticklabels([str(int(b)) for b in batches], color=INK, fontsize=8.5)
    ax_b.minorticks_off()
    ax_b.set_xlim(batches.min() * 0.7, batches.max() * 2.4)
    ax_b.set_ylim(static_mb * 0.45, total.max() * 3.2)
    ax_b.set_xlabel("Inference batch size (windows)", color=INK, fontsize=9)
    ax_b.set_ylabel("Peak RAM (MB, log scale)", color=INK, fontsize=9)
    for bx, tv in zip(batches, total):
        ax_b.annotate(f"{tv:.2f} MB" if tv < 10 else f"{tv:.0f} MB",
                      (bx, tv), textcoords="offset points", xytext=(0, 9),
                      ha="center", color=INK, fontsize=8)
    ax_b.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    _style_axis(ax_b, axis="both")
    ax_b.set_title("(b) RAM requirements at inference",
                   fontsize=10, color=INK, fontweight="bold", loc="left")

    fig.tight_layout()
    png = os.path.join(outdir, "fig_computational_complexity.png")
    pdf = os.path.join(outdir, "fig_computational_complexity.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    if copy_to_paper and os.path.isdir(PAPER_FIG_DIR):
        for src in (png, pdf):
            shutil.copy2(src, os.path.join(PAPER_FIG_DIR, os.path.basename(src)))

    return png, pdf


# ==========================================================================
# 7. OUTPUT: PAPER PARAGRAPH
# ==========================================================================
def write_paper_paragraph(path, res):
    p = res["params"]
    f = res["flops"]
    m = res["memory"]
    lat = res["latency"]
    b1 = next(r for r in res["memory"]["per_batch"] if r["batch_size"] == 1)

    total_mflops = f["total"] / 1e6
    wst_share = 100.0 * f["wst"]["total"] / f["total"]
    lstm_share = 100.0 * f["by_block"].get("lstm", 0) / f["total"]
    dec_share = 100.0 * f["by_block"].get("decoder", 0) / f["total"]
    cnn_share = 100.0 * f["by_block"].get("cnn_encoder", 0) / f["total"]

    rt_budget_ms = GEN_SEC * 1000.0
    rtf = b1["latency_ms_median"] / rt_budget_ms

    lines = []
    lines.append("# Computational complexity\n")
    lines.append("_Auto-generated by "
                 "`scripts/computational_complexity/analyze_computational_complexity.py`. "
                 "Do not edit by hand — re-run the script instead._\n")

    lines.append("\n## Paragraph for the paper\n")
    lines.append(
        f"**Computational complexity.** The deployed generator is deliberately compact: "
        f"it holds only **{p['trainable']:,} trainable parameters** "
        f"({p['trainable_mb']:.2f} MB in single precision), plus {p['buffers']:,} "
        f"non-trainable BatchNorm statistics and {p['wst_filter_elems']:,} fixed wavelet "
        f"coefficients that define the scattering front-end and are never updated by "
        f"back-propagation. The recurrent core dominates the parameter budget "
        f"({100.0 * p['by_block'].get('lstm', 0) / p['trainable']:.0f}% of the weights), "
        f"while the convolutional encoder and the decoder head account for "
        f"{100.0 * p['by_block'].get('cnn_encoder', 0) / p['trainable']:.0f}% and "
        f"{100.0 * p['by_block'].get('decoder', 0) / p['trainable']:.0f}%, respectively. "
        f"A single inference window — {X_SEC} s of PPG context at {TARGET_FS} Hz "
        f"(3x{SEQ_LEN} samples) mapped to {GEN_SEC} s of ECG (1x{TARGET_LEN} samples) — "
        f"costs **{total_mflops:.1f} MFLOPs** (1 MAC = 2 FLOPs). Of these, "
        f"{f['wst']['total'] / 1e6:.2f} MFLOPs ({wst_share:.1f}%) are spent in the "
        f"wavelet-scattering front-end, {f['by_block'].get('cnn_encoder', 0) / 1e6:.2f} "
        f"MFLOPs ({cnn_share:.1f}%) in the CNN encoder, "
        f"{f['by_block'].get('lstm', 0) / 1e6:.2f} MFLOPs ({lstm_share:.1f}%) in the "
        f"BiLSTM, and {f['by_block'].get('decoder', 0) / 1e6:.2f} MFLOPs "
        f"({dec_share:.1f}%) in the decoder head. Peak memory at single-window "
        f"inference is **{b1['total_mb']:.2f} MB** "
        f"({m['static_mb']:.2f} MB of weights and filters plus "
        f"{b1['peak_dynamic_mb']:.2f} MB of transient activations, of which only "
        f"{b1['peak_backbone_mb']:.2f} MB belong to the learned layers — the rest is the "
        f"FFT working space of the scattering transform), and it grows "
        f"linearly with the batch size, reaching {res['memory']['per_batch'][-1]['total_mb']:.1f} MB at a "
        f"batch of {res['memory']['per_batch'][-1]['batch_size']} windows. Training the same network "
        f"requires only {m['training_mb']:.2f} MB for weights, gradients and the two Adam "
        f"moments. On the CPU used for this measurement ({lat['device_name']}, no GPU "
        f"acceleration) a single window is generated in {b1['latency_ms_median']:.2f} ms, "
        f"i.e. {1.0 / rtf:.0f}x faster than the {rt_budget_ms:.0f} ms of ECG it produces; "
        f"since streaming operation is autoregressive, reconstructing T seconds of ECG "
        f"costs T x {total_mflops:.1f} MFLOPs and T x {b1['latency_ms_median']:.2f} ms "
        f"while the memory footprint stays constant. The model therefore sustains "
        f"real-time generation with a wide margin on commodity CPU hardware, and its "
        f"sub-megabyte weight footprint together with a ~{b1['total_mb']:.0f} MB working "
        f"set places it within the envelope of a smartwatch-class application processor "
        f"without requiring a dedicated accelerator."
    )

    lines.append("\n\n## Numbers at a glance\n")
    lines.append("| Quantity | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Trainable parameters | {p['trainable']:,} |")
    lines.append(f"| Non-trainable buffers (BatchNorm) | {p['buffers']:,} |")
    lines.append(f"| Fixed WST filter coefficients | {p['wst_filter_elems']:,} |")
    lines.append(f"| Model size (fp32) | {p['trainable_mb']:.2f} MB |")
    lines.append(f"| Model size (int8 weights, 1 B/param — projected) | {p['trainable'] / MB:.2f} MB |")
    lines.append(f"| FLOPs per inference window ({GEN_SEC} s of ECG) | {total_mflops:.2f} M |")
    lines.append(f"|   - WST front-end | {f['wst']['total'] / 1e6:.2f} M ({wst_share:.1f}%) |")
    lines.append(f"|   - CNN encoder | {f['by_block'].get('cnn_encoder', 0) / 1e6:.2f} M ({cnn_share:.1f}%) |")
    lines.append(f"|   - BiLSTM | {f['by_block'].get('lstm', 0) / 1e6:.2f} M ({lstm_share:.1f}%) |")
    lines.append(f"|   - Decoder head | {f['by_block'].get('decoder', 0) / 1e6:.2f} M ({dec_share:.1f}%) |")
    lines.append(f"| MACs per inference window | {f['total_macs'] / 1e6:.2f} M |")
    lines.append(f"| Static memory (weights + buffers + WST filters) | {m['static_mb']:.2f} MB |")
    lines.append(f"| Peak inference RAM, batch = 1 | {b1['total_mb']:.2f} MB |")
    lines.append(f"|   - transient activations (full model) | {b1['peak_dynamic_mb']:.2f} MB |")
    lines.append(f"|   - transient activations (learned layers only) | {b1['peak_backbone_mb']:.2f} MB |")
    lines.append(f"| Peak inference RAM, batch = {res['memory']['per_batch'][-1]['batch_size']} | "
                 f"{res['memory']['per_batch'][-1]['total_mb']:.1f} MB |")
    lines.append(f"| Training memory (weights + grads + Adam) | {m['training_mb']:.2f} MB |")
    lines.append(f"| Full Python + PyTorch process footprint | {m['process_rss_mb']:.0f} MB |")
    lines.append(f"| Latency per window ({lat['device_name']}) | "
                 f"{b1['latency_ms_median']:.2f} ms |")
    lines.append(f"| Real-time factor ({GEN_SEC} s generated / latency) | {1.0 / rtf:.0f}x |")
    lines.append(f"| Throughput at batch {res['memory']['per_batch'][-1]['batch_size']} | "
                 f"{1000.0 / res['memory']['per_batch'][-1]['latency_ms_per_window']:,.0f} windows/s |")

    lines.append("\n## Figure\n")
    lines.append("![Computational complexity](script_for_paper/fig_computational_complexity.png)\n")
    lines.append("`fig_computational_complexity.pdf` — (a) FLOPs per inference window broken "
                 "down by stage; (b) peak RAM at inference versus batch size, split into the "
                 "constant weight/filter footprint and the transient activation memory.\n")

    lines.append("\n## Reproduction\n")
    lines.append("```bash\nconda activate ecg_gen_env\n"
                 "python scripts/computational_complexity/analyze_computational_complexity.py\n```\n")
    lines.append(f"\nCheckpoint: `{os.path.relpath(res['meta']['checkpoint'], PROJECT_ROOT)}`  \n")
    lines.append(f"Measured on: {res['meta']['host']} / {res['meta']['device']} / "
                 f"torch {res['meta']['torch_version']} — {res['meta']['timestamp']}\n")
    lines.append("\nFLOP convention: 1 MAC = 2 FLOPs; element-wise ops (bias, non-linearity, "
                 "pooling, interpolation) = 1 FLOP/element. The scattering front-end is costed "
                 "analytically from kymatio's FFT-domain implementation "
                 "(complex FFT of length N = 5*N*log2(N) FLOPs). BatchNorm is counted at "
                 "2 FLOPs/element; folding it into the preceding convolution at deployment "
                 "time removes that term.\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Computational-complexity analysis of the deployed LightweightHybrid model")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    parser.add_argument("--runs", type=int, default=50, help="timed repetitions per batch size")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--paper_md", type=str, default=DEFAULT_PAPER_MD)
    parser.add_argument("--no-copy-to-paper", dest="copy_to_paper", action="store_false",
                        help="do not copy the figure into paper_utils/script_for_paper/")
    parser.set_defaults(copy_to_paper=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("  COMPUTATIONAL COMPLEXITY — LightweightHybrid (PPG -> ECG)")
    print("=" * 78)
    print(f"  checkpoint : {os.path.relpath(args.ckpt, PROJECT_ROOT)}")
    print(f"  device     : {device}")
    print(f"  window     : ({IN_CH}, {SEQ_LEN}) [{X_SEC}s @ {TARGET_FS}Hz]"
          f"  ->  (1, {TARGET_LEN}) [{GEN_SEC}s]")
    print("-" * 78)

    # ---- 1. model ----
    model, wst_filter_elems = build_model(args.ckpt, device)

    # ---- 2. parameters ----
    per_module, blocks, totals = count_parameters(model)

    # ---- 3. FLOPs ----
    net_records, out_shape = profile_network(model, device)
    assert out_shape == (1, 1, TARGET_LEN), f"unexpected output shape {out_shape}"
    scattering = get_active_scattering(device)
    if scattering is None:
        raise RuntimeError("WST cache empty — the model did not run the scattering transform.")
    wst = wst_flops(scattering, IN_CH)

    total_net_macs = sum(r["macs"] for r in net_records)
    (md_path, tex_path, csv_path, block_flops,
     total_net_flops, total_flops) = write_parameter_tables(
        args.outdir, per_module, blocks, totals, net_records, wst, wst_filter_elems)

    # ---- 4. memory + latency ----
    static_bytes = (totals["trainable"] + totals["frozen"] + totals["buffers"]
                    + wst_filter_elems) * BYTES_FP32
    mem_rows = measure_memory_and_latency(model, device, args.batch_sizes, args.runs)
    for r in mem_rows:
        r["peak_dynamic_mb"] = r["peak_dynamic_bytes"] / MB
        r["peak_backbone_mb"] = r["peak_backbone_bytes"] / MB
        r["analytic_activation_mb"] = analytic_activation_bytes(
            net_records, wst, r["batch_size"]) / MB
        r["total_mb"] = static_bytes / MB + r["peak_dynamic_mb"]

    process_rss_mb = psutil.Process(os.getpid()).memory_info().rss / MB

    device_name = (torch.cuda.get_device_name(device) if device.type == "cuda"
                   else cpu_model_name())

    # ---- 5. assemble results ----
    results = {
        "meta": {
            "checkpoint": os.path.abspath(args.ckpt),
            "model": "LightweightHybrid",
            "device": str(device),
            "device_name": device_name,
            "host": platform.node(),
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_shape": [IN_CH, SEQ_LEN],
            "output_shape": [1, TARGET_LEN],
            "target_fs": TARGET_FS,
            "context_sec": X_SEC,
            "generated_sec": GEN_SEC,
            "flop_convention": "1 MAC = 2 FLOPs; element-wise ops = 1 FLOP/element",
        },
        "params": {
            "trainable": totals["trainable"],
            "frozen": totals["frozen"],
            "buffers": totals["buffers"],
            "wst_filter_elems": wst_filter_elems,
            "trainable_mb": totals["trainable"] * BYTES_FP32 / MB,
            "by_block": {k: v["trainable"] for k, v in blocks.items()},
            "per_module": per_module,
        },
        "flops": {
            "total": total_flops,
            "network": total_net_flops,
            "total_macs": total_net_macs,
            "by_block": {k: int(v) for k, v in block_flops.items()},
            "wst": wst,
            "per_layer": net_records,
        },
        "memory": {
            "static_bytes": static_bytes,
            "static_mb": static_bytes / MB,
            "training_mb": totals["trainable"] * 16 / MB,  # w + grad + Adam m,v (fp32)
            "process_rss_mb": process_rss_mb,
            "per_batch": mem_rows,
        },
        "latency": {
            "device_name": device_name,
            "runs": args.runs,
            "per_batch": [{k: r[k] for k in
                           ("batch_size", "latency_ms_mean", "latency_ms_std",
                            "latency_ms_median", "latency_ms_per_window")} for r in mem_rows],
        },
    }

    json_path = os.path.join(args.outdir, "complexity_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- 6. figure ----
    stage_flops = OrderedDict()
    stage_flops["WST front-end\n(J=2, Q=8)"] = wst["total"]
    stage_flops["CNN encoder"] = block_flops.get("cnn_encoder", 0)
    stage_flops["BiLSTM"] = block_flops.get("lstm", 0)
    stage_flops["Upsampling"] = block_flops.get("upsample", 0)
    stage_flops["Decoder head"] = block_flops.get("decoder", 0)
    png, pdf = make_figure(args.outdir, stage_flops, total_flops, mem_rows,
                           static_bytes / MB, args.copy_to_paper)

    # ---- 7. paper paragraph ----
    md_paper = write_paper_paragraph(args.paper_md, results)

    # ---- 8. console report ----
    print(f"  Trainable parameters ....... {totals['trainable']:,}"
          f"  ({totals['trainable'] * BYTES_FP32 / MB:.2f} MB fp32)")
    print(f"  Non-trainable buffers ...... {totals['buffers']:,}")
    print(f"  Fixed WST filter coeffs .... {wst_filter_elems:,}")
    print(f"  MACs per window ............ {total_net_macs / 1e6:.2f} M")
    print(f"  FLOPs per window ........... {total_flops / 1e6:.2f} M "
          f"(network {total_net_flops / 1e6:.2f} M + WST {wst['total'] / 1e6:.2f} M)")
    print("-" * 78)
    print(f"  {'batch':>6}  {'peak RAM':>11}  {'of which net':>13}  "
          f"{'latency':>12}  {'per window':>12}")
    for r in mem_rows:
        print(f"  {r['batch_size']:>6}  {r['total_mb']:>8.2f} MB  "
              f"{r['peak_backbone_mb']:>10.2f} MB  "
              f"{r['latency_ms_median']:>9.2f} ms  {r['latency_ms_per_window']:>9.3f} ms")
    print("-" * 78)
    for label, path in (("json", json_path), ("table (md)", md_path),
                        ("table (tex)", tex_path), ("table (csv)", csv_path),
                        ("figure", png), ("figure", pdf), ("paper", md_paper)):
        print(f"  [{label:>11}] {os.path.relpath(path, PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
