"""
count_model_complexity.py
=========================
Calcola e stampa per ogni modello:
  - Numero di parametri trainable
  - FLOPs totali (MACs × 2) per un singolo campione
  - Stima VRAM occupata da pesi + gradienti + stati Adam (fp32)
  - Tempo di inferenza medio su CPU (batch 256, 5 run)

Uso:
    python scripts/count_model_complexity.py
    python scripts/count_model_complexity.py --batch 128
    python scripts/count_model_complexity.py --wst   # include FLOPs della WST

Dipendenze aggiuntive (già in requirements-local.txt):
    pip install torchinfo
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torchinfo import summary

from src.mimic_generation_PINN.models.ha_cnn_bilstm_autoregressive import HACNNBiLSTM_AR
from src.mimic_generation_PINN.models.dual_branch_hybrid import DualBranchHybrid
from src.mimic_generation_PINN.models.lightweight_hybrid import LightweightHybrid
from src.mimic_generation_PINN.models.bio_transformer import BioTransformer
from src.mimic_generation_PINN.models.ppg_wavenet import PPGWaveNet
from src.mimic_generation_PINN.models.ecg_unet1d import ECGUNet1D

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI  (devono rispecchiare la configurazione usata nell'addestramento)
# ─────────────────────────────────────────────────────────────────────────────
IN_CH      = 3     # [PPG, ΔPPG, ECG_past]
SEQ_LEN    = 875   # 7 s × 125 Hz
TARGET_LEN = 125   # 1 s × 125 Hz
N_RUNS     = 5     # ripetizioni per la stima del tempo

MODELS = [
    ('ha_cnn_bilstm_ar',   HACNNBiLSTM_AR,   'HA-CNN-BiLSTM-AR'),
    ('lightweight_hybrid', LightweightHybrid, 'LightweightHybrid'),
    ('ppg_wavenet',        PPGWaveNet,        'PPGWaveNet'),
    ('ecg_unet1d',         ECGUNet1D,         'ECGUNet1D'),
    ('bio_transformer',    BioTransformer,    'PhysioFormer (BioTransformer)'),
    ('dual_branch_hybrid', DualBranchHybrid,  'DualBranchHybrid'),
]


def measure_inference_time(model: torch.nn.Module, dummy: torch.Tensor, n_runs: int) -> float:
    """Restituisce il tempo medio di forward pass in ms (CPU)."""
    model.eval()
    # Warmup
    with torch.no_grad():
        model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="Conta parametri e FLOPs dei modelli PPG→ECG")
    parser.add_argument('--batch', type=int, default=256, help="Batch size per la stima del tempo (default: 256)")
    parser.add_argument('--wst',   action='store_true',   help="Usa WST come preprocessing (più FLOPs)")
    args = parser.parse_args()

    cfg = {
        'input_channels': IN_CH,
        'actual_seq_len': SEQ_LEN,
        'target_len':     TARGET_LEN,
        'apply_wst':      args.wst,
        'normalize_01':   False,
    }

    dummy_single = torch.rand(1,         IN_CH, SEQ_LEN)
    dummy_batch  = torch.rand(args.batch, IN_CH, SEQ_LEN)

    wst_tag = " [WST ON]" if args.wst else ""
    print(f"\n{'='*80}")
    print(f"  Complessità modelli PPG→ECG  |  Input: ({IN_CH}, {SEQ_LEN})  →  Target: ({TARGET_LEN},){wst_tag}")
    print(f"{'='*80}\n")

    results = []
    for key, Cls, label in MODELS:
        print(f"── {label} ──")
        model = Cls(cfg, seq_len=TARGET_LEN)

        # ── Parametri ──────────────────────────────────────────────────────
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ── FLOPs (MACs × 2) tramite torchinfo ─────────────────────────────
        try:
            info = summary(model, input_data=dummy_single, verbose=0,
                           col_names=["output_size", "num_params", "mult_adds"])
            total_macs = info.total_mult_adds
            flops = total_macs * 2
        except Exception as e:
            print(f"   [WARN] FLOPs non calcolabili: {e}")
            flops = None

        # ── VRAM stimata ────────────────────────────────────────────────────
        # pesi fp32 (4 B) + gradienti (4 B) + Adam m+v (8 B) = 16 B / param
        vram_mb = n_params * 16 / 1024**2

        # ── Tempo di inferenza CPU ──────────────────────────────────────────
        t_ms = measure_inference_time(model, dummy_batch, N_RUNS)

        results.append({
            'key':      key,
            'label':    label,
            'params':   n_params,
            'flops':    flops,
            'vram_mb':  vram_mb,
            't_ms':     t_ms,
        })
        print()

    # ── Tabella riepilogativa (ordinata per parametri) ──────────────────────
    results.sort(key=lambda r: r['params'])

    col_w = 28
    print(f"{'─'*90}")
    print(f"  {'Modello':<{col_w}}  {'Params':>10}  {'FLOPs (M)':>10}  {'VRAM* (MB)':>11}  {'CPU fwd ms':>12}")
    print(f"  {'':─<{col_w}}  {'':─>10}  {'':─>10}  {'':─>11}  {'':─>12}")
    for r in results:
        flops_str = f"{r['flops']/1e6:.1f}" if r['flops'] else "N/A"
        print(f"  {r['label']:<{col_w}}  {r['params']:>10,}  {flops_str:>10}  {r['vram_mb']:>9.1f} MB  "
              f"{r['t_ms']:>9.1f} ms  (batch={args.batch})")
    print(f"{'─'*90}")
    print(f"  * VRAM = pesi + gradienti + stati Adam (fp32). Esclude attivazioni e data batch.\n")

    print("  Ordine consigliato di addestramento su GPU A40 (48 GB VRAM):")
    for i, r in enumerate(results, 1):
        vram_str = f"~{r['vram_mb']:.0f} MB pesi"
        flops_str = f"{r['flops']/1e6:.0f}M FLOPs" if r['flops'] else ""
        extra = f"  ({vram_str}{', ' + flops_str if flops_str else ''})"
        print(f"    {i}. {r['label']}{extra}")
    print()


if __name__ == '__main__':
    main()
