import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D


# ─────────────────────────────────────────────────────────────────────────────
# PPGWaveNet  —  Dilated Temporal Convolutional Network con Gated Activations
#
# Ispirato a WaveNet (van den Oord et al., 2016) e adattato al task di
# regressione ECG da PPG.  Caratteristiche principali:
#
#   • Stack di WaveNetResidualBlock con dilatazioni esponenziali cicliche
#     [1,2,4,8,16,32] × n_cycles  →  receptive field molto ampio
#   • Gated activation: tanh(W_f*x) ⊙ σ(W_g*x)  (filtro + gate)
#   • Skip connections accumulate → output head comune
#   • Architettura puramente convoluzionale: nessun LSTM, nessun Transformer
#     → alta parallelizzabilità, efficiente su GPU
#   • Conditioning locale opzionale: canali aggiuntivi (PPG_diff, ECG_past)
#     sono fusi nell'embedding iniziale
#
# Diversa da tutti gli altri modelli nel progetto:
#   HA-CNN-BiLSTM   → CNN + LSTM (ricorrente)
#   LightweightHybrid → CNN + LSTM (ricorrente)
#   BioTransformer  → Transformer con attenzione globale
#   DualBranchHybrid → CNN 2D + Dual LSTM
#   ECGUNet1D       → encoder-decoder con skip connections spaziali
#   PPGWaveNet      → TCN puro con gating (questo modello)
# ─────────────────────────────────────────────────────────────────────────────


class WaveNetResidualBlock(nn.Module):
    """
    Blocco residuale WaveNet con gated activation e skip connection.

        h = tanh(W_f * x) ⊙ σ(W_g * x)
        residual = W_r * h + x          (residual path, mantiene la dimensione)
        skip     = W_s * h              (contributo allo skip aggregato)
    """

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # 'same' padding
        self.filter_conv = nn.Conv1d(channels, channels, kernel_size,
                                     dilation=dilation, padding=pad)
        self.gate_conv   = nn.Conv1d(channels, channels, kernel_size,
                                     dilation=dilation, padding=pad)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv     = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn            = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        h = self.bn(h)
        # Allinea la dimensione temporale in caso di padding asimmetrico
        if h.shape[-1] != x.shape[-1]:
            h = h[..., :x.shape[-1]]
        skip     = self.skip_conv(h)
        residual = self.residual_conv(h) + x
        return residual, skip


class PPGWaveNet(nn.Module):
    """
    WaveNet non causale per la regressione ECG da PPG.

    Hyperparametri via configs dict:
        wn_channels     (int,   default 64)   – canali interni dei residual block
        wn_n_cycles     (int,   default 2)    – numero di cicli di dilatazioni
        wn_kernel_size  (int,   default 3)    – kernel delle dilated conv
        apply_wst       (bool,  default False)
        input_channels  (int,   default 3)
        target_len      (int,   default seq_len)
        actual_seq_len  (int,   default 875)  – usato solo se apply_wst=True
        normalize_01    (bool,  default False)
    """

    def __init__(self, configs: dict, seq_len: int = 125):
        super().__init__()

        # ── 1. Configurazione WST ─────────────────────────────────────────
        self.apply_wst          = configs.get('apply_wst', False)
        self.target_len         = configs.get('target_len', seq_len)
        self.normalize_01       = configs.get('normalize_01', False)
        self.raw_input_channels = configs.get('input_channels', 3)

        if self.apply_wst:
            input_size = configs.get('actual_seq_len', 875)
            self.scattering = Scattering1D(J=2, shape=(input_size,), Q=8)
            with torch.no_grad():
                dummy = torch.zeros(1, input_size)
                coeffs_per_sig = self.scattering(dummy).shape[1]
            self.input_dim = coeffs_per_sig * self.raw_input_channels
            print(f"[PPGWaveNet] WST ON  → {self.input_dim} canali")
        else:
            self.scattering = None
            self.input_dim  = self.raw_input_channels
            print(f"[PPGWaveNet] WST OFF → {self.input_dim} canali")

        # ── 2. Hyperparametri WaveNet ──────────────────────────────────────
        channels    = configs.get('wn_channels',    64)
        n_cycles    = configs.get('wn_n_cycles',    2)
        kernel_size = configs.get('wn_kernel_size', 3)

        # ── 3. Embedding iniziale ─────────────────────────────────────────
        self.input_embedding = nn.Sequential(
            nn.Conv1d(self.input_dim, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        # ── 4. Stack di Residual Blocks ───────────────────────────────────
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList([
            WaveNetResidualBlock(channels, d, kernel_size)
            for _ in range(n_cycles)
            for d in dilations
        ])

        # ── 5. Output Head ────────────────────────────────────────────────
        # Aggregazione delle skip connections + 2-step refinement
        self.output_head = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_channels, T)
        Returns:
            (B, 1, target_len)
        """
        # ── WST opzionale ─────────────────────────────────────────────────
        if self.apply_wst and self.scattering is not None:
            expected = self.scattering.shape[0]
            if x.shape[-1] != expected:
                x = F.interpolate(x, size=expected, mode='linear', align_corners=True)
            wst_list = [self.scattering(x[:, i, :].contiguous()) for i in range(x.shape[1])]
            x = torch.nan_to_num(torch.cat(wst_list, dim=1), nan=0.0)

        # ── Embedding ─────────────────────────────────────────────────────
        x = self.input_embedding(x)     # (B, channels, T)

        # ── WaveNet Residual Stack ────────────────────────────────────────
        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # Il residuo finale (x) viene sommato allo skip aggregato: garantisce
        # che residual_conv dell'ultimo blocco riceva gradiente e contribuisce
        # allo stato "profondo" della rete all'output (analogo alla skip finale in ResNet).
        out = self.output_head(skip_sum + x)     # (B, 1, T)
        out = F.interpolate(out, size=self.target_len,
                            mode='linear', align_corners=True)

        if self.normalize_01:
            out = self.sigmoid(out)

        return out
