import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D


# ─────────────────────────────────────────────────────────────────────────────
# ECGUNet1D  —  U-Net 1D con skip connections per la generazione ECG da PPG
#
# Adatta l'architettura U-Net (Ronneberger et al., 2015), nata per la
# segmentazione biomedica di immagini, alla generazione di segnali 1D.
#
# Motivazione teorica:
#   • Le skip connections permettono al decoder di recuperare dettagli
#     morfologici precisi (es. posizione picchi R, durata QRS) persi durante
#     il downsampling dell'encoder.
#   • L'encoder cattura feature multi-scala: pattern locali (morfologia P/QRS/T)
#     nei layer superficiali, pattern globali (ritmo, RR interval) in profondità.
#   • Il bottleneck BiLSTM integra contesto sequenziale a lungo raggio.
#
# Architettura:
#   Encoder  E1 (32ch) → Pool → E2 (64ch) → Pool → E3 (128ch) → Pool → E4 (256ch)
#   Bottleneck  BiLSTM(256) → proiezione 256ch
#   Decoder  D4 (128ch) ← skip E3 → D3 (64ch) ← skip E2 → D2 (32ch) ← skip E1
#   Output head  Conv1d(32, 1) + resize a target_len
#
# Diversa da tutti gli altri modelli nel progetto:
#   HA-CNN-BiLSTM   → no skip connections, nessun decoder
#   LightweightHybrid → no skip connections
#   BioTransformer  → Transformer puro, nessuna struttura encoder-decoder
#   DualBranchHybrid → due rami paralleli, no skip U-Net
#   PPGWaveNet      → TCN puro, nessun decoder
#   ECGUNet1D       → encoder-decoder con skip connections (questo modello)
# ─────────────────────────────────────────────────────────────────────────────


class ConvBlock1D(nn.Module):
    """Due strati Conv1d + BatchNorm + LeakyReLU (building block di encoder e decoder)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGUNet1D(nn.Module):
    """
    U-Net 1D con 3 livelli di downsampling e bottleneck BiLSTM.

    Hyperparametri via configs dict:
        unet_base_ch    (int,   default 32)   – canali al primo livello
        unet_bidir      (bool,  default True) – BiLSTM nel bottleneck
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
            print(f"[ECGUNet1D] WST ON  → {self.input_dim} canali")
        else:
            self.scattering = None
            self.input_dim  = self.raw_input_channels
            print(f"[ECGUNet1D] WST OFF → {self.input_dim} canali")

        # ── 2. Hyperparametri U-Net ───────────────────────────────────────
        b        = configs.get('unet_base_ch', 32)   # canali base
        bidir    = configs.get('unet_bidir',   True)

        # ── 3. Encoder ────────────────────────────────────────────────────
        self.enc1 = ConvBlock1D(self.input_dim, b)       # → (B, b,    T)
        self.enc2 = ConvBlock1D(b,              b * 2)   # → (B, 2b,   T/2)
        self.enc3 = ConvBlock1D(b * 2,          b * 4)   # → (B, 4b,   T/4)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # ── 4. Bottleneck (BiLSTM) ────────────────────────────────────────
        # Cattura le dipendenze a lungo raggio sulla sequenza compressa
        hidden    = b * 4
        self.bidir = bidir
        lstm_out   = hidden * 2 if bidir else hidden
        self.bottleneck_lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=2, batch_first=True,
            bidirectional=bidir, dropout=0.2,
        )
        self.bottleneck_proj = nn.Sequential(
            nn.Conv1d(lstm_out, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
        )

        # ── 5. Decoder ────────────────────────────────────────────────────
        # Ogni livello: upsample ×2 → concatenazione skip → ConvBlock
        self.dec3 = ConvBlock1D(b * 4 + b * 4, b * 4)   # skip da enc3 (4b)
        self.dec2 = ConvBlock1D(b * 4 + b * 2, b * 2)   # skip da enc2 (2b)
        self.dec1 = ConvBlock1D(b * 2 + b,     b)        # skip da enc1 (b)

        # ── 6. Output head ────────────────────────────────────────────────
        self.output_conv = nn.Conv1d(b, 1, kernel_size=1)

        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Allinea la dimensione temporale di x a quella di ref (±1 campione)."""
        if x.shape[-1] != ref.shape[-1]:
            x = F.interpolate(x, size=ref.shape[-1], mode='linear', align_corners=True)
        return x

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

        # ── Encoder ───────────────────────────────────────────────────────
        e1 = self.enc1(x)                    # (B, b,  T)
        e2 = self.enc2(self.pool(e1))        # (B, 2b, T/2)
        e3 = self.enc3(self.pool(e2))        # (B, 4b, T/4)
        bt = self.pool(e3)                   # (B, 4b, T/8)  → bottleneck input

        # ── Bottleneck ────────────────────────────────────────────────────
        bt_seq, _ = self.bottleneck_lstm(bt.permute(0, 2, 1))   # (B, T/8, lstm_out)
        bt_out    = self.bottleneck_proj(bt_seq.permute(0, 2, 1))  # (B, 4b, T/8)

        # ── Decoder ───────────────────────────────────────────────────────
        d3 = F.interpolate(bt_out, scale_factor=2, mode='linear', align_corners=True)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))   # (B, 4b, T/4)

        d2 = F.interpolate(d3, scale_factor=2, mode='linear', align_corners=True)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))   # (B, 2b, T/2)

        d1 = F.interpolate(d2, scale_factor=2, mode='linear', align_corners=True)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))   # (B, b,  T)

        # ── Output ────────────────────────────────────────────────────────
        out = self.output_conv(d1)                    # (B, 1, T)
        out = F.interpolate(out, size=self.target_len,
                            mode='linear', align_corners=True)

        if self.normalize_01:
            out = self.sigmoid(out)

        return out
