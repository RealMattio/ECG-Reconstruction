import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from kymatio.torch import Scattering1D


# ─────────────────────────────────────────────────────────────────────────────
# PhysioFormer  (registrato come 'bio_transformer' nella ModelFactory)
#
# Architettura multi-scala per la ricostruzione ECG da PPG.
# Rispetto alla versione precedente (BioTransformer), introduce:
#   1. Multi-Scale Tokenizer  – 3 branch Conv1d (kernel 1, 5, 15) concatenati
#   2. Learnable Positional Embedding  – più flessibile dell'encoding sinusoidale
#   3. Transformer Encoder più profondo  (default: 4 layer, d_model=128, nhead=8)
#   4. Decoder progressivo  – upsampling ×2 iterativo + Conv di raffinamento
#   5. Residual PPG Bypass  – il primo canale di input (PPG raw) è proiettato
#      e sommato all'output (inductive bias: ECG e PPG condividono l'HR)
# ─────────────────────────────────────────────────────────────────────────────


class MultiScaleTokenizer(nn.Module):
    """
    Proietta l'input in embedding di dimensione d_model usando 3 kernel di
    diverse scale temporali (fine=1, media=5, grossa=15).
    L'output è una feature map (B, d_model, T).
    """
    def __init__(self, input_channels: int, d_model: int):
        super().__init__()
        b1_dim = d_model // 3
        b2_dim = d_model // 3
        b3_dim = d_model - b1_dim - b2_dim   # residuo garantisce d_model esatto
        self.branch1 = nn.Conv1d(input_channels, b1_dim, kernel_size=1,  padding=0)
        self.branch2 = nn.Conv1d(input_channels, b2_dim, kernel_size=5,  padding=2)
        self.branch3 = nn.Conv1d(input_channels, b3_dim, kernel_size=15, padding=7)
        self.norm    = nn.BatchNorm1d(d_model)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
        ], dim=1)))


class LearnablePositionalEncoding(nn.Module):
    """Positional embedding appreso, più adattivo di quello sinusoidale fisso."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout  = nn.Dropout(p=dropout)
        self.emb      = nn.Embedding(max_len, d_model)
        self._max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        return self.dropout(x + self.emb(pos))


class ProgressiveDecoder(nn.Module):
    """
    Decoder che porta la sequenza da T_in a target_len tramite raddoppi successivi
    seguiti da un blocco Conv + LayerNorm.
    """
    def __init__(self, d_model: int, target_len: int):
        super().__init__()
        self.target_len = target_len
        self.refine = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, T_in)
        x = F.interpolate(x, size=self.target_len, mode='linear', align_corners=True)
        return self.refine(x)


class BioTransformer(nn.Module):
    """
    PhysioFormer — architettura Transformer multi-scala per PPG → ECG.

    Hyperparametri via configs dict:
        trans_d_model   (int,   default 128)  – dimensione embedding
        trans_nhead     (int,   default 8)    – teste di attenzione
        trans_layers    (int,   default 4)    – numero di layer
        trans_dim_ff    (int,   default 256)  – dim feedforward
        trans_dropout   (float, default 0.1)
        apply_wst       (bool,  default False)
        input_channels  (int,   default 3)
        target_len      (int,   default seq_len)
        actual_seq_len  (int,   default 875)  – usato solo se apply_wst=True
        normalize_01    (bool,  default False)
        use_ppg_bypass  (bool,  default True) – residual dal primo canale PPG
    """

    def __init__(self, configs: dict, seq_len: int = 125):
        super().__init__()

        # ── 1. Configurazione WST ─────────────────────────────────────────
        self.apply_wst          = configs.get('apply_wst', False)
        self.target_len         = configs.get('target_len', seq_len)
        self.normalize_01       = configs.get('normalize_01', False)
        self.use_ppg_bypass     = configs.get('use_ppg_bypass', True)
        self.raw_input_channels = configs.get('input_channels', 3)

        if self.apply_wst:
            input_size = configs.get('actual_seq_len', 875)
            self.scattering = Scattering1D(J=2, shape=(input_size,), Q=8)
            with torch.no_grad():
                dummy = torch.zeros(1, input_size)
                coeffs_per_sig = self.scattering(dummy).shape[1]
            self.input_dim = coeffs_per_sig * self.raw_input_channels
            print(f"[PhysioFormer] WST ON  → {self.input_dim} canali")
        else:
            self.scattering = None
            self.input_dim  = self.raw_input_channels
            print(f"[PhysioFormer] WST OFF → {self.input_dim} canali")

        # ── 2. Hyperparametri Transformer ────────────────────────────────
        d_model         = configs.get('trans_d_model',  128)
        nhead           = configs.get('trans_nhead',    8)
        num_layers      = configs.get('trans_layers',   4)
        dim_feedforward = configs.get('trans_dim_ff',   256)
        dropout         = configs.get('trans_dropout',  0.1)

        # ── 3. Tokenizer multi-scala ──────────────────────────────────────
        self.tokenizer   = MultiScaleTokenizer(self.input_dim, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=5000, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN: più stabile durante il training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── 4. Decoder progressivo ────────────────────────────────────────
        self.decoder = ProgressiveDecoder(d_model, self.target_len)

        # ── 5. Residual PPG Bypass ────────────────────────────────────────
        # Il primo canale di input (PPG raw) viene proiettato direttamente
        # sull'output; questo fornisce un forte inductive bias (ECG e PPG
        # hanno la stessa frequenza cardiaca fondamentale).
        if self.use_ppg_bypass:
            self.bypass_proj = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(8, 1, kernel_size=7, padding=3),
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
        # Salva il canale PPG originale prima di eventuali trasformazioni
        ppg_raw = x[:, 0:1, :]   # (B, 1, T)

        # ── WST opzionale ─────────────────────────────────────────────────
        if self.apply_wst and self.scattering is not None:
            expected = self.scattering.shape[0]
            if x.shape[-1] != expected:
                x = F.interpolate(x, size=expected, mode='linear', align_corners=True)
            wst_list = [self.scattering(x[:, i, :].contiguous()) for i in range(x.shape[1])]
            x = torch.nan_to_num(torch.cat(wst_list, dim=1), nan=0.0)

        # ── Tokenizzazione multi-scala ────────────────────────────────────
        tokens = self.tokenizer(x)                          # (B, d_model, T')
        tokens = tokens.permute(0, 2, 1)                    # (B, T', d_model)

        # ── Positional Encoding + Transformer ────────────────────────────
        tokens = self.pos_encoder(tokens)
        feat   = self.transformer_encoder(tokens)           # (B, T', d_model)
        feat   = feat.permute(0, 2, 1)                      # (B, d_model, T')

        # ── Decoder ───────────────────────────────────────────────────────
        out = self.decoder(feat)                            # (B, 1, target_len)

        # ── Residual PPG Bypass ───────────────────────────────────────────
        if self.use_ppg_bypass:
            bypass = F.interpolate(ppg_raw, size=self.target_len,
                                   mode='linear', align_corners=True)
            out = out + self.bypass_proj(bypass)

        if self.normalize_01:
            out = self.sigmoid(out)

        return out
