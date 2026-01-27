import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding per Transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerECGGenerator(nn.Module):
    """
    Transformer-based ECG Generator con architettura Encoder-Decoder
    Più semplice del GRU ma potenzialmente più potente
    """
    def __init__(self, target_len=2048, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.target_len = target_len
        self.d_model = d_model
        
        # === INPUT PROJECTIONS ===
        # PPG: 512 punti -> d_model features
        self.ppg_proj = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 512 -> 256
            nn.Conv1d(d_model // 4, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # ACC: 256 punti, 3 canali -> d_model features
        self.acc_proj = nn.Sequential(
            nn.Conv1d(3, d_model // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # EDA: 32 punti -> d_model features
        self.eda_proj = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Upsample(size=128, mode='linear', align_corners=True),  # 32 -> 128
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Prev ECG: 2048 punti -> d_model features
        self.prev_ecg_proj = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 2048 -> 512
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 512 -> 256
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)
        
        # === ENCODER TRANSFORMER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === DECODER TRANSFORMER ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # === OUTPUT PROJECTION ===
        # Proietta i token del decoder (256 token) ai 2048 punti dell'ECG
        self.output_upsampler = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model // 2, kernel_size=4, stride=2, padding=1),  # 256 -> 512
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1),  # 512 -> 1024
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.ConvTranspose1d(d_model // 4, d_model // 8, kernel_size=4, stride=2, padding=1),  # 1024 -> 2048
            nn.BatchNorm1d(d_model // 8),
            nn.ReLU(),
            nn.Conv1d(d_model // 8, 1, kernel_size=1)
        )
        
        # Learnable query tokens per il decoder
        self.query_tokens = nn.Parameter(torch.randn(1, 256, d_model))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, ppg, acc, eda, prev_ecg):
        batch_size = ppg.size(0)
        
        # === 1. FEATURE EXTRACTION ===
        ppg_feat = self.ppg_proj(ppg).transpose(1, 2)           # (B, 256, d_model)
        acc_feat = self.acc_proj(acc).transpose(1, 2)           # (B, 128, d_model)
        eda_feat = self.eda_proj(eda).transpose(1, 2)           # (B, 128, d_model)
        prev_feat = self.prev_ecg_proj(prev_ecg).transpose(1, 2)  # (B, 256, d_model)
        
        # Allinea tutte le features a 256 timesteps
        acc_feat = F.interpolate(acc_feat.transpose(1, 2), size=256, mode='linear', align_corners=True).transpose(1, 2)
        eda_feat = F.interpolate(eda_feat.transpose(1, 2), size=256, mode='linear', align_corners=True).transpose(1, 2)
        
        # Concatena lungo la dimensione temporale
        encoder_input = torch.cat([ppg_feat, acc_feat, eda_feat, prev_feat], dim=1)  # (B, 1024, d_model)
        
        # Riduci la sequenza per efficienza computazionale
        encoder_input = F.adaptive_avg_pool1d(encoder_input.transpose(1, 2), 512).transpose(1, 2)  # (B, 512, d_model)
        
        # === 2. POSITIONAL ENCODING ===
        encoder_input = self.pos_encoder(encoder_input)
        
        # === 3. ENCODER ===
        memory = self.transformer_encoder(encoder_input)  # (B, 512, d_model)
        
        # === 4. DECODER ===
        # Usa query tokens learnable
        query = self.query_tokens.expand(batch_size, -1, -1)  # (B, 256, d_model)
        query = self.pos_encoder(query)
        
        decoder_output = self.transformer_decoder(query, memory)  # (B, 256, d_model)
        
        # === 5. OUTPUT UPSAMPLING ===
        # (B, 256, d_model) -> (B, d_model, 256) -> (B, 1, 2048)
        output = self.output_upsampler(decoder_output.transpose(1, 2))
        
        # Assicura dimensione corretta
        if output.size(-1) != self.target_len:
            output = F.interpolate(output, size=self.target_len, mode='linear', align_corners=True)
        
        return output.squeeze(1)  # (B, 2048)


# VERSIONE LEGGERA per training più veloce
class LightTransformerECG(nn.Module):
    """Versione più leggera per esperimenti rapidi"""
    def __init__(self, target_len=2048):
        super().__init__()
        d_model = 128  # Ridotto
        nhead = 4       # Ridotto
        num_layers = 3  # Ridotto
        
        # Riusa la stessa architettura ma con parametri ridotti
        self.model = TransformerECGGenerator(
            target_len=target_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.1
        )
    
    def forward(self, ppg, acc, eda, prev_ecg):
        return self.model(ppg, acc, eda, prev_ecg)