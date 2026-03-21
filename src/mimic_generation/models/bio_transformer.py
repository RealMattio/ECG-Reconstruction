import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from kymatio.torch import Scattering1D

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BioTransformer(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(BioTransformer, self).__init__()
        
        # --- 1. Configurazione WST ---
        self.apply_wst = configs.get('apply_wst', False)
        self.target_len = configs.get('target_len', seq_len)
        self.normalize_01 = configs.get('normalize_01', False)
        
        if self.apply_wst:
            input_size = configs.get('actual_seq_len', 875)
            self.scattering = Scattering1D(J=2, shape=(input_size,), Q=8)
            
            # Rilevamento automatico canali tramite dummy tensor
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_size)
                dummy_output = self.scattering(dummy_input)
                coeffs_per_sig = dummy_output.shape[1]
            
            # PPG + ECG Past = 2 segnali
            self.input_dim = coeffs_per_sig * 2
            print(f"[MODEL] BioTransformer con WST: {self.input_dim} canali")
        else:
            self.scattering = None
            self.input_dim = configs.get('input_channels', 2)
            print(f"[MODEL] BioTransformer in Time Domain: {self.input_dim} canali")

        # --- 2. Iperparametri Transformer ---
        d_model = configs.get('trans_d_model', 64)
        nhead = configs.get('trans_nhead', 4)
        num_layers = configs.get('trans_layers', 2)
        dim_feedforward = configs.get('trans_dim_ff', 128)
        dropout = configs.get('trans_dropout', 0.2)
        
        # Proiezione canali -> d_model
        self.input_projection = nn.Conv1d(self.input_dim, d_model, kernel_size=3, padding=1)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. Regression Head ---
        self.output_projection = nn.Conv1d(d_model, 1, kernel_size=1)
        self.temporal_resizer = nn.AdaptiveAvgPool1d(self.target_len)
        self.final_refine = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        
        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        
        # --- FIX DI SICUREZZA PER KYMATIO ---
        if self.apply_wst and self.scattering is not None:
            expected_len = self.scattering.shape[0]
            actual_len = x.shape[-1]
            
            if actual_len != expected_len:
                # Se è diverso, forziamo la dimensione corretta tramite interpolazione o taglio
                # L'interpolazione è più sicura per non perdere informazioni
                x = F.interpolate(x, size=expected_len, mode='linear', align_corners=True)
            
            # Ora calcoliamo WST in sicurezza
            ppg_wst = self.scattering(x[:, 0, :].contiguous())
            ecg_wst = self.scattering(x[:, 1, :].contiguous())
            x = torch.cat([ppg_wst, ecg_wst], dim=1)
            x = torch.nan_to_num(x, nan=0.0)

        # 1. Embedding
        # (Batch, d_model, Time_in)
        x_emb = self.input_projection(x)
        
        # 2. Permute per Transformer: (Batch, Time_in, d_model)
        x_emb = x_emb.permute(0, 2, 1) 
        
        # 3. Positional Encoding
        x_pos = self.pos_encoder(x_emb)
        
        # 4. Attention Processing
        feat = self.transformer_encoder(x_pos)
        
        # 5. Decoding
        # Torniamo a (Batch, d_model, Time_in)
        feat = feat.permute(0, 2, 1)
        
        # Proiezione a 1 canale
        out_raw = self.output_projection(feat)
        
        # Adattamento alla lunghezza target (es. 125)
        out_resized = self.temporal_resizer(out_raw)
        
        # Smoothing finale
        out = self.final_refine(out_resized)
        
        if self.normalize_01:
            out = self.sigmoid(out)
            
        return out