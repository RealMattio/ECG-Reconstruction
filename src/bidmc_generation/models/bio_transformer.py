import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creazione della matrice di encoding posizionale una volta sola
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Aggiungiamo la dimensione batch: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # Aggiungiamo il positional encoding fino alla lunghezza attuale della sequenza
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BioTransformer(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(BioTransformer, self).__init__()
        
        # --- Iperparametri ---
        input_dim = configs.get('input_channels', 2)
        self.target_len = configs.get('target_len', seq_len)
        
        # Dimensioni interne (Teniamole contenute per evitare overfitting)
        d_model = configs.get('trans_d_model', 64)   # Dimensione latente
        nhead = configs.get('trans_nhead', 4)        # Numero di teste di attenzione
        num_layers = configs.get('trans_layers', 2)  # Numero di blocchi encoder
        dim_feedforward = configs.get('trans_dim_ff', 128) # Dimensione FFN interna
        dropout = configs.get('trans_dropout', 0.2)
        
        # 1. Input Embedding
        # Proietta i canali input (2 o 38) nello spazio d_model
        # Usiamo Conv1d invece di Linear per catturare subito un minimo di contesto locale
        self.input_projection = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu', # GELU performa meglio di ReLU nei transformer moderni
            batch_first=True   # Importante: input (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Regression Head (Decoder)
        # Dobbiamo mappare (Batch, Seq_In, d_model) -> (Batch, 1, Target_Len)
        
        # Step A: Comprimere i canali d_model -> 1
        self.output_projection = nn.Conv1d(d_model, 1, kernel_size=1)
        
        # Step B: Adattare la lunghezza temporale
        # Da Seq_In (es. 875) a Target_Len (es. 125)
        self.temporal_resizer = nn.AdaptiveAvgPool1d(self.target_len)
        
        # Step C: Refinement finale (opzionale, per lisciare)
        self.final_refine = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        
        self.normalize_01 = configs.get('normalize_01', False)
        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channels, Time) -> es. (32, 2, 875)
        
        # 1. Embedding
        # (Batch, d_model, Time)
        x_emb = self.input_projection(x)
        
        # 2. Preparazione per Transformer (Batch, Time, d_model)
        x_emb = x_emb.permute(0, 2, 1) 
        
        # 3. Positional Encoding
        x_pos = self.pos_encoder(x_emb)
        
        # 4. Transformer Processing
        # Output: (Batch, Time, d_model) - Stessa shape dell'input
        # Qui avviene la magia dell'attenzione globale
        feat = self.transformer_encoder(x_pos)
        
        # 5. Decoding
        # Torniamo a (Batch, d_model, Time) per le convoluzioni
        feat = feat.permute(0, 2, 1)
        
        # Proiezione canali: (Batch, 1, Time)
        out_raw = self.output_projection(feat)
        
        # Resize temporale: (Batch, 1, Target_Len)
        # Nota: AdaptiveAvgPool è intelligente, ma potremmo anche prendere solo gli ultimi token.
        # Per ora usiamo il pooling che è più stabile.
        out_resized = self.temporal_resizer(out_raw)
        
        # Refinement finale
        out = self.final_refine(out_resized)
        
        if self.normalize_01:
            out = self.sigmoid(out)
            
        return out