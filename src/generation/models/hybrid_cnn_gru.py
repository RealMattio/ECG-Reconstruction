import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFeatureExtractor(nn.Module):
    """Estrae features da ogni sensore usando CNN 1D"""
    def __init__(self, in_channels, out_channels, input_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)  # Riduce la lunghezza per il GRU
        )
        self.output_len = input_len // 2
    
    def forward(self, x):
        return self.conv(x)  # (B, out_channels, L//2)


class AttentionLayer(nn.Module):
    """Attention mechanism per decoder"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, hidden_size)
        # encoder_outputs: (B, seq_len, hidden_size)
        
        seq_len = encoder_outputs.size(1)
        
        # Ripeti decoder_hidden per ogni timestep
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calcola attention scores
        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)  # (B, seq_len)
        
        # Softmax per ottenere weights
        attn_weights = F.softmax(attention, dim=1)
        
        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, hidden_size)
        
        return context.squeeze(1), attn_weights


class HybridCNNGRUModel(nn.Module):
    """
    Architettura Ibrida per generazione ECG:
    1. CNN Feature Extractors per ogni sensore
    2. GRU Bidirectional Encoder
    3. Attention-based GRU Decoder
    """
    def __init__(self, target_len=2048, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.target_len = target_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # === FEATURE EXTRACTORS (CNN) ===
        # PPG: 512 -> 256, canali: 1 -> 64
        self.ppg_extractor = ConvFeatureExtractor(1, 64, 512)
        
        # ACC: 256 -> 128, canali: 3 -> 64
        self.acc_extractor = ConvFeatureExtractor(3, 64, 256)
        
        # EDA: 32 -> 64 (upsample), poi -> 32, canali: 1 -> 32
        self.eda_pre = nn.Upsample(size=64, mode='linear', align_corners=True)
        self.eda_extractor = ConvFeatureExtractor(1, 32, 64)
        
        # Prev ECG: 2048 -> 1024, canali: 1 -> 128
        self.prev_ecg_extractor = ConvFeatureExtractor(1, 128, 2048)
        
        # Total channels dopo feature extraction: 64+64+32+128 = 288
        total_features = 288
        
        # Projection layer per ridurre dimensionalità prima del GRU
        self.feature_projection = nn.Sequential(
            nn.Linear(total_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # === ENCODER (Bidirectional GRU) ===
        self.encoder_gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection per ridurre da bidirectional (2*hidden) a hidden
        self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # === ATTENTION ===
        self.attention = AttentionLayer(hidden_size)
        
        # === DECODER (GRU Autoregressivo) ===
        self.decoder_gru = nn.GRU(
            hidden_size + 1,  # hidden + input precedente
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # concat(decoder_out, context)
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def extract_features(self, ppg, acc, eda, prev_ecg):
        """Estrae features da tutti i sensori"""
        # Feature extraction con CNN
        ppg_feat = self.ppg_extractor(ppg)           # (B, 64, 256)
        acc_feat = self.acc_extractor(acc)           # (B, 64, 128)
        eda_feat = self.eda_extractor(self.eda_pre(eda))  # (B, 32, 32)
        prev_feat = self.prev_ecg_extractor(prev_ecg)     # (B, 128, 1024)
        
        # Allinea tutte le features alla stessa lunghezza temporale (128)
        ppg_feat = F.avg_pool1d(ppg_feat, 2)        # 256 -> 128
        prev_feat = F.avg_pool1d(prev_feat, 8)      # 1024 -> 128
        eda_feat = F.interpolate(eda_feat, size=128, mode='linear', align_corners=True)
        
        # Concatena lungo la dimensione dei canali
        combined = torch.cat([ppg_feat, acc_feat, eda_feat, prev_feat], dim=1)  # (B, 288, 128)
        
        # Trasforma in (B, seq_len, features) per il GRU
        combined = combined.transpose(1, 2)  # (B, 128, 288)
        
        return combined
    
    def encode(self, features):
        """Encoder: Bidirectional GRU"""
        # Project features
        x = self.feature_projection(features)  # (B, 128, hidden_size)
        
        # Bidirectional GRU
        encoder_outputs, encoder_hidden = self.encoder_gru(x)  # outputs: (B, 128, 2*hidden)
        
        # Project back to hidden_size
        encoder_outputs = self.encoder_projection(encoder_outputs)  # (B, 128, hidden_size)
        
        # Combina forward e backward hidden states
        # encoder_hidden: (2*num_layers, B, hidden_size)
        # Prendiamo solo l'ultimo layer e combiniamo forward/backward
        forward_hidden = encoder_hidden[-2, :, :]   # (B, hidden_size)
        backward_hidden = encoder_hidden[-1, :, :]  # (B, hidden_size)
        decoder_hidden = torch.tanh(self.encoder_projection(
            torch.cat([forward_hidden, backward_hidden], dim=1)
        )).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_size)
        
        return encoder_outputs, decoder_hidden
    
    def decode(self, encoder_outputs, decoder_hidden, target_len):
        """Decoder con Attention corretto"""
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Inizializza il primo input: (B, 1)
        # Usiamo solo 2 dimensioni qui per semplificare il cat successivo
        decoder_input = torch.zeros(batch_size, 1).to(device)
        
        outputs = []
        
        for t in range(target_len):
            # 1. Attention context: (B, hidden_size)
            context, _ = self.attention(
                decoder_hidden[-1], 
                encoder_outputs
            )
            
            # 2. Concatena input (B, 1) con context (B, hidden_size) -> (B, hidden_size + 1)
            # Poi aggiungi la dimensione temporale per il GRU -> (B, 1, hidden_size + 1)
            decoder_input_combined = torch.cat([decoder_input, context], dim=1).unsqueeze(1)
            
            # 3. GRU step
            # decoder_output: (B, 1, hidden_size)
            decoder_output, decoder_hidden = self.decoder_gru(
                decoder_input_combined, 
                decoder_hidden
            )
            
            # 4. Output projection
            # Concateniamo l'uscita del GRU con il contesto (skip connection dell'attention)
            combined = torch.cat([
                decoder_output.squeeze(1), 
                context
            ], dim=1)  # (B, hidden_size * 2)
            
            output = self.output_projection(combined)  # (B, 1)
            outputs.append(output)
            
            # 5. IMPORTANTE: Il prossimo input è l'output corrente (B, 1)
            # Manteniamo 2 dimensioni per il prossimo ciclo
            decoder_input = output
        
        # Stack finale: (B, target_len)
        return torch.stack(outputs, dim=1).squeeze(-1)
    
    def forward(self, ppg, acc, eda, prev_ecg):
        # 1. Extract features
        features = self.extract_features(ppg, acc, eda, prev_ecg)
        
        # 2. Encode
        encoder_outputs, decoder_hidden = self.encode(features)
        
        # 3. Decode
        output = self.decode(encoder_outputs, decoder_hidden, self.target_len)
        
        return output