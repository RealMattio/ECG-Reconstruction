import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightHybrid(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(LightweightHybrid, self).__init__()
        
        input_dim = configs.get('input_channels', 2)
        self.target_len = configs.get('target_len', seq_len)
        
        # ============================================================
        # 1. Feature Extractor (CNN 1D Semplificata)
        # ============================================================
        # Riduce la dimensione temporale e aumenta i canali
        self.cnn_encoder = nn.Sequential(
            # Layer 1: Estrazione feature basso livello
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2), # Dimezza il tempo
            
            # Layer 2: Estrazione feature alto livello
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2)  # Dimezza di nuovo (Totale /4)
        )
        
        # ============================================================
        # 2. Sequential Modeling (LSTM Singola)
        # ============================================================
        # La LSTM lavora sulla sequenza compressa dalla CNN
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, 
                            num_layers=1, # Solo 1 layer per ridurre parametri
                            batch_first=True, 
                            bidirectional=True) # Output size = 64*2 = 128
        
        # ============================================================
        # 3. Decoder & Output
        # ============================================================
        # Dobbiamo riportare la dimensione temporale al target richiesto
        self.upsample = nn.Upsample(size=self.target_len, mode='linear', align_corners=True)
        
        self.decoder = nn.Sequential(
            nn.Dropout(0.2), # Regolarizzazione importante con pochi dati
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=1) # Proiezione finale a 1 canale (ECG)
        )
        
        self.normalize_01 = configs.get('normalize_01', False)
        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Channels, Time_Original)
        
        # 1. CNN Encoder
        # Output: (Batch, 64, Time_Original/4)
        features = self.cnn_encoder(x)
        
        # 2. LSTM Processing
        # Permutiamo per LSTM: (Batch, Time, Channels)
        lstm_in = features.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        # Torniamo a (Batch, Channels, Time): (Batch, 128, Time_Original/4)
        lstm_out = lstm_out.transpose(1, 2)
        
        # 3. Upsampling al Target
        # Riportiamo la dimensione temporale esattamente alla lunghezza di generazione richiesta
        upsampled = self.upsample(lstm_out)
        
        # 4. Generazione Finale
        out = self.decoder(upsampled)
        
        if self.normalize_01:
            out = self.sigmoid(out)
            
        return out