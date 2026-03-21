import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D

class LightweightHybrid(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(LightweightHybrid, self).__init__()
        
        # --- 1. Configurazione WST ---
        self.apply_wst = configs.get('apply_wst', False)
        self.target_len = configs.get('target_len', seq_len)
        self.normalize_01 = configs.get('normalize_01', False)
        
        if self.apply_wst:
            input_size = configs.get('actual_seq_len', 875)
            self.scattering = Scattering1D(J=2, shape=(input_size,), Q=8)
            
            # Calcolo automatico dei canali in uscita tramite dummy tensor
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_size)
                dummy_output = self.scattering(dummy_input)
                coeffs_per_sig = dummy_output.shape[1]
            
            # PPG + ECG Past (2 segnali)
            self.input_dim = coeffs_per_sig * 2
            print(f"[MODEL] LightweightHybrid con WST: {self.input_dim} canali")
        else:
            self.scattering = None
            self.input_dim = configs.get('input_channels', 2)
            print(f"[MODEL] LightweightHybrid in Time Domain: {self.input_dim} canali")

        # ============================================================
        # 1. Feature Extractor (CNN 1D Semplificata)
        # ============================================================
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2)
        )
        
        # ============================================================
        # 2. Sequential Modeling (LSTM Singola)
        # ============================================================
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True) 
        
        # ============================================================
        # 3. Decoder & Output
        # ============================================================
        self.upsample = nn.Upsample(size=self.target_len, mode='linear', align_corners=True)
        
        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=1)
        )
        
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
            # Concatenazione: (B, input_dim, T_reduced)
            x = torch.cat([ppg_wst, ecg_wst], dim=1)
            x = torch.nan_to_num(x, nan=0.0)

        # 1. CNN Encoder
        features = self.cnn_encoder(x)
        
        # 2. LSTM Processing
        lstm_in = features.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.transpose(1, 2)
        
        # 3. Upsampling al Target (125 campioni)
        upsampled = self.upsample(lstm_out)
        
        # 4. Generazione Finale
        out = self.decoder(upsampled)
        
        if self.normalize_01:
            out = self.sigmoid(out)
            
        return out