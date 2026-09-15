import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D

# =========================================================================
# FIX MULTI-GPU: Cache globale per aggirare il bug di Kymatio
# Salverà un'istanza esatta della WST per ogni singola GPU del cluster.
# =========================================================================
_SCATTERING_CACHE = {}

class LightweightHybrid(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(LightweightHybrid, self).__init__()
        
        # --- 1. Configurazione WST ---
        self.apply_wst = configs.get('apply_wst', False)
        self.target_len = configs.get('target_len', seq_len)
        self.normalize_01 = configs.get('normalize_01', False)
        self.raw_input_channels = configs.get('input_channels', 3) 
        self.expected_len = configs.get('actual_seq_len', 875)
        
        if self.apply_wst:
            # Istanziamo temporaneamente su CPU solo per capire le dimensioni in uscita
            temp_scattering = Scattering1D(J=2, shape=(self.expected_len,), Q=8)
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.expected_len)
                dummy_output = temp_scattering(dummy_input)
                coeffs_per_sig = dummy_output.shape[1]
            
            # Dinamico
            self.input_dim = coeffs_per_sig * self.raw_input_channels
            print(f"[MODEL] LightweightHybrid con WST: {self.input_dim} canali in ingresso")
        else:
            self.input_dim = self.raw_input_channels
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
        # x shape: (Batch_diviso, Channels, Time)
        
        if self.apply_wst:
            actual_len = x.shape[-1]
            if actual_len != self.expected_len:
                x = F.interpolate(x, size=self.expected_len, mode='linear', align_corners=True)
            
            # --- FIX DEFINITIVO PER MULTI-GPU ---
            # Identifichiamo su quale GPU si trova questo specifico frammento di Dati
            device_str = str(x.device)
            
            # Se la trasformata non è mai stata creata per questa GPU, la creiamo ORA!
            if device_str not in _SCATTERING_CACHE:
                _SCATTERING_CACHE[device_str] = Scattering1D(J=2, shape=(self.expected_len,), Q=8).to(x.device)
            
            # Peschiamo la WST locale specifica per questa scheda video
            local_scattering = _SCATTERING_CACHE[device_str]
            
            wst_channels = []
            for i in range(x.shape[1]):
                wst_channels.append(local_scattering(x[:, i, :].contiguous()))
            
            x = torch.cat(wst_channels, dim=1)
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