import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D

class HACNNBiLSTM_AR(nn.Module):
    def __init__(self, configs, seq_len=125):
        """
        Modello Autoregressivo HACNN-BiLSTM:
        Input: (Batch, Canali, Seq_Len_X)
        Output: (Batch, 1, Target_Len) -> 1 Secondo di ECG futuro
        """
        super(HACNNBiLSTM_AR, self).__init__()
        
        # --- 1. Configurazione WST ---
        self.apply_wst = configs.get('apply_wst', False)
        self.target_seq_len = configs.get('target_len', seq_len)
        self.normalize_01 = configs.get('normalize_01', False)
        self.raw_input_channels = configs.get('input_channels', 3) # Ora è dinamico
        
        if self.apply_wst:
            input_size = configs.get('actual_seq_len', 875)
            self.scattering = Scattering1D(J=2, shape=(input_size,), Q=8)
            
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_size)
                dummy_output = self.scattering(dummy_input)
                coeffs_per_sig = dummy_output.shape[1]
            
            self.input_dim = coeffs_per_sig * self.raw_input_channels
            print(f"[MODEL] HACNN con WST: {self.input_dim} canali")
        else:
            self.scattering = None
            self.input_dim = self.raw_input_channels
            print(f"[MODEL] HACNN in Time Domain: {self.input_dim} canali")

        # --- 2. STAGE CONVOLUZIONALE ---
        self.cnn_stage = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # --- 3. BILSTM ---
        self.bi_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        
        # --- 4. UPSAMPLING LAYER ---
        self.upsample = nn.Upsample(size=self.target_seq_len, mode='linear', align_corners=True)
        
        # --- 5. ATTENTION GATE ---
        self.attention_gate = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 6. REGRESSION HEAD ---
        layers = [
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=1)
        ]
        
        if self.normalize_01:
            layers.append(nn.Sigmoid())
            
        self.regression = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        
        # --- FIX DI SICUREZZA PER KYMATIO ---
        if self.apply_wst and self.scattering is not None:
            expected_len = self.scattering.shape[0]
            actual_len = x.shape[-1]
            
            if actual_len != expected_len:
                x = F.interpolate(x, size=expected_len, mode='linear', align_corners=True)
            
            # Dinamico per N canali
            wst_channels = []
            for i in range(x.shape[1]):
                wst_channels.append(self.scattering(x[:, i, :].contiguous()))
            
            x = torch.cat(wst_channels, dim=1)
            x = torch.nan_to_num(x, nan=0.0)

        # 1. Feature Extraction: (Batch, 128, T_reduced)
        cnn_out = self.cnn_stage(x)
        
        # 2. BiLSTM: (Batch, T_reduced, 256)
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, _ = self.bi_lstm(lstm_in)
        lstm_out = lstm_out.transpose(1, 2) 
        
        # 3. Upsample al target (1 secondo): (Batch, 256, target_len)
        fused_upsampled = self.upsample(lstm_out)
        
        # 4. Attention & Regression
        att_weights = self.attention_gate(fused_upsampled)
        fused = fused_upsampled * att_weights
        
        return self.regression(fused)