import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D

class DualBranchHybrid(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(DualBranchHybrid, self).__init__()
        
        self.apply_wst = configs.get('apply_wst', False)
        self.target_len = configs.get('target_len', seq_len)
        self.normalize_01 = configs.get('normalize_01', False)
        
        # 1. --- Configurazione Canali e WST ---
        # Garantiamo dinamicità con fallback a 3 (PPG, PPG_diff, ECG_Past)
        self.raw_input_channels = configs.get('input_channels', 3) 
        
        if self.apply_wst:
            self.expected_wst_len = configs.get('actual_seq_len', 875)
            self.scattering = Scattering1D(J=2, shape=(self.expected_wst_len,), Q=8)
            
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.expected_wst_len)
                dummy_output = self.scattering(dummy_input)
                coeffs_per_sig = dummy_output.shape[1]
                
            self.img_height = coeffs_per_sig * self.raw_input_channels 
            print(f"[MODEL] WST ON: {self.img_height} canali totali")
        else:
            self.scattering = None
            self.img_height = self.raw_input_channels
            print(f"[MODEL] WST OFF: {self.img_height} canali (PPG, Diff, ECG_Past)")

        self.split_dim = self.img_height // 2 
        cnn_base_channels = 32

        # ============================================================
        # RAMO 1: CNN 2D ADATTIVA
        # ============================================================
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(1, cnn_base_channels, kernel_size=(3, 3), padding=(1, 1)), 
            nn.BatchNorm2d(cnn_base_channels),
            nn.LeakyReLU(0.1)
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv2d(cnn_base_channels, cnn_base_channels*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_base_channels*2),
            nn.LeakyReLU(0.1)
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2)) if self.img_height > 1 else nn.MaxPool2d(kernel_size=(1, 2))

        self.enc_block3 = nn.Sequential(
            nn.Conv2d(cnn_base_channels*2, cnn_base_channels*4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.LeakyReLU(0.1)
        )
        self.enc_block4 = nn.Sequential(
            nn.Conv2d(cnn_base_channels*4, cnn_base_channels*4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.LeakyReLU(0.1)
        )

        h_after_pool1 = self.img_height // 2 if self.img_height > 1 else 1
        if h_after_pool1 > 1:
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 1)) 
            self.h_out = h_after_pool1 // 2
        else:
            self.pool2 = nn.Identity()
            self.h_out = h_after_pool1
        
        self.cnn_projector = nn.Sequential(
            nn.Conv1d(cnn_base_channels*4 * self.h_out, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )

        # ============================================================
        # RAMO 2: Dual Bi-LSTM
        # ============================================================
        self.ppg_split = (self.img_height // 3) * 2 # PPG e PPG_Diff vanno qui
        self.ecg_split = self.img_height - self.ppg_split # ECG_Past va qui

        self.lstm_ppg = nn.LSTM(input_size=self.ppg_split, hidden_size=128, 
                                num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.lstm_ecg = nn.LSTM(input_size=self.ecg_split, hidden_size=128, 
                                num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.lstm_fusion = nn.Sequential(
            nn.Linear(128*4, 256),
            nn.LayerNorm(256),
            nn.Tanh()
        )

        # ============================================================
        # ATTENZIONE E DECODER
        # ============================================================
        self.attention_gate = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.final_regression = nn.Sequential(
            nn.Conv1d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=1) # Dimensione finale target garantita
        )
        
        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, 3, T)
        
        if self.apply_wst and self.scattering is not None:
            if x.shape[-1] != self.expected_wst_len:
                x = F.interpolate(x, size=self.expected_wst_len, mode='linear', align_corners=True)

            wst_0 = self.scattering(x[:, 0, :].contiguous())
            wst_1 = self.scattering(x[:, 1, :].contiguous())
            wst_2 = self.scattering(x[:, 2, :].contiguous()) # Aggiunto il terzo canale
            
            x = torch.cat([wst_0, wst_1, wst_2], dim=1)
            x = torch.nan_to_num(x, nan=0.0)

        # --- 1. RAMO CNN ---
        x_img = x.unsqueeze(1)
        x_img = self.enc_block1(x_img)
        x_img = self.enc_block2(x_img)
        x_img = self.pool1(x_img)
        x_img = self.enc_block3(x_img)
        x_img = self.enc_block4(x_img)
        x_img = self.pool2(x_img)
        
        B, C, H, T_cnn = x_img.size()
        cnn_feat = self.cnn_projector(x_img.view(B, C * H, T_cnn))
        
        # --- 2. RAMO LSTM ---
        x_ppg_side = x[:, :self.ppg_split, :]
        x_ecg_side = x[:, self.ppg_split:, :]
        
        out_ppg, _ = self.lstm_ppg(x_ppg_side.transpose(1, 2))
        out_ecg, _ = self.lstm_ecg(x_ecg_side.transpose(1, 2))
        
        lstm_feat = self.lstm_fusion(torch.cat([out_ppg, out_ecg], dim=2)).transpose(1, 2)
        
        # --- 3. ALLINEAMENTO E ATTENZIONE ---
        if cnn_feat.shape[-1] != lstm_feat.shape[-1]:
             cnn_feat = F.interpolate(cnn_feat, size=lstm_feat.shape[-1], mode='linear', align_corners=True)
             
        combined = torch.cat([cnn_feat, lstm_feat], dim=1).transpose(1, 2)
        weights = self.attention_gate(combined)
        
        cnn_w = cnn_feat * weights[:, :, 0].unsqueeze(1)
        lstm_w = lstm_feat * weights[:, :, 1].unsqueeze(1)
        
        # --- 4. OUTPUT ---
        final_input = torch.cat([
            F.interpolate(cnn_w[:, :, -self.target_len:], size=self.target_len, mode='linear', align_corners=True),
            F.interpolate(lstm_w[:, :, -self.target_len:], size=self.target_len, mode='linear', align_corners=True)
        ], dim=1)
        
        out = self.final_regression(final_input)
        return self.sigmoid(out) if self.normalize_01 else out