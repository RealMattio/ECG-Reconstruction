import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_prob=0.0):
        super().__init__()
        layers = [
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2)
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout1d(dropout_prob))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class EncoderDecoderUNet(nn.Module):
    def __init__(self, target_len=2048):
        super().__init__()
        
        # --- ENCODER PPG (Input: 512) ---
        self.ppg_enc1 = conv_block(1, 16)   # 512
        self.ppg_enc2 = conv_block(16, 32)  # 256 (dopo pool)
        self.ppg_enc3 = conv_block(32, 64)  # 128 (dopo pool)
        
        # --- ENCODER ACC (Input: 256) ---
        self.acc_enc1 = conv_block(3, 16)   # 256
        self.acc_enc2 = conv_block(16, 64)  # 128 (dopo pool)

        # --- ENCODER EDA (Input: 32) ---
        # Portiamo l'EDA da 32 a 128 punti per la fusione
        self.eda_enc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode='linear', align_corners=True), # 32 -> 128
            conv_block(16, 32)
        )

        # --- BOTTLENECK: 64 (PPG) + 64 (ACC) + 32 (EDA) = 160 canali ---
        self.bottleneck = conv_block(160, 128, dropout_prob=0.5)
        
        # --- DECODER (Output: target_len) ---
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4) # 128 -> 512
        self.dec1 = conv_block(64, 32, dropout_prob=0.3)
        
        self.up2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)  # 512 -> 1024
        self.dec2 = conv_block(16, 8, dropout_prob=0.2)
        
        self.up3 = nn.ConvTranspose1d(8, 8, kernel_size=2, stride=2)    # 1024 -> 2048
        self.final_conv = nn.Conv1d(8, 1, kernel_size=1)

    def forward(self, ppg, acc, eda):
        # 1. Processing PPG: 512 -> 128
        p1 = self.ppg_enc1(ppg)
        p2 = self.ppg_enc2(F.max_pool1d(p1, 2))
        p3 = self.ppg_enc3(F.max_pool1d(p2, 2)) # (Batch, 64, 128)
        
        # 2. Processing ACC: 256 -> 128
        a1 = self.acc_enc1(acc)
        a2 = self.acc_enc2(F.max_pool1d(a1, 2)) # (Batch, 64, 128)

        # 3. Processing EDA: 32 -> 128
        e1 = self.eda_enc(eda) # (Batch, 32, 128)
        
        # 4. Feature Fusion (Concatenazione sui canali)
        # Uniamo le caratteristiche estratte da tutti i sensori
        combined = torch.cat([p3, a2, e1], dim=1) # (Batch, 160, 128)
        
        # 5. Bottleneck (Compressione e Regolarizzazione)
        bn = self.bottleneck(combined)
        
        # 6. Decoder (Ricostruzione ECG)
        d1 = self.dec1(self.up1(bn))
        d2 = self.dec2(self.up2(d1))
        d3 = self.final_conv(self.up3(d2))
        
        return d3.squeeze(1)