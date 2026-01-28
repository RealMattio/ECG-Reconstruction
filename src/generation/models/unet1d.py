import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_prob=0.0):
        super().__init__()
        layers = [
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout1d(dropout_prob))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class EncoderDecoderUNet(nn.Module):
    def __init__(self, target_len=2048):
        super().__init__()
        
        self.target_len = target_len
        
        # --- ENCODER PPG (Input: 512 -> 128) ---
        self.ppg_enc1 = conv_block(1, 16)   # 512
        self.ppg_enc2 = conv_block(16, 32)  # 256
        self.ppg_enc3 = conv_block(32, 64)  # 128
        
        # --- ENCODER ACC (Input: 256 -> 128) ---
        self.acc_enc1 = conv_block(3, 16)   # 256
        self.acc_enc2 = conv_block(16, 64)  # 128

        # --- ENCODER EDA (Input: 32 -> 128) ---
        self.eda_enc = nn.Sequential(
            conv_block(1, 16),
            nn.Upsample(size=128, mode='linear', align_corners=True), # 32 -> 128
            conv_block(16, 32)
        )

        # --- NUOVO: ENCODER ECG PRECEDENTE (Input: 2048 -> 128) ---
        # Richiede più passaggi di pooling per arrivare a 128
        self.prev_ecg_enc = nn.Sequential(
            conv_block(1, 16),      # 2048
            nn.MaxPool1d(4),        # 512
            conv_block(16, 32),
            nn.MaxPool1d(2),        # 256
            conv_block(32, 64),
            nn.MaxPool1d(2),        # 128
            conv_block(64, 64)
        )

        # --- BOTTLENECK AGGIORNATO ---
        # 64 (PPG) + 64 (ACC) + 32 (EDA) + 64 (PREV_ECG) = 224 canali
        self.bottleneck = nn.Sequential(
            conv_block(224, 128, dropout_prob=0.3),
            conv_block(128, 128)
        )
        
        # --- DECODER (Output: 2048) ---
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0)
        self.dec1 = conv_block(64, 64, dropout_prob=0.2)
        
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = conv_block(32, 32, dropout_prob=0.1)
        
        self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec3 = conv_block(16, 16)
        
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ppg, acc, eda, prev_ecg):
        # 1. Processing PPG: 512 -> 128
        p1 = self.ppg_enc1(ppg)
        p2 = self.ppg_enc2(F.max_pool1d(p1, 2))
        p3 = self.ppg_enc3(F.max_pool1d(p2, 2))
        
        # 2. Processing ACC: 256 -> 128
        a1 = self.acc_enc1(acc)
        a2 = self.acc_enc2(F.max_pool1d(a1, 2))

        # 3. Processing EDA: 32 -> 128
        e1 = self.eda_enc(eda)

        # 4. Processing ECG PREV: 2048 -> 128
        prev1 = self.prev_ecg_enc(prev_ecg)
        
        # 5. Feature Fusion con ECG Precedente
        combined = torch.cat([p3, a2, e1, prev1], dim=1) # (B, 224, 128)
        
        # 6. Bottleneck
        bn = self.bottleneck(combined)
        
        # 7. Decoder
        d1 = self.dec1(self.up1(bn))
        d2 = self.dec2(self.up2(d1))
        d3 = self.dec3(self.up3(d2))
        
        output = self.final_conv(d3)
        
        if output.size(-1) != self.target_len:
            output = F.interpolate(output, size=self.target_len, mode='linear', align_corners=True)
        
        return output.squeeze(1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c, dropout_prob=0.0):
#         super().__init__()
#         layers = [
#             nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_c),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_c),
#             nn.LeakyReLU(0.2, inplace=True)
#         ]
#         if dropout_prob > 0:
#             layers.append(nn.Dropout1d(dropout_prob))
        
#         self.conv = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.conv(x)

# class EncoderDecoderUNet(nn.Module):
#     def __init__(self, target_len=2048):
#         super().__init__()
        
#         self.target_len = target_len
        
#         # --- ENCODER PPG (Input: 512) ---
#         self.ppg_enc1 = conv_block(1, 16)   # 512
#         self.ppg_enc2 = conv_block(16, 32)  # 256 (dopo pool)
#         self.ppg_enc3 = conv_block(32, 64)  # 128 (dopo pool)
        
#         # --- ENCODER ACC (Input: 256) ---
#         self.acc_enc1 = conv_block(3, 16)   # 256
#         self.acc_enc2 = conv_block(16, 64)  # 128 (dopo pool)

#         # --- ENCODER EDA (Input: 32) ---
#         # Migliore upsampling per EDA
#         self.eda_enc = nn.Sequential(
#             conv_block(1, 16),
#             nn.Upsample(size=128, mode='linear', align_corners=True),  # 32 -> 128
#             conv_block(16, 32)
#         )

#         # --- BOTTLENECK: 64 (PPG) + 64 (ACC) + 32 (EDA) = 160 canali ---
#         self.bottleneck = nn.Sequential(
#             conv_block(160, 128, dropout_prob=0.3),  # Ridotto dropout
#             conv_block(128, 128)  # Doppio bottleneck per più capacità
#         )
        
#         # --- DECODER (Output: target_len) ---
#         # Percorso più graduale: 128 -> 512 -> 1024 -> 2048
        
#         # 128 -> 512 (x4)
#         self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0)
#         self.dec1 = conv_block(64, 64, dropout_prob=0.2)
        
#         # 512 -> 1024 (x2)
#         self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.dec2 = conv_block(32, 32, dropout_prob=0.1)
        
#         # 1024 -> 2048 (x2)
#         self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
#         self.dec3 = conv_block(16, 16)
        
#         # Output layer - IMPORTANTE: senza attivazione!
#         self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        
#         # Inizializzazione dei pesi
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Inizializzazione Xavier/He per una migliore convergenza."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, ppg, acc, eda):
#         # 1. Processing PPG: 512 -> 128
#         p1 = self.ppg_enc1(ppg)                    # (B, 16, 512)
#         p2 = self.ppg_enc2(F.max_pool1d(p1, 2))    # (B, 32, 256)
#         p3 = self.ppg_enc3(F.max_pool1d(p2, 2))    # (B, 64, 128)
        
#         # 2. Processing ACC: 256 -> 128
#         a1 = self.acc_enc1(acc)                    # (B, 16, 256)
#         a2 = self.acc_enc2(F.max_pool1d(a1, 2))    # (B, 64, 128)

#         # 3. Processing EDA: 32 -> 128
#         e1 = self.eda_enc(eda)                     # (B, 32, 128)
        
#         # 4. Feature Fusion
#         combined = torch.cat([p3, a2, e1], dim=1)  # (B, 160, 128)
        
#         # 5. Bottleneck
#         bn = self.bottleneck(combined)             # (B, 128, 128)
        
#         # 6. Decoder
#         d1 = self.dec1(self.up1(bn))               # (B, 64, 512)
#         d2 = self.dec2(self.up2(d1))               # (B, 32, 1024)
#         d3 = self.dec3(self.up3(d2))               # (B, 16, 2048)
        
#         output = self.final_conv(d3)               # (B, 1, 2048)
        
#         # Verifica dimensione finale e adatta se necessario
#         if output.size(-1) != self.target_len:
#             output = F.interpolate(output, size=self.target_len, mode='linear', align_corners=True)
        
#         return output.squeeze(1)                   # (B, 2048)