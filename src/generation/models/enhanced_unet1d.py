import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedContextFusion(nn.Module):
    """
    Modula le feature della PPG usando il contesto (ACC, EDA, Prev ECG).
    """
    def __init__(self, ppg_channels, context_channels):
        super().__init__()
        # Genera un peso (0-1) per ogni canale della PPG
        self.gate = nn.Sequential(
            nn.Linear(context_channels, ppg_channels),
            nn.Sigmoid()
        )
        # Genera un offset (bias) per correggere la morfologia
        self.bias_layer = nn.Sequential(
            nn.Linear(context_channels, ppg_channels),
            nn.Tanh()
        )

    def forward(self, ppg_feat, context_feat):
        # Global Average Pooling del contesto per estrarre il descrittore semantico
        ctx_vector = torch.mean(context_feat, dim=-1) # (B, Context_C)
        
        weights = self.gate(ctx_vector).unsqueeze(-1) # (B, PPG_C, 1)
        bias = self.bias_layer(ctx_vector).unsqueeze(-1) # (B, PPG_C, 1)
        
        # Modulazione: PPG pesata dal contesto + correzione additiva
        return (ppg_feat * weights) + bias

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c)
        )
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class EnhancedUNet1D(nn.Module):
    def __init__(self, target_len=2048):
        super().__init__()
        
        # ENCODER PRIMARIO: PPG (Pesi aumentati a 64 filtri)
        self.ppg_enc = nn.Sequential(ResidualBlock(1, 64), nn.MaxPool1d(4)) # 512 -> 128
        
        # ENCODER CONTESTO: Secondari
        self.acc_enc = nn.Sequential(ResidualBlock(3, 32), nn.MaxPool1d(2)) # 256 -> 128
        self.eda_enc = nn.Sequential(nn.Upsample(size=128, mode='linear', align_corners=True), ResidualBlock(1, 16)) 
        self.prev_enc = nn.Sequential(ResidualBlock(1, 32), nn.MaxPool1d(16)) # 2048 -> 128

        # FUSIONE: PPG (64) guidata da Contesto (32+16+32 = 80)
        self.fusion = GatedContextFusion(ppg_channels=64, context_channels=80)

        # BOTTLENECK: Lavora sulle 64 feature PPG modulate
        self.bottleneck = ResidualBlock(64, 128, dropout=0.3)

        # DECODER: Ricostruzione progressiva
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4) # 128 -> 512
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) # 512 -> 1024
        self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1) # 1024 -> 2048
        
        self.final = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, ppg, acc, eda, prev_ecg):
        # 1. Feature Extraction
        p_feat = self.ppg_enc(ppg)
        
        # 2. Creazione Contesto
        a = self.acc_enc(acc)
        e = self.eda_enc(eda)
        pr = self.prev_enc(prev_ecg)
        context = torch.cat([a, e, pr], dim=1) # (B, 80, 128)
        
        # 3. Gated Fusion (Pesa la PPG in base al contesto)
        fused = self.fusion(p_feat, context) # (B, 64, 128)
        
        # 4. Bottleneck e Decoding
        bn = self.bottleneck(fused)
        
        d1 = self.up1(bn)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        
        return self.final(d3).squeeze(1)