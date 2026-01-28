import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c)
        )
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.dropout = nn.Dropout1d(dropout)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.dropout(self.conv(x) + self.shortcut(x)))

class PPGtoECG_UNet(nn.Module):
    def __init__(self, target_len=1792):
        super().__init__()
        self.target_len = target_len
        
        # --- ENCODER (Input: 448 campioni) ---
        self.enc1 = ResidualBlock(1, 32)    # 448
        self.enc2 = ResidualBlock(32, 64)   # 224 (dopo pool)
        self.enc3 = ResidualBlock(64, 128)  # 112 (dopo pool)
        self.enc4 = ResidualBlock(128, 256) # 56  (dopo pool)
        
        # --- BOTTLENECK ---
        self.bottleneck = ResidualBlock(256, 512, dropout=0.3)
        
        # --- DECODER (Upsampling verso 1792) ---
        # 56 -> 112
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualBlock(256 + 256, 256) 
        
        # 112 -> 224
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualBlock(128 + 128, 128)
        
        # 224 -> 448
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualBlock(64 + 64, 64)
        
        # 448 -> 896 (x2)
        self.up4 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec4 = ResidualBlock(32, 32)
        
        # 896 -> 1792 (x2) - Totale x4 rispetto all'input
        self.up5 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec5 = ResidualBlock(16, 16)
        
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(self, ppg):
        if ppg.dim() == 2: ppg = ppg.unsqueeze(1) # (B, 1, 448)

        # Encoder
        e1 = self.enc1(ppg)                          # (B, 32, 448)
        e2 = self.enc2(F.max_pool1d(e1, 2))          # (B, 64, 224)
        e3 = self.enc3(F.max_pool1d(e2, 2))          # (B, 128, 112)
        e4 = self.enc4(F.max_pool1d(e3, 2))          # (B, 256, 56)
        
        # Bottleneck
        bn = self.bottleneck(F.max_pool1d(e4, 2))    # (B, 512, 28)
        
        # Decoder con Skip Connections
        # Nota: usiamo F.interpolate per assicurarci che le skip matches coincidano perfettamente
        d1 = self.up1(bn)                            # 56
        d1 = torch.cat([F.interpolate(d1, size=e4.size(-1)), e4], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)                            # 112
        d2 = torch.cat([F.interpolate(d2, size=e3.size(-1)), e3], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)                            # 224
        d3 = torch.cat([F.interpolate(d3, size=e2.size(-1)), e2], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.dec4(self.up4(d3))                 # 896
        d5 = self.dec5(self.up5(d4))                 # 1792
        
        out = self.final_conv(d5)
        
        # Sicurezza finale
        if out.size(-1) != self.target_len:
            out = F.interpolate(out, size=self.target_len, mode='linear', align_corners=True)
            
        return out.squeeze(1)