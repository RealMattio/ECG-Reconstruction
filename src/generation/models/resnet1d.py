import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResNetBlock1D, self).__init__()
        # Assicuriamoci che il padding mantenga la dimensione per stride=1
        # Se kernel_size è pari, usiamo un padding che preserva la simmetria
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Recupero lo shortcut
        identity = self.shortcut(x)
        
        # FIX: Se c'è un mismatch di 1 o 2 pixel dovuto al padding/kernel pari, 
        # tagliamo il tensore più lungo per farlo combaciare con quello prodotto dalle conv
        if out.shape[2] != identity.shape[2]:
            diff = identity.shape[2] - out.shape[2]
            if diff > 0:
                identity = identity[:, :, :out.shape[2]]
            else:
                out = out[:, :, :identity.shape[2]]

        out += identity
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, n_block, n_classes):
        super(ResNet1D, self).__init__()
        
        # Primo strato
        padding = kernel_size // 2
        self.first_conv = nn.Conv1d(in_channels, base_filters, kernel_size, stride, padding=padding)
        self.first_bn = nn.BatchNorm1d(base_filters)
        
        self.blocks = nn.ModuleList()
        current_filters = base_filters
        
        for i in range(n_block):
            next_filters = base_filters * (2 ** ((i + 1) // 2))
            # Manteniamo stride=1 come da tua modifica per stabilità
            self.blocks.append(ResNetBlock1D(current_filters, next_filters, kernel_size, stride=1))
            current_filters = next_filters
        
        self.final_conv = nn.Conv1d(current_filters, 1, kernel_size=1) 
        self.flatten = nn.Flatten()
        
        # Supponendo che dopo i blocchi la lunghezza sia L_out, 
        # il layer lineare deve mappare (Batch, 1 * L_out) -> (Batch, 2048)
        # Dobbiamo calcolare L_out o usare un AdaptivePool più grande
        self.adaptive_pool = nn.AdaptiveAvgPool1d(512) # Teniamo 512 punti di feature temporali
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.relu(self.first_bn(self.first_conv(x)))
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x) # Riduce i canali a 1
        x = self.adaptive_pool(x).squeeze(1) # (Batch, 512)
        x = self.fc(x) # (Batch, 2048)
        return x