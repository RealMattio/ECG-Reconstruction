import torch
import torch.nn as nn
import torch.nn.functional as F

class HACNNBiLSTM(nn.Module):
    def __init__(self, configs, seq_len=120):
        super(HACNNBiLSTM, self).__init__()
        input_dim = configs.get('input_channels', 1)
        self.target_seq_len = seq_len # Salviamo il target (120)
        
        # --- STAGE CONVOLUZIONALE ---
        # Manteniamo i MaxPool se vuoi estrarre feature robuste, 
        # ma dobbiamo compensarli dopo.
        self.cnn_stage = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2), # 120 -> 60
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)  # 60 -> 30
        )
        
        # --- BILSTM ---
        # L'input della BiLSTM avrà ora dimensione 128 (canali della CNN)
        self.bi_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        
        # --- UPSAMPLING LAYER ---
        # Riporta la sequenza da 30 a 120
        self.upsample = nn.Upsample(size=self.target_seq_len, mode='linear', align_corners=True)
        
        # --- ATTENTION GATE (Lavora sulla sequenza upsamplata) ---
        self.attention_gate = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # REGRESSION HEAD
        layers = [
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=1)
        ]
        
        # Aggiunta modulare Sigmoid se usiamo scala 0-1
        if configs.get('normalize_01', False):
            layers.append(nn.Sigmoid())
            
        self.regression = nn.Sequential(*layers)

    def forward(self, x):
        # 1. CNN: (Batch, 1, 120) -> (Batch, 128, 30)
        cnn_out = self.cnn_stage(x)
        
        # 2. BiLSTM: (Batch, 128, 30) -> (Batch, 30, 256)
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, _ = self.bi_lstm(lstm_in)
        lstm_out = lstm_out.transpose(1, 2) # (Batch, 256, 30)
        
        # 3. UPSAMPLE: (Batch, 256, 30) -> (Batch, 256, 120)
        # Fondamentale per far combaciare l'output con il target ECG
        fused_upsampled = self.upsample(lstm_out)
        
        # 4. Attention & Regression (su 120 campioni)
        att_weights = self.attention_gate(fused_upsampled)
        fused = fused_upsampled * att_weights
        
        return self.regression(fused)