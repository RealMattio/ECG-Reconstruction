import torch
import torch.nn as nn
import torch.nn.functional as F

class HACNNBiLSTM_AR(nn.Module):
    def __init__(self, configs, seq_len=125):
        """
        Modello Autoregressivo:
        Input: (Batch, 2, Seq_Len_X) -> PPG + ECG Past
        Output: (Batch, 1, Target_Len) -> 1 Secondo di ECG futuro
        """
        super(HACNNBiLSTM_AR, self).__init__()
        # Riceve 2 canali (PPG e ECG passato)
        input_dim = configs.get('input_channels', 2)
        # Il target è ora la lunghezza di 1 secondo (es. 125 campioni)
        self.target_seq_len = configs.get('target_len', seq_len) 
        
        # --- STAGE CONVOLUZIONALE ---
        self.cnn_stage = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # --- BILSTM ---
        self.bi_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        
        # --- UPSAMPLING LAYER ---
        # Riporta la sequenza estratta alla dimensione del TARGET (1 secondo)
        self.upsample = nn.Upsample(size=self.target_seq_len, mode='linear', align_corners=True)
        
        # --- ATTENTION GATE ---
        self.attention_gate = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # --- REGRESSION HEAD ---
        layers = [
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=1)
        ]
        
        if configs.get('normalize_01', False):
            layers.append(nn.Sigmoid())
            
        self.regression = nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, 2, Seq_Len_X)
        
        # 1. Feature Extraction: (Batch, 128, Seq_Len_X / 4)
        cnn_out = self.cnn_stage(x)
        
        # 2. BiLSTM: (Batch, Seq_Len_X / 4, 256)
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, _ = self.bi_lstm(lstm_in)
        lstm_out = lstm_out.transpose(1, 2)
        
        # 3. Upsample al target (1 secondo): (Batch, 256, target_len)
        fused_upsampled = self.upsample(lstm_out)
        
        # 4. Attention & Regression
        att_weights = self.attention_gate(fused_upsampled)
        fused = fused_upsampled * att_weights
        
        return self.regression(fused)