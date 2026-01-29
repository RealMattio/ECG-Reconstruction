import torch
import torch.nn as nn
import torch.nn.functional as F

class HACNNBiLSTM(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, seq_len=120):
        """
        Architettura HA-CNN-BILSTM basata sulla Tabella 1 del paper.
        input_dim: 1 per PPG grezzo (anche se il paper suggerisce 19 per WST [cite: 154, 318]).
        """
        super(HACNNBiLSTM, self).__init__()
        
        # --- STAGE CONVOLUZIONALE (Local Feature Extraction) ---
        # Conv -> BN -> ReLU -> MaxPool (x2) 
        self.cnn_stage = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Seq: 120 -> 60
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Seq: 60 -> 30
        )
        
        # --- FULLY CONNECTED PRE-LSTM ---
        self.fc_pre_lstm = nn.Linear(128 * 30, 256)
        
        # --- BILSTM LAYER (Temporal Feature Extraction) ---
        # Bidirectional mitigates gradient vanishing [cite: 26, 318]
        self.bi_lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # --- ATTENTION & FUSION LAYERS (Tabella 1) ---
        self.fc_post_lstm = nn.Linear(256, 256)
        self.tanh = nn.Tanh() # Tanh posizionato a metà rete 
        
        self.fc_attention = nn.Linear(256, 256)
        
        # --- REGRESSION LAYER (Final Output) ---
        # Include Dropout per prevenire overfitting [cite: 296, 318]
        self.regression = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len) # Regression output lineare per ampiezze reali 
        )

    def forward(self, x):
        # x: (Batch, 1, 120)
        batch_size = x.size(0)
        
        # 1. CNN Stage
        features = self.cnn_stage(x) # (Batch, 128, 30)
        features = features.view(batch_size, -1)
        
        # 2. Fully Connected
        features = self.fc_pre_lstm(features) 
        
        # 3. BiLSTM (input: Batch, Seq=1, H=256)
        lstm_out, _ = self.bi_lstm(features.unsqueeze(1)) 
        lstm_out = lstm_out.squeeze(1) # (Batch, 256)
        
        # 4. Attention & Fusion Logic 
        # Concatenated addition e multiplier
        res = self.fc_post_lstm(lstm_out)
        activated = self.tanh(res)
        
        # Calcolo pesi attenzione (Multiplier Layer)
        attn_weights = self.fc_attention(activated)
        fused = activated * attn_weights 
        
        # 5. Regression Output
        out = self.regression(fused) # (Batch, 120)
        return out.unsqueeze(1) # Ritorna (Batch, 1, 120)