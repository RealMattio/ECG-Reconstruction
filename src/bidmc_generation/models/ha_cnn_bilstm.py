import torch
import torch.nn as nn
import torch.nn.functional as F

class HACNNBiLSTM(nn.Module):
    def __init__(self, configs, seq_len=120):
        super(HACNNBiLSTM, self).__init__()
        
        # Usiamo i valori rilevati dinamicamente dalla pipeline
        input_dim = configs.get('input_channels', 1)
        # La lunghezza del segnale in ingresso (es. 120 se raw, 30 se WST)
        current_seq_len = configs.get('actual_seq_len', seq_len)
        
        # --- STAGE CONVOLUZIONALE ---
        self.cnn_stage = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calcolo dinamico dei neuroni post-CNN basato sulla sequenza REALE
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, current_seq_len)
            self.cnn_out_dim = self.cnn_stage(dummy_input).view(1, -1).size(1)
            print(f"DEBUG: CNN Output dimension = {self.cnn_out_dim} (Input Seq: {current_seq_len})")

        self.fc_cnn = nn.Linear(self.cnn_out_dim, 256)
        
        # --- BILSTM ---
        # L'input della BiLSTM è sempre la dimensione delle feature (1 o 19/8)
        self.bi_lstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(256, 256)

        # --- ATTENTION & REGRESSION ---
        self.tanh = nn.Tanh()
        self.attention_fc = nn.Linear(256, 256)
        
        self.regression = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len) # L'output torna sempre a 120 per il confronto
        )

    def forward(self, x):
        # x: (Batch, Channels, Seq) -> Seq può essere 120 o 30
        batch_size = x.size(0)
        
        # 1. CNN
        cnn_out = self.cnn_stage(x) 
        cnn_feat = self.fc_cnn(cnn_out.view(batch_size, -1))
        
        # 2. BiLSTM
        lstm_in = x.transpose(1, 2)
        lstm_out, _ = self.bi_lstm(lstm_in)
        lstm_feat = self.fc_lstm(lstm_out[:, -1, :]) 
        
        # 3. Attention & Fusion
        combined = self.tanh(cnn_feat + lstm_feat)
        weights = torch.sigmoid(self.attention_fc(combined))
        fused = combined * weights 
        
        # 4. Output (120 campioni)
        return self.regression(fused).unsqueeze(1)