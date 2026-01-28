import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        # x: (Batch, Seq, Hidden)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        return torch.matmul(attn_weights, v)

class BiLSTMGenerator(nn.Module):
    def __init__(self, input_channels, hidden_size=128, num_layers=2):
        super(BiLSTMGenerator, self).__init__()
        
        # 1. Pesi apprendibili per i canali di input (Channel Importance)
        self.channel_weights = nn.Parameter(torch.ones(input_channels))
        
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 2. Self-Attention Layer (dopo la BiLSTM)
        self.attention = SelfAttention(hidden_size * 2)
        
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (Batch, Seq_Len, Channels)
        # Applicazione pesi ai canali: x * self.channel_weights
        weighted_x = x * self.channel_weights
        
        lstm_out, _ = self.lstm(weighted_x)
        
        # Applicazione Self-Attention
        attended_out = self.attention(lstm_out)
        
        out = self.fc(attended_out)
        return self.tanh(out).transpose(1, 2) # (Batch, 1, Seq_Len)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, cond_channels):
        super(PatchGANDiscriminator, self).__init__()
        in_channels = cond_channels + 1 
        
        def conv_block(in_c, out_c, stride=3):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=stride, padding=2),
                nn.InstanceNorm1d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv1d(512, 1, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, ecg, condition):
        input_data = torch.cat([ecg, condition], dim=1)
        return self.model(input_data)