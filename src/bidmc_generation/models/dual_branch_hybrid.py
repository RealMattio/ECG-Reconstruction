import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchHybrid(nn.Module):
    def __init__(self, configs, seq_len=125):
        super(DualBranchHybrid, self).__init__()
        
        # --- Configurazione Dimensionale ---
        self.img_height = configs.get('input_channels', 2) 
        self.target_len = configs.get('target_len', seq_len)
        self.split_dim = self.img_height // 2
        
        # Se l'input è raw (2 canali), l'immagine è alta 2. 
        # Se è WST, è alta ~38. Adattiamo i canali della CNN di conseguenza.
        cnn_base_channels = 32 # Aumentiamo la capacità base (era 16)

        # ============================================================
        # RAMO 1: CNN 2D POTENZIATA (ResNet-block style idea)
        # ============================================================
        self.cnn_2d = nn.Sequential(
            # Block 1
            nn.Conv2d(1, cnn_base_channels, kernel_size=(3, 3), padding=(1, 1)), 
            nn.BatchNorm2d(cnn_base_channels),
            nn.LeakyReLU(0.1),
            
            # Block 2
            nn.Conv2d(cnn_base_channels, cnn_base_channels*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_base_channels*2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 2)), # Downsample spaziale e temporale
            
            # Block 3 (Profondità aggiunta)
            nn.Conv2d(cnn_base_channels*2, cnn_base_channels*4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.LeakyReLU(0.1),
            
            # Block 4
            nn.Conv2d(cnn_base_channels*4, cnn_base_channels*4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 1)) # Downsample solo spaziale (features) per mantenere tempo
        )
        
        # Calcolo dimensione output CNN
        # Altezza: ridotta di 2 volte dal primo pool e 2 volte dal secondo -> /4
        self.h_out = self.img_height // 4
        if self.h_out == 0: self.h_out = 1 
        
        # Proiezione: (128 channels * h_out) -> 256 feature vector temporale
        self.cnn_projector = nn.Sequential(
            nn.Conv1d(cnn_base_channels*4 * self.h_out, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )

        # ============================================================
        # RAMO 2: Dual Bi-LSTM POTENZIATA (Più hidden units)
        # ============================================================
        lstm_hidden = 128 # Aumentato da 64 a 128 per maggiore capacità
        
        self.lstm_ppg = nn.LSTM(input_size=self.split_dim, hidden_size=lstm_hidden, 
                                num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.lstm_ecg = nn.LSTM(input_size=self.split_dim, hidden_size=lstm_hidden, 
                                num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Fusione LSTM: 256 (PPG) + 256 (ECG) = 512 -> Proiettato a 256
        self.lstm_fusion = nn.Sequential(
            nn.Linear(lstm_hidden*4, 256),
            nn.LayerNorm(256), # LayerNorm aiuta la stabilità delle LSTM
            nn.Tanh()
        )

        # ============================================================
        # ATTENZIONE CROSS-MODALE (CNN features <-> LSTM features)
        # ============================================================
        # Invece di un'attenzione interna alla LSTM, usiamo un'attenzione che impara
        # a pesare il contributo della CNN (spaziale) rispetto alla LSTM (temporale)
        self.attention_gate = nn.Sequential(
            nn.Linear(512, 128), # Input: CNN(256) + LSTM(256)
            nn.Tanh(),
            nn.Linear(128, 2), # Output: 2 pesi (alpha per CNN, beta per LSTM)
            nn.Softmax(dim=1)
        )

        # ============================================================
        # DECODER FINALE
        # ============================================================
        self.upsample = nn.Upsample(size=self.target_len, mode='linear', align_corners=True)
        
        self.final_regression = nn.Sequential(
            # Input: 512 (Concatenazione CNN + LSTM pesata)
            nn.Conv1d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), # BatchNorm anche qui per velocizzare convergenza
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(64, 1, kernel_size=1)
        )
        
        self.normalize_01 = configs.get('normalize_01', False)
        if self.normalize_01:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        
        # --- 1. RAMO CNN ---
        x_img = x.unsqueeze(1) # (B, 1, C, T)
        cnn_out = self.cnn_2d(x_img) # (B, 128, H_out, T_out)
        
        B, C, H, T = cnn_out.size()
        cnn_flat = cnn_out.view(B, C * H, T) 
        cnn_feat = self.cnn_projector(cnn_flat) # (B, 256, T_out)
        
        # --- 2. RAMO LSTM ---
        x_ppg = x[:, :self.split_dim, :]
        x_ecg = x[:, self.split_dim:, :]
        
        out_ppg, _ = self.lstm_ppg(x_ppg.transpose(1, 2))
        out_ecg, _ = self.lstm_ecg(x_ecg.transpose(1, 2))
        
        # Concatenazione e Fusione
        lstm_concat = torch.cat([out_ppg, out_ecg], dim=2) # (B, T, 512)
        lstm_feat = self.lstm_fusion(lstm_concat).transpose(1, 2) # (B, 256, T)
        
        # --- 3. ALLINEAMENTO TEMPORALE ---
        # CNN ha subito pooling temporale (/2), LSTM no.
        # Dobbiamo allinearle PRIMA della fusione finale per il target.
        
        # Upsample della CNN alla dimensione temporale originale (o target se necessario)
        # Qui riportiamo CNN alla dimensione della LSTM per calcolare l'attenzione passo-passo
        if cnn_feat.shape[-1] != lstm_feat.shape[-1]:
             cnn_feat = F.interpolate(cnn_feat, size=lstm_feat.shape[-1], mode='linear')
             
        # --- 4. ATTENZIONE DINAMICA ---
        # Calcoliamo pesi per ogni istante temporale: (B, T, 2)
        combined_context = torch.cat([cnn_feat, lstm_feat], dim=1).transpose(1, 2) # (B, T, 512)
        weights = self.attention_gate(combined_context) # (B, T, 2)
        
        alpha = weights[:, :, 0].unsqueeze(2) # Peso CNN
        beta = weights[:, :, 1].unsqueeze(2)  # Peso LSTM
        
        # Applicazione pesi
        cnn_weighted = cnn_feat * alpha.transpose(1, 2)
        lstm_weighted = lstm_feat * beta.transpose(1, 2)
        
        # --- 5. PREPARAZIONE OUTPUT ---
        # Taglio/Upsample per il Target (Futuro)
        cnn_cut = cnn_weighted[:, :, -self.target_len:]
        lstm_cut = lstm_weighted[:, :, -self.target_len:]
        
        # Sicurezza dimensionale finale (se target_len è diverso dal taglio)
        if cnn_cut.shape[-1] != self.target_len:
             cnn_cut = F.interpolate(cnn_cut, size=self.target_len, mode='linear')
        if lstm_cut.shape[-1] != self.target_len:
             lstm_cut = F.interpolate(lstm_cut, size=self.target_len, mode='linear')

        # Concatenazione feature pesate
        final_input = torch.cat([cnn_cut, lstm_cut], dim=1) # (B, 512, Target_Len)
        
        out = self.final_regression(final_input)
        
        if self.normalize_01:
            out = self.sigmoid(out)
            
        return out