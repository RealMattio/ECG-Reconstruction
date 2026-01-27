import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Meccanismo di attenzione ibrida come descritto nel paper (Eq. 12-14).
    Assegna pesi alle caratteristiche in base alla loro rilevanza morfologica[cite: 281, 294].
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W_c = nn.Linear(hidden_dim, hidden_dim) # Pesi per DCNN [cite: 281, 283]
        self.W_b = nn.Linear(hidden_dim, hidden_dim) # Pesi per BiLSTM [cite: 281, 283]
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dcnn_out, bilstm_out):
        # Calcolo dello score di attenzione (st) [cite: 281, 282]
        # dcnn_out: (B, L, H), bilstm_out: (B, L, H)
        score = self.tanh(self.W_c(dcnn_out) + self.W_b(bilstm_out))
        
        # Normalizzazione tramite Softmax (at) [cite: 285, 289]
        attention_weights = self.softmax(score)
        
        # Accumulo pesato delle caratteristiche (s) [cite: 291, 292]
        context = attention_weights * dcnn_out
        return context

class HA_CNN_BiLSTM(nn.Module):
    def __init__(self, target_len=1792):
        super(HA_CNN_BiLSTM, self).__init__()
        self.target_len = target_len
        
        # --- RAMO DCNN (Dilated CNN) ---
        # Utilizza convoluzioni dilatate per espandere il campo ricettivo senza pooling[cite: 20, 200].
        # Aiuta a gestire la dispersione del gradiente[cite: 162].
        self.dcnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, dilation=1), # rate=1 [cite: 215, 218]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2), # rate=2 [cite: 216, 218]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4), # rate=4 [cite: 217, 218]
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # --- RAMO BiLSTM ---
        # Estrae feature globali e mitiga il problema del vanishing gradient[cite: 26, 163, 164].
        self.bilstm = nn.LSTM(
            input_size=1, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True # [cite: 542]
        )
        
        # --- FUSION & ATTENTION ---
        # Dimensione BiLSTM (128*2) coincide con DCNN (256) per la fusione[cite: 166].
        self.attention = AttentionLayer(256)
        
        # --- REGRESSION LAYER ---
        # Trasforma le feature fuse nel segnale ECG target[cite: 295, 318].
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # [cite: 296, 318]
            nn.Linear(128, 1)
        )

    def forward(self, ppg):
        # ppg shape: (B, 1, 448)
        
        # 1. Analisi Frequenziale (Fourier) opzionale come feature input
        # Qui potremmo integrare una STFT, ma il paper suggerisce che la 
        # DCNN estrae già feature spaziali ottimali[cite: 195].
        
        # 2. Ramo DCNN (Locale)
        dcnn_feat = self.dcnn(ppg) # (B, 256, 448)
        dcnn_feat = dcnn_feat.transpose(1, 2) # (B, 448, 256)
        
        # 3. Ramo BiLSTM (Globale)
        ppg_trans = ppg.transpose(1, 2) # (B, 448, 1)
        bilstm_feat, _ = self.bilstm(ppg_trans) # (B, 448, 256)
        
        # 4. Attention Mechanism
        # Pesa le feature ibride per identificare le zone critiche del battito[cite: 167, 273].
        attended_feat = self.attention(dcnn_feat, bilstm_feat)
        
        # 5. Ricostruzione (Regressione)
        # Ogni campione temporale viene mappato nel corrispondente punto ECG.
        output = self.fc(attended_feat) # (B, 448, 1)
        output = output.transpose(1, 2) # (B, 1, 448)
        
        # 6. Upsampling finale a 1792 (per pareggiare i 256Hz dell'ECG)
        # Il paper usa beat-by-beat a lunghezza fissa, noi adattiamo alla tua pipeline[cite: 86, 318].
        output = F.interpolate(output, size=self.target_len, mode='linear', align_corners=True)
        
        return output.squeeze(1)


class HA_CNN_BiLSTM_Fourier(nn.Module):
    def __init__(self, target_len=1792, device='cuda'):
        super(HA_CNN_BiLSTM_Fourier, self).__init__()
        self.target_len = target_len
        self.device = device

        # --- RAMO DCNN (Dilated CNN) ---
        # Estrae caratteristiche spaziali locali[cite: 195].
        # Il numero di canali in ingresso (33) deriva da n_fft=64 (onesided: n_fft // 2 + 1).
        self.dcnn = nn.Sequential(
            nn.Conv1d(33, 64, kernel_size=3, padding=1, dilation=1), 
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2), # rate=2 [cite: 206]
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4), # rate=4 [cite: 217]
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        # --- RAMO BiLSTM ---
        # Cattura le caratteristiche temporali globali[cite: 163].
        self.bilstm = nn.LSTM(
            input_size=1, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )

        # --- HYBRID ATTENTION ---
        # Calcola l'importanza relativa delle feature[cite: 272, 281].
        self.attention_w_c = nn.Linear(256, 256) 
        self.attention_w_b = nn.Linear(256, 256) 
        
        # --- REGRESSION LAYER ---
        # Output finale per la ricostruzione dell'ECG[cite: 146, 298].
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5), # Previene l'overfitting [cite: 296]
            nn.Linear(128, 1)
        )

    def forward(self, ppg):
        # ppg shape: (B, 1, L)
        batch_size = ppg.size(0)
        
        # 1. TRASFORMATA DI FOURIER (STFT)
        # CORREZIONE: squeeze(1) per passare da (B, 1, L) a (B, L)
        x_2d = ppg.squeeze(1) 
        
        stft = torch.stft(
            x_2d, 
            n_fft=64, 
            hop_length=1, 
            normalized=True,
            return_complex=True, 
            window=torch.hann_window(64).to(self.device)
        )
        mag = torch.abs(stft) # Magnitudo: (B, 33, Time_steps)

        # 2. FEATURE EXTRACTION
        # Ramo DCNN (Locale) [cite: 162]
        # Adattiamo la dimensione temporale dello spettrogramma a quella dell'input originale
        mag_resampled = F.interpolate(mag, size=ppg.size(-1), mode='linear', align_corners=True)
        dcnn_out = self.dcnn(mag_resampled).transpose(1, 2) # (B, L, 256)

        # Ramo BiLSTM (Temporale) [cite: 164]
        bilstm_in = ppg.transpose(1, 2)
        bilstm_out, _ = self.bilstm(bilstm_in) # (B, L, 256)

        # 3. HYBRID ATTENTION [cite: 158, 281]
        score = torch.tanh(self.attention_w_c(dcnn_out) + self.attention_w_b(bilstm_out))
        attn_weights = F.softmax(score, dim=-1)
        fused_features = attn_weights * dcnn_out # Fusione locale-globale [cite: 166]

        # 4. RECONSTRUCTION
        out = self.fc(fused_features).transpose(1, 2) # (B, 1, L)
        
        # Upsampling finale alla target_len dell'ECG (es. 1792)
        if out.size(-1) != self.target_len:
            out = F.interpolate(out, size=self.target_len, mode='linear', align_corners=True)
        
        return out.squeeze(1)