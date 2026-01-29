import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

class HACNNBiLSTMTrainer:
    def __init__(self, model, device, configs):
        self.model = model.to(device)
        self.device = device
        self.configs = configs
        self.mse_criterion = nn.MSELoss()
        self.optimizer = self._get_optimizer(
            configs.get('optimizer_type', 'SGDM'), 
            configs.get('lr', 0.001)
        )

    def _get_optimizer(self, opt_type, lr):
        if opt_type.upper() == 'ADAM':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type.upper() == 'SGDM': # Ottimizzatore top per RMSE
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=lr)

    def rmse_loss(self, output, target):
        mse = self.mse_criterion(output, target)
        return torch.sqrt(mse + 1e-8)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for ppg, ecg in train_loader:
            ppg, ecg = ppg.to(self.device), ecg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(ppg)
            loss = self.rmse_loss(output, ecg)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for ppg, ecg in val_loader:
                ppg, ecg = ppg.to(self.device), ecg.to(self.device)
                output = self.model(ppg)
                loss = self.rmse_loss(output, ecg)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=100, patience=15):
        """Esegue il ciclo di addestramento con Early Stopping."""
        best_val_loss = float('inf')
        patience_counter = 0 # Contatore per la pazienza
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"--- Training Start (Max Epochs: {epochs}, Patience: {patience}) ---")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            epoch_duration = time.time() - start_time
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Log Epoca
            print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                  f"Time: {epoch_duration:.2f}s | "
                  f"Train RMSE: {train_loss:.4f} | "
                  f"Val RMSE: {val_loss:.4f}", end="")

            # Logica Early Stopping e salvataggio miglior checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 # Reset pazienza
                save_path = os.path.join(self.configs['model_save_path'], 'best_ha_cnn_bilstm.pth')
                torch.save(self.model.state_dict(), save_path)
                print(" -> BEST Model Saved! ✓")
            else:
                patience_counter += 1
                print(f" -> Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n[Early Stopping] L'addestramento si ferma: nessun miglioramento per {patience} epoche.")
                break
                
        return history