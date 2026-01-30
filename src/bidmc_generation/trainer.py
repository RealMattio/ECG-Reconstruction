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
        
        # Flag per la modularità della loss
        self.use_morph = configs.get('use_morphological_loss', False)
        self.alpha = configs.get('morph_loss_weight', 0.5) # Peso della componente morfologica

        self.optimizer = self._get_optimizer(
            configs.get('optimizer_type', 'SGDM'), 
            configs.get('lr', 0.001)
        )

    def _get_optimizer(self, opt_type, lr):
        if opt_type.upper() == 'ADAM':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type.upper() == 'SGDM':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=lr)

    def pearson_loss(self, y_pred, y_true):
        """
        Calcola la perdita basata sulla correlazione di Pearson:
        $$L_{pearson} = 1 - \frac{cov(y_{pred}, y_{true})}{\sigma_{y_{pred}}\sigma_{y_{true}}}$$
        """
        # Centralizzazione dei segnali
        y_pred_mean = torch.mean(y_pred, dim=-1, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=-1, keepdim=True)
        
        y_pred_cent = y_pred - y_pred_mean
        y_true_cent = y_true - y_true_mean
        
        # Numeratore: Covarianza
        numerator = torch.sum(y_pred_cent * y_true_cent, dim=-1)
        
        # Denominatore: Prodotto delle deviazioni standard
        denominator = torch.sqrt(
            torch.sum(y_pred_cent**2, dim=-1) * torch.sum(y_true_cent**2, dim=-1) + 1e-8
        )
        
        correlation = numerator / denominator
        # Restituiamo 1 - correlazione media del batch (vogliamo minimizzarla)
        return 1 - torch.mean(correlation)

    def compute_total_loss(self, output, target):
        """Calcola la loss combinata con MSE Pesata e Pearson."""
        
        # --- 1. Weighted MSE (Novità) ---
        # Calcoliamo una mappa di pesi basata sull'ampiezza del target assoluto.
        # lambda_peak regola quanto "spingere" sui picchi. Prova valori tra 2.0 e 5.0.
        lambda_peak = self.configs.get('peak_loss_weight', 3.0)
        
        # Il peso è 1.0 sulla linea di base e aumenta linearmente con l'ampiezza
        weights = 1.0 + lambda_peak * torch.abs(target)
        
        # Calcolo MSE pesata: mean(weights * (error)^2)
        weighted_mse = torch.mean(weights * (output - target)**2)
        rmse_weighted = torch.sqrt(weighted_mse + 1e-8)
        
        # --- 2. Combinazione con Loss Morfologica (Esistente) ---
        if self.use_morph:
            p_loss = self.pearson_loss(output, target)
            # Loss Totale = (1-alpha)*Weighted_RMSE + alpha*Pearson_Loss
            return (1 - self.alpha) * rmse_weighted + self.alpha * p_loss
        
        return rmse_weighted

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for ppg, ecg in train_loader:
            ppg, ecg = ppg.to(self.device), ecg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(ppg)
            
            # Utilizzo della loss modulare
            loss = self.compute_total_loss(output, ecg)
            
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
                loss = self.compute_total_loss(output, ecg)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    # ... (metodo fit invariato) ...

    def fit(self, train_loader, val_loader, epochs=100, patience=15):
        best_val_loss = float('inf')
        patience_counter = 0 
        history = {'train_loss': [], 'val_loss': []}
        
        # Estrazione informazioni sui batch e finestre
        num_batches = len(train_loader)
        # Il numero di finestre per batch corrisponde alla batch_size impostata nel loader
        windows_per_batch = train_loader.batch_size 
        total_windows = num_batches * windows_per_batch

        print(f"--- Training Start (Max Epochs: {epochs}, Patience: {patience}) ---")
        # Nuovo print informativo richiesto
        print(f"Batches per Epoch: {num_batches} | Windows per Batch: {windows_per_batch}")
        print(f"Total Training Windows per Epoch: {total_windows}")
        print("-" * 50)
        
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            epoch_duration = time.time() - start_time
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 
                save_path = os.path.join(self.configs['model_save_path'], 'best_ha_cnn_bilstm.pth')
                torch.save(self.model.state_dict(), save_path)
                print(" -> BEST Model Saved! ✓")
            else:
                patience_counter += 1
                print(f" -> Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Stop a epoca {epoch+1}")
                break
                
        return history