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
        
        self.use_morph = configs.get('use_morphological_loss', False)
        self.alpha = configs.get('morph_loss_weight', 0.5)
        self.lambda_peak = configs.get('peak_loss_weight', 3.0)

        self.optimizer = self._get_optimizer(configs.get('optimizer_type', 'SGDM'), configs.get('lr', 0.001))

    def _get_optimizer(self, opt_type, lr):
        if opt_type.upper() == 'ADAM': return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type.upper() == 'SGDM': return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=lr)

    def pearson_loss(self, y_pred, y_true):
        """Calcola la perdita basata sulla correlazione di Pearson."""
        y_pred_mean = torch.mean(y_pred, dim=-1, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=-1, keepdim=True)
        y_pred_cent = y_pred - y_pred_mean
        y_true_cent = y_true - y_true_mean
        numerator = torch.sum(y_pred_cent * y_true_cent, dim=-1)
        denominator = torch.sqrt(torch.sum(y_pred_cent**2, dim=-1) * torch.sum(y_true_cent**2, dim=-1) + 1e-8)
        correlation = numerator / denominator
        return 1 - torch.mean(correlation)

    def compute_detailed_losses(self, output, target):
        """Calcola separatamente i componenti della loss."""
        # 1. Weighted RMSE
        weights = 1.0 + self.lambda_peak * torch.abs(target)
        weighted_mse = torch.mean(weights * (output - target)**2)
        rmse_w = torch.sqrt(weighted_mse + 1e-8)
        
        # 2. Pearson Loss
        p_loss = self.pearson_loss(output, target) if self.use_morph else torch.tensor(0.0).to(self.device)
        
        # 3. Total Loss
        total_loss = (1 - self.alpha) * rmse_w + self.alpha * p_loss if self.use_morph else rmse_w
        
        return total_loss, rmse_w, p_loss

    def train_epoch(self, train_loader):
        self.model.train()
        m = {'total': 0, 'rmse': 0, 'pearson': 0}
        for ppg, ecg in train_loader:
            ppg, ecg = ppg.to(self.device), ecg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(ppg)
            
            total, rmse, pear = self.compute_detailed_losses(output, ecg)
            total.backward()
            self.optimizer.step()
            
            m['total'] += total.item()
            m['rmse'] += rmse.item()
            m['pearson'] += pear.item()
            
        return {k: v / len(train_loader) for k, v in m.items()}

    def evaluate(self, val_loader):
        self.model.eval()
        m = {'total': 0, 'rmse': 0, 'pearson': 0}
        with torch.no_grad():
            for ppg, ecg in val_loader:
                ppg, ecg = ppg.to(self.device), ecg.to(self.device)
                output = self.model(ppg)
                total, rmse, pear = self.compute_detailed_losses(output, ecg)
                m['total'] += total.item(); m['rmse'] += rmse.item(); m['pearson'] += pear.item()
        return {k: v / len(val_loader) for k, v in m.items()}

    def fit(self, train_loader, val_loader, epochs=100, patience=15):
        best_val_loss = float('inf')
        patience_counter = 0 
        history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [], 'train_pearson': [], 'val_pearson': []}
        
        print(f"--- Training Start | Batches: {len(train_loader)} ---")
        
        for epoch in range(epochs):
            start_time = time.time()
            t_m = self.train_epoch(train_loader)
            v_m = self.evaluate(val_loader)
            
            # Salvataggio history
            history['train_loss'].append(t_m['total']); history['val_loss'].append(v_m['total'])
            history['train_rmse'].append(t_m['rmse']); history['val_rmse'].append(v_m['rmse'])
            history['train_pearson'].append(t_m['pearson']); history['val_pearson'].append(v_m['pearson'])
            
            duration = time.time() - start_time
            print(f"Epoch [{epoch+1:03d}/{epochs}] | {duration:.1f}s | Train Loss: {t_m['total']:.4f} | Val Loss: {v_m['total']:.4f}", end="")

            if v_m['total'] < best_val_loss:
                best_val_loss = v_m['total']
                patience_counter = 0 
                torch.save(self.model.state_dict(), os.path.join(self.configs['model_save_path'], 'best_ha_cnn_bilstm.pth'))
                print(" -> BEST ✓")
            else:
                patience_counter += 1
                print(f" -> P: {patience_counter}")
            
            if patience_counter >= patience: break
        return history