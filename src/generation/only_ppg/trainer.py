import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, shutil
from tqdm import tqdm

class Approach2Trainer:
    def __init__(self, model, device, configs):
        self.model = model.to(device)
        self.device = device
        self.configs = configs
        
        self.l1_loss = nn.L1Loss()
        
        # Pesi per la combinazione della loss (Morphology-focused)
        self.w_l1 = configs.get('loss_weights', {}).get('l1', 1.0)
        self.w_pearson = configs.get('loss_weights', {}).get('pearson', 1.0)
        
        # Selezione ottimizzatore basata sul paper [cite: 318, 340]
        # Il paper suggerisce SGDM per il minor RMSE 
        opt_type = configs.get('optimizer_type', 'adamw').lower()
        lr = configs.get('lr', 0.001) # Default paper: 0.001 [cite: 306]
        
        if opt_type == 'sgdm':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=1e-5
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=1e-5
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.patience = configs.get('patience', 20)
        self.patience_counter = 0

    def pearson_correlation_loss(self, x, y):
        """Calcola 1 - Pearson Correlation Coefficient."""
        centered_x = x - x.mean(dim=-1, keepdim=True)
        centered_y = y - y.mean(dim=-1, keepdim=True)
        
        covariance = (centered_x * centered_y).sum(dim=-1)
        bessel_x = torch.sqrt((centered_x**2).sum(dim=-1) + 1e-8)
        bessel_y = torch.sqrt((centered_y**2).sum(dim=-1) + 1e-8)
        
        corr = covariance / (bessel_x * bessel_y)
        return 1 - corr.mean()

    def combined_loss(self, output, target):
        # Allineamento dinamico delle lunghezze se necessario
        if output.shape != target.shape:
            output = F.interpolate(output.unsqueeze(1), size=target.shape[-1], mode='linear').squeeze(1)
        
        l1 = self.l1_loss(output, target)
        pearson = self.pearson_correlation_loss(output, target)
        
        # Il paper usa MSE/RMSE come riferimento [cite: 318]
        # La nostra combinazione aiuta a mantenere la sincronia dei picchi
        total_loss = (self.w_l1 * l1) + (self.w_pearson * pearson)
        return total_loss, l1, pearson

    def calculate_rmse(self, output, target):
        """Calcola il Root Mean Square Error come da paper[cite: 305, 544]."""
        return torch.sqrt(F.mse_loss(output, target))

    def train_epoch(self, dataloader):
        self.model.train()
        metrics_sum = {'loss': 0, 'l1': 0, 'pearson_loss': 0, 'corr': 0, 'rmse': 0}
        
        for batch in tqdm(dataloader, desc="Training"):
            ppg, targets = [b.to(self.device) for b in batch]
            targets = targets.squeeze(1) if targets.dim() == 3 else targets

            if self.model.training and torch.rand(1).item() > 0.5:
                ppg = self.apply_augmentation(ppg)

            self.optimizer.zero_grad()
            output = self.model(ppg) 
            
            loss, l1, p_loss = self.combined_loss(output, targets)
            loss.backward()
            
            # Clipping necessario per BiLSTM per evitare gradient exploding 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics_sum['loss'] += loss.item()
            metrics_sum['l1'] += l1.item()
            metrics_sum['pearson_loss'] += p_loss.item()
            metrics_sum['corr'] += (1 - p_loss.item())
            metrics_sum['rmse'] += self.calculate_rmse(output.detach(), targets).item()
        
        n = len(dataloader)
        return {k: v/n for k, v in metrics_sum.items()}

    def validate_epoch(self, dataloader):
        self.model.eval()
        metrics_sum = {'loss': 0, 'l1': 0, 'pearson_loss': 0, 'corr': 0, 'rmse': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                ppg, targets = [b.to(self.device) for b in batch]
                targets = targets.squeeze(1) if targets.dim() == 3 else targets
                
                output = self.model(ppg)
                loss, l1, p_loss = self.combined_loss(output, targets)
                
                metrics_sum['loss'] += loss.item()
                metrics_sum['l1'] += l1.item()
                metrics_sum['pearson_loss'] += p_loss.item()
                metrics_sum['corr'] += (1 - p_loss.item())
                metrics_sum['rmse'] += self.calculate_rmse(output, targets).item()
                
        n = len(dataloader)
        results = {k: v/n for k, v in metrics_sum.items()}
        self.scheduler.step(results['loss'])
        return results

    def apply_augmentation(self, x):
        # Noise inject leggero per migliorare la robustezza come suggerito [cite: 19, 66]
        noise = torch.randn_like(x) * 0.001
        return x + noise

    def save_checkpoint(self, epoch, save_dir, is_best=False):
        last_path = os.path.join(save_dir, "last_checkpoint")
        os.makedirs(last_path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(last_path, "checkpoint.pth"))
        if is_best:
            best_path = os.path.join(save_dir, "best_model")
            if os.path.exists(best_path): shutil.rmtree(best_path)
            os.makedirs(best_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(best_path, "model.pth"))