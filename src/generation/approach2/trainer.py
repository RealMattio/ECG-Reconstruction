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
        
        # Loss standard
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Coefficienti per il bilanciamento della loss
        # Ispirati alla letteratura: alpha (ricostruzione), gamma (morfologia/picchi)
        self.alpha_mse = 0.7
        self.alpha_l1 = 0.3
        self.gamma = configs.get('loss_weights', {}).get('gamma', 2.0) # Valore consigliato: 2.0
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=configs.get('lr', 5e-4),
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.patience = configs.get('patience', 20)
        self.patience_counter = 0

    def derivative_loss(self, output, target):
        """
        Calcola la differenza tra le derivate prime dei segnali.
        Penalizza il modello se non riproduce correttamente le pendenze ripide (picchi R).
        """
        # torch.diff calcola la differenza tra elementi adiacenti (B, L-1)
        d_output = torch.diff(output, dim=-1)
        d_target = torch.diff(target, dim=-1)
        
        # Usiamo L1 per la derivata perché è più reattiva ai cambiamenti bruschi
        return self.l1_loss(d_output, d_target)

    def combined_loss(self, output, target):
        """Loss totale con componente per i picchi e derivata"""
        # 1. Loss di ricostruzione (forma generale)
        mse = self.mse_loss(output, target)
        l1 = self.l1_loss(output, target)
        
        # 2. Derivative Loss (nitidezza pendenze)
        der = self.derivative_loss(output, target)
        
        # 3. Peak Loss (ampiezza picchi R)
        peak = self.peak_loss(output, target)
        
        # Combinazione pesata (Esempio di pesi)
        # Aumenta peak_weight se le ampiezze sono ancora basse
        peak_weight = 2.0
        der_weight = 1.0
        
        total_loss = (0.7 * mse + 0.3 * l1) + (der_weight * der) + (peak_weight * peak)
        
        return total_loss, mse, der, peak
    
    def peak_loss(self, output, target):
        """
        Pesa l'errore in modo esponenziale rispetto all'ampiezza del target.
        Utile per forzare la ricostruzione dei complessi QRS.
        """
        # Creiamo una mappa di pesi: valori alti nel target (picchi) pesano di più
        # Usiamo il valore assoluto del target per pesare sia i picchi R che S
        weights = torch.abs(target) + 1.0 # Offset di 1.0 per non annullare le zone piatte
        
        # Errore quadratico pesato
        return torch.mean(weights * (output - target)**2)
    
    def calculate_metrics(self, output, target):
        """Calcola metriche standard per la regressione di segnali."""
        with torch.no_grad():
            mse = F.mse_loss(output, target)
            rmse = torch.sqrt(mse)
            mae = F.l1_loss(output, target)
            
            output_flat = output.view(-1)
            target_flat = target.view(-1)
            
            vx = output_flat - torch.mean(output_flat)
            vy = target_flat - torch.mean(target_flat)
            
            correlation = torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
            )
            
        return rmse.item(), mae.item(), correlation.item()

    def train_epoch(self, dataloader):
        self.model.train()
        metrics_sum = {'loss': 0, 'rmse': 0, 'mae': 0, 'corr': 0, 'der': 0}
        
        for batch in tqdm(dataloader, desc="Training"):
            batch_data = [b.to(self.device) for b in batch]
            
            if len(batch_data) == 5:
                ppg, eda, acc, prev_ecg, targets = batch_data
            else:
                ppg, eda, acc, targets = batch_data
                prev_ecg = None
            
            targets = targets.squeeze(1) if targets.dim() == 3 else targets

            if torch.rand(1).item() > 0.3:
                ppg = self.apply_augmentation(ppg, intensity=0.5)
                acc = self.apply_augmentation(acc, intensity=0.5)
                eda = self.apply_augmentation(eda, intensity=0.5)
                if prev_ecg is not None:
                    prev_ecg = self.apply_augmentation(prev_ecg, intensity=0.3)

            self.optimizer.zero_grad()
            
            if prev_ecg is not None:
                output = self.model(ppg, acc, eda, prev_ecg)
            else:
                output = self.model(ppg, acc, eda)
            
            # Calcolo loss con componente derivativa
            loss, _, _, der = self.combined_loss(output, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Calcolo metriche
            rmse, mae, corr = self.calculate_metrics(output, targets)
            metrics_sum['loss'] += loss.item()
            metrics_sum['rmse'] += rmse
            metrics_sum['mae'] += mae
            metrics_sum['corr'] += corr
            metrics_sum['der'] += der.item()
        
        n = len(dataloader)
        results = {k: v/n for k, v in metrics_sum.items()}
        results['total'] = results['loss']
        return results

    def validate_epoch(self, dataloader):
        self.model.eval()
        metrics_sum = {'loss': 0, 'rmse': 0, 'mae': 0, 'corr': 0, 'der': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch_data = [b.to(self.device) for b in batch]
                
                if len(batch_data) == 5:
                    ppg, eda, acc, prev_ecg, targets = batch_data
                else:
                    ppg, eda, acc, targets = batch_data
                    prev_ecg = None
                
                targets = targets.squeeze(1) if targets.dim() == 3 else targets
                
                if prev_ecg is not None:
                    output = self.model(ppg, acc, eda, prev_ecg)
                else:
                    output = self.model(ppg, acc, eda)
                    
                loss, _, _, der = self.combined_loss(output, targets)
                
                rmse, mae, corr = self.calculate_metrics(output, targets)
                metrics_sum['loss'] += loss.item()
                metrics_sum['rmse'] += rmse
                metrics_sum['mae'] += mae
                metrics_sum['corr'] += corr
                metrics_sum['der'] += der.item()
                
        n = len(dataloader)
        results = {k: v/n for k, v in metrics_sum.items()}
        results['total'] = results['loss']
        self.scheduler.step(results['loss'])
        return results

    # apply_augmentation e save_checkpoint rimangono invariati
    def apply_augmentation(self, x, intensity=1.0):
        if not self.model.training: return x
        noise_level = 0.005 * intensity
        noise = torch.randn_like(x) * noise_level
        scale_range = 0.02 * intensity
        scale = torch.FloatTensor(1).uniform_(1.0 - scale_range, 1.0 + scale_range).to(self.device)
        return (x + noise) * scale
    
    def save_checkpoint(self, epoch, save_dir, is_best=False):
        last_path = os.path.join(save_dir, "last_checkpoint")
        os.makedirs(last_path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(last_path, "checkpoint.pth"))
        if is_best:
            best_path = os.path.join(save_dir, "best_model")
            if os.path.exists(best_path): shutil.rmtree(best_path)
            os.makedirs(best_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(best_path, "model.pth"))
            with open(os.path.join(best_path, "info.txt"), "w") as f:
                f.write(f"Epoch: {epoch}\nBest Val Loss: {self.best_val_loss:.6f}")
            print(f"--- [SAVE] Nuovo miglior modello salvato con Val Loss: {self.best_val_loss:.6f} ---")