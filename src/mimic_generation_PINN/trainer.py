import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import math
import numpy as np
from src.evaluation.visualization import plot_validation_snapshot, plot_autoregressive_epoch

class Trainer:
    def __init__(self, model, device, configs, preprocessor=None):
        self.model = model.to(device)
        self.device = device
        self.configs = configs
        self.preprocessor = preprocessor
        self.val_step = configs.get('val_step', None)
        self.optimizer = self._get_optimizer(configs.get('optimizer_type', 'ADAM'), configs.get('lr', 0.001))
        
        self.use_early_stopping = configs.get('use_early_stopping', True)
        self.use_lr_scheduler = configs.get('use_lr_scheduler', False)
        self.base_loss_type = configs.get('base_loss_type', 'MAE').upper()
        
        if self.use_lr_scheduler:
            step_size = configs.get('lr_step_size', 15)
            gamma = configs.get('lr_gamma', 0.5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self.alpha = configs.get('morph_loss_weight', 0.4) 
        self.beta = configs.get('ode_loss_weight', 0.1)    
        self.lambda_peak = configs.get('peak_loss_weight', 3.0)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 1 # <-- Variabile per tracciare l'epoca migliore
        self.fold_dir = configs['model_save_path']
        
        self.mcsharry_a = torch.tensor([1.0, -5.0, 30.0, -7.5, 0.55], device=self.device, dtype=torch.float32)
        self.mcsharry_b = torch.tensor([0.25, 0.1, 0.1, 0.1, 0.3], device=self.device, dtype=torch.float32)
        self.mcsharry_theta = torch.tensor([-1.05, -0.25, 0.0, 0.25, 1.4], device=self.device, dtype=torch.float32)
        self.z0 = 0.0 

        print(f"[TRAINER] Composizione Loss ibrida in uso:")
        print(f"          - Ampiezza  ({self.base_loss_type}): Peso 1.0 (Peak Penalty: x{self.lambda_peak})")
        print(f"          - Morfologia (PEARSON): Peso {self.alpha}")
        print(f"          - Fisica     (ODE PINN): Peso {self.beta}")
        print("-" * 50)

    def _get_optimizer(self, opt_type, lr):
        if opt_type.upper() == 'ADAM': return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type.upper() == 'SGDM': return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=lr)

    def _compute_physics_derivative(self, z, theta):
        theta_exp = theta.unsqueeze(-1)
        dtheta = (theta_exp - self.mcsharry_theta) % (2 * math.pi)
        dtheta = torch.where(dtheta > math.pi, dtheta - 2 * math.pi, dtheta)
        term = self.mcsharry_a * dtheta * torch.exp(-0.5 * (dtheta / self.mcsharry_b)**2)
        dz_phys = -torch.sum(term, dim=-1) - (z - self.z0)
        return dz_phys

    def compute_detailed_losses(self, output, target, theta, omega):
        if len(target.shape) == 2: target = target.unsqueeze(1)
        if len(theta.shape) == 2: theta = theta.unsqueeze(1)
        
        weights = 1.0 + self.lambda_peak * torch.abs(target)
        
        if self.base_loss_type == 'HUBER':
            unweighted_huber = F.huber_loss(output, target, reduction='none', delta=1.0)
            amp_loss = torch.mean(weights * unweighted_huber)
        elif self.base_loss_type == 'MAE':
            amp_loss = torch.mean(weights * torch.abs(output - target))
        else:
            weighted_mse = torch.mean(weights * (output - target)**2)
            amp_loss = torch.sqrt(weighted_mse + 1e-8)
        
        y_pred_mean = torch.mean(output, dim=-1, keepdim=True)
        y_true_mean = torch.mean(target, dim=-1, keepdim=True)
        num = torch.sum((output - y_pred_mean) * (target - y_true_mean), dim=-1)
        den = torch.sqrt(torch.sum((output - y_pred_mean)**2, dim=-1) * torch.sum((target - y_true_mean)**2, dim=-1) + 1e-8)
        pearson = 1 - torch.mean(num / den)
        
        fs = self.configs.get('target_fs', 125)
        dz_emp = output[:,:,1:] - output[:,:,:-1]
        dz_phys = self._compute_physics_derivative(output[:,:,:-1], theta[:,:,:-1]) / fs
        ode_loss = torch.mean((dz_emp - dz_phys)**2)
        
        loss = 1.0 * amp_loss + self.alpha * pearson + self.beta * ode_loss
        return loss, amp_loss, pearson, ode_loss

    def evaluate(self, val_loader):
        self.model.eval()
        m = {'loss': 0, 'amp_loss': 0, 'pearson': 0, 'ode': 0}
        with torch.no_grad():
            for ppg, ecg, theta, omega, info in val_loader:
                ppg, ecg = ppg.to(self.device), ecg.to(self.device)
                theta, omega = theta.to(self.device), omega.to(self.device)
                
                output = self.model(ppg)
                l, a, p, o = self.compute_detailed_losses(output, ecg, theta, omega)
                
                m['loss'] += l.item(); m['amp_loss'] += a.item(); m['pearson'] += p.item(); m['ode'] += o.item()
        return {k: v / len(val_loader) for k, v in m.items()}

    def train_epoch(self, train_loader):
        self.model.train()
        m = {'loss': 0, 'amp_loss': 0, 'pearson': 0, 'ode': 0}
        
        for batch_idx, (ppg, ecg, theta, omega, info) in enumerate(train_loader):
            ppg, ecg = ppg.to(self.device), ecg.to(self.device)
            theta, omega = theta.to(self.device), omega.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(ppg)
            
            loss, amp_loss, pear, ode = self.compute_detailed_losses(output, ecg, theta, omega)
            
            loss.backward()
            self.optimizer.step()
            
            m['loss'] += loss.item(); m['amp_loss'] += amp_loss.item(); m['pearson'] += pear.item(); m['ode'] += ode.item()

        return {k: v / len(train_loader) for k, v in m.items()}

    def fit(self, train_loader, val_loader=None, epochs=100, patience=15):
        # NOTA: Ora accetta val_loader=None per il training finale
        history = {'train_loss': [], 'val_loss': [], 'train_amp_loss': [], 'val_amp_loss': [], 
                   'train_pearson': [], 'val_pearson': [], 'train_ode': [], 'val_ode': []}
        
        fold_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            t_m = self.train_epoch(train_loader)
            
            if self.use_lr_scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            stop_training = False
            saved = False

            epoch_duration = time.time() - epoch_start_time
            mins = int(epoch_duration // 60)
            secs = int(epoch_duration % 60)
            time_str = f"{mins}m {secs}s"
            loss_name = self.base_loss_type

            # --- SE ESISTE IL VALIDATION SET (K-FOLD) ---
            if val_loader is not None:
                v_m = self.evaluate(val_loader)
                
                if v_m['loss'] < self.best_val_loss:
                    self.best_val_loss = v_m['loss']
                    self.patience_counter = 0 
                    self.best_epoch = epoch + 1  # Salviamo l'epoca migliore!
                    torch.save(self.model.state_dict(), os.path.join(self.fold_dir, f'best_{self.configs.get("model_type", "model")}.pth'))
                    saved = True
                    
                    try:
                        plot_validation_snapshot(self.model, train_loader, self.device, self.fold_dir, epoch+1, prefix='train')
                        plot_validation_snapshot(self.model, val_loader, self.device, self.fold_dir, epoch+1, prefix='val')
                        if self.preprocessor:
                            plot_autoregressive_epoch(self.model, val_loader.dataset, self.preprocessor, self.device, self.configs, epoch+1, self.fold_dir)
                    except Exception as e:
                        print(f"   [PLOT ERROR] Impossibile generare i plot per l'epoca {epoch+1}: {e}")
                else:
                    self.patience_counter += 1
                
                if self.use_early_stopping and self.patience_counter >= patience: 
                    stop_training = True

                status = "BEST ✓" if saved else f"P: {self.patience_counter}"
                log_msg = f"[Epoch {epoch+1:03d} | {time_str} | LR: {current_lr:.2e}] TRAIN: L {t_m['loss']:.4f}, {loss_name} {t_m['amp_loss']:.4f}, Pear {t_m['pearson']:.4f}, ODE {t_m['ode']:.4f} | VAL: L {v_m['loss']:.4f}, {loss_name} {v_m['amp_loss']:.4f}, Pear {v_m['pearson']:.4f}, ODE {v_m['ode']:.4f} | {status}"
                print(log_msg)
                
                for k in ['loss', 'amp_loss', 'pearson', 'ode']:
                    history[f'train_{k}'].append(t_m[k])
                    history[f'val_{k}'].append(v_m[k])
                    
            # --- SE NON C'E' VALIDATION SET (ADDESTRAMENTO FINALE) ---
            else:
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), os.path.join(self.fold_dir, f'best_{self.configs.get("model_type", "model")}.pth'))
                
                for k in ['loss', 'amp_loss', 'pearson', 'ode']:
                    history[f'train_{k}'].append(t_m[k])
                    
                log_msg = f"[Epoch {epoch+1:03d} | {time_str} | LR: {current_lr:.2e}] TRAIN: L {t_m['loss']:.4f}, {loss_name} {t_m['amp_loss']:.4f}, Pear {t_m['pearson']:.4f}, ODE {t_m['ode']:.4f} | [NO VAL - SAVED ✓]"
                print(log_msg)

            if stop_training:
                print(f"   [Early Stopping] Pazienza esaurita ({patience} epoche senza miglioramenti).")
                break
        
        fold_duration = time.time() - fold_start_time
        f_mins = int(fold_duration // 60)
        f_secs = int(fold_duration % 60)
        print(f"\n✅ Addestramento completato in {f_mins} minuti e {f_secs} secondi.")
        
        history['best_epoch'] = self.best_epoch # Agganciamo l'epoca migliore all'output
        return history