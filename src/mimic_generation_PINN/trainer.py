import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import math
import numpy as np
import gc
from src.evaluation.visualization import plot_validation_snapshot, plot_autoregressive_epoch

class Trainer:
    def __init__(self, model, device, configs, preprocessor=None):
        self.model = model.to(device)
        self.device = device
        self.configs = configs
        self.preprocessor = preprocessor
        
        self.log_step = configs.get('val_step', 500) 
        
        self.optimizer = self._get_optimizer(configs.get('optimizer_type', 'ADAM'), configs.get('lr', 0.001))
        self.use_early_stopping = configs.get('use_early_stopping', True)
        self.use_lr_scheduler = configs.get('use_lr_scheduler', False)
        self.base_loss_type = configs.get('base_loss_type', 'MAE').upper()
        
        self.use_advanced_loss = configs.get('use_advanced_loss', False)
        
        if self.use_lr_scheduler:
            step_size = configs.get('lr_step_size', 15)
            gamma = configs.get('lr_gamma', 0.5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self.alpha = configs.get('morph_loss_weight', 0.4) 
        self.beta = configs.get('ode_loss_weight', 0.1)    
        self.lambda_peak = configs.get('peak_loss_weight', 3.0)
        self.gamma_pat = configs.get('pat_loss_weight', 0.5)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 1 
        self.fold_dir = configs['model_save_path']
        
        self.mcsharry_a = torch.tensor([1.0, -5.0, 30.0, -7.5, 0.55], device=self.device, dtype=torch.float32)
        self.mcsharry_b = torch.tensor([0.25, 0.1, 0.1, 0.1, 0.3], device=self.device, dtype=torch.float32)
        self.mcsharry_theta = torch.tensor([-1.05, -0.25, 0.0, 0.25, 1.4], device=self.device, dtype=torch.float32)
        self.z0 = 0.0 

        print(f"[TRAINER] Composizione Loss in uso:")
        print(f"          - Ampiezza  ({self.base_loss_type}): Peso 1.0 (Peak Penalty: x{self.lambda_peak})")
        if self.use_advanced_loss:
            print(f"          - Morfologia (COSINE SIMILARITY): Peso {self.alpha}")
            print(f"          - Sfasamento PAT (DELAY SHIFT): Peso {self.gamma_pat}")
        else:
            print(f"          - Morfologia (PEARSON): Peso {self.alpha}")
            print(f"          - Sfasamento PAT: DISATTIVATO")
        print(f"          - Fisica     (ODE PINN): Peso {self.beta}")
        print("-" * 50)

    def _get_model_state(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

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

    def _compute_pat_delay_loss(self, output, target, max_shift=15):
        batch_size, channels, seq_len = output.shape
        out_norm = output - output.mean(dim=-1, keepdim=True)
        tgt_norm = target - target.mean(dim=-1, keepdim=True)
        
        out_std = out_norm.std(dim=-1, keepdim=True) + 1e-8
        tgt_std = tgt_norm.std(dim=-1, keepdim=True) + 1e-8
        
        out_norm = out_norm / out_std
        tgt_norm = tgt_norm / tgt_std

        padded_out = F.pad(out_norm, (max_shift, max_shift), mode='constant', value=0)
        padded_out_grouped = padded_out.view(1, batch_size * channels, -1)
        tgt_grouped = tgt_norm.view(batch_size * channels, 1, seq_len)
        
        cross_corr = F.conv1d(padded_out_grouped, tgt_grouped, groups=batch_size * channels)
        cross_corr = cross_corr.view(batch_size, channels, -1)
        
        corr_probs = F.softmax(cross_corr / 10.0, dim=-1)
        
        shift_indices = torch.arange(-max_shift, max_shift + 1, device=output.device, dtype=torch.float32)
        shift_indices = shift_indices.view(1, 1, -1)
        
        expected_shift = torch.sum(corr_probs * shift_indices, dim=-1)
        delay_loss = torch.mean((expected_shift / max_shift)**2)
        
        return delay_loss

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
            
        fs = self.configs.get('target_fs', 125)
        dz_emp = output[:,:,1:] - output[:,:,:-1]
        dz_phys = self._compute_physics_derivative(output[:,:,:-1], theta[:,:,:-1]) / fs
        ode_loss = torch.mean((dz_emp - dz_phys)**2)

        if self.use_advanced_loss:
            peak_mask = (target > 0.7).float()
            focal_peak_loss = torch.mean(peak_mask * (output - target)**2) * 10.0
            
            cos_sim = F.cosine_similarity(output, target, dim=-1)
            morph_loss = 1.0 - torch.mean(cos_sim)
            
            pat_loss = self._compute_pat_delay_loss(output, target, max_shift=15)
            total_loss = (amp_loss + focal_peak_loss) + (self.alpha * morph_loss) + (self.gamma_pat * pat_loss) + (self.beta * ode_loss)
            
        else:
            focal_peak_loss = torch.tensor(0.0, device=self.device)
            
            y_pred_mean = torch.mean(output, dim=-1, keepdim=True)
            y_true_mean = torch.mean(target, dim=-1, keepdim=True)
            num = torch.sum((output - y_pred_mean) * (target - y_true_mean), dim=-1)
            den = torch.sqrt(torch.sum((output - y_pred_mean)**2, dim=-1) * torch.sum((target - y_true_mean)**2, dim=-1) + 1e-8)
            morph_loss = 1 - torch.mean(num / den)
            
            pat_loss = torch.tensor(0.0, device=self.device)
            total_loss = 1.0 * amp_loss + self.alpha * morph_loss + self.beta * ode_loss

        return total_loss, amp_loss, morph_loss, ode_loss, pat_loss

    def evaluate(self, val_loader):
        self.model.eval()
        m = {'loss': 0, 'amp_loss': 0, 'pearson': 0, 'ode': 0, 'pat': 0}
        
        batch_count = 0
        with torch.no_grad():
            for ppg, ecg, theta, omega, info in val_loader:
                ppg, ecg = ppg.to(self.device), ecg.to(self.device)
                theta, omega = theta.to(self.device), omega.to(self.device)
                
                output = self.model(ppg)
                l, a, morph, o, pat = self.compute_detailed_losses(output, ecg, theta, omega)
                
                m['loss'] += l.item()
                m['amp_loss'] += a.item()
                m['pearson'] += morph.item() 
                m['ode'] += o.item()
                m['pat'] += pat.item()
                batch_count += 1
                
        if batch_count == 0:
            batch_count = 1 
            
        return {k: v / batch_count for k, v in m.items()}

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        m = {'loss': 0, 'amp_loss': 0, 'pearson': 0, 'ode': 0, 'pat': 0}
        
        batch_count = 0
        start_time = time.time()
        
        for batch_idx, (ppg, ecg, theta, omega, info) in enumerate(train_loader):
            ppg, ecg = ppg.to(self.device), ecg.to(self.device)
            theta, omega = theta.to(self.device), omega.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(ppg)
            
            loss, amp_loss, morph, ode, pat = self.compute_detailed_losses(output, ecg, theta, omega)
            
            loss.backward()
            self.optimizer.step()
            
            m['loss'] += loss.item()
            m['amp_loss'] += amp_loss.item()
            m['pearson'] += morph.item()
            m['ode'] += ode.item()
            m['pat'] += pat.item()
            batch_count += 1
            
            if batch_count % self.log_step == 0:
                elapsed = time.time() - start_time
                batches_per_sec = batch_count / elapsed
                avg_l = m['loss'] / batch_count
                avg_a = m['amp_loss'] / batch_count
                avg_pat = m['pat'] / batch_count
                print(f"      [Epoca {epoch+1:03d} | Batch {batch_count}] L: {avg_l:.4f} | Amp: {avg_a:.4f} | PAT: {avg_pat:.4f} | Speed: {batches_per_sec:.2f} b/s")

        if batch_count == 0:
            batch_count = 1
            
        return {k: v / batch_count for k, v in m.items()}

    def fit(self, train_loader, val_loader=None, epochs=100, patience=15):
        # IL DIZIONARIO COMPLETO (INCLUSE LE NUOVE METRICHE)
        history = {'train_loss': [], 'val_loss': [], 'train_amp_loss': [], 'val_amp_loss': [], 
                   'train_pearson': [], 'val_pearson': [], 'train_ode': [], 'val_ode': [],
                   'train_pat': [], 'val_pat': []}
        
        start_epoch = 0
        checkpoint_path = os.path.join(self.fold_dir, 'latest_checkpoint.pth')
        
        if os.path.exists(checkpoint_path):
            print(f"\n🔄 Trovato checkpoint! Ripresa dall'ultimo stato salvato in {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.use_lr_scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            self.best_epoch = checkpoint.get('best_epoch', 1) 
            start_epoch = checkpoint.get('epoch', 0)
            
            # --- FIX CHIAVI MANCANTI NEL CHECKPOINT ---
            loaded_history = checkpoint.get('history', {})
            # Ricostruiamo il dizionario base assicurandoci che tutte le chiavi correnti esistano.
            # Se nel vecchio dizionario mancavano, riempiamole con zeri fittizi per mantenere la 
            # lunghezza dell'array coerente con l'epoca da cui si riparte.
            for key in history.keys():
                if key in loaded_history:
                    history[key] = loaded_history[key]
                else:
                    # Inserisce zeri per le epoche precedenti se la metrica (es. 'pat') non esisteva
                    history[key] = [0.0] * start_epoch 

            print(f"▶ Ripresa dall'epoca {start_epoch + 1}/{epochs}\n")
        
        fold_start_time = time.time()
        morph_label = "Cos " if self.use_advanced_loss else "Pear"
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            t_m = self.train_epoch(train_loader, epoch)
            
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

            if val_loader is not None:
                v_m = self.evaluate(val_loader)
                
                if v_m['loss'] < self.best_val_loss:
                    self.best_val_loss = v_m['loss']
                    self.patience_counter = 0 
                    self.best_epoch = epoch + 1  
                    torch.save(self._get_model_state(), os.path.join(self.fold_dir, f'best_{self.configs.get("model_type", "model")}.pth'))
                    saved = True
                    
                    try:
                        plot_validation_snapshot(self.model, val_loader, self.device, self.fold_dir, epoch+1, prefix='val')
                    except Exception as e:
                        print(f"   [PLOT ERROR] Impossibile generare i plot per l'epoca {epoch+1}: {e}")
                else:
                    self.patience_counter += 1
                
                if self.use_early_stopping and self.patience_counter >= patience: 
                    stop_training = True

                status = "BEST ✓" if saved else f"P: {self.patience_counter}"
                log_msg = f"[Epoch {epoch+1:03d} | {time_str} | LR: {current_lr:.2e}] TRAIN: L {t_m['loss']:.4f}, {loss_name} {t_m['amp_loss']:.4f}, {morph_label} {t_m['pearson']:.4f}, PAT {t_m['pat']:.4f} | VAL: L {v_m['loss']:.4f}, {loss_name} {v_m['amp_loss']:.4f}, {morph_label} {v_m['pearson']:.4f}, PAT {v_m['pat']:.4f} | {status}"
                print(log_msg)
                
                for k in ['loss', 'amp_loss', 'pearson', 'ode', 'pat']:
                    history[f'train_{k}'].append(t_m[k])
                    history[f'val_{k}'].append(v_m[k])
                    
            else:
                self.best_epoch = epoch + 1
                torch.save(self._get_model_state(), os.path.join(self.fold_dir, f'best_{self.configs.get("model_type", "model")}.pth'))
                
                for k in ['loss', 'amp_loss', 'pearson', 'ode', 'pat']:
                    history[f'train_{k}'].append(t_m[k])
                    
                log_msg = f"[Epoch {epoch+1:03d} | {time_str} | LR: {current_lr:.2e}] TRAIN: L {t_m['loss']:.4f}, {loss_name} {t_m['amp_loss']:.4f}, {morph_label} {t_m['pearson']:.4f}, PAT {t_m['pat']:.4f} | [NO VAL - SAVED ✓]"
                print(log_msg)

            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': self._get_model_state(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'best_epoch': self.best_epoch, 
                'history': history
            }
            if self.use_lr_scheduler:
                checkpoint_state['scheduler_state_dict'] = self.scheduler.state_dict()
                
            torch.save(checkpoint_state, checkpoint_path)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if stop_training:
                print(f"   [Early Stopping] Pazienza esaurita ({patience} epoche senza miglioramenti).")
                break
        
        fold_duration = time.time() - fold_start_time
        f_mins = int(fold_duration // 60)
        f_secs = int(fold_duration % 60)
        print(f"\n✅ Addestramento completato in {f_mins} minuti e {f_secs} secondi.")
        
        history['best_epoch'] = self.best_epoch
        return history