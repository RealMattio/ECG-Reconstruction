import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
from src.evaluation.visualization import plot_validation_snapshot, plot_autoregressive_epoch

class Trainer:
    def __init__(self, model, device, configs, preprocessor=None):
        self.model = model.to(device)
        self.device = device
        self.configs = configs
        self.preprocessor = preprocessor
        self.val_step = configs.get('val_step', None)
        self.optimizer = self._get_optimizer(configs.get('optimizer_type', 'ADAM'), configs.get('lr', 0.001))
        
        self.alpha = configs.get('morph_loss_weight', 0.5)
        self.lambda_peak = configs.get('peak_loss_weight', 3.0)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.fold_dir = configs['model_save_path']

    def _get_optimizer(self, opt_type, lr):
        if opt_type.upper() == 'ADAM': return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type.upper() == 'SGDM': return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=lr)

    def compute_detailed_losses(self, output, target):
        if len(target.shape) == 2: target = target.unsqueeze(1)
        # 1. Weighted RMSE
        weights = 1.0 + self.lambda_peak * torch.abs(target)
        weighted_mse = torch.mean(weights * (output - target)**2)
        rmse_w = torch.sqrt(weighted_mse + 1e-8)
        # 2. Pearson Loss
        y_pred_mean = torch.mean(output, dim=-1, keepdim=True)
        y_true_mean = torch.mean(target, dim=-1, keepdim=True)
        num = torch.sum((output - y_pred_mean) * (target - y_true_mean), dim=-1)
        den = torch.sqrt(torch.sum((output - y_pred_mean)**2, dim=-1) * torch.sum((target - y_true_mean)**2, dim=-1) + 1e-8)
        pearson = 1 - torch.mean(num / den)
        # 3. Grad Loss
        grad_loss = torch.mean((output[:,:,1:] - output[:,:,:-1] - (target[:,:,1:] - target[:,:,:-1]))**2)
        
        loss = (1 - self.alpha) * rmse_w + self.alpha * pearson + 0.1 * grad_loss
        return loss, rmse_w, pearson

    def _run_validation_and_check(self, val_loader, context_str, patience, epoch_idx, step_idx=None, train_m=None):
        self.model.eval()
        v_m = self.evaluate(val_loader)
        stop_training = False
        saved = False

        try:
            plot_validation_snapshot(self.model, val_loader, self.device, self.fold_dir, epoch_idx, step_idx)
        except: pass

        if v_m['loss'] < self.best_val_loss:
            self.best_val_loss = v_m['loss']
            self.patience_counter = 0 
            torch.save(self.model.state_dict(), os.path.join(self.fold_dir, f'best_{self.configs.get("model_type", "model")}.pth'))
            saved = True
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= patience: stop_training = True

        # --- LOGGING RICHIESTO ---
        status = "BEST ✓" if saved else f"P: {self.patience_counter}"
        log_msg = f"\n   [{context_str}] "
        if train_m:
            log_msg += f"TRAIN: Loss {train_m['loss']:.4f}, RMSE {train_m['rmse']:.4f}, Pear {train_m['pearson']:.4f} | "
        log_msg += f"VAL: Loss {v_m['loss']:.4f}, RMSE {v_m['rmse']:.4f}, Pear {v_m['pearson']:.4f} | {status}"
        tqdm.write(log_msg)
        
        return v_m, stop_training

    def train_epoch(self, train_loader, val_loader, epoch_idx, patience):
        self.model.train()
        m = {'loss': 0, 'rmse': 0, 'pearson': 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}", leave=False)
        
        for batch_idx, (ppg, ecg) in enumerate(pbar):
            ppg, ecg = ppg.to(self.device), ecg.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(ppg)
            loss, rmse, pear = self.compute_detailed_losses(output, ecg)
            loss.backward()
            self.optimizer.step()
            
            m['loss'] += loss.item(); m['rmse'] += rmse.item(); m['pearson'] += pear.item()
            pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'RMSE': f"{rmse.item():.3f}"})

            if self.val_step and (batch_idx + 1) % self.val_step == 0:
                cur_m = {k: v / (batch_idx + 1) for k, v in m.items()}
                _, stop_early = self._run_validation_and_check(val_loader, f"Step {batch_idx+1}", patience, epoch_idx, batch_idx+1, train_m=cur_m)
                self.model.train()
                if stop_early: return None

        return {k: v / len(train_loader) for k, v in m.items()}

    def evaluate(self, val_loader):
        self.model.eval()
        m = {'loss': 0, 'rmse': 0, 'pearson': 0}
        with torch.no_grad():
            for ppg, ecg in val_loader:
                ppg, ecg = ppg.to(self.device), ecg.to(self.device)
                output = self.model(ppg)
                l, r, p = self.compute_detailed_losses(output, ecg)
                m['loss'] += l.item(); m['rmse'] += r.item(); m['pearson'] += p.item()
        return {k: v / len(val_loader) for k, v in m.items()}

    def fit(self, train_loader, val_loader, epochs=100, patience=15):
        history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [], 'train_pearson': [], 'val_pearson': []}
        for epoch in range(epochs):
            t_m = self.train_epoch(train_loader, val_loader, epoch + 1, patience)
            if t_m is None: break
            
            v_m, stop = self._run_validation_and_check(val_loader, f"End Epoch {epoch+1}", patience, epoch+1, None, train_m=t_m)
            
            # Salvataggio history corretta
            for k in ['loss', 'rmse', 'pearson']:
                history[f'train_{k}'].append(t_m[k])
                history[f'val_{k}'].append(v_m[k])

            if self.preprocessor:
                plot_autoregressive_epoch(self.model, val_loader.dataset, self.preprocessor, self.device, self.configs, epoch+1, self.fold_dir)
            if stop: break
        return history