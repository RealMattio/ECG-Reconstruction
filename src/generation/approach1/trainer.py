import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import shutil

class Approach1Trainer:
    def __init__(self, models_dict, device, configs):
        """
        Args:
            models_dict (dict): {'ppg': model, 'acc': model, 'eda': model, 'meta': model}
            device (torch.device): 'cuda' o 'cpu'
            configs (dict): {
                'lr': float,
                'loss_weights': {'alpha': float, 'beta': float, 'gamma': float, 'delta': float},
                'use_scheduler': bool (opzionale),
                'patience': int (opzionale, per early stopping)
            }
        """
        self.models = models_dict
        self.device = device
        self.configs = configs
        
        # Spostiamo tutti i modelli sul device
        for m in self.models.values():
            m.to(self.device)

        # Criterio di perdita
        self.criterion = nn.SmoothL1Loss()
        
        # Normalizza i pesi della loss (dovrebbero sommare a 1)
        weights = configs['loss_weights']
        total_weight = sum(weights.values())
        self.loss_weights = {k: v/total_weight for k, v in weights.items()}
        print(f"Pesi loss normalizzati: {self.loss_weights}")
        
        # Ottimizzatore unico per tutti i parametri
        all_params = []
        for m in self.models.values():
            all_params += list(m.parameters())
            
        self.optimizer = optim.Adam(all_params, lr=configs.get('lr', 1e-4))
        
        # Learning rate scheduler (opzionale ma consigliato)
        if configs.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        else:
            self.scheduler = None
        
        # Per tracking del best model
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = configs.get('patience', 10)

    def train_epoch(self, dataloader):
        """Esegue un'epoca di training."""
        for m in self.models.values(): 
            m.train()
        
        total_loss = 0
        total_losses = {'ppg': 0, 'acc': 0, 'eda': 0, 'final': 0}

        for batch in tqdm(dataloader, desc="Training"):
            # Correzione dello spacchettamento:
            ppg, eda, acc, targets = [b.to(self.device) for b in batch]
            
            # LOG DI DEBUG 
            # if epoch == 0: 
            #    print(f"Shape Output: {out_ppg.shape}, Shape Target: {targets.shape}")

            if targets.dim() == 3 and targets.size(1) == 1:
                targets = targets.squeeze(1)

            self.optimizer.zero_grad()

            # 1. Forward pass dei rami individuali
            out_ppg = self.models['ppg'](ppg)
            out_acc = self.models['acc'](acc)
            out_eda = self.models['eda'](eda)

            # 2. Late Fusion con Meta-Learner
            combined_features = torch.cat([out_ppg, out_acc, out_eda], dim=1)
            out_final = self.models['meta'](combined_features)

            # 3. Calcolo Perdita Composita
            w = self.loss_weights
            loss_ppg = self.criterion(out_ppg, targets)
            loss_acc = self.criterion(out_acc, targets)
            loss_eda = self.criterion(out_eda, targets)
            loss_final = self.criterion(out_final, targets)

            loss_total = (w['alpha'] * loss_ppg + 
                          w['beta'] * loss_acc + 
                          w['gamma'] * loss_eda + 
                          w['delta'] * loss_final)

            # Gradient clipping per stabilità
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for m in self.models.values() for p in m.parameters()], 
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss_total.item()
            total_losses['ppg'] += loss_ppg.item()
            total_losses['acc'] += loss_acc.item()
            total_losses['eda'] += loss_eda.item()
            total_losses['final'] += loss_final.item()

        n = len(dataloader)
        avg_losses = {k: v/n for k, v in total_losses.items()}
        avg_losses['total'] = total_loss / n
        
        return avg_losses

    def validate_epoch(self, dataloader):
        """Esegue validazione (senza aggiornamento pesi)."""
        for m in self.models.values(): 
            m.eval()
        
        total_loss = 0
        total_losses = {'ppg': 0, 'acc': 0, 'eda': 0, 'final': 0}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                ppg, eda, acc, targets = [b.to(self.device) for b in batch]
                
                if targets.dim() == 3 and targets.size(1) == 1:
                    targets = targets.squeeze(1)

                # Forward pass
                out_ppg = self.models['ppg'](ppg)
                out_acc = self.models['acc'](acc)
                out_eda = self.models['eda'](eda)
                
                combined_features = torch.cat([out_ppg, out_acc, out_eda], dim=1)
                out_final = self.models['meta'](combined_features)

                # Calcolo loss
                w = self.loss_weights
                loss_ppg = self.criterion(out_ppg, targets)
                loss_acc = self.criterion(out_acc, targets)
                loss_eda = self.criterion(out_eda, targets)
                loss_final = self.criterion(out_final, targets)

                loss_total = (w['alpha'] * loss_ppg + 
                              w['beta'] * loss_acc + 
                              w['gamma'] * loss_eda + 
                              w['delta'] * loss_final)

                total_loss += loss_total.item()
                total_losses['ppg'] += loss_ppg.item()
                total_losses['acc'] += loss_acc.item()
                total_losses['eda'] += loss_eda.item()
                total_losses['final'] += loss_final.item()

        n = len(dataloader)
        avg_losses = {k: v/n for k, v in total_losses.items()}
        avg_losses['total'] = total_loss / n
        
        return avg_losses

    def check_early_stopping(self, val_loss):
        """Controlla se fermare il training anticipatamente."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False  # Continua training
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping! Nessun miglioramento per {self.patience} epoche.")
                return True  # Ferma training
        return False

    def save_models(self, epoch, save_dir, is_best=False):
        """
        Salva i modelli e pulisce i checkpoint precedenti meno performanti.
        """
        # 1. Rimuovi la cartella dell'epoca precedente (se non è il "best")
        # In questo modo teniamo solo l'ultimo checkpoint temporale
        if epoch > 0:
            old_path = os.path.join(save_dir, f"epoch_{epoch-1}")
            if os.path.exists(old_path):
                shutil.rmtree(old_path)

        # 2. Crea la nuova cartella per l'epoca corrente
        path = os.path.join(save_dir, f"epoch_{epoch}")
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(path, f"model_{name}.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        
        # 3. Se è il miglior modello, sovrascrivi la cartella 'best_model'
        if is_best:
            best_path = os.path.join(save_dir, "best_model")
            # Rimuoviamo la vecchia directory 'best_model' prima di ricrearla
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            os.makedirs(best_path, exist_ok=True)
            
            for name, model in self.models.items():
                torch.save(model.state_dict(), os.path.join(best_path, f"model_{name}.pth"))
            print(f"✓ Nuovo Best Model rilevato (Loss: {self.best_val_loss:.4f}) e salvato.")

    def load_models(self, load_dir):
        """Carica i modelli da una directory."""
        for name, model in self.models.items():
            path = os.path.join(load_dir, f"model_{name}.pth")
            model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Modelli caricati da {load_dir}")

