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
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=configs.get('lr', 1e-3), weight_decay=1e-4) 
        
        self.best_val_loss = float('inf')
        self.patience = configs.get('patience', 15)
        self.patience_counter = 0

    def calculate_metrics(self, output, target):
        """Calcola metriche standard per la regressione di segnali."""
        with torch.no_grad():
            mse = F.mse_loss(output, target)
            rmse = torch.sqrt(mse)
            mae = F.l1_loss(output, target)
        return rmse.item(), mae.item()

    def train_epoch(self, dataloader):
        self.model.train()
        metrics_sum = {'loss': 0, 'rmse': 0, 'mae': 0}
        
        for batch in tqdm(dataloader, desc="Training"):
            ppg, eda, acc, targets = [b.to(self.device) for b in batch]
            targets = targets.squeeze(1) if targets.dim() == 3 else targets

            # Augmentation (usando la tua funzione esistente)
            ppg, acc, eda = self.apply_augmentation(ppg), self.apply_augmentation(acc), self.apply_augmentation(eda)

            self.optimizer.zero_grad()
            output = self.model(ppg, acc, eda) 
            
            loss = self.criterion(output, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calcolo metriche
            rmse, mae = self.calculate_metrics(output, targets)
            metrics_sum['loss'] += loss.item()
            metrics_sum['rmse'] += rmse
            metrics_sum['mae'] += mae
        
        n = len(dataloader)
        results = {k: v/n for k, v in metrics_sum.items()}
        # Aggiungiamo 'total' come alias di 'loss' per la pipeline
        results['total'] = results['loss']
        return results

    def validate_epoch(self, dataloader):
        self.model.eval()
        metrics_sum = {'loss': 0, 'rmse': 0, 'mae': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                ppg, eda, acc, targets = [b.to(self.device) for b in batch]
                targets = targets.squeeze(1) if targets.dim() == 3 else targets
                
                output = self.model(ppg, acc, eda)
                loss = self.criterion(output, targets)
                
                rmse, mae = self.calculate_metrics(output, targets)
                metrics_sum['loss'] += loss.item()
                metrics_sum['rmse'] += rmse
                metrics_sum['mae'] += mae
                
        n = len(dataloader)
        # Usiamo 'total' per compatibilità con la tua pipeline esistente
        results = {k: v/n for k, v in metrics_sum.items()}
        results['total'] = results['loss'] 
        return results

    # ... (apply_augmentation e save_checkpoint rimangono invariati)
    def apply_augmentation(self, x):
        """Aggiunge rumore e variazione di scala per combattere l'overfitting."""
        if not self.model.training:
            return x
        # 1. Rumore Gaussiano leggero
        noise = torch.randn_like(x) * 0.01
        # 2. Random Scaling (moltiplica per un fattore tra 0.9 e 1.1)
        scale = torch.FloatTensor(1).uniform_(0.9, 1.1).to(self.device)
        return (x + noise) * scale
    
    # ... (save_checkpoint rimane uguale)
    def save_checkpoint(self, epoch, save_dir, is_best=False):
        """
        Gestisce il salvataggio dei modelli:
        - Salva sempre l'ultimo modello (last_checkpoint) per monitorare il progresso.
        - Se è il migliore, lo salva/sovrascrive nella cartella 'best_model'.
        """
        # 1. Percorso per l'ultimo modello (sovrascrive quello dell'epoca precedente)
        last_path = os.path.join(save_dir, "last_checkpoint")
        os.makedirs(last_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(last_path, "model.pth"))
        # Salviamo anche l'ottimizzatore se volessimo riprendere il training in futuro
        torch.save(self.optimizer.state_dict(), os.path.join(last_path, "optimizer.pth"))

        # 2. Se è il miglior modello basato sulla validation loss
        if is_best:
            best_path = os.path.join(save_dir, "best_model")
            
            # Pulizia della vecchia cartella best_model per sicurezza
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            
            os.makedirs(best_path, exist_ok=True)
            
            # Salvataggio del modello migliore
            torch.save(self.model.state_dict(), os.path.join(best_path, "model.pth"))
            
            # Opzionale: salva un piccolo file di testo con l'epoca e la loss
            with open(os.path.join(best_path, "info.txt"), "w") as f:
                f.write(f"Epoch: {epoch}\nBest Val Loss: {self.best_val_loss:.6f}")
            
            print(f"--- [SAVE] Nuovo miglior modello salvato con Val Loss: {self.best_val_loss:.6f} ---")