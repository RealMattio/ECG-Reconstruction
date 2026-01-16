import torch
from src.generation.approach1.trainer import Approach1Trainer
# Esempio di utilizzo completo
if __name__ == "__main__":
    from src.generation.approach1.approach1_models import Approach1LateFusion
    
    # Configurazione
    configs = {
        'lr': 1e-4,
        'loss_weights': {
            'alpha': 0.25,   # PPG
            'beta': 0.25,    # ACC
            'gamma': 0.25,   # EDA
            'delta': 0.25    # Final (meta-learner)
        },
        'use_scheduler': True,
        'patience': 10
    }
    
    # Inizializza modelli
    approach1 = Approach1LateFusion(target_len=2048)
    models_dict = {
        'ppg': approach1.get_ppg_model(),
        'acc': approach1.get_acc_model(),
        'eda': approach1.get_eda_model(),
        'meta': approach1.get_meta_learner()
    }
    
    # Inizializza trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Approach1Trainer(models_dict, device, configs)
    print("Trainer per l'Approccio 1 inizializzato correttamente.")
    # Training loop (esempio)
    # for epoch in range(num_epochs):
    #     train_losses = trainer.train_epoch(train_loader)
    #     val_losses = trainer.validate_epoch(val_loader)
    #     
    #     if trainer.scheduler:
    #         trainer.scheduler.step(val_losses['total'])
    #     
    #     is_best = val_losses['total'] < trainer.best_val_loss
    #     trainer.save_models(epoch, save_dir='checkpoints', is_best=is_best)
    #     
    #     if trainer.check_early_stopping(val_losses['total']):
    #         break