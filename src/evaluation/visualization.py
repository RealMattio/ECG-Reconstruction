import matplotlib.pyplot as plt
import os
import json

def save_training_plots(history, save_dir):
    """Genera e salva i grafici delle perdite."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train']) + 1)
    loss_keys = ['total', 'ppg', 'acc', 'eda', 'final']
    
    for key in loss_keys:
        plt.figure(figsize=(10, 6))
        train_loss = [h[key] for h in history['train']]
        val_loss = [h[key] for h in history['val']]
        
        plt.plot(epochs, train_loss, 'b-', label=f'Train {key.upper()}')
        plt.plot(epochs, val_loss, 'r-', label=f'Val {key.upper()}')
        plt.title(f'Andamento Loss: {key.upper()}')
        plt.xlabel('Epoche')
        plt.ylabel('Smooth L1 Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f"loss_plot_{key}.png"))
        plt.close()

def save_history_to_json(history, save_dir):
    """
    Salva l'intera cronologia in un file JSON.
    Utile per monitorare l'efficacia della perdita composita[cite: 461, 498].
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "training_history.json")
    
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"✓ Cronologia addestramento salvata in: {file_path}")