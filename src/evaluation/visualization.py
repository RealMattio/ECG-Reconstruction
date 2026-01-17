import matplotlib.pyplot as plt
import os

def save_training_plots(history, save_dir):
    """
    Genera e salva i grafici delle perdite per ogni ramo e per la loss totale.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train']) + 1)
    
    # Lista delle chiavi di loss presenti nel trainer
    loss_keys = ['total', 'ppg', 'acc', 'eda', 'final']
    
    for key in loss_keys:
        plt.figure(figsize=(10, 6))
        
        # Estrazione dati
        train_loss = [h[key] for h in history['train']]
        val_loss = [h[key] for h in history['val']]
        
        plt.plot(epochs, train_loss, 'b-', label=f'Train {key.upper()}')
        plt.plot(epochs, val_loss, 'r-', label=f'Val {key.upper()}')
        
        plt.title(f'Andamento Loss: {key.upper()}')
        plt.xlabel('Epoche')
        plt.ylabel('Smooth L1 Loss')
        plt.legend()
        plt.grid(True)
        
        # Salvataggio senza visualizzazione
        file_name = f"loss_plot_{key}.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close() # Libera la memoria
        
    print(f"Grafici di addestramento salvati in: {save_dir}")