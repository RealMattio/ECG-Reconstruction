import pandas as pd
import os

def save_training_history(history, save_dir):
    """Salva la history dell'addestramento in CSV."""
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    path = os.path.join(save_dir, 'training_history.csv')
    df.to_csv(path)
    print(f"History salvata in: {path}")