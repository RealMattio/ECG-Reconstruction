import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import dei moduli interni
from src.data_loader.data_loader import DaliaDataLoader
from src.preprocessing.preprocessor import DaliaPreprocessor
from src.generation.approach1.approach1_models import Approach1LateFusion
from src.generation.approach1.trainer import Approach1Trainer
from src.evaluation.visualization import save_training_plots

def run_approach1_pipeline(subject_ids, base_path, configs):
    """
    Orchestra l'intera pipeline per l'Approccio 1:
    Caricamento -> Preprocessing -> Training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancio pipeline su: {device}")

    # --- 1. CARICAMENTO DATI ---
    loader = DaliaDataLoader()
    # Carichiamo i dati originali (is_resampled=False) come richiesto
    raw_data = loader.load_subjects(subject_ids, base_path, is_resampled=False)

    # --- 2. PREPROCESSING ---
    preprocessor = DaliaPreprocessor()
    
    # Splitting dei soggetti
    train_subs = configs['split']['train']
    val_subs = configs['split']['val']
    test_subs = configs['split']['test']
    
    data_split = preprocessor.split_data(raw_data, train_subs, val_subs, test_subs)
    
    # Normalizzazione globale (calcolata solo sul train set) [cite: 121]
    normalized_data = preprocessor.compute_and_apply_normalization(data_split)
    
    # Windowing (8s finestra, 2s shift) [cite: 119, 120]
    # Genera una lista di dizionari con chiavi 'input' (ppg, eda, acc) e 'target' (ecg)
    windowed_data = preprocessor.create_windows(
        normalized_data, 
        is_resampled=False, 
        window_size=configs.get('window_size', 8),
        window_shift=configs.get('window_shift', 2)
    )

    # --- 3. CREAZIONE DATALOADER PYTORCH ---
    def prepare_loader(data_list, batch_size, shuffle=True):
        # Estraiamo i tensori e regoliamo le dimensioni per Conv1D: (Batch, Channels, Length)
        ppg = torch.stack([torch.tensor(d['input'][0]).float().unsqueeze(0) for d in data_list])
        eda = torch.stack([torch.tensor(d['input'][1]).float().unsqueeze(0) for d in data_list])
        acc = torch.stack([torch.tensor(d['input'][2]).float().transpose(0, 1) for d in data_list])
        target = torch.stack([torch.tensor(d['target']).float().unsqueeze(0) for d in data_list])
        
        dataset = TensorDataset(ppg, eda, acc, target)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = prepare_loader(windowed_data['train'], configs['batch_size'])
    val_loader = prepare_loader(windowed_data['val'], configs['batch_size'], shuffle=False)

    # --- 4. INIZIALIZZAZIONE MODELLI ---
    # target_len deve corrispondere ai campioni della finestra ECG (es: 8s * 256Hz = 2048)
    factory = Approach1LateFusion(target_len=configs['target_len'])
    
    models_dict = {
        'ppg': factory.get_ppg_model(),
        'acc': factory.get_acc_model(),
        'eda': factory.get_eda_model(),
        'meta': factory.get_meta_learner()
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(current_dir, "..", "models", "approach1"))

    os.makedirs(save_dir, exist_ok=True)
    print(f"I modelli verranno salvati in: {save_dir}")

    # --- 5. TRAINING ---
    trainer = Approach1Trainer(models_dict, device, configs)
    
    # save_dir = os.path.join(base_path, 'models', 'approach_1')
    # os.makedirs(save_dir, exist_ok=True)
    history = {'train': [], 'val': []}
    print("\nInizio addestramento multimodale...")
    for epoch in range(configs['epochs']):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)

        # Salva le metriche per il plot finale
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
            
        print(f"Epoca {epoch}: Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")
        
        # Scheduler update basato sulla validation loss
        if trainer.scheduler:
            trainer.scheduler.step(val_metrics['total'])
            
        # Salvataggio e Early Stopping
        is_best = val_metrics['total'] < trainer.best_val_loss
        trainer.save_models(epoch, save_dir, is_best=is_best)
        
        if trainer.check_early_stopping(val_metrics['total']):
            break

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.abspath(os.path.join(current_dir, "..", "models", "approach1"))
    save_training_plots(history, plot_path)

    return trainer, windowed_data['test']