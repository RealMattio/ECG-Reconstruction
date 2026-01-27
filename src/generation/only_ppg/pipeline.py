import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader.data_loader import DaliaDataLoader
from src.preprocessing.preprocessor import DaliaPreprocessor
from .approach2_model import Approach2Model
from .trainer import Approach2Trainer
from src.evaluation.visualization import (
    save_training_plots, 
    save_history_to_json, 
    plot_random_sample, 
    plot_inference_comparison, 
    plot_ppg_ecg_comparison
)

def diagnose_data(windowed_data, name="train"):
    print(f"\n=== DIAGNOSTICA DATI ({name.upper()}) ===")
    sample = windowed_data[0]
    
    # Input: PPG (64Hz) -> 448 campioni per 7s
    ppg = sample['input'][0]
    # Target: ECG (256Hz) -> 1792 campioni per 7s
    target = sample['target']
    
    print(f"PPG (Input)  - Shape: {ppg.shape}, Range: [{ppg.min():.4f}, {ppg.max():.4f}]")
    print(f"ECG (Target) - Shape: {target.shape}, Range: [{target.min():.4f}, {target.max():.4f}]")
    print("=" * 50)

def run_approach2_pipeline(subject_ids, base_path, configs, show_random_sample=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizio Pipeline su dispositivo: {device}")
    
    # 1. Caricamento e Preprocessing
    loader = DaliaDataLoader()
    raw_data = loader.load_subjects(subject_ids, base_path, is_resampled=False)
    preprocessor = DaliaPreprocessor()
    data_split = preprocessor.split_data(raw_data, configs['split']['train'], configs['split']['val'], configs['split']['test'])
    normalized_data = preprocessor.compute_and_apply_normalization(data_split)

    # Segmentazione in finestre (es. 7 secondi)
    windowed_data = preprocessor.create_windows(
        normalized_data, 
        is_resampled=False,
        window_size=configs['window_size'],
        window_shift=configs.get('window_shift', 2),
        include_prev_ecg=False 
    )
    
    diagnose_data(windowed_data['train'], "train")
    
    # 2. Dataloaders Semplificati (Input PPG -> Target ECG)
    def prepare_loader(data_list, batch_size, shuffle=True):
        ppg = torch.stack([torch.tensor(d['input'][0]).float().unsqueeze(0) for d in data_list])
        target = torch.stack([torch.tensor(d['target']).float().unsqueeze(0) for d in data_list])
        
        print(f"DataLoader creato - Input: {ppg.shape}, Target: {target.shape}")
        return DataLoader(TensorDataset(ppg, target), batch_size=batch_size, shuffle=shuffle)

    train_loader = prepare_loader(windowed_data['train'], configs['batch_size'], shuffle=True)
    val_loader = prepare_loader(windowed_data['val'], configs['batch_size'], shuffle=False)

    # 3. Inizializzazione Modello HA-CNN-BILSTM
    # Assicurati che configs['model_type'] sia 'ha_cnn_bilstm' o 'unet_only_ppg'
    model_factory = Approach2Model(target_len=configs['target_len'])
    model = model_factory.get_model(model_type=configs.get('model_type', 'ha_cnn_bilstm')).to(device)

    # Test forward pass rapido per validare le dimensioni (448 -> 1792)
    with torch.no_grad():
        test_ppg, _ = next(iter(train_loader))
        test_output = model(test_ppg.to(device))
        print(f"Forward Pass Validato - Output: {test_output.shape}")
    
    trainer = Approach2Trainer(model, device, configs)

    # 4. Training Loop
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "approach2_ppg_only"))
    best_plots_dir = os.path.join(save_dir, "best_model_plots")
    os.makedirs(best_plots_dir, exist_ok=True)
    
    history = {'train': [], 'val': []}
    
    print(f"\nAddestramento con ottimizzatore: {configs.get('optimizer_type', 'adamw').upper()}")
    
    for epoch in range(configs['epochs']):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.validate_epoch(val_loader)
        
        history['train'].append(train_m)
        history['val'].append(val_m)
        
        # Inseriamo il log dell'RMSE per monitorare la precisione morfologica [cite: 28, 529]
        print(f"Epoch {epoch+1:03d} | "
              f"T-Loss: {train_m['loss']:.4f} [RMSE: {train_m['rmse']:.4f}] | "
              f"V-Loss: {val_m['loss']:.4f} [RMSE: {val_m['rmse']:.4f}] | "
              f"Corr: {val_m['corr']:.3f}")
        
        # Logica del miglior modello basata sulla Val Loss
        is_best = val_m['loss'] < trainer.best_val_loss
        
        if is_best:
            trainer.best_val_loss = val_m['loss']
            trainer.patience_counter = 0
            
            # Salvataggio del miglior modello e grafico di inferenza
            trainer.save_checkpoint(epoch, save_dir, is_best=True)
            
            plot_path = os.path.join(best_plots_dir, f"best_inference_epoch_{epoch+1:03d}.png")
            plot_ppg_ecg_comparison(
                model=model, 
                val_data=windowed_data['val'], 
                device=device, 
                save_path=plot_path,
                epoch=epoch+1
            )
            print(f"--- [BEST] Miglioramento rilevato! Plot salvato in epoch {epoch+1} ---")
            
        else:
            trainer.patience_counter += 1
            trainer.save_checkpoint(epoch, save_dir, is_best=False)

        # Early stopping basato sulla pazienza definita in configs
        if trainer.patience_counter >= trainer.patience:
            print(f"\n--- [STOP] Early Stopping attivato a epoca {epoch+1} ---")
            break
    
    # 5. Salvataggio finale statistiche
    save_training_plots(history, save_dir)
    save_history_to_json(history, save_dir)
    
    return trainer, windowed_data['test']