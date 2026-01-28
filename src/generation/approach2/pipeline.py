import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader.data_loader import DaliaDataLoader
from src.preprocessing.preprocessor import DaliaPreprocessor
from src.generation.approach2.approach2_model import Approach2Model
from src.generation.approach2.trainer import Approach2Trainer
from src.evaluation.visualization import save_training_plots, save_history_to_json, plot_random_sample, plot_inference_comparison

def diagnose_data(windowed_data, name="train"):
    print(f"\n=== DIAGNOSTICA DATI ({name.upper()}) ===")
    sample = windowed_data[0]
    
    # AGGIORNAMENTO: Supporto per 4 input
    ppg, eda, acc = sample['input'][:3]
    prev_ecg = sample['input'][3] if len(sample['input']) > 3 else None
    target = sample['target']
    
    print(f"PPG      - Shape: {ppg.shape}, Range: [{ppg.min():.4f}, {ppg.max():.4f}]")
    print(f"EDA      - Shape: {eda.shape}, Range: [{eda.min():.4f}, {eda.max():.4f}]")
    print(f"ACC      - Shape: {acc.shape}, Range: [{acc.min():.4f}, {acc.max():.4f}]")
    if prev_ecg is not None:
        print(f"Prev ECG - Shape: {prev_ecg.shape}, Range: [{prev_ecg.min():.4f}, {prev_ecg.max():.4f}]")
    print(f"Target   - Shape: {target.shape}, Range: [{target.min():.4f}, {target.max():.4f}]")
    
    if target.std() < 0.01:
        print("⚠️  WARNING: Target ha varianza molto bassa!")
    print("=" * 50)

def run_approach2_pipeline(subject_ids, base_path, configs, show_random_sample=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Caricamento e Preprocessing
    loader = DaliaDataLoader()
    raw_data = loader.load_subjects(subject_ids, base_path, is_resampled=False)
    preprocessor = DaliaPreprocessor()
    data_split = preprocessor.split_data(raw_data, configs['split']['train'], configs['split']['val'], configs['split']['test'])
    normalized_data = preprocessor.compute_and_apply_normalization(data_split)
    
    # AGGIORNAMENTO: Abilitazione ECG precedente e window_shift
    windowed_data = preprocessor.create_windows(
        normalized_data, 
        is_resampled=False,
        window_size=configs['window_size'],
        window_shift=configs.get('window_shift', 2),
        include_prev_ecg=True # <--- Attivazione modifica
    )
    
    diagnose_data(windowed_data['train'], "train")
    diagnose_data(windowed_data['val'], "val")
    
    if show_random_sample:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eval_path = os.path.abspath(os.path.join(current_dir, "..", "..", "evaluation", "random_train_sample.png"))
        plot_random_sample(windowed_data['train'], save_path=eval_path, window_size=configs.get('window_size', 8))
    
    # 2. Dataloaders
    def prepare_loader(data_list, batch_size, shuffle=True):
        # AGGIORNAMENTO: Estrazione di tutti e 4 gli input
        ppg = torch.stack([torch.tensor(d['input'][0]).float().unsqueeze(0) for d in data_list])
        eda = torch.stack([torch.tensor(d['input'][1]).float().unsqueeze(0) for d in data_list])
        acc = torch.stack([torch.tensor(d['input'][2]).float().transpose(0, 1) for d in data_list])
        prev_ecg = torch.stack([torch.tensor(d['input'][3]).float().unsqueeze(0) for d in data_list]) # <--- Nuovo
        target = torch.stack([torch.tensor(d['target']).float().unsqueeze(0) for d in data_list])
        
        print(f"\nDataLoader shapes - PPG: {ppg.shape}, Prev ECG: {prev_ecg.shape}, Target: {target.shape}")
        
        return DataLoader(TensorDataset(ppg, eda, acc, prev_ecg, target), batch_size=batch_size, shuffle=shuffle)

    train_loader = prepare_loader(windowed_data['train'], configs['batch_size'], shuffle=True)
    val_loader = prepare_loader(windowed_data['val'], configs['batch_size'], shuffle=False)

    # 3. Inizializzazione Modello
    model_factory = Approach2Model(target_len=configs['target_len'])
    model_type = configs.get('model_type', 'unet')
    model = model_factory.get_model(model_type=model_type).to(device)

    # Test forward pass
    print("\n=== TEST FORWARD PASS ===")
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_ppg, test_eda, test_acc, test_prev, test_target = [b.to(device) for b in test_batch]
        test_output = model(test_ppg, test_acc, test_eda, test_prev)
        print(f"Output shape: {test_output.shape}")
    print("=" * 50)
    
    trainer = Approach2Trainer(model, device, configs)

    # 4. Training Loop
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "approach2"))
    os.makedirs(save_dir, exist_ok=True)
    history = {'train': [], 'val': []}
    
    for epoch in range(configs['epochs']):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.validate_epoch(val_loader)
        
        history['train'].append(train_m)
        history['val'].append(val_m)
        
        print(f"Epoch {epoch+1:03d}/{configs['epochs']} | "
              f"Train Loss: {train_m['loss']:.4f} [Corr: {train_m['corr']:.3f}] | "
              f"Val Loss: {val_m['loss']:.4f} [Corr: {val_m['corr']:.3f}] | "
              f"Best: {trainer.best_val_loss:.4f} | LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        is_best = val_m['total'] < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = val_m['total']
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
        
        trainer.save_checkpoint(epoch, save_dir, is_best=is_best)
        if trainer.patience_counter >= trainer.patience: break
    
    # 5. Inferenza finale
    best_model_path = os.path.join(save_dir, "best_model", "model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    plot_inference_comparison(
        model=model, 
        val_data=windowed_data['val'], 
        device=device, 
        save_path=os.path.abspath(os.path.join(save_dir, "inference_comparison.png"))
    )

    save_training_plots(history, save_dir)
    save_history_to_json(history, save_dir)
    return trainer, windowed_data['test']