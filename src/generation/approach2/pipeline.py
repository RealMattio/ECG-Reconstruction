import os
from src.generation.approach1 import trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader.data_loader import DaliaDataLoader
from src.preprocessing.preprocessor import DaliaPreprocessor
from src.generation.approach2.approach2_model import Approach2Model
from src.generation.approach2.trainer import Approach2Trainer
from src.evaluation.visualization import save_training_plots, save_history_to_json, plot_random_sample, plot_inference_comparison

def run_approach2_pipeline(subject_ids, base_path, configs, show_random_sample=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Caricamento e Preprocessing (come Approccio 1)
    loader = DaliaDataLoader()
    raw_data = loader.load_subjects(subject_ids, base_path, is_resampled=False)
    preprocessor = DaliaPreprocessor()
    data_split = preprocessor.split_data(raw_data, configs['split']['train'], configs['split']['val'], configs['split']['test'])
    normalized_data = preprocessor.compute_and_apply_normalization(data_split)
    windowed_data = preprocessor.create_windows(
        normalized_data, 
        is_resampled=False,  # <--- Aggiungi questo parametro
        window_size=configs['window_size'], 
        window_shift=configs['window_shift']
    )
    if show_random_sample:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Risaliamo fino a src/ e scendiamo in evaluation/
        eval_path = os.path.abspath(os.path.join(current_dir, "..", "..", "evaluation", "random_train_sample.png"))
        
        plot_random_sample(
            windowed_data['train'], 
            save_path=eval_path, 
            window_size=configs.get('window_size', 8)
        )
    
    # 2. Dataloaders
    def prepare_loader(data_list, batch_size):
        ppg = torch.stack([torch.tensor(d['input'][0]).float().unsqueeze(0) for d in data_list])
        eda = torch.stack([torch.tensor(d['input'][1]).float().unsqueeze(0) for d in data_list])
        acc = torch.stack([torch.tensor(d['input'][2]).float().transpose(0, 1) for d in data_list])
        target = torch.stack([torch.tensor(d['target']).float().unsqueeze(0) for d in data_list])
        return DataLoader(TensorDataset(ppg, eda, acc, target), batch_size=batch_size, shuffle=True)

    train_loader = prepare_loader(windowed_data['train'], configs['batch_size'])
    val_loader = prepare_loader(windowed_data['val'], configs['batch_size'])

    # 3. Inizializzazione Modello e Trainer
    model_factory = Approach2Model(target_len=configs['target_len'])
    model = model_factory.get_model()
    trainer = Approach2Trainer(model, device, configs)

    # 4. Training Loop
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "approach2"))
    history = {'train': [], 'val': []}
    
    for epoch in range(configs['epochs']):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.validate_epoch(val_loader)
        
        history['train'].append(train_m); history['val'].append(val_m)
        # In src/generation/approach2/pipeline.py riga 61 (circa)
        print(f"Epoch {epoch+1}/{configs['epochs']} | "
            f"Train Loss: {train_m['loss']:.4f} [RMSE: {train_m['rmse']:.4f}] | "
            f"Val Loss: {val_m['loss']:.4f} [RMSE: {val_m['rmse']:.4f}] | "
            f"Best Val: {trainer.best_val_loss:.4f}")
        is_best = val_m['total'] < trainer.best_val_loss
        
        if is_best: trainer.best_val_loss = val_m['total']; trainer.patience_counter = 0
        else: trainer.patience_counter += 1
        trainer.save_checkpoint(epoch, save_dir, is_best=is_best)
        if trainer.patience_counter >= trainer.patience: break
    
    # --- 5. INFERENZA FINALE DI TEST ---
    print("\nEsecuzione inferenza di prova su validation set...")
    
    # Carichiamo i pesi del miglior modello appena salvato
    best_model_path = os.path.join(save_dir, "best_model", "model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Definiamo il percorso del grafico di confronto
    inference_plot_path = os.path.abspath(os.path.join(save_dir, "inference_comparison.png"))
    
    # Eseguiamo l'inferenza e salviamo il grafico
    plot_inference_comparison(
        model=model, 
        val_data=windowed_data['val'], 
        device=device, 
        save_path=inference_plot_path
    )

    save_training_plots(history, save_dir)
    save_history_to_json(history, save_dir)
    return trainer, windowed_data['test']