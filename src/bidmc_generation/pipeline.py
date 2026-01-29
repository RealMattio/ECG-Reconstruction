import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# Importazione dei moduli precedentemente definiti
from src.data_loader.bidmc_data_loader import BidmcDataLoader
from src.preprocessing.bidmc_preprocessor import BidmcPreprocessor
from src.bidmc_generation.models.ha_cnn_bilstm import HACNNBiLSTM
from src.bidmc_generation.trainer import HACNNBiLSTMTrainer
from src.evaluation.visualization import save_inference_plot

def run_ha_cnn_bilstm_pipeline(base_path, configs):
    """
    Pipeline principale per la ricostruzione ECG da PPG (Approccio 4).
    Coordina caricamento, preprocessing e addestramento.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    # 1. CARICAMENTO DATI (BIDMC 01-53)
    # Il dataset comprende 53 pazienti medici[cite: 759].
    subject_ids = [str(i).zfill(2) for i in range(1, 54)]
    loader = BidmcDataLoader()
    raw_data = loader.load_subjects(subject_ids, base_path)

    # 2. PREPROCESSING MODULARE
    # Implementa filtraggio 0.5-8Hz e segmentazione a 120 campioni[cite: 766, 774].
    preprocessor = BidmcPreprocessor(fs=125, beat_len=120)
    
    all_ppg_beats = []
    all_ecg_beats = []

    print("Inizio preprocessing e segmentazione beat-by-beat...")
    for s_id, data in raw_data['subjects_data'].items():
        ppg_beats, ecg_beats = preprocessor.process_subject(data['PPG'], data['ECG'], configs)
        if len(ppg_beats) > 0:
            all_ppg_beats.append(ppg_beats)
            all_ecg_beats.append(ecg_beats)

    # Concatenazione di tutti i battiti estratti
    X = np.concatenate(all_ppg_beats, axis=0)
    y = np.concatenate(all_ecg_beats, axis=0)
    
    # Conversione in tensori PyTorch (Batch, Channels, Seq_Len)
    X_tensor = torch.tensor(X).float().unsqueeze(1)
    y_tensor = torch.tensor(y).float().unsqueeze(1)

    print(f"Totale battiti processati: {X_tensor.shape[0]}")

    # 3. SPLIT DATASET (90% Train, 10% Test)
    # Seguendo la divisione descritta nel paper[cite: 749].
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False)

    # 4. INIZIALIZZAZIONE MODELLO E TRAINER
    # Modello HA-CNN-BILSTM con DCNN, BiLSTM e Attention[cite: 473, 605].
    model = HACNNBiLSTM(input_dim=1, output_dim=1, seq_len=120)
    os.makedirs(configs['model_save_path'], exist_ok=True)
    print(f"Cartella di salvataggio verificata: {configs['model_save_path']}")
    trainer = HACNNBiLSTMTrainer(model, device, configs)

    # 5. ADDESTRAMENTO
    # Il paper suggerisce 50 epoche con RMSE come loss[cite: 753, 754].
    history = trainer.fit(
        train_loader, 
        test_loader, 
        epochs=configs['epochs'], 
        patience=configs['patience'] # Assicurati di passare questo!
    )
    # 6. SALVATAGGIO FINALE
    save_path = os.path.join(configs['model_save_path'], "ha_cnn_bilstm_final.pth")
    os.makedirs(configs['model_save_path'], exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print("Generazione grafico di inferenza finale...")
    # Definisci il percorso per il grafico
    plot_save_path = os.path.join(configs['model_save_path'], "final_validation_inference.png")
    
    # Chiama la funzione di visualizzazione
    # Nota: model è già l'istanza addestrata, test_loader è il validation set
    save_inference_plot(
        model=model,
        val_loader=test_loader, # Usiamo il test/validation loader
        device=device,
        save_path=plot_save_path,
        title="Final Model Inference on Random Validation Beat"
    )

    print(f"Pipeline completata. Modello salvato in: {save_path}")

    return history

if __name__ == "__main__":
    # Configurazione basata sui risultati ottimali del paper (SGDM optimizer).
    configurations = {
        'lr': 0.001,
        'batch_size': 20, # Minimo suggerito dal paper [cite: 754]
        'epochs': 50,
        'optimizer_type': 'SGDM', # SGDM ha ottenuto l'RMSE più basso (0.031) [cite: 477]
        'model_save_path': './models/approach4/'
    }

    # Percorso dei dati CSV
    data_folder = './bidmc_data'

    run_ha_cnn_bilstm_pipeline(data_folder, configurations)