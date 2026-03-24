import os
import sys
import torch
import json
from torch.utils.data import DataLoader

# Aggiungi la root al path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



from src.mimic_generation_PINN.model_factory import ModelFactory
from src.preprocessing.mimic_autoregressive_preprocessor import MimicAutoregressivePreprocessor
from src.evaluation.evaluation import evaluate_test_set_performance
from src.evaluation.visualization import plot_validation_snapshot, plot_autoregressive_epoch
from src.mimic_generation_PINN.pipeline import MimicSmartDataset

def main():
    # --- PATHS (Basati sul tuo log di errore) ---
    RUN_DIR = os.path.join(PROJECT_ROOT, '../', "src", "experiments", "final_mimic_pinn_results", "MAE_loss", "lightweight_hybrid_20260323_154244")
    FINAL_MODEL_DIR = os.path.join(RUN_DIR, "final_full_model")
    MODEL_WEIGHTS = os.path.join(FINAL_MODEL_DIR, "best_lightweight_hybrid.pth")

    preprocessed_data_path = os.path.join(PROJECT_ROOT, '../', 'mimic3wdb-matched_healthy_data') # Controlla che questo path sia corretto
    split_file_path = os.path.join(preprocessed_data_path, 'dataset_split.json')
    manifest_path = os.path.join(preprocessed_data_path, 'dataset_manifest.json')

    # --- CONFIGURAZIONI MINIME ---
    configs = {
        'model_type': 'lightweight_hybrid',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'target_fs': 125,
        'x_sec': 7,
        'gen_sec': 1,
        'stride_sec': 1,
        'apply_preprocessing': False,
        'preprocessed_data': preprocessed_data_path,
        'batch_size': 256,
        'normalize_01': False,
        'apply_wst': True,  # <--- AGGIUNGI QUESTA RIGA!
    }

    print("-> Caricamento dello Split e del Manifest...")
    with open(split_file_path, 'r') as f:
        saved_split = json.load(f)
    test_patients = saved_split["test_patients"]

    with open(manifest_path, 'r') as f:
        full_manifest = json.load(f)

    test_subset = [m for m in full_manifest if m['subject_id'] in test_patients]

    print(f"-> Caricamento del Test Set in RAM ({len(test_patients)} pazienti)...")
    test_ds = MimicSmartDataset(test_subset, configs['preprocessed_data'], configs)
    test_loader = DataLoader(test_ds, batch_size=configs['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Configurazione input dinamica come nella pipeline
    sample_x, sample_y, _, _, _ = test_ds[0] 
    configs['input_channels'] = sample_x.shape[0] 
    configs['actual_seq_len'] = sample_x.shape[-1]
    configs['target_len'] = sample_y.shape[-1]

    print("-> Inizializzazione Modello e caricamento Pesi...")
    model = ModelFactory.get_model(configs).to(configs['device'])
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=configs['device']))
    model.eval()

    print("\n-> Calcolo metriche sul Test Set...")
    metrics = evaluate_test_set_performance(model, test_loader, configs['device'], FINAL_MODEL_DIR, configs)

    print("\n-> Generazione delle 6 Immagini Finali...")
    preprocessor = MimicAutoregressivePreprocessor(fs=125, window_sec=7, gen_sec=1)
    
    for i in range(1, 4):
        plot_validation_snapshot(model, test_loader, configs['device'], FINAL_MODEL_DIR, epoch="TEST", step=i, prefix='test_inference')
        plot_autoregressive_epoch(model, test_ds, preprocessor, configs['device'], configs, epoch=f"TEST_{i}", save_dir=FINAL_MODEL_DIR)

    print("\n-> Salvataggio Report Finale...")
    final_report = {
        "model_type": configs['model_type'],
        "total_epochs_trained": 61, # Come da tuo log
        "test_set_metrics": metrics,
        "test_patients_count": len(test_patients)
    }
    report_path = os.path.join(FINAL_MODEL_DIR, "final_model_report.json")
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=4)

    print(f"\n✅ RECUPERO COMPLETATO! Risultati e grafici sono in: {FINAL_MODEL_DIR}")

if __name__ == "__main__":
    main()