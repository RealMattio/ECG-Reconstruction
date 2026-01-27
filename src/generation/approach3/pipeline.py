import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import resample
from src.data_loader.data_loader import DaliaDataLoader
from .approach3_model import Approach3Model
from .trainer import WGANTrainer

def run_approach3_pipeline(base_path, subjects, configs):
    # DEFINIZIONE PROJECT_ROOT
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
    
    device = configs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    target_fs = 125 # Frequenza MIMIC-II/Paper [cite: 120]
    window_samples = 7 * target_fs # 875 campioni

    loader = DaliaDataLoader()
    all_data = loader.load_subjects(subjects, base_path, is_resampled=False)

    for test_sub in subjects:
        print(f"\nLOOCV: Testing on {test_sub}")
        train_subs = [s for s in subjects if s != test_sub]
        
        def get_windows(sub_ids):
            """Prepara finestre contigue di 7s scalate tra -1 e 1."""
            all_ppg, all_acc, all_eda, all_ecg = [], [], [], []
            for s_id in sub_ids:
                if s_id not in all_data['subjects_data']: continue
                d = all_data['subjects_data'][s_id]
                ppg = (d['PPG'] - d['PPG'].mean()) / (d['PPG'].std() + 1e-8)
                ecg = (d['ECG'] - d['ECG'].mean()) / (d['ECG'].std() + 1e-8)
                num_target = int(len(ppg) * (target_fs / d['fs_ppg']))
                p_res = resample(ppg, num_target)
                e_res = resample(ecg, num_target)
                a_res = np.array([resample(d['ACC'][:, i], num_target) for i in range(3)]).T
                ed_res = resample(d['EDA'], num_target)
                for i in range(0, len(p_res) - window_samples, window_samples):
                    all_ppg.append(p_res[i:i+window_samples])
                    all_acc.append(a_res[i:i+window_samples])
                    all_eda.append(ed_res[i:i+window_samples])
                    all_ecg.append(e_res[i:i+window_samples])
            return (torch.tensor(np.array(all_ppg)).float().unsqueeze(1), 
                    torch.tensor(np.array(all_acc)).float().transpose(1, 2), 
                    torch.tensor(np.array(all_eda)).float().unsqueeze(1), 
                    torch.tensor(np.array(all_ecg)).float().unsqueeze(1))

        train_loader = DataLoader(TensorDataset(*get_windows(train_subs)), batch_size=configs['batch_size'], shuffle=True)

        # UNICA FASE DI TRAINING (200 epoche)
        model_factory = Approach3Model(pretrain_mode=True) # Sempre 6 canali
        gen, disc = model_factory.get_models()
        trainer = WGANTrainer(gen, disc, device, configs)
        
        total_epochs = 200
        transition = 100

        for epoch in range(total_epochs):
            g_epoch, d_epoch = 0, 0
            current_skip = 0
            for p, a, ed, ec in train_loader:
                g_l, d_l, skip = trainer.train_step(
                    p.to(device), a.to(device), ed.to(device), ec.to(device), 
                    current_epoch=epoch, transition_steps=transition
                )
                g_epoch += g_l; d_epoch += d_l; current_skip = skip
            
            print(f"Epoch [{epoch+1}/{total_epochs}] - Skip Prob: {current_skip:.2f} | Loss G: {g_epoch/len(train_loader):.4f} | Loss D: {d_epoch/len(train_loader):.4f}")

        # --- PLOT E SALVATAGGIO ---
        save_dir = os.path.join(PROJECT_ROOT, 'src', 'generation', 'models', 'approach3')
        os.makedirs(save_dir, exist_ok=True)
        
        gen.eval()
        with torch.no_grad():
            p, a, ed, ec_real = next(iter(train_loader))
            # Test finale senza ECG (ultimo canale a zero)
            in_test = torch.cat([p, a, ed, torch.zeros_like(ec_real)], dim=1).to(device)
            ec_gen = gen(in_test.transpose(1, 2)).cpu().numpy()
            
            for i in range(2):
                fig, axes = plt.subplots(4, 1, figsize=(12, 12))
                axes[0].plot(p[i, 0]); axes[0].set_title("Input PPG")
                axes[1].plot(a[i].numpy().T); axes[1].set_title("Input ACC (3-axis)")
                axes[2].plot(ed[i, 0]); axes[2].set_title("Input EDA")
                axes[3].plot(ec_real[i, 0], alpha=0.4, label="Real ECG"); axes[3].plot(ec_gen[i, 0], color='red', label="Generated ECG")
                axes[3].legend(); axes[3].set_title("Inference Result (Sensors only)")
                plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"final_inference_{test_sub}_win_{i}.png")); plt.close()

        torch.save(gen.state_dict(), os.path.join(save_dir, f"model_unified_app3_{test_sub}.pth"))
    return True