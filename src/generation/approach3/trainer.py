import torch
import torch.nn as nn
import torch.optim as optim
import random

class WGANTrainer:
    def __init__(self, gen, disc, device, configs):
        self.gen = gen.to(device)
        self.disc = disc.to(device)
        self.device = device
        self.lambda_gp = configs['loss_weights'].get('lambda_gp', 10.0) # [cite: 213]
        self.lambda_extrema = configs['loss_weights'].get('lambda_extrema', 80.0) # [cite: 238]
        
        # Optimizer Adam con parametri da paper P2E-WGAN [cite: 213]
        self.opt_G = optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def compute_gradient_penalty(self, real, fake, cond):
        """GP per garantire il vincolo di 1-Lipschitz del discriminatore[cite: 145, 168]."""
        alpha = torch.rand(real.size(0), 1, 1).to(self.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolates = self.disc(interpolates, cond)
        fake_grad = torch.ones(d_interpolates.shape).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=fake_grad, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def train_step(self, ppg, acc, eda, ecg_target, current_epoch, transition_steps=100):
        """Gestisce lo Scheduled Skipping dell'ECG in ingresso."""
        # La condizione per il discriminatore sono sempre i sensori (5 canali)
        condition = torch.cat([ppg, acc, eda], dim=1) 
        
        # LOGICA PROBABILITÀ DINAMICA
        if current_epoch < transition_steps:
            # Probabilità di skip che aumenta linearmente da 0 a 1 nelle prime 100 epoche
            skip_prob = current_epoch / transition_steps
        else:
            # Dopo le 100 epoche, l'ECG è sempre rimosso
            skip_prob = 1.0
            
        # Decisione: passiamo l'ECG reale o un segnale nullo?
        if random.random() < skip_prob:
            ecg_input = torch.zeros_like(ecg_target)
        else:
            ecg_input = ecg_target
            
        # Il generatore riceve sempre 6 canali (PPG, 3xACC, EDA, ECG/Zeri)
        gen_input = torch.cat([condition, ecg_input], dim=1)
        gen_input_lstm = gen_input.transpose(1, 2) # (B, Seq, Ch) per BiLSTM

        # --- UPDATE DISCRIMINATOR (WGAN-GP) [cite: 143, 147] ---
        self.opt_D.zero_grad()
        fake_ecg = self.gen(gen_input_lstm)
        real_val = self.disc(ecg_target, condition)
        fake_val = self.disc(fake_ecg.detach(), condition)
        gp = self.compute_gradient_penalty(ecg_target, fake_ecg.detach(), condition)
        
        d_loss = -torch.mean(real_val) + torch.mean(fake_val) + gp
        d_loss.backward()
        self.opt_D.step()

        # --- UPDATE GENERATOR (Adv + Extrema Loss) [cite: 154] ---
        self.opt_G.zero_grad()
        fake_val_g = self.disc(fake_ecg, condition)
        # Loss morfologica (MSE) per preservare i picchi PQRST [cite: 152, 172]
        g_ext_loss = nn.MSELoss()(fake_ecg, ecg_target)
        
        g_loss = -torch.mean(fake_val_g) + (self.lambda_extrema * g_ext_loss)
        g_loss.backward()
        self.opt_G.step()

        return g_loss.item(), d_loss.item(), skip_prob