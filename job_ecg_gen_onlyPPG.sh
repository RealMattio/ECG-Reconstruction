#!/bin/bash
#SBATCH --job-name=PPG_ecg_gen_mattia
#SBATCH --partition=gn02_a40-high      # Cambiato su A40
#SBATCH --gres=gpu:1                   # 1 GPU NVIDIA A40 (48GB VRAM)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Alzato a 8: ottimo per il data loading senza eccedere
#SBATCH --mem=32G                      # Alzato a 32GB: i segnali ECG possono essere pesanti in RAM
#SBATCH --time=10:00:00                # Massimo consentito per questa partizione

# --- GESTIONE OUTPUT ---
# Usiamo percorsi assoluti per sicurezza
#SBATCH --output=/home/mmerone/mattia/ecg_generation/logs/output_%j.txt
#SBATCH --error=/home/mmerone/mattia/ecg_generation/logs/error_%j.err

# 1. Risolviamo il problema dei moduli (PYTHONPATH)
export PYTHONPATH=$PYTHONPATH:/home/mmerone/mattia/ecg_generation

# 2. Entriamo nella cartella
cd /home/mmerone/mattia/ecg_generation/

# 3. Esecuzione con percorso assoluto (evita errori Conda)
/home/mmerone/.conda/envs/ecg_gen_env/bin/python -u src/main_onlyPPG_BIDMC.py