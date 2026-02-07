from huggingface_hub import snapshot_download
import os

# Configurazione dei parametri
REPO_ID = "kyegorov/mcd_rppg"
DEST_FOLDER = "./mcd_rppg_data"

print(f"Avvio del download selettivo da: {REPO_ID}")

# Esecuzione del download filtrato
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    ignore_patterns=["video/*"],      # Esclude ricorsivamente tutto il contenuto di 'video'
    local_dir=DEST_FOLDER,           # Scarica i file direttamente in questa cartella
    local_dir_use_symlinks=False,    # Scarica i file reali invece di link simbolici alla cache
    token=None                       # Inserisci il tuo token HF se il dataset diventasse privato
)

print(f"\nDownload completato con successo!")
print(f"I file (ECG, PPG, Meta) sono disponibili in: {os.path.abspath(DEST_FOLDER)}")