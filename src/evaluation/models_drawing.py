import torch
from torchview import draw_graph
import os
import sys
# 1. IMPORTA LA TUA CLASSE
# Visto che hai i file in src/mimic_generation/models/bio_transformer.py,
# adatta questo import con il nome esatto della tua classe principale
# 1. GESTIONE PERCORSI
# 1. GESTIONE PERCORSI (Sali di un livello per arrivare alla root del progetto)
cartella_corrente = os.path.dirname(os.path.abspath(__file__)) # Questo punta a /.../ecg_generation/tests
root_progetto = os.path.dirname(cartella_corrente)             # Questo punta a /.../ecg_generation

if root_progetto not in sys.path:
    sys.path.insert(0, root_progetto)

# 2. IMPORTA LA TUA CLASSE (percorso corretto in base al tuo albero)
from src.mimic_generation_PINN.model_factory import ModelFactory

# --- CONFIGURAZIONE ---
percorso_file_pt = "/home/mmerone/mattia/ecg_generation/src/experiments/mimic_pinn_results/bio_transformer_20260223_014515/fold_1/best_bio_transformer.pth"

# Sostituisci con le dimensioni reali di un singolo input del tuo dataset!
# Esempio: (Batch Size=1, Canali/Sensori=12, Lunghezza_Sequenza=5000)
forma_input_dati = (1, 3, 219) 
# ----------------------

configs = {
        # --- PARAMETRI SEGNALE ---
        'target_fs': 125,               
        'x_sec': 7,                     
        'model_type': 'bio_transformer', # Cambia al modello che stai testando
        'input_channels': 3,              # Cambia in base al numero di canali del tuo dataset
        # --- PREPROCESSING ON-THE-FLY ---
        'normalize_01': False,           # Fondamentale che sia False per la PINN
        'apply_wst': True,              
        
        # --- DATA AUGMENTATION (Anti-Overfitting per run lunghe) ---
        'apply_augmentation': True,     
        'aug_random_gain': True,        
        'aug_additive_noise': True,     
        'aug_context_noise': 0.02,      
        'apply_context_augmentation': True, 
    }

print("1. Costruzione dell'architettura vuota...")
# Inserisci le parentesi i parametri che la tua classe richiede (se ne richiede)
modello = ModelFactory.get_model(configs)

print("2. Caricamento dei pesi addestrati...")
# map_location='cpu' è fondamentale se sposti il modello da una GPU al tuo PC portatile!
pesi_salvati = torch.load(percorso_file_pt, map_location=torch.device('cpu'), weights_only=True)

# Inseriamo i pesi nello scheletro
modello.load_state_dict(pesi_salvati)

# Mettiamo il modello in modalità "lettura" (spegne Dropout e BatchNorm)
modello.eval()

print("3. Disegno dell'architettura in corso...")
modello_grafo = draw_graph(
    modello, 
    input_size=forma_input_dati, 
    graph_name=configs['model_type'],
    filename=f"Architettura_{configs['model_type']}.gv",
    save_graph=True,
    roll=True, # Se il grafico è troppo largo, metti a True per arrotolarlo verticalmente
    expand_nested=True, # Metti a False se il grafico diventa troppo gigantesco,
    directory=os.path.join(root_progetto, 'evaluation', 'model_graphs'), # Salva in src/evaluation/model_graphs
    show_shapes=False, # Mostra le forme dei tensori nei nodi
    graph_dir = 'LR' # Usa il layout "TB" (Top-Bottom) "LR" (Left-Right) o "RL" (Right-Left) per una migliore leggibilità
)

print("Fatto! Immagine salvata come 'Architettura_Modello_Reale.gv.png'")