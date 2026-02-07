import torch.nn as nn
from src.bidmc_generation.models.ha_cnn_bilstm_autoregressive import HACNNBiLSTM_AR
from src.bidmc_generation.models.dual_branch_hybrid import DualBranchHybrid
from src.bidmc_generation.models.lightweight_hybrid import LightweightHybrid
from src.bidmc_generation.models.bio_transformer import BioTransformer

class ModelFactory:
    """
    Factory class per istanziare i modelli di generazione ECG in base alla configurazione.
    """
    
    @staticmethod
    def get_model(configs):
        # Normalizziamo la stringa in minuscolo per evitare errori di case sensitivity
        model_type = configs.get('model_type', '').lower().strip()
        
        # Recuperiamo la lunghezza del target (default 125 se non specificata)
        target_len = configs.get('target_len', 125)
        
        print(f"[FACTORY] Richiesto modello: '{model_type}'")

        if model_type == 'ha_cnn_bilstm_ar':
            # Il modello Autoregressivo classico (CNN 1D + BiLSTM)
            return HACNNBiLSTM_AR(configs, seq_len=target_len)
        
        elif model_type == 'dual_branch_hybrid':
            # Il nuovo modello ibrido (CNN 2D + Dual BiLSTM)
            return DualBranchHybrid(configs, seq_len=target_len)
        elif model_type == 'lightweight_hybrid':
            # Modello ibrido leggero (CNN 1D semplificata + singola LSTM)
            return LightweightHybrid(configs, seq_len=target_len)
        
        elif model_type == 'bio_transformer':
            # Modello basato su Transformer (BioTransformer)
            return BioTransformer(configs, seq_len=target_len)
        else:
            raise ValueError(f"[ERROR] ModelFactory: Tipo di modello '{model_type}' non riconosciuto. "
                             f"Opzioni valide: ['ha_cnn_bilstm_ar', 'dual_branch_hybrid']")