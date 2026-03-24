import torch.nn as nn
from src.mimic_generation_PINN.models.ha_cnn_bilstm_autoregressive import HACNNBiLSTM_AR
from src.mimic_generation_PINN.models.dual_branch_hybrid import DualBranchHybrid
from src.mimic_generation_PINN.models.lightweight_hybrid import LightweightHybrid
from src.mimic_generation_PINN.models.bio_transformer import BioTransformer
from src.mimic_generation_PINN.models.ppg_wavenet import PPGWaveNet
from src.mimic_generation_PINN.models.ecg_unet1d import ECGUNet1D

# Mappa completa dei modelli disponibili
_MODEL_REGISTRY = {
    'ha_cnn_bilstm_ar':   HACNNBiLSTM_AR,    # CNN 1D + BiLSTM (baseline da paper)
    'dual_branch_hybrid': DualBranchHybrid,   # CNN 2D + Dual BiLSTM
    'lightweight_hybrid': LightweightHybrid,  # CNN 1D leggera + BiLSTM (miglior baseline)
    'bio_transformer':    BioTransformer,     # PhysioFormer: Transformer multi-scala
    'ppg_wavenet':        PPGWaveNet,         # WaveNet-style dilated TCN con gated activations
    'ecg_unet1d':         ECGUNet1D,          # U-Net 1D encoder-decoder con skip connections
}


class ModelFactory:
    """
    Factory class per istanziare i modelli di generazione ECG in base alla configurazione.

    Modelli disponibili (configs['model_type']):
        'ha_cnn_bilstm_ar'   – CNN 1D + BiLSTM autoregressivo (baseline da paper)
        'dual_branch_hybrid' – CNN 2D + Dual BiLSTM con attention gate
        'lightweight_hybrid' – CNN 1D leggera + BiLSTM singola
        'bio_transformer'    – PhysioFormer: Transformer multi-scala con residual PPG bypass
        'ppg_wavenet'        – WaveNet-style dilated TCN con gated activations (pure-CNN)
        'ecg_unet1d'         – U-Net 1D encoder-decoder con skip connections e bottleneck BiLSTM
    """

    @staticmethod
    def get_model(configs):
        model_type = configs.get('model_type', '').lower().strip()
        target_len = configs.get('target_len', 125)

        print(f"[FACTORY] Richiesto modello: '{model_type}'")

        if model_type not in _MODEL_REGISTRY:
            valid = list(_MODEL_REGISTRY.keys())
            raise ValueError(
                f"[ERROR] ModelFactory: tipo '{model_type}' non riconosciuto. "
                f"Opzioni valide: {valid}"
            )

        return _MODEL_REGISTRY[model_type](configs, seq_len=target_len)