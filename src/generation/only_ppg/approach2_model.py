from src.generation.models.unet1d import EncoderDecoderUNet
from src.generation.models.hybrid_cnn_gru import HybridCNNGRUModel
from src.generation.models.transformer_model import LightTransformerECG, TransformerECGGenerator
from src.generation.models.enhanced_unet1d import EnhancedUNet1D
from src.generation.models.unet1d_onlyPPG import PPGtoECG_UNet
from src.generation.models.ha_cnn_bilstm import HA_CNN_BiLSTM, HA_CNN_BiLSTM_Fourier

class Approach2Model:
    """
    Gestisce l'ensemble basato su Encoder-Decoder per l'Approccio 2.
    """
    def __init__(self, target_len=2048):
        self.target_len = target_len

    def get_model(self, model_type='unet'):
        # Restituisce il modello unico che integra la fusione interna
        if model_type == 'unet':
            print("Using UNet Model")
            return EncoderDecoderUNet(target_len=self.target_len)
        elif model_type == 'hybrid_cnn_gru':
            print("Using Hybrid CNN-GRU Model")
            return HybridCNNGRUModel(target_len=self.target_len)
        elif model_type == 'light_transformer':
            print("Using Light Transformer Model")
            return LightTransformerECG(target_len=self.target_len)
        elif model_type == 'transformer':
            print("Using Transformer ECG Generator Model")
            return TransformerECGGenerator(target_len=self.target_len)
        elif model_type == 'enhanced_unet':
            print("Using Enhanced UNet Model")
            return EnhancedUNet1D(target_len=self.target_len)
        elif model_type == 'unet_only_ppg':
            print("Using UNet Only PPG Model")
            return PPGtoECG_UNet(target_len=self.target_len)
        elif model_type == 'ha_cnn_bilstm':
            print("Using HA-CNN-BiLSTM Model")
            return HA_CNN_BiLSTM(target_len=self.target_len)
        elif model_type == 'ha_cnn_bilstm_fourier':
            print("Using HA-CNN-BiLSTM with Fourier Features Model")
            return HA_CNN_BiLSTM_Fourier(target_len=self.target_len)
        else:
            raise ValueError("Model type not supported")