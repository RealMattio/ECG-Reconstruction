from src.generation.models.unet1d import EncoderDecoderUNet

class Approach2Model:
    """
    Gestisce l'ensemble basato su Encoder-Decoder per l'Approccio 2.
    """
    def __init__(self, target_len=2048):
        self.target_len = target_len

    def get_model(self):
        # Restituisce il modello unico che integra la fusione interna
        return EncoderDecoderUNet(target_len=self.target_len)