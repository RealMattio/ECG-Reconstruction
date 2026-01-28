from src.generation.models.bilstm import BiLSTMGenerator, PatchGANDiscriminator

class Approach3Model:
    def __init__(self, pretrain_mode=True):
        # Il Generatore cambia input tra le fasi
        # Pretrain: PPG(1) + ACC(3) + EDA(1) + ECG(1) = 6
        # Train: PPG(1) + ACC(3) + EDA(1) = 5
        self.gen_in_channels = 6 if pretrain_mode else 5
        
        # Il Discriminatore riceve sempre (ECG) + (PPG + ACC + EDA)
        # Quindi 1 + 5 = 6 canali totali in ingresso [cite: 193]
        self.cond_channels = 5 
        
        self.generator = BiLSTMGenerator(input_channels=self.gen_in_channels)
        self.discriminator = PatchGANDiscriminator(cond_channels=self.cond_channels)

    def get_models(self):
        return self.generator, self.discriminator