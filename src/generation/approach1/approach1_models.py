import torch
import torch.nn as nn
from src.generation.models.resnet1d import ResNet1D

class Approach1LateFusion:
    """
    Definisce i modelli per l'Approccio 1: Late Fusion Ensemble.
    Ogni segnale ha un proprio ramo ResNet-1D dedicato.
    """
    def __init__(self, target_len=2048):
        """
        Args:
            target_len (int): Lunghezza dell'ECG generato (es. 2048 campioni per 8s a 256Hz).
        """
        self.target_len = target_len

    def get_ppg_model(self):
        """
        Ramo PPG: Analizza la forma d'onda pulsatoria (64 Hz).
        Input: 1 canale (PPG).
        """
        model = ResNet1D(
            in_channels=1, 
            base_filters=64, 
            kernel_size=16, 
            stride=2, 
            n_block=8, 
            n_classes=self.target_len  # Output diretto come sequenza ECG
        )
        return model

    def get_acc_model(self):
        """
        Ramo ACC: Identifica artefatti da movimento (32 Hz).
        Input: 3 canali (Assi X, Y, Z).
        """
        model = ResNet1D(
            in_channels=3, 
            base_filters=32, 
            kernel_size=8, 
            stride=2, 
            n_block=4, 
            n_classes=self.target_len
        )
        return model

    def get_eda_model(self):
        """
        Ramo EDA: Cattura lo stato di attivazione simpatica (4 Hz).
        Input: 1 canale (EDA).
        """
        model = ResNet1D(
            in_channels=1, 
            base_filters=16, 
            kernel_size=4, 
            stride=1, 
            n_block=3, 
            n_classes=self.target_len
        )
        return model

    def get_meta_learner(self):
        """
        Aggregatore: Combina i tre ECG candidati per produrre il segnale finale.
        Prende in input 3 sequenze ECG e restituisce 1 sequenza ECG ottimizzata.
        """
        meta_learner = nn.Sequential(
            nn.Linear(self.target_len * 3, self.target_len * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.target_len * 2, self.target_len)
        )
        return meta_learner

# Nota: ResNet1D può essere importata da librerie specializzate come 'fastai' o 
# implementata come modulo helper basato su blocchi residui 1D standard.