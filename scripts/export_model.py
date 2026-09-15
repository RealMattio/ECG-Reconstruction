import torch
import torch.nn as nn
import os
import sys

from src.mimic_generation_PINN.model_factory import ModelFactory

# =========================================================
# CONFIGURAZIONI DI ESPORTAZIONE
# =========================================================
# Scegli il formato desiderato: "ONNX", "PT" (TorchScript), oppure "BOTH" (entrambi)
EXPORT_FORMAT = "BOTH"

# Percorso del modello addestrato (.pth) da cui leggere i pesi
MODEL_WEIGHTS_PATH = "src/experiments/final_mimic_pinn_results/MAE_loss/lightweight_hybrid_20260608_192241/final_full_model/best_lightweight_hybrid.pth"

# Percorsi di output
ONNX_OUTPUT_PATH = "ecg_generator_mobile.onnx"
PT_OUTPUT_PATH   = "ecg_generator_mobile.pt"
# =========================================================


class _ONNXBackbone(nn.Module):
    """
    Wrapper ONNX-compatibile per LightweightHybrid.

    kymatio Scattering1D usa view_as_complex internamente, operatore non
    supportato dall'esportatore ONNX. Questo wrapper riceve le feature WST
    già pre-calcolate (shape: [B, input_dim, T_wst]) ed esegue solo la parte
    CNN-encoder → LSTM → decoder, che è interamente ONNX-compatibile.

    Per PyTorch Mobile (.pt) il modello completo (WST inclusa) viene invece
    esportato via TorchScript, che supporta nativamente le operazioni complesse.

    Flusso di inferenza con il file .onnx su mobile:
        1. Pre-calcola le feature WST (es. con il modello .pt separato o
           un'implementazione nativa dell'app).
        2. Passa le feature a questo modello ONNX per ottenere l'ECG.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.cnn_encoder  = model.cnn_encoder
        self.lstm         = model.lstm
        self.upsample     = model.upsample
        self.decoder      = model.decoder
        self.normalize_01 = model.normalize_01
        if model.normalize_01:
            self.sigmoid = model.sigmoid

    def forward(self, wst_features: torch.Tensor) -> torch.Tensor:
        # wst_features: (Batch, input_dim, T_wst)
        features  = self.cnn_encoder(wst_features)
        lstm_in   = features.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out  = lstm_out.transpose(1, 2)
        upsampled = self.upsample(lstm_out)
        out       = self.decoder(upsampled)
        if self.normalize_01:
            out = self.sigmoid(out)
        return out


def export_model_for_mobile():
    print(f"Avvio esportazione modello. Formato richiesto: {EXPORT_FORMAT}")

    # 1. Stesse configurazioni usate per addestrare
    configs = {
        'model_type':     'lightweight_hybrid',
        'target_fs':      125,
        'x_sec':          7,
        'gen_sec':        1,
        'apply_wst':      True,
        'input_channels': 3,
        'actual_seq_len': 875,   # 7s * 125 Hz
        'target_len':     125,
    }

    # 2. Architettura + pesi
    print("-> Inizializzazione architettura modello...")
    model = ModelFactory.get_model(configs)

    print(f"-> Caricamento pesi da: {MODEL_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()

    # 3. Dummy input grezzo [B, 3, 875]
    print("-> Generazione tensore di input fittizio (Dummy Input)...")
    dummy_raw = torch.randn(1, 3, 875)

    # ==========================================
    # ESPORTAZIONE IN ONNX (backbone senza WST)
    # ==========================================
    if EXPORT_FORMAT in ["ONNX", "BOTH"]:
        print("\n[ONNX] Inizio esportazione...")
        print("       kymatio usa view_as_complex (non supportato in ONNX).")
        print("       Esportazione del backbone (CNN+LSTM+Decoder) con feature WST pre-calcolate.")

        # Pre-calcolo delle feature WST fuori dal grafo ONNX
        with torch.no_grad():
            wst_channels = []
            for i in range(dummy_raw.shape[1]):
                wst_channels.append(model.scattering(dummy_raw[:, i, :].contiguous()))
            wst_features = torch.cat(wst_channels, dim=1)
            wst_features = torch.nan_to_num(wst_features, nan=0.0)

        print(f"       Feature WST calcolate — shape: {list(wst_features.shape)}")

        backbone = _ONNXBackbone(model)
        backbone.eval()

        torch.onnx.export(
            backbone,
            wst_features,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['wst_features'],
            output_names=['generated_ecg'],
            dynamic_axes={
                'wst_features':  {0: 'batch_size'},
                'generated_ecg': {0: 'batch_size'},
            },
        )
        print(f"✅ Modello ONNX esportato con successo in: {ONNX_OUTPUT_PATH}")
        print(f"   Input atteso dal modello ONNX: wst_features {list(wst_features.shape)}")

    # ==========================================
    # ESPORTAZIONE PER PYTORCH MOBILE (TorchScript / .pt)
    # — modello completo, WST inclusa
    # ==========================================
    if EXPORT_FORMAT in ["PT", "BOTH"]:
        print("\n[TorchScript] Inizio esportazione (modello completo con WST)...")
        traced_script_module = torch.jit.trace(model, dummy_raw)
        traced_script_module.save(PT_OUTPUT_PATH)
        print(f"✅ Modello TorchScript esportato con successo in: {PT_OUTPUT_PATH}")
        print(f"   Input atteso dal modello PT: raw PPG signals [B, 3, 875]")

    print("\nProcesso completato!")


if __name__ == "__main__":
    export_model_for_mobile()
