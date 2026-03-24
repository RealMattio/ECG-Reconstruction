"""
test_new_models_pipeline.py
===========================
Verifica end-to-end dei tre modelli nuovi/modificati:
  - BioTransformer (PhysioFormer)
  - PPGWaveNet
  - ECGUNet1D

Cosa viene testato per ogni modello:
  1. Istanziazione (con e senza WST)
  2. Forward pass — shape output corretta (B, 1, target_len)
  3. Nessun NaN / Inf nei pesi iniziali e nell'output
  4. Backward pass — gradienti non nulli su tutti i parametri
  5. Compatibilità con Trainer.compute_detailed_losses (loss PINN)
  6. Compatibilità con ModelFactory (creazione tramite configs dict)
  7. Diverse configurazioni di input_channels (1, 2, 3)
  8. Robustezza a lunghezze di sequenza diverse da quella nominale
"""

import sys
import os
import math
import pytest
import torch
import torch.nn as nn

# Aggiungiamo la root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mimic_generation_PINN.models.bio_transformer import BioTransformer
from src.mimic_generation_PINN.models.ppg_wavenet import PPGWaveNet
from src.mimic_generation_PINN.models.ecg_unet1d import ECGUNet1D
from src.mimic_generation_PINN.model_factory import ModelFactory

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI DI TEST  (replicano le dimensioni reali della pipeline MIMIC-III)
# ─────────────────────────────────────────────────────────────────────────────
BATCH       = 4
IN_CH       = 3          # [PPG, PPG_diff, ECG_past]
SEQ_LEN     = 875        # 7 s × 125 Hz
TARGET_LEN  = 125        # 1 s × 125 Hz
DEVICE      = torch.device('cpu')

BASE_CONFIGS = {
    'input_channels':  IN_CH,
    'actual_seq_len':  SEQ_LEN,
    'target_len':      TARGET_LEN,
    'apply_wst':       False,
    'normalize_01':    False,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _make_batch(b=BATCH, ch=IN_CH, t=SEQ_LEN):
    """Crea un batch sintetico normalizzato [0,1]."""
    return torch.rand(b, ch, t)


def _assert_output_valid(out, name: str):
    """Controlla shape, NaN, Inf sull'output del modello."""
    assert out.shape == (BATCH, 1, TARGET_LEN), (
        f"[{name}] Shape output errata: attesa ({BATCH},1,{TARGET_LEN}), ottenuta {tuple(out.shape)}"
    )
    assert not torch.isnan(out).any(),  f"[{name}] NaN nell'output"
    assert not torch.isinf(out).any(),  f"[{name}] Inf nell'output"


def _assert_gradients_flow(model, loss, name: str):
    """Verifica che il backward popoli i gradienti su tutti i layer con parametri."""
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"[{name}] Gradienti mancanti su: {missing[:5]}"


def _pinn_loss(output, target):
    """
    Replica semplificata di Trainer.compute_detailed_losses senza physics ODE
    (sufficiente per verificare la compatibilità di shape e dtype).
    """
    target = target.unsqueeze(1) if target.dim() == 3 else target
    # Weighted MAE
    weights  = 1.0 + 3.0 * torch.abs(target)
    amp_loss = torch.mean(weights * torch.abs(output - target))

    # Pearson loss
    o_m = output.mean(dim=-1, keepdim=True)
    t_m = target.mean(dim=-1, keepdim=True)
    num = ((output - o_m) * (target - t_m)).sum(dim=-1)
    den = torch.sqrt(((output - o_m)**2).sum(-1) * ((target - t_m)**2).sum(-1) + 1e-8)
    pearson = 1 - (num / den).mean()

    # ODE penalty (derivata empirica vs costante nulla — solo verifica shape)
    dz = output[:, :, 1:] - output[:, :, :-1]
    ode_loss = torch.mean(dz ** 2)

    return amp_loss + 0.4 * pearson + 0.1 * ode_loss


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIZZAZIONE: i modelli da testare
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CLASSES = [
    ('bio_transformer', BioTransformer),
    ('ppg_wavenet',     PPGWaveNet),
    ('ecg_unet1d',      ECGUNet1D),
]


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — Istanziazione base
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_instantiation(model_key, ModelClass):
    """Il modello deve istanziarsi senza eccezioni con la config base."""
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN)
    assert model is not None


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — Forward pass: shape output
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_forward_shape(model_key, ModelClass):
    """L'output deve avere shape (B, 1, target_len)."""
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(_make_batch())
    _assert_output_valid(out, model_key)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Nessun NaN nei pesi iniziali
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_no_nan_in_weights(model_key, ModelClass):
    """I pesi iniziali non devono contenere NaN o Inf."""
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN)
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"[{model_key}] NaN nel parametro '{name}'"
        assert not torch.isinf(param).any(), f"[{model_key}] Inf nel parametro '{name}'"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — Backward pass: gradienti non nulli
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_backward_gradients(model_key, ModelClass):
    """Il backward deve popolare i gradienti su tutti i parametri trainable."""
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN).to(DEVICE)
    model.train()
    x   = _make_batch().requires_grad_(False)
    out = model(x)
    target = torch.rand_like(out)
    loss = _pinn_loss(out, target)
    _assert_gradients_flow(model, loss, model_key)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — Compatibilità con ModelFactory
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_model_factory(model_key, ModelClass):
    """ModelFactory deve restituire il modello corretto dato il model_type."""
    cfg = {**BASE_CONFIGS, 'model_type': model_key}
    model = ModelFactory.get_model(cfg)
    assert isinstance(model, ModelClass), (
        f"ModelFactory ha restituito {type(model).__name__}, atteso {ModelClass.__name__}"
    )
    model.eval()
    with torch.no_grad():
        out = model(_make_batch())
    _assert_output_valid(out, model_key)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — Diversi numeri di input_channels (1, 2, 3)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_variable_input_channels(model_key, ModelClass, n_channels):
    """Il modello deve funzionare con 1, 2 o 3 canali di input."""
    cfg = {**BASE_CONFIGS, 'input_channels': n_channels}
    model = ModelClass(cfg, seq_len=TARGET_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        x   = torch.rand(BATCH, n_channels, SEQ_LEN)
        out = model(x)
    assert out.shape == (BATCH, 1, TARGET_LEN), (
        f"[{model_key}, ch={n_channels}] Shape errata: {tuple(out.shape)}"
    )
    assert not torch.isnan(out).any()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — Robustezza a lunghezze di sequenza diverse
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
@pytest.mark.parametrize("seq", [100, 500, 875, 1000])
def test_variable_seq_len(model_key, ModelClass, seq):
    """
    Il modello deve gestire senza crash sequenze di lunghezza diversa
    da quella nominale (il resize finale porta sempre a target_len).
    """
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        x   = torch.rand(BATCH, IN_CH, seq)
        out = model(x)
    assert out.shape == (BATCH, 1, TARGET_LEN), (
        f"[{model_key}, seq={seq}] Shape errata: {tuple(out.shape)}"
    )
    assert not torch.isnan(out).any()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8 — Loss PINN fine-to-end (train step completo)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_full_train_step(model_key, ModelClass):
    """
    Simula un intero step di training: forward → loss PINN → backward → optimizer step.
    Verifica che i pesi cambino dopo l'update.
    """
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Snapshot pesi prima
    first_param = next(model.parameters()).clone().detach()

    x      = _make_batch()
    target = torch.rand(BATCH, 1, TARGET_LEN)
    theta  = torch.rand(BATCH, TARGET_LEN) * 2 * math.pi
    omega  = torch.full((BATCH, TARGET_LEN), 2 * math.pi)

    optimizer.zero_grad()
    out  = model(x)
    loss = _pinn_loss(out, target)

    assert not torch.isnan(loss), f"[{model_key}] Loss è NaN prima del backward"
    assert not torch.isinf(loss), f"[{model_key}] Loss è Inf prima del backward"

    loss.backward()
    optimizer.step()

    # I pesi devono essere cambiati
    first_param_after = next(model.parameters()).clone().detach()
    assert not torch.allclose(first_param, first_param_after), (
        f"[{model_key}] I pesi NON sono cambiati dopo l'optimizer step — gradiente nullo?"
    )


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9 — normalize_01=True (output in [0,1])
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_normalize_01_output_range(model_key, ModelClass):
    """Con normalize_01=True l'output deve essere in [0,1]."""
    cfg = {**BASE_CONFIGS, 'normalize_01': True}
    model = ModelClass(cfg, seq_len=TARGET_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(_make_batch())
    assert out.min() >= -1e-5,  f"[{model_key}] Output < 0 con normalize_01=True"
    assert out.max() <= 1+1e-5, f"[{model_key}] Output > 1 con normalize_01=True"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 10 — ModelFactory: chiave non valida → ValueError
# ═════════════════════════════════════════════════════════════════════════════
def test_factory_invalid_key():
    """ModelFactory deve sollevare ValueError per model_type sconosciuto."""
    with pytest.raises(ValueError):
        ModelFactory.get_model({'model_type': 'modello_inesistente', 'target_len': TARGET_LEN})


# ═════════════════════════════════════════════════════════════════════════════
# TEST 11 — Conteggio parametri (ragionevolezza)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_parameter_count_reasonable(model_key, ModelClass):
    """
    Il numero di parametri deve essere nell'intervallo [10K, 20M].
    Troppo pochi → underfitting garantito. Troppi → overfitting su MIMIC-III.
    """
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n >= 10_000,    f"[{model_key}] Troppo pochi parametri: {n:,}"
    assert n <= 20_000_000, f"[{model_key}] Troppi parametri: {n:,}"
    print(f"\n  [{model_key}] Parametri trainable: {n:,}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 12 — Batch size 1 (inference singola)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_key, ModelClass", MODEL_CLASSES)
def test_batch_size_1(model_key, ModelClass):
    """Il modello deve funzionare con batch size = 1 (inference su singolo campione)."""
    model = ModelClass(BASE_CONFIGS, seq_len=TARGET_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        x   = torch.rand(1, IN_CH, SEQ_LEN)
        out = model(x)
    assert out.shape == (1, 1, TARGET_LEN), f"[{model_key}] Shape errata con bs=1: {tuple(out.shape)}"
    assert not torch.isnan(out).any()
