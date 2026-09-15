# Computational complexity — LightweightHybrid (deployed model)

Input `(3, 875)` = 7 s @ 125 Hz  ->  output `(1, 125)` = 1 s @ 125 Hz


## Table 1 — Trainable parameters and FLOPs per layer

| Block | Layer | Type | Output shape | Trainable params | FLOPs |
|---|---|---|---|---:|---:|
| WST front-end | `scattering` | Scattering1D (J=2, Q=8) | (1, 24, 219) | 0 | 1,517,568 |
| CNN encoder | `cnn_encoder.0` | Conv1d | 1x32x219 | 5,408 | 2,361,696 |
| CNN encoder | `cnn_encoder.1` | BatchNorm1d | 1x32x219 | 64 | 14,016 |
| CNN encoder | `cnn_encoder.2` | LeakyReLU | 1x32x219 | 0 | 7,008 |
| CNN encoder | `cnn_encoder.3` | MaxPool1d | 1x32x109 | 0 | 3,488 |
| CNN encoder | `cnn_encoder.4` | Conv1d | 1x64x109 | 10,304 | 2,239,296 |
| CNN encoder | `cnn_encoder.5` | BatchNorm1d | 1x64x109 | 128 | 13,952 |
| CNN encoder | `cnn_encoder.6` | LeakyReLU | 1x64x109 | 0 | 6,976 |
| CNN encoder | `cnn_encoder.7` | MaxPool1d | 1x64x54 | 0 | 3,456 |
| BiLSTM | `lstm` | LSTM | 1x54x128 | 66,560 | 7,195,392 |
| Upsampling | `upsample` | Upsample | 1x128x125 | 0 | 48,000 |
| Decoder head | `decoder.0` | Dropout | 1x128x125 | 0 | 0 |
| Decoder head | `decoder.1` | Conv1d | 1x64x125 | 24,640 | 6,152,000 |
| Decoder head | `decoder.2` | LeakyReLU | 1x64x125 | 0 | 8,000 |
| Decoder head | `decoder.3` | Conv1d | 1x1x125 | 65 | 16,125 |
| **Total** | | | | **107,169** | **19,586,973** |

## Table 2 — Trainable parameters per block

| Block | Trainable params | Share |
|---|---:|---:|
| CNN encoder | 15,904 | 14.8% |
| BiLSTM | 66,560 | 62.1% |
| Upsampling | 0 | 0.0% |
| Decoder head | 24,705 | 23.1% |
| **Total trainable** | **107,169** | **100.0%** |
| Non-trainable buffers (BatchNorm statistics) | 194 | — |
| Fixed WST filter coefficients | 11,776 | — |
