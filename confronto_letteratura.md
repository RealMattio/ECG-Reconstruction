Nella maggior parte degli studi, i dataset di addestramento e di test provengono dalla stessa banca dati (es. MIMIC, BIDMC) suddivisi con proporzioni standard (tipicamente 80% training e 20% testing). In alcuni casi, la validazione incrociata o dataset indipendenti vengono usati per il test.

### Tabella Estesa: Modelli di Generazione PPG $\rightarrow$ ECG con Performance

| Modello / Articolo | Dataset Addestramento | Dataset Test | Lunghezza Input | Lunghezza Output | Freq. Campionamento | Metriche Usate | Valori Ottenuti |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WaveNet / 3P2P-PPG2ECG-TI** | MIMIC-II (80% train, 10% val) | MIMIC-II (10% test) | 3-beats (cicli) | **Stessa dell'input** | 128 campioni/battito | r, RMSE, FD, PRD, MAE, SD | **r:** 0.9831; **RMSE:** 0.0373; **FD:** 0.1174; **PRD:** 12.32%; **MAE:** 0.0262 |
| **CLEP-GAN (Att. U-Net/VQ-VAE)** | BIDMC, CapnoBase, Synthetic ODE | Stessi dataset (15%) | Segmenti fissi (4s) | **Stessa dell'input (512 punti)** | 125 Hz | rHI, rRMSE, rEMD, KL, KS, RMSE, FD, MAE_HR | **BIDMC:** RMSE=0.37, FD=22.27, MAE_HR=0.84 bpm.<br>**CapnoBase:** RMSE=0.33, FD=32.45, MAE_HR=1.29 bpm. |
| **CardioFlow (Rectified Flow)** | WESAD, DALIA (9 sogg.) | WESAD, DALIA (3 sogg.) | 4 secondi | **Stessa dell'input (512 punti)** | 128 Hz (512 punti) | RMSE, FD, MAE_HR | *Non esplicitati nei frammenti forniti* |
| **CardioGAN (Dual Discriminator)** | BIDMC, CAPNO, DALIA, WESAD (80%) | Stessi dataset (20%) | 4 secondi | **Stessa dell'input (512 punti)** | 128 Hz (512 punti) | MAE_HR, RMSE, PRD, FD | **Media:** RMSE=0.364, PRD=8.356, FD=0.694.<br>**MAE_HR:** da 0.7 (BIDMC) a 8.6 bpm (WESAD). |
| **XDJDL (Dizionari congiunti)** | MIMIC-III (80%) | MIMIC-III (20%) | Beat-by-beat | **Stessa dell'input (300 punti)** | 125 Hz (norm. 300 pt) | r, rRMSE, MAE intervalli | **r:** 0.88 (base) / 0.92 (LC-XDJDL).<br>**rRMSE:** 0.39. |
| **Trasformata DCT (Lineare - Pilot)** | Capnobase TBME-RR (80%) | Capnobase TBME-RR | Ciclo cardiaco | **Stessa dell'input** | 300 Hz | r, rRMSE | **r:** ~0.98 (media);<br>**rRMSE:** ~0.14-0.15 (dai grafici). |
| **HA-CNN-BiLSTM** | Physionet MIMIC II (90%) | Physionet MIMIC II | 1024 (o 120 pt per beat) | **Stessa dell'input** | 125 Hz | RMSE | **RMSE**: 0.052 |
| **BiLSTM-CNN GAN** | MIT-BIH Arrhythmia | MIT-BIH Arrhythmia | Fino a 3120 punti | **Stessa dell'input (es. 3120 punti)** | 360 Hz | PRD, RMSE, FD | **RMSE:** ~0.2-0.3;<br>**FD:** ~0.6-0.8 (stimati dai grafici). |
| **Performer (Transformer SPA)** | UQVSD, DaLiA, BIDMC, MIMIC-III, PPG-BP | Stessi dataset | 4 secondi | **Stessa dell'input (512 punti)** | 128 Hz (512 punti) | RMSE, Accuratezza diagnosi | **RMSE:** 0.29 (su BIDMC).<br>**Accuratezza CVD:** 95.9% (MIMIC-III), 75.9% (PPG-BP). |
| **DCT Regression (SM e GM)** | Capnobase, MIMIC-III, UMD | Capnobase, MIMIC-III, UMD | Ciclo cardiaco | **Stessa dell'input** | 300 Hz / 125 Hz | r, rRMSE | **TBME-RR (GM):** r=0.859, rRMSE=0.499.<br>**UMD (SM):** r=0.904, rRMSE=0.426. |
| **Att-U-Net con BiGRU** | PulseDB, MIMIC III, VitalDB | PulseDB, MIMIC III, VitalDB | 2s, 4s, o beat | **Stessa dell'input (es. 250, 500 pt)** | 125 Hz | r (p), RMSE | **MIMIC-III (4s):** p=0.86, RMSE=0.09.<br>**VitalDB (4s):** p=0.89, RMSE=0.10. *(Fino a p=0.94 con RR+pad)* |
| **P2E-LGAN (LSTM GAN)** | BIDMC, MIMIC-II (80%) | BIDMC, MIMIC-II | Circa 1000 punti | **Stessa dell'input** | N/D | RMSE, MAE_HR, PRD | **RMSE:** 0.234;<br>**MAE_HR:** 2.995 bpm;<br>**PRD:** 7.54%. |
| **P2E-WGAN** | MIMIC II (80%) | MIMIC II (20%) | 3 secondi | **Stessa dell'input (375 punti)** | 125 Hz (375 punti) | r, RMSE, FD | **r:** 0.835;<br>**RMSE:** 0.162;<br>**FD:** 0.375 (o 0.357). |
| **PGANs (Personalized GAN)** | MIT-BIH (48 record) | Test *leave-one-out* | 3 cicli cardiaci | **1 ciclo cardiaco (centrale)** | 360 Hz | AUC per classificazione | *Non esplicitati nei frammenti forniti* |
| **PPG2ECGps (W-Net)** | Cuffless Blood Pressure (80%) | Cuffless BP (20%) | 1024 campioni (~8.2s) | **Stessa dell'input (1024 punti)** | 125 Hz | r, RMSE, DTW Norm. | **r:** 0.977;<br>**RMSE:** 0.037 mV;<br>**DTW norm.:** 0.010 mV. |
| **PhotoECG (Feature-based)** | Capnobase, iPhone | Stessi dataset (40%) | 1024 o 256 pt | **Parametri estratti (non un'onda)** | 300 Hz / 30 Hz | Accuratezza detezione | **Accuratezza:** ~80% per stima intervalli PR/QRS/QT. |
| **P2ERM (SAE-MPNN-LSTM)** | MIMIC II/III, Volontari | Stessi dataset (50%) | Beat-by-beat | **Stessa dell'input (Beat ECG)** | 125 Hz | MAE, RMSE, MSE, r | **r:** 0.952;<br>**RMSE:** 0.0259 mV;<br>**MAE:** 0.0178 mV. |
| **Transformed Attentional NN** | UQVSD, BIDMC (10-fold CV) | UQVSD, BIDMC | Finestre 200/256 pt | **Stessa dell'input (200/256 punti)** | 100 Hz / 125 Hz | L1QRS, NRMSE, MLE | **UQVSD:** L1QRS=0.339, NRMSE=0.107, RFAIL=3.67%. |
| **BiLSTM (Subject-Based)** | MIMIC III (100 record) | MIMIC III | Finestre 1s a 4s | **Stessa dell'input** | 125 Hz | r, RMSE, DTW | **r:** 0.818;<br>**RMSE:** 0.083 mV;<br>**DTW:** 2.12 mV/s. |
| **Conv-BiLSTM Autoenc.** | MIMIC II, BIDMC, MIMIC | Cross-subject | Beat-by-beat (128 pt)| **Stessa dell'input (128 punti)** | 125 Hz / 500 Hz | r, rRMSE, MSE | **r:** 0.91 - 0.923;<br>**rRMSE:** 0.31 - 0.35;<br>**MSE:** 0.0086. |
| **Cross-Lead ECG (PPG $\rightarrow$ Multi)** | SensSmartTech (80%) | SensSmartTech (20%) | 4 secondi | **Stessa dell'input (512 punti)** | 128 Hz (512 punti) | RMSE, FD | **RMSE:** 0.24;<br>**FD:** 3.108. |

### Note Aggiuntive Emmerse dall'Analisi
*   **Frequenza e Finestre:** La grandissima maggioranza degli studi recenti propende a fare un ricampionamento del segnale a **125 Hz** o **128 Hz**, indipendentemente dalla frequenza originale di acquisizione. Le finestre temporali più diffuse vanno dai 3 ai 4 secondi (quindi dai 375 ai 512 campioni in ingresso alla rete), per assicurare di catturare da 3 a 5 complessi QRS, oppure utilizzano metodiche di input *beat-by-beat* scalando il battito a un numero fisso di campioni (ad esempio 128 punti o 300 punti).
*   **Metriche dominanti:** La bontà morfologica della ricostruzione dell'onda ECG è universalmente misurata tramite il **Coefficiente di Correlazione di Pearson (r)** e la **Root Mean Square Error (RMSE)**. Modelli più avanzati orientati alla fedeltà temporale introducono anche misurazioni come la *Fréchet Distance (FD)* e calcoli specifici sugli intervalli fiduciali (es. errore sul tratto PR, QRS o Q-T) per valutare l'effettiva utilizzabilità in diagnostica cardiaca.



### Tassonomia delle Metriche per Dataset

| Dataset | Metriche Morfologiche | Metriche Clinico-Temporali | Metriche Fisiologiche e Diagnostiche (incl. Picchi R) |
| :--- | :--- | :--- | :--- |
| **MIMIC-II** (e derivati es. Cuffless BP) | **r / p, RMSE, MSE, MAE, rRMSE, PRD, FD, SD, Normalized DTW, Q1, Q2** | Errori di stima intervalli in ms/campioni (**T, QT, P, PR, QRS**) | - |
| **MIMIC-III** | **r, RMSE, rRMSE, MSE, DTW** | MAE su lunghezza intervalli in secondi (**PR, QRS, QT**) | Accuratezza di Classificazione e Matrice di Confusione (CVD: CAD, CHF, MI, HoTN) |
| **BIDMC** | **r, RMSE, rRMSE, PRD, FD** | Metriche su distribuzioni RR: **rHI, rEMD, KL Divergence, KS Test** | - Errore HR: **MAE_HR, HRV**<br>- Metriche Picchi R/QRS: **L1QRS, L1nQRS, NMAE, NRMSE, MLE, MME, RFAIL** |
| **CapnoBase** (TBME-RR) | **r, RMSE, rRMSE, PRD, FD** | Metriche su distribuzioni RR: **rHI, rEMD, KL, KS** | - Errore HR: **MAE_HR**<br>- Accuratezza (%) posizionamento in range fisiologici normali/bassi/alti (PR, QRS, QT) |
| **WESAD e DALIA** | **RMSE, FD, PRD** | - | - Errore HR: **MAE_HR**<br>- Metriche affettive: **Accuratezza, F1-score** (Stress, Neutralità, Divertimento) |
| **UQVSD** | **RMSE** | - | - Metriche Picchi R/QRS: **L1QRS, L1nQRS, NMAE, NRMSE, MLE, MME, RFAIL** |
| **MIT-BIH Arrhythmia** | **RMSE, FD, PRD** | - | Diagnosi Aritmie: **AUC** (Area Under the Curve) |
| **SensSmartTech** (Multi-lead) | **RMSE, FD** | Errori normalizzati intervallo **Q-T** e **R-R** | - |
| **PPG-BP** | - | - | Accuratezza e Matrice di Confusione (Diagnosi Diabete) |
| **UMD** | **r, rRMSE** | - | - |
| **PulseDB / VitalDB** | **p, RMSE** | - | - |

**Legenda rapida per le metriche sui Picchi R (inserite in colonna 4 per UQVSD e BIDMC):**
*   **L1QRS / L1nQRS:** Loss L1 per le aree QRS e non-QRS.
*   **NMAE / NRMSE:** Errori medi assoluti e quadratici normalizzati ai picchi.
*   **MLE (Mean Location Error):** Errore temporale di posizionamento del picco R.
*   **MME (Mean Magnitude Error):** Errore di ampiezza del picco R.
*   **RFAIL:** Tasso di fallimento nel rilevamento del picco R.