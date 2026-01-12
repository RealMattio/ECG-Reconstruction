![Python](https://img.shields.io/badge/Python-3.10.19%20-blue?logo=python)

| [🇬🇧 Read in English](README_en.md) | [🇮🇹 Leggi in Italiano](README.md) | 
| :--- | :--- |
## 📊 Il Dataset

Il dataset **PPG-DaLiA** (PPG-based Heart Rate Estimation Dataset for Daily Life Activities) è stato creato per affrontare la sfida della stima della frequenza cardiaca tramite fotopletismografia (PPG) in presenza di artefatti da movimento. A differenza dei dataset di laboratorio, questo include registrazioni di lunga durata effettuate durante attività di vita quotidiana.

### 📂 Struttura delle Cartelle

Il dataset è organizzato per soggetti. Esistono **15 soggetti** totali (7 uomini, 8 donne). Ogni soggetto ha la propria cartella identificata da un ID (es. `S1`, `S2`, ... `S15`).

```text
PPG-FieldStudy/
├── S1/
│   ├── S1.pkl           # Dati sincronizzati e pre-elaborati (Raccomandato)
│   ├── S1_quest.csv     # Metadati del soggetto (età, peso, fitness)
│   ├── S1_activity.csv  # Timestamp di inizio delle attività
│   ├── S1_RespiBAN.h5   # Dati grezzi dal sensore toracico
│   └── S1_E4.zip        # Dati grezzi dal sensore al polso
├── S2/
...
```
---

### 📄 Dettaglio dei File

#### 1. Il File Master: `SX.pkl` (Consigliato per il Machine Learning)

Questo file è un dizionario Python (`pickle`) che contiene tutti i dati già **sincronizzati e pronti all'uso**. È la risorsa principale se vuoi iniziare subito a addestrare modelli.

*  **`signal`**: Contiene i dati grezzi sincronizzati da entrambi i dispositivi:
    * `wrist`: Dati del sensore Empatica E4 (ACC, BVP, EDA, TEMP).
    * `chest`: Dati del sensore RespiBAN (ACC, ECG, RESP).
* **`label`**: La *Ground Truth* della frequenza cardiaca (calcolata dall'ECG) fornita per finestre di 8 secondi con uno shift di 2 secondi.
* **`activity`**: Etichette delle attività corrispondenti ai dati.
*  **`questionnaire`**: Informazioni demografiche sul soggetto.
* **`rpeaks`**: Gli indici dei picchi R estratti dal segnale ECG.

#### 2. Dati dei Sensori (Raw Data)

Se preferisci lavorare con i dati non processati, sono disponibili due sorgenti:

*  **RespiBAN (Torace)**: Campionato a **700 Hz**. Include segnali ECG (usati per la verità di base), respirazione e accelerometro 3D .
* **Empatica E4 (Polso)**: Include diversi sensori con frequenze differenti:
    * **BVP (PPG)**: 64 Hz (Il segnale principale per la stima HR).
    * **ACC**: 32 Hz (3 assi, fondamentale per compensare il movimento).
    * **EDA / TEMP**: 4 Hz.

#### 3. Metadati e Protocollo: `SX_quest.csv` e `SX_activity.csv`

* **Metadati**: Età, genere, altezza, peso, tipo di pelle (Fitzpatrick scale) e livello di fitness .

* **Attività**: Il dataset copre 8 attività diverse svolte in condizioni naturali:


| ID | Attività | Descrizione | Durata Media |
| --- | --- | --- | --- |
| 1 | Sitting | Seduti a leggere (baseline) | 10 min |
| 2 | Stairs | Salire e scendere 6 piani di scale | 5 min |
| 3 | Table Soccer | Partita a calcetto 1 vs 1 | 5 min |
| 4 | Cycling | Ciclismo all'aperto su vari terreni | 8 min |
| 5 | Driving | Guida in città e su strade extraurbane | 15 min |
| 6 | Lunch Break | Coda in mensa, mangiare e parlare | 30 min |
| 7 | Walking | Camminata di ritorno in ufficio  | 10 min |
| 8 | Working | Lavoro al computer in ufficio | 20 min |
----

### ⚠️ Note Importanti

*  **Soggetto S6**: A causa di un guasto hardware, i dati di S6 sono validi solo per i primi 90 minuti della raccolta.


*  **Sincronizzazione**: I dispositivi sono stati sincronizzati manualmente tramite un gesto di "doppio tocco" sul petto, registrato dagli accelerometri di entrambi i sensori.

Ecco una proposta per la sezione del file `README.md` che descrive la struttura del tuo progetto, integrando le informazioni tecniche del dataset **PPG-DaLiA**.

---

## 📂 Struttura del Progetto

Il progetto è organizzato in modo modulare per gestire le quattro fasi principali: pre-processing, fusione multimodale, generazione di segnali ECG e valutazione delle performance.

```text
PPG-ECG-Generation/
│
├── data/                       # Directory locale per la gestione dei dati
│   ├── raw/                    # File originali del dataset SX.pkl (esclusi da git poiché troppo grandi)
│   └── processed/              # Segnali memorizzati dopo la fase di pre-elaborazione
│
├── src/                        # Codice sorgente principale
│   ├── __init__.py
│   │
│   ├── data_loader/            # Moduli per il caricamento dati e gestione Dataset PyTorch
│   │   ├── dalia_loader.py     # Caricamento dei file .pkl con dati sincronizzati e etichettati
│   │   └── transforms.py       # Operazioni di normalizzazione e data augmentation
│   │
│   ├── preprocessing/          # Fase 1: Elaborazione dei segnali canale per canale
│   │   ├── filters.py          # Implementazione di filtri passa-banda e rimozione artefatti
│   │   └── segmentation.py     # Segmentazione tramite sliding window (8s di finestra, 2s di shift)
│   │
│   ├── fusion/                 # Fase 2: Architettura di Fusione Multimodale
│   │   ├── attention.py        # Implementazione di meccanismi di Self e Cross-Attention
│   │   └── fusion_layers.py    # Definizione della struttura di fusione (Early/Late/Hybrid)
│   │
│   ├── generation/             # Fase 3: Definizione del Modello e Addestramento
│   │   ├── models/             # Architetture generative (es. GAN, Diffusion o UNet)
│   │   ├── trainer.py          # Gestione del loop di training e salvataggio dei pesi
│   │   └── inference.py        # Generazione di ECG partendo da nuovi input PPG
│   │
│   └── evaluation/             # Fase 4: Metriche di Valutazione e Testing
│       ├── metrics.py          # Calcolo di RMSE, correlazione e errore sulla HR
│       └── ablation.py         # Script per l'esecuzione di studi di ablazione
│
├── notebooks/                  # Jupyter Notebooks per analisi esplorativa e visualizzazioni
├── configs/                    # File .yaml o .json per la gestione degli iperparametri
├── scripts/                    # Script shell per avviare rapidamente addestramento o test
├── tests/                      # Unit test per la validazione dei singoli moduli
├── requirements.txt            # Dipendenze del progetto
└── README.md                   # Documentazione principale

```

### Dettagli sui Componenti Core
_Da definire_