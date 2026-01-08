| [🇮🇹 Leggi in Italiano](README.md) | [🇬🇧 Read in English](README_en.md) |
| :--- | :--- |
## 📊 Dataset

The **PPG-DaLiA** (PPG-based Heart Rate Estimation Dataset for Daily Life Activities) dataset was created to address the challenge of estimating heart rate using photoplethysmography (PPG) in the presence of motion artifacts. Unlike laboratory datasets, this one includes long-term recordings made during daily life activities.

### 📂 Folder Structure

The dataset is organized by subject. There are **15 subjects** in total (7 men, 8 women). Each subject has their own folder identified by an ID (e.g., `S1`, `S2`, ... `S15`).

```text
PPG-FieldStudy/
├── S1/
│   ├── S1.pkl           # Synchronized and pre-processed data (Recommended)
│   ├── S1_quest.csv     # Subject metadata (age, weight, fitness)
│   ├── S1_activity.csv  # Activity start timestamps
│   ├── S1_RespiBAN.h5   # Raw data from chest sensor
│   └── S1_E4.zip        # Raw data from wrist sensor
├── S2/...

```
---

### 📄 File Details

#### 1. The Master File: `SX.pkl` (Recommended for Machine Learning)

This file is a Python dictionary (`pickle`) containing all data already **synchronized and ready to use**. It is the main resource if you want to start training models right away.

*  **`signal`**: Contains raw data synchronized from both devices:
    * `wrist`: Empatica E4 sensor data (ACC, BVP, EDA, TEMP).
    * `chest`: RespiBAN sensor data (ACC, ECG, RESP).
* **`label`**: The *Ground Truth* of the heart rate (calculated from the ECG) provided for 8-second windows with a 2-second shift.
* **`activity`**: Activity labels corresponding to the data.
* **`questionnaire`**: Demographic information about the subject.
* **`rpeaks`**: R peak indices extracted from the ECG signal.

#### 2. Sensor Data (Raw Data)

If you prefer to work with unprocessed data, two sources are available:

*  **RespiBAN (Chest)**: Sampled at **700 Hz**. Includes ECG signals (used for baseline truth), respiration, and 3D accelerometer.
* **Empatica E4 (Wrist)**: Includes several sensors with different frequencies:
    * **BVP (PPG)**: 64 Hz (The main signal for HR estimation).
    * **ACC**: 32 Hz (3 axes, essential for compensating for movement).
    * **EDA / TEMP**: 4 Hz.

#### 3. Metadata and Protocol: `SX_quest.csv` and `SX_activity.csv`

* **Metadata**: Age, gender, height, weight, skin type (Fitzpatrick scale), and fitness level.

* **Activity**: The dataset covers 8 different activities performed under natural conditions:


| ID | Activity | Description | Average Duration |
| --- | --- | --- | --- |
| 1 | Sitting | Sitting and reading (baseline) | 10 min |
| 2 | Stairs | Climbing and descending 6 flights of stairs | 5 min |
| 3 | Table Soccer | 1 vs 1 table soccer match | 5 min |
| 4 | Cycling | Outdoor cycling on various terrains | 8 min |
| 5 | Driving | Driving in the city and on suburban roads | 15 min |
| 6 | Lunch Break | Queuing in the cafeteria, eating and talking | 30 min |
| 7 | Walking | Walking back to the office  | 10 min |
| 8 | Working | Working on the computer in the office | 20 min |
----

### ⚠️ Important Notes

*  **Subject S6**: Due to a hardware failure, S6 data is only valid for the first 90 minutes of collection.


*  **Synchronization**: Devices were synchronized manually via a “double tap” gesture on the chest, recorded by the accelerometers of both sensors.
