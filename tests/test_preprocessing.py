import os
import numpy as np
from src.data_loader.data_loader import DaliaDataLoader
from src.preprocessing.preprocessor import DaliaPreprocessor
def main():
    # 1. Configurazione percorsi e parametri
    # Otteniamo il percorso assoluto della root del progetto
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data')
    
    # Selezioniamo un piccolo gruppo di soggetti per il test
    all_subjects = ["S1", "S2", "S3", "S4", "S5"]
    train_subs = ["S1", "S2", "S3"]
    val_subs = ["S4"]
    test_subs = ["S5"]

    # Parametri finestra definiti dalla letteratura del dataset [cite: 16, 105, 107]
    WINDOW_SIZE = 8 
    WINDOW_SHIFT = 2
    IS_RESAMPLED = False # Testiamo sui dati originali (asincroni)

    print("--- INIZIO TEST PIPELINE ---")

    # 2. Caricamento Dati
    loader = DaliaDataLoader()
    raw_data = loader.load_subjects(all_subjects, data_path, is_resampled=IS_RESAMPLED)
    
    if not raw_data['subjects_data']:
        print("Errore: Nessun dato caricato. Verifica la cartella data/processed/")
        return

    # 3. Inizializzazione Preprocessor
    preprocessor = DaliaPreprocessor()

    # 4. Step 1: Splitting
    print(f"\nEseguendo lo split: Train {train_subs}, Val {val_subs}, Test {test_subs}...")
    split_dict = preprocessor.split_data(raw_data, train_subs, val_subs, test_subs)

    # 5. Step 2: Normalizzazione Globale
    # Calcola media/std sul train e applica a tutti [cite: 112]
    print("Calcolo della normalizzazione globale sul set di Training...")
    normalized_data = preprocessor.compute_and_apply_normalization(split_dict)
    
    # Sostituzione della riga del print nel ciclo delle statistiche
    for key, stat in preprocessor.stats.items():
        if isinstance(stat['mean'], np.ndarray):
            # Se è un array (come per l'ACC), formattiamo ogni elemento
            m_str = ", ".join([f"{x:.2f}" for x in stat['mean']])
            s_str = ", ".join([f"{x:.2f}" for x in stat['std']])
            print(f"  - {key}: Media=[{m_str}], Std=[{s_str}]")
        else:
            # Se è un numero singolo (PPG, EDA, ECG)
            print(f"  - {key}: Media={stat['mean']:.2f}, Std={stat['std']:.2f}")

    # 6. Step 3: Windowing
    print(f"\nGenerazione finestre di {WINDOW_SIZE}s con shift {WINDOW_SHIFT}s...")
    final_windows = preprocessor.create_windows(
        normalized_data, 
        is_resampled=IS_RESAMPLED, 
        window_size=WINDOW_SIZE, 
        window_shift=WINDOW_SHIFT
    )

    # 7. Verifica Risultati
    print("\n--- RISULTATI FINALI ---")
    for group in ['train', 'val', 'test']:
        num_win = len(final_windows[group])
        print(f"Set {group.upper()}: {num_win} finestre generate.")
        
        if num_win > 0:
            sample = final_windows[group][0]
            ppg, eda, acc = sample['input']
            target = sample['target']
            
            # Verifica lunghezze basate sulle frequenze originali [cite: 21, 22, 80, 89, 91]
            print(f"  Esempio finestra {group}:")
            print(f"    Input PPG: {len(ppg)} campioni (64Hz)")
            print(f"    Input ACC: {len(acc)} campioni (32Hz)")
            print(f"    Input EDA: {len(eda)} campioni (4Hz)")
            print(f"    Target ECG: {len(target)} campioni (256Hz)") # o 700Hz se non ricampionato

    print("\nTest completato con successo!")

if __name__ == "__main__":
    main()