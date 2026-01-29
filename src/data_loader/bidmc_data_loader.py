import os
import pandas as pd
import numpy as np

class BidmcDataLoader:
    """
    Classe dedicata esclusivamente al caricamento dei dati grezzi dal dataset BIDMC.
    I file CSV contengono segnali acquisiti a 125 Hz[cite: 761].
    """

    def load_subjects(self, subject_ids, base_path):
        """
        Carica i segnali PLETH (PPG) e II (ECG) dai file CSV originali.

        Args:
            subject_ids (list): Lista di ID soggetti (es. [1, 2, 10] o ['01', '02']).
            base_path (str): Percorso della cartella 'bidmc_data'.

        Returns:
            dict: Un dizionario contenente i segnali grezzi per ogni soggetto.
        """
        all_data = {
            'is_resampled': False, # I dati BIDMC sono nativamente a 125Hz [cite: 761]
            'subjects_data': {}
        }

        for s_id in subject_ids:
            # Formattazione dell'ID per corrispondere al pattern bidmc_##_Signals.csv
            s_num = str(s_id).zfill(2)
            file_name = f"bidmc_{s_num}_Signals.csv"
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                print(f"Avviso: File non trovato per il soggetto {s_num} in {file_path}")
                continue

            try:
                # Caricamento del file CSV
                df = pd.read_csv(file_path)
                
                # Pulizia dei nomi delle colonne per evitare problemi con spazi bianchi
                df.columns = df.columns.str.strip()

                # Verifica della presenza delle colonne necessarie [cite: 761]
                if 'PLETH' not in df.columns or 'II' not in df.columns:
                    print(f"Errore: Colonne 'PLETH' o 'II' non trovate nel file {file_name}")
                    continue

                # Estrazione dei segnali grezzi
                # PLETH rappresenta il fotopletismogramma (PPG) [cite: 761]
                # II rappresenta l'elettrocardiogramma lead II (ECG) [cite: 761]
                all_data['subjects_data'][f"S{s_num}"] = {
                    'PPG': df['PLETH'].values,
                    'ECG': df['II'].values,
                    'fs_ppg': 125, # Frequenza campionamento BIDMC [cite: 761]
                    'fs_ecg': 125
                }
                print(f"Caricato correttamente: Soggetto {s_num} (Campioni: {len(df)})")

            except Exception as e:
                print(f"Errore durante il caricamento del soggetto {s_num}: {e}")

        return all_data