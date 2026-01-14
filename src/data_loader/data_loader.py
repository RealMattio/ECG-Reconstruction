import pickle
import os

class DaliaDataLoader:
    """
    Classe dedicata al caricamento dei dati del dataset PPG-DaLiA.
    Gestisce sia i dati originali sincronizzati che quelli ricampionati.
    """

    def load_subjects(self, subject_ids, base_path, is_resampled=True):
        """
        Carica i dati per una lista di soggetti e li incapsula in un dizionario.

        Args:
            subject_ids (list): Lista di ID soggetti (es. ['S1', 'S2', 'S3'])
            base_path (str): Il percorso relativo alla radice dove cercare i dati
            is_resampled (bool): Se True, cerca i file ricampionati (.pkl). 
                                Se False, cerca i file originali sincronizzati (.pkl).

        Returns:
            dict: Un dizionario contenente il flag 'is_resampled' e i dati dei soggetti.
        """
        all_data = {
            'is_resampled': is_resampled,
            'subjects_data': {}
        }

        for s_id in subject_ids:
            # Definizione del percorso in base al tipo di dato
            if is_resampled:
                # Cerca i file nella cartella processed generata precedentemente
                file_path = os.path.join(base_path, 'processed', f"{s_id}_resampled.pkl")
            else:
                # Cerca i file originali sincronizzati nella struttura data/processed/SX/SX.pkl 
                file_path = os.path.join(base_path, 'processed', f"{s_id}_original.pkl")

            if not os.path.exists(file_path):
                print(f"Avviso: File non trovato per il soggetto {s_id} in {file_path}")
                continue

            try:
                with open(file_path, 'rb') as f:
                    # Caricamento del dizionario del soggetto [cite: 109]
                    subject_data = pickle.load(f, encoding='latin1')
                    all_data['subjects_data'][s_id] = subject_data
                    print(f"Caricato correttamente: {s_id} ({'Resampled' if is_resampled else 'Raw'})")
            except Exception as e:
                print(f"Errore durante il caricamento di {s_id}: {e}")

        return all_data