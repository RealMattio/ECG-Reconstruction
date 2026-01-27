import numpy as np
import random # Aggiunto per lo shuffling

class DaliaPreprocessor:
    def __init__(self):
        self.stats = {}

    def split_data(self, all_data, train_subjects, val_subjects, test_subjects):
        """
        Divide il dizionario dei soggetti in tre gruppi distinti: train, val, test.
        Args:
            all_data (dict): Dizionario contenente i dati di tutti i soggetti.
            train_subjects (list): Lista di ID soggetti per il training set.
            val_subjects (list): Lista di ID soggetti per il validation set.
            test_subjects (list): Lista di ID soggetti per il test set.
        Returns:
            dict: Dizionario con le suddivisioni 'train', 'val' e 'test'. Ognuna contenente...
        """
        data_split = {
            'train': {s: all_data['subjects_data'][s] for s in train_subjects if s in all_data['subjects_data']},
            'val': {s: all_data['subjects_data'][s] for s in val_subjects if s in all_data['subjects_data']},
            'test': {s: all_data['subjects_data'][s] for s in test_subjects if s in all_data['subjects_data']}
        }
        return data_split

    def compute_and_apply_normalization(self, data_split):
        train_signals = {'PPG': [], 'EDA': [], 'ACC': [], 'ECG': []}
        
        for sub_id in data_split['train']:
            sub = data_split['train'][sub_id]
            train_signals['PPG'].append(sub['PPG'])
            train_signals['EDA'].append(sub['EDA'])
            train_signals['ACC'].append(sub['ACC'])
            train_signals['ECG'].append(sub['ECG'])

        for key in train_signals:
            all_vals = np.concatenate(train_signals[key], axis=0)
            self.stats[key] = {
                'mean': np.mean(all_vals, axis=0),
                'std': np.std(all_vals, axis=0) + 1e-8
            }

        for group in ['train', 'val', 'test']:
            for sub_id in data_split[group]:
                for key in ['PPG', 'EDA', 'ACC', 'ECG']:
                    s = data_split[group][sub_id]
                    s[key] = (s[key] - self.stats[key]['mean']) / self.stats[key]['std']
        
        return data_split

    def create_windows(self, data_split, is_resampled, window_size=8, window_shift=2, include_prev_ecg=False, shuffle_train=True):
        """
        Esegue il windowing e opzionalmente include l'ECG della finestra temporale precedente.
        
        Args:
            window_shift: Definisce lo scorrimento. L'overlap è (window_size - window_shift).
            include_prev_ecg: Se True, aggiunge l'ECG della finestra precedente agli input.
        """
        windowed_data = {'train': [], 'val': [], 'test': []}

        for group in ['train', 'val', 'test']:
            for sub_id in data_split[group]:
                sub = data_split[group][sub_id]
                
                fs_in = 64 if is_resampled else sub['fs_ppg']
                fs_ecg = 256 if is_resampled else sub['fs_ecg']
                
                samples_win_in = window_size * fs_in
                samples_shift_in = window_shift * fs_in
                samples_win_ecg = window_size * fs_ecg
                
                total_samples_ppg = len(sub['PPG'])
                
                # Inizializziamo il riferimento per la finestra precedente
                prev_ecg_w = None
                
                for start_in in range(0, total_samples_ppg - samples_win_in, samples_shift_in):
                    end_in = start_in + samples_win_in
                    
                    # Estrazione input standard
                    ppg_w = sub['PPG'][start_in:end_in]
                    eda_w = sub['EDA'][int(start_in * (sub['fs_eda']/fs_in)):int(end_in * (sub['fs_eda']/fs_in))] if not is_resampled else sub['EDA'][start_in:end_in]
                    acc_w = sub['ACC'][int(start_in * (sub['fs_acc']/fs_in)):int(end_in * (sub['fs_acc']/fs_in))] if not is_resampled else sub['ACC'][start_in:end_in]
                    
                    # Estrazione target (ECG attuale)
                    start_ecg = int((start_in / fs_in) * fs_ecg)
                    end_ecg = start_ecg + samples_win_ecg
                    ecg_w = sub['ECG'][start_ecg:end_ecg]

                    if len(ecg_w) == samples_win_ecg:
                        # Logica di condizionamento
                        if include_prev_ecg:
                            # Se è la prima finestra del soggetto, non abbiamo una precedente.
                            # Saltiamo la prima per mantenere la coerenza dimensionale degli input nel batch.
                            if prev_ecg_w is not None:
                                windowed_data[group].append({
                                    'input': (ppg_w, eda_w, acc_w, prev_ecg_w),
                                    'target': ecg_w,
                                    'subject': sub_id
                                })
                        else:
                            # Comportamento originale
                            windowed_data[group].append({
                                'input': (ppg_w, eda_w, acc_w),
                                'target': ecg_w,
                                'subject': sub_id
                            })
                        
                        # Aggiorniamo l'ECG precedente per la prossima iterazione
                        prev_ecg_w = ecg_w
                
                # Resettiamo prev_ecg_w quando cambiamo soggetto
                prev_ecg_w = None
        
        # Shuffling del set di training
        if shuffle_train and len(windowed_data['train']) > 0:
            print(f"Mescolamento di {len(windowed_data['train'])} finestre di training...")
            random.seed(42) 
            random.shuffle(windowed_data['train'])
        
        return windowed_data