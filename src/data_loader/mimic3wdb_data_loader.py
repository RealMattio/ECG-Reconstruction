import os
import wfdb
import numpy as np
from scipy.signal import resample
from tqdm import tqdm
import gc

class MimicDataLoader:
    """
    DataLoader ottimizzato per MIMIC-III.
    Supporta 'Lazy Loading': indicizza i file senza caricare i segnali in RAM.
    """

    def __init__(self, data_dir, target_fs=125):
        self.data_dir = data_dir
        self.target_fs = target_fs
        # Cache interna per i metadati (percorsi, lunghezza, fs originale)
        self.subjects_index = {} 

    def _get_all_records(self):
        """Scansiona le cartelle per trovare i file .hea."""
        record_paths = []
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory non trovata: {self.data_dir}")

        print("--- Scanning MIMIC-III Directories ---")
        for root, dirs, files in os.walk(self.data_dir):
            # Cerca file .hea validi (escludendo i layout 'p...')
            hea_files = [f for f in files if f.endswith('.hea') and not f.startswith('p')]
            for hea in hea_files:
                base_name = hea.replace('.hea', '')
                if (base_name + '.dat') in files: # Verifica coppia
                    record_paths.append(os.path.join(root, base_name))
        
        return sorted(record_paths)

    def _resample_signal(self, signal, original_fs):
        """Ricampiona usando float32 per risparmiare memoria."""
        if original_fs == self.target_fs:
            return signal.astype(np.float32)
        
        duration_sec = len(signal) / original_fs
        target_samples = int(duration_sec * self.target_fs)
        
        # Casting a float32 PRIMA del resample per ridurre l'impronta memoria
        signal = signal.astype(np.float32)
        resampled = resample(signal, target_samples)
        return resampled.astype(np.float32)

    def load_data(self, lazy=True, limit=None):
        """
        Costruisce l'indice dei dati.
        
        Args:
            lazy (bool): Se True (Default), carica solo metadati (veloce, poca RAM).
                         Se False, carica TUTTI i segnali (lento, rischio OOM).
            limit (int): Limita il numero di record (utile per debug).
        """
        record_paths = self._get_all_records()
        if limit: record_paths = record_paths[:limit]

        # Struttura dati di ritorno
        dataset_container = {
            'is_lazy': lazy,
            'target_fs': self.target_fs,
            'subjects_data': {} 
        }

        print(f"--- Indicizzazione ({'LAZY' if lazy else 'FULL'}) di {len(record_paths)} file ---")
        
        for rec_path in tqdm(record_paths, desc="Indexing Records"):
            try:
                # --- MODIFICA CRITICA: Estrazione ID Paziente Reale ---
                # Estraiamo il vero ID paziente dalla cartella (es. "p000052")
                # rec_path è tipo: ".../p00/p000052/3533390_0011"
                parent_dir = os.path.dirname(rec_path)
                patient_id = os.path.basename(parent_dir) 
                # ------------------------------------------------------

                # OTTIMIZZAZIONE: Leggiamo solo l'header testuale (.hea)
                # rdheader è molto più veloce e leggero di rdrecord
                header = wfdb.rdheader(rec_path)
                
                signals = header.sig_name
                
                # Check rapido canali necessari
                if 'II' not in signals or 'PLETH' not in signals:
                    continue

                record_key = header.record_name
                
                # In modalità LAZY salviamo solo dove trovare il file
                dataset_container['subjects_data'][record_key] = {
                    'patient_path': rec_path,
                    'subject_id': patient_id, # <--- SALVIAMO QUESTO NUOVO CAMPO
                    'original_fs': header.fs,
                    'sig_len': header.sig_len,
                    'signals': signals # Salviamo l'ordine dei canali
                }

                # In modalità FULL carichiamo subito (sconsigliato per MIMIC intero)
                if not lazy:
                    data = self.load_signal(record_key, dataset_container['subjects_data'][record_key])
                    # Sovrascriviamo i metadati con i dati veri
                    dataset_container['subjects_data'][record_key].update(data)

            except Exception:
                continue

        self.subjects_index = dataset_container['subjects_data']
        print(f"[COMPLETO] Indicizzati {len(self.subjects_index)} record validi.")
        return dataset_container

    def load_signal(self, key, metadata=None):
        """
        Carica i segnali ON-DEMAND per un singolo paziente.
        Da chiamare dentro il loop di training/preprocessing.
        """
        # Recupera metadati se non passati
        if metadata is None:
            metadata = self.subjects_index.get(key)
            if metadata is None:
                raise ValueError(f"Key {key} non trovata nell'indice.")

        rec_path = metadata['patient_path']
        
        try:
            # Caricamento effettivo dal disco
            record = wfdb.rdrecord(rec_path)
            
            # Estrazione canali
            # Nota: Usiamo i nomi salvati o li cerchiamo al volo
            idx_ecg = record.sig_name.index('II')
            idx_ppg = record.sig_name.index('PLETH')
            
            ecg_raw = record.p_signal[:, idx_ecg]
            ppg_raw = record.p_signal[:, idx_ppg]
            
            # Gestione NaN rapida
            ecg_raw = np.nan_to_num(ecg_raw).astype(np.float32)
            ppg_raw = np.nan_to_num(ppg_raw).astype(np.float32)

            # Resampling
            ecg_final = self._resample_signal(ecg_raw, record.fs)
            ppg_final = self._resample_signal(ppg_raw, record.fs)
            
            # Pareggiamento lunghezze
            min_len = min(len(ecg_final), len(ppg_final))
            
            return {
                'ECG': ecg_final[:min_len],
                'PPG': ppg_final[:min_len],
                'fs': self.target_fs
            }
            
        except Exception as e:
            print(f"Errore caricamento lazy {key}: {e}")
            return {'ECG': np.array([]), 'PPG': np.array([])}