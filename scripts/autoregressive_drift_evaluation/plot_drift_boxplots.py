import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAZIONI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation", "drift_performance_results.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "drift_evaluation")

# --- DEFINIZIONE DEI GRUPPI DI METRICHE ---
GROUP_1 = {
    'name': "Morfologia e Non-Linearità (Ultima Finestra 10s)",
    'filename': "drift_group1_morphology.png",
    'metrics': ['MAE_10s', 'Pearson_10s', 'Wasserstein_10s', 'SampEn_Error_10s', 'LZ_Complexity_Error_10s']
}

GROUP_2 = {
    'name': "Metriche Cliniche e Fisiologiche (Errore in ms sul Cumulativo)",
    'filename': "drift_group2_clinical.png",
    'metrics': ['Error_Cumulative_HRV_RMSSD', 'Error_Cumulative_QRS_ms', 'Error_Cumulative_PR_ms', 'Error_Cumulative_QT_ms']
}

GROUP_3 = {
    'name': "Metriche Cumulative (Media su Intera Generazione)",
    'filename': "drift_group3_cumulative.png",
    'metrics': ['Cumulative_RMSD', 'Cumulative_CosineSim']
}

HORIZONS = ["1m", "30m", "1h", "6h", "12h", "24h"]

def load_and_prepare_data():
    """Legge il JSON, filtra i collassi numerici e prepara il DataFrame."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"File risultati non trovato: {RESULTS_PATH}")

    with open(RESULTS_PATH, 'r') as f:
        all_results = json.load(f)

    records = []
    explosion_counts = {}

    for horizon in HORIZONS:
        if horizon not in all_results or not all_results[horizon]:
            continue
            
        data = all_results[horizon]
        exploded = 0
        total_patients = len(data)
        
        for entry in data:
            metrics = entry.get('metrics', {})
            
            # Controllo divergenza numerica (Exposure Bias Explosion)
            if pd.isna(metrics.get('MAE_10s')) or metrics.get('MAE_10s') is None:
                exploded += 1
                continue 
                
            # Aggiungiamo i dati in formato 'Tidy' (Lungo) perfetto per Seaborn
            for m_key, m_val in metrics.items():
                if m_val is not None and not pd.isna(m_val):
                    records.append({
                        'Horizon': horizon,
                        'Subject': entry['subject_id'],
                        'Metric': m_key,
                        'Value': float(m_val)
                    })
                    
        explosion_counts[horizon] = (exploded, total_patients)

    df = pd.DataFrame(records)
    return df, explosion_counts

def plot_metric_group(df, group_config):
    """Genera una singola figure con 6 subplot per un gruppo specifico di metriche."""
    # Filtriamo il DataFrame tenendo solo le metriche di questo gruppo
    df_group = df[df['Metric'].isin(group_config['metrics'])].copy()
    
    # Per una migliore formattazione visiva eliminiamo suffissi troppo lunghi
    df_group['Metric'] = df_group['Metric'].str.replace('_10s', '').str.replace('Error_Cumulative_', '').str.replace('Cumulative_', '')
    
    # Configuriamo lo stile Seaborn
    sns.set_theme(style="whitegrid", context="paper")
    
    # 2 righe x 3 colonne
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(group_config['name'], fontsize=18, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # Ordine in cui far comparire le metriche
    metric_order = [m.replace('_10s', '').replace('Error_Cumulative_', '').replace('Cumulative_', '') for m in group_config['metrics']]
    
    for i, horizon in enumerate(HORIZONS):
        ax = axes[i]
        
        # Estraiamo i dati di questo specifico orizzonte
        df_horizon = df_group[df_group['Horizon'] == horizon]
        
        if df_horizon.empty:
            ax.set_title(f"Orizzonte: {horizon} (Nessun dato valido)", color='red')
            ax.axis('off')
            continue
            
        # Generiamo i boxplot affiancati
        sns.boxplot(
            data=df_horizon, 
            x='Metric', 
            y='Value', 
            ax=ax,
            order=metric_order,
            palette="Set2", 
            showfliers=True, # Mostriamo gli outlier (punti estremi)
            boxprops=dict(alpha=0.8)
        )
        
        ax.set_title(f"Orizzonte: {horizon}", fontsize=14, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Valore Metrica" if i % 3 == 0 else "")
        
        # Ruotiamo le etichette sull'asse x per farle leggere bene
        ax.tick_params(axis='x', rotation=25)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(OUTPUT_DIR, group_config['filename'])
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"📊 Grafico '{group_config['name']}' salvato in: {plot_path}")

def main():
    print("=" * 60)
    print(" GENERAZIONE GRAFICI: DRIFT EVALUATION")
    print("=" * 60)
    
    try:
        df, explosion_counts = load_and_prepare_data()
    except FileNotFoundError as e:
        print(e)
        return

    # Stampiamo il report delle esplosioni numeriche
    print("\nREPORT DIVERGENZA NUMERICA (EXPLOSION RATE)")
    print("-" * 60)
    for horizon in HORIZONS:
        if horizon in explosion_counts:
            expl, tot = explosion_counts[horizon]
            pct = (expl / tot) * 100 if tot > 0 else 0
            print(f" - Orizzonte {horizon:<4}: {expl}/{tot} pazienti collassati ({pct:.1f}%)")
    print("-" * 60 + "\n")

    # Generiamo i 3 grafici
    plot_metric_group(df, GROUP_1)
    plot_metric_group(df, GROUP_2)
    plot_metric_group(df, GROUP_3)

    print("\n✅ Generazione completata con successo!")

if __name__ == "__main__":
    main()