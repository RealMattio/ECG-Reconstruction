import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR = "./mimic_clinical_data"
DIAGNOSES_CSV     = "DIAGNOSES_ICD.csv"
PATHOLOGIES_CSV   = "unique_present_pathologies.csv"
REPORT_FILE       = "exclusion_report.txt"
CHART_FILE        = "exclusion_barchart.png"
# ----------------------

EXCLUDED_KEYWORDS = [
    'heart', 'cardiac', 'myocard', 'infarct', 'ischemi', 'arrhythmia',
    'fibrillat', 'tachycardia', 'bradycardia', 'bundle branch', 'block',
    'pacemaker', 'defibrillator', 'valv', 'coronary', 'angina', 'ventric',
    'atrial', 'potassium', 'hypokalemia', 'hyperkalemia', 'calcium',
    'prolonged qt', 'st elevation', 'st depression', 'pericard',
]

CARDIAC_V_CODES = frozenset({'V421', 'V422', 'V433', 'V450', 'V533'})

# Ordered list of (category_key, label for report, short label for chart)
CATEGORY_META = [
    ('circulatory',         'Circulatory System Diseases (ICD-9: 390-459)',
                            'Circulatory System\nDiseases (390-459)'),
    ('congenital',          'Congenital Cardiac Anomalies (ICD-9: 745-747)',
                            'Congenital Cardiac\nAnomalies (745-747)'),
    ('electrolytic',        'Electrolytic Disorders (ICD-9: 276)',
                            'Electrolytic\nDisorders (276)'),
    ('cardiovascular_signs','Cardiovascular Signs & Symptoms (ICD-9: 785)',
                            'Cardiovascular\nSigns & Symptoms (785)'),
    ('cardiac_devices',     'Cardiac Interventions & Devices (V-codes)',
                            'Cardiac Interventions\n& Devices (V-codes)'),
    ('keyword_only',        'Other Cardiac Keyword Matches (text-based)',
                            'Other Cardiac\nKeyword Matches'),
]

PHASE1_KEYS = frozenset({
    'circulatory', 'congenital', 'electrolytic',
    'cardiovascular_signs', 'cardiac_devices',
})


def get_exclusion_category(icd9_code: str, title: str) -> str | None:
    """
    Returns the exclusion category key for a diagnosis, or None if it is safe.
    Mirrors the logic in filter_ecg_patologies.py: V-codes first, then
    numeric ranges, then keyword fallback.
    """
    code = str(icd9_code).strip().upper().replace('.', '')
    title_l = str(title).strip().lower() if title else ''

    if code.startswith('V'):
        if any(code.startswith(vc) for vc in CARDIAC_V_CODES):
            return 'cardiac_devices'
        return None

    if code.startswith('E'):
        return None

    try:
        macro = int(code[:3])
        if 390 <= macro <= 459:
            return 'circulatory'
        if 745 <= macro <= 747:
            return 'congenital'
        if macro == 276:
            return 'electrolytic'
        if macro == 785:
            return 'cardiovascular_signs'
    except ValueError:
        pass

    if any(kw in title_l for kw in EXCLUDED_KEYWORDS):
        return 'keyword_only'

    return None


def load_data(data_dir: str):
    diag_path = os.path.join(data_dir, DIAGNOSES_CSV)
    path_path = os.path.join(data_dir, PATHOLOGIES_CSV)

    print(f" -> Loading {DIAGNOSES_CSV}...")
    df_diag = pd.read_csv(diag_path, dtype={'SUBJECT_ID': int, 'ICD9_CODE': str})
    df_diag = df_diag.dropna(subset=['SUBJECT_ID', 'ICD9_CODE'])

    if os.path.exists(path_path):
        print(f" -> Loading {PATHOLOGIES_CSV} for diagnosis titles...")
        df_titles = pd.read_csv(path_path, dtype={'ICD9_CODE': str})[['ICD9_CODE', 'LONG_TITLE']]
        df = df_diag.merge(df_titles, on='ICD9_CODE', how='left')
        df['LONG_TITLE'] = df['LONG_TITLE'].fillna('')
    else:
        print(f"    [WARNING] {PATHOLOGIES_CSV} not found — keyword filtering will be skipped.")
        df = df_diag.copy()
        df['LONG_TITLE'] = ''

    return df


def compute_exclusions(df: pd.DataFrame):
    print(" -> Classifying each diagnosis by exclusion category...")
    df = df.copy()
    df['category'] = df.apply(
        lambda r: get_exclusion_category(r['ICD9_CODE'], r['LONG_TITLE']), axis=1
    )

    total_patients = df['SUBJECT_ID'].nunique()

    patients_phase1 = set(df.loc[df['category'].isin(PHASE1_KEYS), 'SUBJECT_ID'])
    patients_keyword = set(df.loc[df['category'] == 'keyword_only', 'SUBJECT_ID'])
    patients_phase2_only = patients_keyword - patients_phase1
    all_excluded = set(df.loc[df['category'].notna(), 'SUBJECT_ID'])
    retained = total_patients - len(all_excluded)

    counts_per_category = {
        key: int(df.loc[df['category'] == key, 'SUBJECT_ID'].nunique())
        for key, _, _ in CATEGORY_META
    }

    return {
        'total_patients': total_patients,
        'patients_phase1': len(patients_phase1),
        'patients_phase2_only': len(patients_phase2_only),
        'all_excluded': len(all_excluded),
        'retained': retained,
        'counts_per_category': counts_per_category,
    }


def write_report(stats: dict, data_dir: str):
    report_path = os.path.join(data_dir, REPORT_FILE)
    width = 72

    lines = [
        "=" * width,
        "  MIMIC-III ECG COHORT SELECTION — EXCLUSION REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * width,
        "",
        "  PIPELINE SUMMARY",
        "-" * width,
        f"  {'Total patients in MIMIC-III database:':<48} {stats['total_patients']:>8,}",
        f"  {'Excluded — Phase 1 (ICD-9 code ranges + V-codes):':<48} {stats['patients_phase1']:>8,}",
        f"  {'Excluded — Phase 2 (keyword matching only):':<48} {stats['patients_phase2_only']:>8,}",
        f"  {'Total excluded patients:':<48} {stats['all_excluded']:>8,}",
        f"  {'Retained (healthy ECG cohort):':<48} {stats['retained']:>8,}",
        "",
        "  BREAKDOWN BY EXCLUSION CATEGORY",
        "  (a patient may appear in multiple categories)",
        "-" * width,
        "",
    ]

    for key, report_label, _ in CATEGORY_META:
        n = stats['counts_per_category'][key]
        lines.append(f"  {report_label:<55} {n:>6,} patients")

    lines += [
        "",
        "=" * width,
    ]

    report_text = "\n".join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text + "\n")

    print("\n" + report_text)
    print(f"\n -> Report saved to: {report_path}")


def draw_barchart(stats: dict, data_dir: str):
    keys    = [m[0] for m in CATEGORY_META]
    labels  = [m[2] for m in CATEGORY_META]
    values  = [stats['counts_per_category'][k] for k in keys]
    x       = np.arange(len(values))

    # Color palette: blue gradient + warm tones for non-ICD categories
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F4A582', '#D6604D']

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.bar(x, values, width=0.62, color=colors,
                  edgecolor='white', linewidth=0.8, zorder=3)

    # Value labels
    y_max = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.012,
            f'{val:,}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#222222',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, ha='center', linespacing=1.35)
    ax.set_ylabel('Number of Excluded Patients', fontsize=13, labelpad=10)
    ax.set_title(
        'MIMIC-III Cohort Selection: Excluded Patients per Cardiac Pathology Category',
        fontsize=12.5, fontweight='bold', pad=14,
    )

    ax.set_xlim(-0.55, len(values) - 0.45)
    ax.set_ylim(0, y_max * 1.18)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='#bbbbbb', zorder=0)
    ax.set_axisbelow(True)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(axis='both', color='#888888')

    # Legend distinguishing the two filtering phases
    patch_icd  = plt.Rectangle((0, 0), 1, 1, fc='#4393C3', ec='white')
    patch_kw   = plt.Rectangle((0, 0), 1, 1, fc='#D6604D', ec='white')
    patch_dev  = plt.Rectangle((0, 0), 1, 1, fc='#F4A582', ec='white')
    ax.legend(
        [patch_icd, patch_dev, patch_kw],
        ['Phase 1 — ICD-9 numeric ranges', 'Phase 1 — Cardiac V-codes', 'Phase 2 — Keyword matching'],
        fontsize=11, loc='upper right', frameon=True,
        framealpha=0.9, edgecolor='#cccccc',
    )

    plt.tight_layout(pad=1.5)

    chart_path = os.path.join(data_dir, CHART_FILE)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f" -> Bar chart saved to: {chart_path}")


def main():
    print("=" * 60)
    print("  MIMIC-III: Cohort Exclusion Report Generator")
    print("=" * 60)

    df = load_data(DATA_DIR)
    stats = compute_exclusions(df)
    write_report(stats, DATA_DIR)
    draw_barchart(stats, DATA_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
