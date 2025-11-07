"""
Microbiome and Smoking Cessation Analysis Pipeline
===================================================

This script implements the analysis pipeline for investigating the relationship
between gut microbiome composition, short-chain fatty acids (SCFA), and smoking cessation.

Study Groups:
- Group A: Non-smokers
- Group B: Smokers who did not quit
- Group C: Smokers who quit

"""

# ============================================================================
# IMPORTS
# ============================================================================

# Data processing
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind, shapiro
from mimic_da import apply_mimic
from statsmodels.stats.multitest import multipletests
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tableone import TableOne

# Visualization
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

# System
import os
from pathlib import Path

# Machine learning
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

# Explainability
import shap

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the analysis"""

    # Base folder where input files are located and outputs will be saved
    BASE_FOLDER = "Datas/OK195"

    # Output directory (relative to base_folder)
    @property
    def OUTPUT_DIR(self):
        return f"{self.BASE_FOLDER}/analysis_result"

    # File paths (relative to base_folder)
    METADATA_FULL = "metadata_OK195.csv"
    METADATA_TIME_A = "metadata_OK195__T[A].csv"
    SCFA_PATH = "SCFAs_ordered.csv"
    MIPMLP_A_BC = "0-MIPMLP_A-BC_T[A]_overlap.csv"
    METADATA_A_BC = "0-metadata_A-BC_T[A]_overlap.csv"
    MIPMLP_B_C = "1-MIPMLP_B-C_T[A]_overlap.csv"
    METADATA_B_C = "1-metadata_B-C_T[A]_overlap.csv"
    MIPMLP_MEAN = "2-MIPMLP_mean_ordered.csv"
    METADATA_MIPMLP_MEAN = "2-metadata_ordered_correspond_to_MIPMLP_mean.csv"

    # Feature definitions
    SCFA_COLUMNS = [
        'Acetic acid', 'Propionic acid', 'I-Butytic acid',
        'Butyric acid', 'I-Valeric acid', 'Valeric acid'
    ]

    CLINICAL_COLUMNS = [
        'AGE', 'BMI',
        'Cort (awakening)', 'Cort (awakening+30)',
        'Cort (awakening response)', 'Cort (Night)',
        'FEV1/FVC (Z)', 'COppm'
    ]

    ACTIGRAPHY_COLUMNS = [
        'Acti_Start Time_D', 'Acti_End Time_D', 'Acti_Duration',
        'Acti_Onset Latency', 'Acti_Efficiency', 'Acti_WASO', 'Acti_%Sleep'
    ]

    PSG_COLUMNS = [
        'PSG_SleepEfficiency', 'PSG_SustainedSleepEff', 'PSG_SleepLatency', 'PSG_SleepLatencyN2',
        'PSG_DeepSleepLatency', 'PSG_REMlatency', 'PSG_SleepStageChangeIndex', 'PSG_Wake',
        'PSG_WakeIndex', 'PSG_REM', 'PSG_N2', 'PSG_N4', 'PSG_LightSleep', 'PSG_DeepSleep'
    ]


    QUESTIONNAIRE_COLUMNS = [
        'ISI_Sum', 'NRS_Sum', 'PSQI_total_Sum',
        'PSAS_Somatic_Sum', 'PSAS_Cognitive_Sum',
        'Depression_Beck_Sum', 'Anxiety_STAI_State_Sum'
    ]

    COLUMNS_FOR_CORRELATIONS = ['AGE', 'BMI', 'COppm', 'PSAS_Somatic_Sum', 'PSAS_Cognitive_Sum', 'Anxiety_STAI_State_Sum', 'PSQI_total_Sum',]

    # Analysis parameters
    TARGET_COLUMN = 'Tag'  # Group labels (A, B, C)
    RANDOM_STATE = 42
    ALPHA = 0.05  # Statistical significance threshold

    # Machine learning parameters
    SVM_KERNEL = 'rbf'
    SVM_GAMMA = 1
    SVM_PROBABILITY = True


# ============================================================================
# OUTPUT DIRECTORY MANAGEMENT
# ============================================================================

def ensure_output_dir():
    """
    Create output directory if it doesn't exist.

    Returns:
    --------
    Path: Path object for the output directory
    """
    config = Config()
    output_path = Path(config.OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_output_path(filename):
    """
    Get full path for output file.

    Parameters:
    -----------
    filename : str
        Name of the output file

    Returns:
    --------
    str: Full path to output file
    """
    config = Config()
    return str(Path(config.OUTPUT_DIR) / filename)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_all_data(min_nonzero=5):
    """
    Load all required datasets for the analysis and filter sparse microbes
    AFTER subsetting to Time == 'A' (to avoid false sparse retention).
    """

    print("Loading data files...")

    config = Config()
    base_path = Path(config.BASE_FOLDER)
    data = {}

    try:
        # Load all files
        data['metadata_full'] = pd.read_csv(base_path / Config.METADATA_FULL, index_col=0)
        print(f"✓ Loaded full metadata: {data['metadata_full'].shape}")

        data['metadata_A'] = pd.read_csv(base_path / Config.METADATA_TIME_A, index_col=0)
        print(f"✓ Loaded Time A metadata: {data['metadata_A'].shape}")

        data['scfa'] = pd.read_csv(base_path / Config.SCFA_PATH, index_col=0)
        print(f"✓ Loaded SCFA data: {data['scfa'].shape}")

        data['microbiome_a_bc'] = pd.read_csv(base_path / Config.MIPMLP_A_BC, index_col=0)
        data['metadata_a_bc'] = pd.read_csv(base_path / Config.METADATA_A_BC, index_col=0)
        print(f"✓ Loaded A-BC microbiome data: {data['microbiome_a_bc'].shape}")

        data['microbiome_bc'] = pd.read_csv(base_path / Config.MIPMLP_B_C, index_col=0)
        data['metadata_bc'] = pd.read_csv(base_path / Config.METADATA_B_C, index_col=0)
        print(f"✓ Loaded B-C microbiome data: {data['microbiome_bc'].shape}")

        data['microbiome_mean'] = pd.read_csv(base_path / Config.MIPMLP_MEAN, index_col=0)
        data['metadata_mean'] = pd.read_csv(base_path / Config.METADATA_MIPMLP_MEAN, index_col=0)
        print(f"✓ Loaded mean microbiome data: {data['microbiome_mean'].shape}")

        # Subset to Time == 'A' BEFORE filtering
        print(f"\nFiltering samples by Time == 'A'...")
        data['metadata_mean'] = data['metadata_mean'][data['metadata_mean']['Time'] == 'A']

        # Align sample IDs
        common_ids = data['microbiome_mean'].index.intersection(data['metadata_mean'].index)
        data['microbiome_mean'] = data['microbiome_mean'].loc[common_ids]
        data['metadata_mean'] = data['metadata_mean'].loc[common_ids]
        print(f"✓ Retained {len(common_ids)} samples with Time == 'A'")

        # Now apply sparse filtering AFTER subsetting
        print("\n" + "=" * 50)
        print("FILTERING SPARSE MICROBES (after Time == 'A' subset)")
        print("=" * 50)

        print(f"\nFiltering microbiome_bc (B-C groups)...")
        data['microbiome_bc'] = filter_sparse_microbes(
            data['microbiome_bc'],
            min_nonzero=min_nonzero
        )

        print(f"\nFiltering microbiome_mean (Time A only)...")
        data['microbiome_mean'] = filter_sparse_microbes(
            data['microbiome_mean'],
            min_nonzero=min_nonzero
        )

    except FileNotFoundError as e:
        print(f"✗ Error loading file: {e}")
        print(f"Please ensure all data files are in: {base_path}")
        return None

    print("\n" + "=" * 50)
    print("Data loading complete!")
    print("=" * 50 + "\n")

    return data



def filter_sparse_microbes(microbiome_df, min_nonzero=5):
    """
    Filter out sparse microbes with too few non-zero values.

    Parameters:
    -----------
    microbiome_df : pd.DataFrame
        Microbiome abundance data (samples x taxa)
    min_nonzero : int
        Minimum number of non-zero values required to keep a taxon (default: 5)

    Returns:
    --------
    pd.DataFrame: Filtered microbiome data
    """
    print(f"  Initial number of taxa: {microbiome_df.shape[1]}")
    print(f"  Filtering criteria: Keep taxa with ≥ {min_nonzero} non-zero values")

    # Convert all columns to numeric, replacing non-numeric values with NaN
    microbiome_numeric = microbiome_df.copy()
    for col in microbiome_numeric.columns:
        microbiome_numeric[col] = pd.to_numeric(microbiome_numeric[col], errors='coerce')

    # Fill NaN with 0 (treating missing/non-numeric as zero abundance)
    microbiome_numeric = microbiome_numeric.fillna(0)

    # Count non-zero values for each taxon (column)
    #nonzero_counts = (microbiome_numeric > 0).sum(axis=0)
    nonzero_counts = (microbiome_numeric > 1e-6).sum(axis=0)

    # Filter taxa that meet the threshold
    taxa_to_keep = nonzero_counts[nonzero_counts >= min_nonzero].index
    microbiome_filtered = microbiome_numeric[taxa_to_keep].copy()

    n_removed = microbiome_df.shape[1] - microbiome_filtered.shape[1]

    print(f"  Taxa removed: {n_removed}")
    print(f"  Taxa retained: {microbiome_filtered.shape[1]}")
    print(f"  Percentage retained: {100 * microbiome_filtered.shape[1] / microbiome_df.shape[1]:.1f}%")

    # Show statistics about removed taxa
    if n_removed > 0:
        removed_taxa = microbiome_df.columns.difference(taxa_to_keep)
        removed_nonzero = nonzero_counts[removed_taxa]
        print(
            f"  Removed taxa had {removed_nonzero.mean():.1f} non-zero values on average (max: {removed_nonzero.max():.0f})")

    # Show statistics about retained taxa
    if len(taxa_to_keep) > 0:
        retained_nonzero = nonzero_counts[taxa_to_keep]
        print(
            f"  Retained taxa have {retained_nonzero.mean():.1f} non-zero values on average (min: {retained_nonzero.min():.0f}, max: {retained_nonzero.max():.0f})")

    return microbiome_filtered

# ============================================================================
# DIFFERENTIAL ABUNDANCE TESTING
# ============================================================================

def run_mimic_test(microbiome_df, metadata_df, folder, comparison='A_vs_BC'):
    """
    Run mi-Mic test for differential abundance analysis.

    Parameters:
    -----------
    microbiome_df : pd.DataFrame
        Microbiome abundance data (samples x taxa)
    metadata_df : pd.DataFrame
        Metadata with group labels
    comparison : str
        Type of comparison: 'A_vs_BC' or 'B_vs_C'

    Returns:
    --------
    tuple: (taxonomy_selected, samba_output) - significant taxa and results
    """

    print(f"\n{'=' * 60}")
    print(f"MI-MIC TEST - {comparison}")
    print(f"{'=' * 60}")

    # Filter and prepare data based on comparison type
    if comparison == 'A_vs_BC':
        samples_to_keep = metadata_df.index
        microbiome_filtered = microbiome_df.loc[samples_to_keep]
        metadata_filtered = metadata_df.loc[samples_to_keep]
        groups = metadata_filtered[Config.TARGET_COLUMN].replace(
            {'A': 0, 'B': 1, 'C': 1}
        )
        group_names = {0: 'A', 1: 'BC'}

    elif comparison == 'B_vs_C':
        samples_to_keep = metadata_df[
            metadata_df[Config.TARGET_COLUMN].isin(['B', 'C'])
        ].index
        microbiome_filtered = microbiome_df.loc[samples_to_keep]
        metadata_filtered = metadata_df.loc[samples_to_keep]
        groups = metadata_filtered[Config.TARGET_COLUMN].replace(
            {'B': 0, 'C': 1}
        )
        group_names = {0: 'B', 1: 'C'}

    else:
        raise ValueError("comparison must be 'A_vs_BC' or 'B_vs_C'")

    print(f"Testing {microbiome_filtered.shape[1]} taxa across {len(microbiome_filtered)} samples")
    print(f"Group distribution: {groups.value_counts().to_dict()}")

    # Run mi-Mic
    print("\nRunning mi-Mic test...")

    # Step 1: Preprocess
    # processed = apply_mimic(
    #     folder=Config.OUTPUT_DIR,
    #     tag=groups,
    #     mode="preprocess",
    #     preprocess=False,
    #     rawData=microbiome_filtered,
    #     taxnomy_group='sub PCA'
    # )
    processed = microbiome_filtered

    taxonomy_selected = None
    samba_output = None
    folder = f"Datas/OK195/analysis_result/mimic_output_{comparison}"

    if processed is not None:
        print("✓ Preprocessing completed")

        # Step 2: Run differential abundance test
        taxonomy_selected, samba_output = apply_mimic(
            folder=folder,
            tag=groups,
            eval="man",
            threshold_p=0.05,
            processed=processed,
            apply_samba=True,
            save=True
        )

        if taxonomy_selected is not None:
            print(f"\n{'=' * 60}")
            print(f"MI-MIC RESULTS SUMMARY - {comparison}")
            print(f"{'=' * 60}")
            print(f"\nSignificant taxa:")


            # Step 3: Generate plots
            print("\nGenerating mi-Mic plots...")
            try:
                apply_mimic(
                    folder=folder,
                    tag=groups,
                    mode="plot",
                    tax=taxonomy_selected,
                    eval="man",
                    sis='fdr_bh',
                    samba_output=samba_output,
                    save=True,
                    threshold_p=0.05,
                    THRESHOLD_edge=0.5
                )
                print(f"✓ mi-Mic plots saved for {comparison}")
            except Exception as e:
                print(f"⚠ Warning: Could not generate plots: {e}")
        else:
            print(f"\n{'=' * 60}")
            print(f"MI-MIC RESULTS SUMMARY - {comparison}")
            print(f"{'=' * 60}")
            print("No significant taxa found after FDR correction.")
    else:
        print("✗ Preprocessing failed")

    return taxonomy_selected, samba_output


def plot_mann_whitney_heatmap(significant_taxa, comparison, results_df):
    """
    Create heatmap of significant taxa colored by U-statistic strength.

    Parameters:
    -----------
    significant_taxa : pd.DataFrame
        DataFrame with significant taxa results
    comparison : str
        Type of comparison ('A_vs_BC' or 'B_vs_C')
    results_df : pd.DataFrame
        Full results dataframe for context
    """
    if len(significant_taxa) == 0:
        print(f"  No significant taxa to plot for {comparison}")
        return

    print(f"\nCreating heatmap for {len(significant_taxa)} significant taxa...")

    # Prepare data for heatmap
    # Sort by U-statistic for better visualization
    sig_sorted = significant_taxa.sort_values('U_statistic', ascending=False).copy()

    # Create a matrix: taxa x metrics
    # We'll show: U-statistic, -log10(q-value), and fold-change
    sig_sorted['neg_log_qvalue'] = -np.log10(sig_sorted['q_value'] + 1e-300)  # Avoid log(0)
    sig_sorted['log2_fold_change'] = np.log2(sig_sorted['fold_change'])

    # Prepare matrix for heatmap
    heatmap_data = sig_sorted[['U_statistic', 'neg_log_qvalue', 'log2_fold_change']].T
    heatmap_data.columns = sig_sorted['taxon'].values

    # Normalize each row to 0-1 for better color scaling
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data.T).T,
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )

    # Create figure
    fig_height = max(4, len(significant_taxa) * 0.3)
    fig_width = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with original values (not normalized) for better interpretation
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        cbar_kws={'label': 'Value'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        fmt='.2f',
        annot=False  # Don't annotate if many taxa
    )

    # Customize
    ax.set_xlabel('Significant Taxa', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title(f'Significant Taxa - {comparison}\n(Colored by metric strength)',
                 fontsize=14, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=10)

    # Update y-axis labels for clarity
    ax.set_yticklabels(['U-statistic', '-log₁₀(q-value)', 'log₂(Fold-change)'])

    plt.tight_layout()

    # Save
    output_file = get_output_path(f'mann_whitney_heatmap_{comparison}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved as '{output_file}'")
    plt.close()


def plot_mann_whitney_abundance_heatmap(microbiome_df, metadata_df, significant_taxa, comparison, renamed=False):
    """
    Compact, proportional abundance heatmap for significant taxa.
    Keeps consistent per-taxon height even when only few taxa exist.
    """
    if len(significant_taxa) == 0:
        print(f"No significant taxa for {comparison}")
        return

    taxa = significant_taxa['taxon'].values
    if renamed:
        taxa = rename_microbes(taxa)

    # Prepare metadata and grouping
    if comparison == 'A_vs_BC':
        samples = metadata_df.index
        metadata = metadata_df.loc[samples].copy()
        metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN].replace({'A': 'A', 'B': 'BC', 'C': 'BC'})
        groups = ['A', 'BC']
    else:
        samples = metadata_df[metadata_df[Config.TARGET_COLUMN].isin(['B', 'C'])].index
        metadata = metadata_df.loc[samples].copy()
        metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN]
        groups = ['B', 'C']

    # Filter data
    abund = microbiome_df.loc[samples, significant_taxa['taxon']].T
    abund.index = taxa
    abund = abund[metadata.sort_values('Group_Binary').index]
    abund = np.log10(abund + 1e-6)

    # Dynamic compact figure sizing
    n_taxa = len(taxa)
    per_taxon_height = 0.35  # controls height per row
    base_height = 2.5        # fixed base height
    fig_height = base_height + per_taxon_height * n_taxa
    fig_height = min(fig_height, 8)   # cap so large plots don’t get huge
    fig_width = max(8, len(abund.columns) * 0.12)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        abund, cmap="cool",
        cbar_kws={'label': 'log₁₀(Abundance + 10⁻⁶)'},
        linewidths=0.1, linecolor='white',
        xticklabels=False, ax=ax
    )

    # Separator between groups
    gchange = (metadata.loc[abund.columns, 'Group_Binary'] !=
               metadata.loc[abund.columns, 'Group_Binary'].shift()).cumsum()
    for change in gchange.unique()[1:]:
        pos = list(abund.columns).index((gchange == change).idxmax())
        ax.axvline(x=pos, color='black', lw=1.1, ls='--')

    # Labels & formatting
    label_y = 'Significant Taxa (Renamed)' if renamed else 'Significant Taxa'
    title = f'Abundance of Significant Taxa - {comparison}' + (' (Renamed)' if renamed else '')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Samples ({groups[0]} | {groups[1]})', fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # Save
    suffix = "_renamed" if renamed else ""
    out = get_output_path(f"mann_whitney_abundance_heatmap{suffix}_{comparison}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {out}")
    plt.close()



def rename_microbes(features):
    """
    Rename microbe features from full taxonomy to readable format.

    Parameters:
    -----------
    features : list or array
        List of full taxonomy strings

    Returns:
    --------
    list: Renamed features in readable format
    """
    renamed_features = []
    for feature in features:
        parts = feature.split(';')
        Class = next((part.split('__')[1] for part in parts if part.startswith('c__')), '')
        order = next((part.split('__')[1] for part in parts if part.startswith('o__')), '')
        family = next((part.split('__')[1] for part in parts if part.startswith('f__')), '')
        genus = next((part.split('__')[1] for part in parts if part.startswith('g__')), '')
        species = next((part.split('__')[1] for part in parts if part.startswith('s__')), '')

        if species != '' and genus != '':
            renamed_features.append(f"{genus} (g), {species} (s)")
        elif species == '' and genus != '':
            renamed_features.append(f"{genus} (g), __(s)")
        elif genus == '' and family != '':
            renamed_features.append(f"{family} (f), __(g), __(s)")
        elif family == '' and order != '':
            renamed_features.append(f"{order} (o), __(f), __(g), __(s)")
        elif order == '':
            renamed_features.append(f"{Class} (c), __(o), __(f), __(g), __(s)")
        else:
            # Fallback if nothing found
            renamed_features.append(feature)

    return renamed_features


def plot_mann_whitney_heatmap_renamed(significant_taxa, comparison, results_df):
    """
    Create heatmap of significant taxa with renamed (readable) microbe names.

    Parameters:
    -----------
    significant_taxa : pd.DataFrame
        DataFrame with significant taxa results
    comparison : str
        Type of comparison ('A_vs_BC' or 'B_vs_C')
    results_df : pd.DataFrame
        Full results dataframe for context
    """
    if len(significant_taxa) == 0:
        print(f"  No significant taxa to plot renamed heatmap for {comparison}")
        return

    print(f"\nCreating renamed heatmap for {len(significant_taxa)} significant taxa...")

    # Prepare data for heatmap
    # Sort by U-statistic for better visualization
    sig_sorted = significant_taxa.sort_values('U_statistic', ascending=False).copy()

    # Rename taxa
    original_taxa = sig_sorted['taxon'].values
    renamed_taxa = rename_microbes(original_taxa)

    # Create a matrix: taxa x metrics
    # We'll show: U-statistic, -log10(q-value), and fold-change
    sig_sorted['neg_log_qvalue'] = -np.log10(sig_sorted['q_value'] + 1e-300)  # Avoid log(0)
    sig_sorted['log2_fold_change'] = np.log2(sig_sorted['fold_change'])

    # Prepare matrix for heatmap
    heatmap_data = sig_sorted[['U_statistic', 'neg_log_qvalue', 'log2_fold_change']].T
    heatmap_data.columns = renamed_taxa  # Use renamed taxa

    # Create figure
    fig_height = max(4, len(significant_taxa) * 0.3)
    fig_width = max(12, len(significant_taxa) * 0.5)  # Wider for longer names
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with original values (not normalized) for better interpretation
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        cbar_kws={'label': 'Value'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        fmt='.2f',
        annot=False  # Don't annotate if many taxa
    )

    # Customize
    ax.set_xlabel('Significant Taxa (Renamed)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title(f'Significant Taxa - {comparison} (Readable Names)\n(Colored by metric strength)',
                 fontsize=14, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)

    # Update y-axis labels for clarity
    ax.set_yticklabels(['U-statistic', '-log₁₀(q-value)', 'log₂(Fold-change)'])

    plt.tight_layout()

    # Save
    output_file = get_output_path(f'mann_whitney_heatmap_renamed_{comparison}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Renamed heatmap saved as '{output_file}'")
    plt.close()


def plot_mann_whitney_abundance_heatmap_renamed(microbiome_df, metadata_df, significant_taxa, comparison):
    """
    Compact, proportional abundance heatmap for significant taxa (renamed microbes).
    Keeps consistent per-taxon height even when only a few taxa exist.
    """
    if len(significant_taxa) == 0:
        print(f"No significant taxa for {comparison}")
        return

    print(f"\nCreating renamed abundance heatmap for {len(significant_taxa)} taxa...")

    # Prepare taxa names
    taxa = significant_taxa['taxon'].values
    renamed_taxa = rename_microbes(taxa)

    # Prepare metadata and grouping
    if comparison == 'A_vs_BC':
        samples = metadata_df.index
        metadata = metadata_df.loc[samples].copy()
        metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN].replace({'A': 'A', 'B': 'BC', 'C': 'BC'})
        groups = ['A', 'BC']
    else:
        samples = metadata_df[metadata_df[Config.TARGET_COLUMN].isin(['B', 'C'])].index
        metadata = metadata_df.loc[samples].copy()
        metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN]
        groups = ['B', 'C']

    # Filter data
    abund = microbiome_df.loc[samples, significant_taxa['taxon']].T
    abund.index = renamed_taxa
    abund = abund[metadata.sort_values('Group_Binary').index]
    abund = np.log10(abund + 1e-6)

    # Dynamic compact figure sizing
    n_taxa = len(renamed_taxa)
    per_taxon_height = 0.1
    base_height = 2.5
    fig_height = base_height + per_taxon_height * n_taxa
    fig_height = min(fig_height, 8)
    fig_width = max(8, len(abund.columns) * 0.12)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color palette
    sns.heatmap(
        abund, cmap="cool",
        cbar_kws={'label': 'log₁₀(Abundance + 10⁻⁶)'},
        linewidths=0.1, linecolor='white',
        xticklabels=False, ax=ax
    )

    # Add group separators
    gchange = (metadata.loc[abund.columns, 'Group_Binary'] !=
               metadata.loc[abund.columns, 'Group_Binary'].shift()).cumsum()
    for change in gchange.unique()[1:]:
        pos = list(abund.columns).index((gchange == change).idxmax())
        ax.axvline(x=pos, color='black', lw=1.1, ls='--')

    # Labels and formatting
    ax.set_title(f'Abundance of Significant Taxa - {comparison} (Renamed)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Samples ({groups[0]} | {groups[1]})', fontsize=11)
    ax.set_ylabel('Significant Taxa (Renamed)', fontsize=11)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # Save
    out = get_output_path(f"mann_whitney_abundance_heatmap_renamed_{comparison}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {out}")
    plt.close()


def run_mann_whitney_test(microbiome_df, metadata_df, comparison='A_vs_BC', fdr_alpha=0.05):
    """
    Run Mann-Whitney U test with FDR correction for differential abundance.

    Parameters:
    -----------
    microbiome_df : pd.DataFrame
        Microbiome abundance data (samples x taxa)
    metadata_df : pd.DataFrame
        Metadata with group labels
    comparison : str
        Type of comparison: 'A_vs_BC' or 'B_vs_C'
    fdr_alpha : float
        FDR significance threshold (default: 0.05)

    Returns:
    --------
    tuple: (results_df, significant_taxa) - full results and significant taxa only
    """

    print(f"\n{'=' * 60}")
    print(f"MANN-WHITNEY U TEST - {comparison}")
    print(f"{'=' * 60}")

    # Filter and prepare data based on comparison type
    if comparison == 'A_vs_BC':
        samples_to_keep = metadata_df.index
        microbiome_filtered = microbiome_df.loc[samples_to_keep]
        metadata_filtered = metadata_df.loc[samples_to_keep]
        groups = metadata_filtered[Config.TARGET_COLUMN].replace(
            {'A': 'A', 'B': 'BC', 'C': 'BC'}
        )
        group1_name = 'A'
        group2_name = 'BC'

    elif comparison == 'B_vs_C':
        samples_to_keep = metadata_df[
            metadata_df[Config.TARGET_COLUMN].isin(['B', 'C'])
        ].index
        microbiome_filtered = microbiome_df.loc[samples_to_keep]
        metadata_filtered = metadata_df.loc[samples_to_keep]
        groups = metadata_filtered[Config.TARGET_COLUMN]
        group1_name = 'B'
        group2_name = 'C'

    else:
        raise ValueError("comparison must be 'A_vs_BC' or 'B_vs_C'")

    print(f"Testing {microbiome_filtered.shape[1]} taxa across {len(microbiome_filtered)} samples")
    print(f"Group distribution: {groups.value_counts().to_dict()}")

    # Perform Mann-Whitney U test for each taxon
    results_list = []

    print("\nRunning Mann-Whitney U tests...")
    for taxon in microbiome_filtered.columns:
        # Get abundances for each group
        if comparison == 'A_vs_BC':
            group1_data = microbiome_filtered.loc[
                metadata_filtered[Config.TARGET_COLUMN] == 'A', taxon
            ].values
            group2_data = microbiome_filtered.loc[
                metadata_filtered[Config.TARGET_COLUMN].isin(['B', 'C']), taxon
            ].values
        else:  # B_vs_C
            group1_data = microbiome_filtered.loc[
                metadata_filtered[Config.TARGET_COLUMN] == 'B', taxon
            ].values
            group2_data = microbiome_filtered.loc[
                metadata_filtered[Config.TARGET_COLUMN] == 'C', taxon
            ].values

        # Perform Mann-Whitney U test
        try:
            stat, pval = mannwhitneyu(group1_data, group2_data, alternative='two-sided')

            # Calculate mean abundances
            mean_group1 = group1_data.mean()
            mean_group2 = group2_data.mean()

            # Determine enrichment
            if mean_group1 > mean_group2:
                enriched_in = group1_name
                fold_change = mean_group1 / (mean_group2 + 1e-10)
            else:
                enriched_in = group2_name
                fold_change = mean_group2 / (mean_group1 + 1e-10)

            results_list.append({
                'taxon': taxon,
                'U_statistic': stat,
                'p_value': pval,
                f'mean_{group1_name}': mean_group1,
                f'mean_{group2_name}': mean_group2,
                'fold_change': fold_change,
                'enriched_in': enriched_in
            })
        except Exception as e:
            print(f"  Warning: Could not test {taxon}: {e}")
            continue

    # Create results dataframe
    results_df = pd.DataFrame(results_list)

    # Apply FDR correction (Benjamini-Hochberg)
    print(f"\nApplying FDR correction (Benjamini-Hochberg method)...")
    reject, qvals, _, _ = multipletests(
        results_df['p_value'].values,
        alpha=fdr_alpha,
        method='fdr_bh'
    )

    results_df['q_value'] = qvals
    results_df['significant'] = reject

    # Sort by q-value
    results_df = results_df.sort_values('q_value')

    # Get significant results
    significant_taxa = results_df[results_df['significant'] == True].copy()

    print(f"\n{'=' * 60}")
    print(f"MANN-WHITNEY RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total taxa tested: {len(results_df)}")
    print(f"Significant taxa (FDR < {fdr_alpha}): {len(significant_taxa)}")

    if len(significant_taxa) > 0:
        print(f"\nSignificant taxa:")
        for idx, row in significant_taxa.iterrows():
            print(f"  {row['taxon']}: q-value={row['q_value']:.4f}, "
                  f"fold-change={row['fold_change']:.2f}, enriched in {row['enriched_in']}")
    else:
        print(f"\nNo significant taxa found after FDR correction (alpha={fdr_alpha}).")

    # Save results CSV
    output_file = get_output_path(f'mann_whitney_results_{comparison}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Full Mann-Whitney results saved to '{output_file}'")

    # Generate heatmaps for significant taxa
    if len(significant_taxa) > 0:
        # Original heatmaps with full taxonomy names
        print("\n--- Generating heatmaps with full taxonomy names ---")
        plot_mann_whitney_heatmap(significant_taxa, comparison, results_df)
        plot_mann_whitney_abundance_heatmap(
            microbiome_df,
            metadata_df,
            significant_taxa,
            comparison
        )

        # Renamed heatmaps with readable names
        print("\n--- Generating heatmaps with readable names ---")
        plot_mann_whitney_heatmap_renamed(significant_taxa, comparison, results_df)
        plot_mann_whitney_abundance_heatmap_renamed(
            microbiome_df,
            metadata_df,
            significant_taxa,
            comparison
        )

    return results_df, significant_taxa


def plot_combined_abundance_heatmap(microbiome_df, metadata_df, mw_sig, mimic_folder, comparison, renamed=False):
    """
    Create a combined abundance heatmap for taxa significant in either Mann-Whitney or mi-Mic.
    Keeps only taxa that contain 'f__' (family-level annotation).
    """
    mimic_path = Path(mimic_folder) / "just_mimic.csv"
    if not mimic_path.exists():
        print(f"⚠️ No mi-Mic file found at {mimic_path}")
        return

    # Load mi-Mic taxa
    mimic_df = pd.read_csv(mimic_path)
    mimic_df.rename(columns={mimic_df.columns[0]: 'taxon'}, inplace=True)

    # Merge taxa lists
    mw_taxa = set(mw_sig['taxon']) if mw_sig is not None and len(mw_sig) > 0 else set()
    mimic_taxa = set(mimic_df['taxon'])
    combined_taxa = list(mw_taxa.union(mimic_taxa))

    # Keep only taxa with family-level annotation (contain 'f__')
    combined_taxa = [t for t in combined_taxa if isinstance(t, str) and 'f__' in t]

    if len(combined_taxa) == 0:
        print(f"⚠️ No taxa with 'f__' annotation found for {comparison}")
        return

    print(f"\nCreating combined abundance heatmap ({len(combined_taxa)} taxa with family-level annotation)...")

    # Rename if needed
    taxa_display = rename_microbes(combined_taxa) if renamed else combined_taxa

    # Prepare metadata & groups
    if comparison == 'A_vs_BC':
        samples = metadata_df.index
        metadata = metadata_df.loc[samples].copy()
        metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN].replace({'A': 'A', 'B': 'BC', 'C': 'BC'})
        groups = ['A', 'BC']
    else:
        samples = metadata_df[metadata_df[Config.TARGET_COLUMN].isin(['B', 'C'])].index
        metadata = metadata_df.loc[samples].copy()
        metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN]
        groups = ['B', 'C']

    # Fuzzy match taxa between mi-Mic / MW names and microbiome_df columns
    matched_taxa = []
    for t in combined_taxa:
        hits = [col for col in microbiome_df.columns if t in col]
        matched_taxa.extend(hits)

    # Deduplicate while preserving order
    combined_taxa = list(dict.fromkeys(matched_taxa))

    if len(combined_taxa) == 0:
        print(f"⚠️ No taxa matched in microbiome_df columns after fuzzy search for {comparison}")
        return

    print(f"✓ Matched {len(combined_taxa)} taxa to microbiome_df (fuzzy matching).")

    # ✅ Recreate taxa_display AFTER fuzzy matching to keep them aligned
    taxa_display = rename_microbes(combined_taxa) if renamed else combined_taxa

    # Filter microbiome data
    abund = microbiome_df.loc[samples, combined_taxa].T

    # Safety check for alignment
    if len(abund.index) != len(taxa_display):
        print(f"⚠️ Length mismatch fixed automatically: {len(abund.index)} taxa vs {len(taxa_display)} names")
        taxa_display = taxa_display[:len(abund.index)]

    abund.index = taxa_display
    abund = abund[metadata.sort_values('Group_Binary').index]
    abund = np.log10(abund + 1e-6)

    # Dynamic figure sizing
    n_taxa = len(combined_taxa)
    per_taxon_height = 0.1
    base_height = 2.5
    fig_height = min(base_height + per_taxon_height * n_taxa, 8)
    fig_width = max(8, len(abund.columns) * 0.12)

    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        abund,
        cmap="cool",
        cbar_kws={'label': 'log₁₀(Abundance + 10⁻⁶)'},
        linewidths=0.1,
        linecolor='white',
        xticklabels=False,
        ax=ax
    )

    # Group separators
    gchange = (metadata.loc[abund.columns, 'Group_Binary'] !=
               metadata.loc[abund.columns, 'Group_Binary'].shift()).cumsum()
    for change in gchange.unique()[1:]:
        pos = list(abund.columns).index((gchange == change).idxmax())
        ax.axvline(x=pos, color='black', lw=1.1, ls='--')

    # Labels
    title_suffix = " (Renamed)" if renamed else ""
    ax.set_title(f'Combined Abundance Heatmap - {comparison}{title_suffix}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Samples ({groups[0]} | {groups[1]})', fontsize=11)
    ax.set_ylabel('Significant Taxa', fontsize=11)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # Save
    suffix = "_renamed" if renamed else ""
    out = get_output_path(f"combined_abundance_heatmap{suffix}_{comparison}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Combined abundance heatmap saved as '{out}'")
    plt.close()

def plot_combined_abundance_heatmap_dual(
    microbiome_df, metadata_df,
    mw_sig_a_bc, mw_sig_b_c,
    mimic_folder_a_bc, mimic_folder_b_c,
    renamed=True
):
    """
    Create one vertically aligned figure with:
    - Top:  A_vs_BC (larger)
    - Bottom: B_vs_C (smaller)
    Shared colorbar for both.
    """

    # Figure layout
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1.2], hspace=0.5)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # Helper to extract abundances
    def extract_abundance(comparison, mw_sig, mimic_folder):
        mimic_path = Path(mimic_folder) / "just_mimic.csv"
        if not mimic_path.exists():
            print(f"⚠️ No mi-Mic file found for {comparison} at {mimic_path}")
            return None, None, None, None

        mimic_df = pd.read_csv(mimic_path)
        mimic_df.rename(columns={mimic_df.columns[0]: 'taxon'}, inplace=True)

        mw_taxa = set(mw_sig['taxon']) if mw_sig is not None and len(mw_sig) > 0 else set()
        mimic_taxa = set(mimic_df['taxon'])
        combined_taxa = list(mw_taxa.union(mimic_taxa))
        combined_taxa = [t for t in combined_taxa if isinstance(t, str) and 'f__' in t]
        if not combined_taxa:
            return None, None, None, None

        taxa_display = rename_microbes(combined_taxa) if renamed else combined_taxa

        if comparison == 'A_vs_BC':
            samples = metadata_df.index
            metadata = metadata_df.loc[samples].copy()
            metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN].replace({'A': 'A', 'B': 'BC', 'C': 'BC'})
        else:
            samples = metadata_df[metadata_df[Config.TARGET_COLUMN].isin(['B', 'C'])].index
            metadata = metadata_df.loc[samples].copy()
            metadata['Group_Binary'] = metadata[Config.TARGET_COLUMN]

        matched_taxa = []
        for t in combined_taxa:
            hits = [col for col in microbiome_df.columns if t in col]
            matched_taxa.extend(hits)
        combined_taxa = list(dict.fromkeys(matched_taxa))
        if not combined_taxa:
            return None, None, None, None

        taxa_display = rename_microbes(combined_taxa) if renamed else combined_taxa

        abund = microbiome_df.loc[samples, combined_taxa].T
        abund.index = taxa_display[:len(abund)]
        abund = abund[metadata.sort_values('Group_Binary').index]
        abund = np.log10(abund + 1e-6)

        return abund, metadata, taxa_display, comparison

    # Extract both datasets
    abund_a, meta_a, taxa_a, _ = extract_abundance("A_vs_BC", mw_sig_a_bc, mimic_folder_a_bc)
    abund_b, meta_b, taxa_b, _ = extract_abundance("B_vs_C", mw_sig_b_c, mimic_folder_b_c)

    if abund_a is None or abund_b is None:
        print("⚠️ One or both abundance matrices missing; cannot plot dual heatmap.")
        return

    # Determine shared color scale
    shared_vmin = min(abund_a.min().min(), abund_b.min().min())
    shared_vmax = max(abund_a.max().max(), abund_b.max().max())

    # Plot both heatmaps
    def plot_single(ax, abund, metadata, comparison, groups, compact=False):
        sns.heatmap(
            abund,
            cmap="Greys",
            cbar=False,  # disable individual bars
            vmin=shared_vmin, vmax=shared_vmax,
            linewidths=0.1, linecolor='white',
            xticklabels=False,
            ax=ax
        )

        gchange = (metadata.loc[abund.columns, 'Group_Binary'] !=
                   metadata.loc[abund.columns, 'Group_Binary'].shift()).cumsum()
        for change in gchange.unique()[1:]:
            pos = list(abund.columns).index((gchange == change).idxmax())
            ax.axvline(x=pos, color='black', lw=1.1, ls='--')

        ax.set_title(f"{comparison}" if renamed else comparison,
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Significant Taxa', fontsize=18)
        ax.set_xlabel(f'Samples ({groups[0]} | {groups[1]})', fontsize=18)
        ax.tick_params(axis='y', labelrotation=0, labelsize=14)

        if compact:
            #ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va='center')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')

    plot_single(ax_top, abund_a, meta_a, "A_vs_BC", ['A', 'BC'], compact=False)
    plot_single(ax_bottom, abund_b, meta_b, "B_vs_C", ['B', 'C'], compact=True)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.5])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=shared_vmin, vmax=shared_vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('log₁₀(Abundance + 10⁻⁶)', fontsize=11)

    # Save
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for shared colorbar
    out = get_output_path("combined_abundance_heatmap_dual_sharedscale.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Combined aligned dual heatmap saved with shared colorbar as '{out}'")
    plt.show()





# ============================================================================
# TABLE 1: DEMOGRAPHIC AND CLINICAL CHARACTERISTICS
# ============================================================================

def create_table_one(df_metadata, numeric_columns):
    """
    Create Table 1 with demographic and clinical characteristics.
    Includes Mann-Whitney U tests for A vs BC and B vs C comparisons.

    Parameters:
    -----------
    df_metadata : pd.DataFrame
        Metadata dataframe with 'Tag' column (A, B, or C)
    numeric_columns : list
        List of numeric column names to include

    Returns:
    --------
    tuple: (summary_table, pvalue_table)
    """

    print("Creating Table 1: Demographic and Clinical Characteristics")
    print("-" * 60)

    # Create a copy to avoid modifying original
    df = df_metadata.copy()

    # Convert numeric columns to numeric type
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create TableOne for overall summary
    table1 = TableOne(
        data=df,
        columns=numeric_columns,
        groupby=Config.TARGET_COLUMN,
        pval=False
    )

    print("\nOverall Summary by Group:")
    print(table1)

    # Create comparison groups
    df["Group_A_vs_BC"] = df[Config.TARGET_COLUMN].replace(
        {"A": "A", "B": "BC", "C": "BC"}
    )
    df["Group_B_vs_C"] = df[Config.TARGET_COLUMN].replace(
        {"A": np.nan}
    )

    # Calculate Mann-Whitney U test p-values
    print("\nPerforming Mann-Whitney U tests...")
    pvals = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue

        # A vs BC comparison
        group_A = df[df["Group_A_vs_BC"] == "A"][col].dropna()
        group_BC = df[df["Group_A_vs_BC"] == "BC"][col].dropna()

        if len(group_A) > 0 and len(group_BC) > 0:
            stat, pval = mannwhitneyu(group_A, group_BC, alternative='two-sided')
            pvals[f"{col} (A vs BC)"] = pval

        # B vs C comparison
        group_B = df[df["Group_B_vs_C"] == "B"][col].dropna()
        group_C = df[df["Group_B_vs_C"] == "C"][col].dropna()

        if len(group_B) > 0 and len(group_C) > 0:
            stat, pval = mannwhitneyu(group_B, group_C, alternative='two-sided')
            pvals[f"{col} (B vs C)"] = pval

    # Create p-value dataframe
    pval_df = pd.DataFrame.from_dict(
        pvals, orient='index', columns=["Mann-Whitney p-value"]
    )
    pval_df['Significant'] = pval_df['Mann-Whitney p-value'] < Config.ALPHA

    print("\nMann-Whitney U Test Results:")
    print(pval_df)

    # Save results
    output_file = get_output_path('table1_pvalues.csv')
    pval_df.to_csv(output_file)
    print(f"\n✓ Results saved to '{output_file}'")

    return table1, pval_df


# ============================================================================
# SCFA ANALYSIS
# ============================================================================

def statistical_analyze_scfa(df_metadata, scfa_columns):
    """
    Compare SCFA levels between groups using Mann-Whitney U tests.

    Parameters:
    -----------
    df_metadata : pd.DataFrame
        Metadata including SCFA measurements
    scfa_columns : list
        List of SCFA column names

    Returns:
    --------
    pd.DataFrame: Statistical test results
    """
    print("\n" + "=" * 60)
    print("SHORT-CHAIN FATTY ACID (SCFA) ANALYSIS")
    print("=" * 60)

    results = []

    for scfa in scfa_columns:
        if scfa not in df_metadata.columns:
            continue

        print(f"\nAnalyzing {scfa}:")

        # A vs BC
        group_A = df_metadata[df_metadata[Config.TARGET_COLUMN] == 'A'][scfa].dropna()
        group_BC = df_metadata[
            df_metadata[Config.TARGET_COLUMN].isin(['B', 'C'])
        ][scfa].dropna()

        if len(group_A) > 0 and len(group_BC) > 0:
            stat, pval = mannwhitneyu(group_A, group_BC, alternative='two-sided')
            results.append({
                'SCFA': scfa,
                'Comparison': 'A vs BC',
                'Group_A_mean': group_A.mean(),
                'Group_BC_mean': group_BC.mean(),
                'U_statistic': stat,
                'p_value': pval,
                'Significant': pval < Config.ALPHA
            })
            print(f"  A vs BC: p={pval:.4f}")

        # B vs C
        group_B = df_metadata[df_metadata[Config.TARGET_COLUMN] == 'B'][scfa].dropna()
        group_C = df_metadata[df_metadata[Config.TARGET_COLUMN] == 'C'][scfa].dropna()

        if len(group_B) > 0 and len(group_C) > 0:
            stat, pval = mannwhitneyu(group_B, group_C, alternative='two-sided')
            results.append({
                'SCFA': scfa,
                'Comparison': 'B vs C',
                'Group_B_mean': group_B.mean(),
                'Group_C_mean': group_C.mean(),
                'U_statistic': stat,
                'p_value': pval,
                'Significant': pval < Config.ALPHA
            })
            print(f"  B vs C: p={pval:.4f}")

    results_df = pd.DataFrame(results)
    output_file = get_output_path('scfa_analysis_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ SCFA analysis results saved to '{output_file}'")

    return results_df


def remove_scfa_outliers(df, scfa_columns, z_thresh=3.0):
    """Remove SCFA outliers based on per-feature Z-scores (>|z_thresh|)."""
    df_clean = df.copy()
    z_scores = ((df_clean[scfa_columns] - df_clean[scfa_columns].mean()) /
                df_clean[scfa_columns].std(ddof=0))
    mask = (np.abs(z_scores) > z_thresh)
    outlier_counts = mask.sum()
    total_outliers = int(mask.sum().sum())

    for scfa, count in outlier_counts.items():
        if count > 0:
            print(f"  Removed {count} outliers from {scfa}")

    df_clean[scfa_columns] = df_clean[scfa_columns].mask(mask)
    df_clean = df_clean.dropna(subset=scfa_columns, how='all')

    print(f"✓ Total outliers removed: {total_outliers}")
    return df_clean


def analyze_scfa(df_metadata, scfa_columns, target_col='Tag', alpha=0.05):
    """
    Compare SCFA levels between groups using adaptive statistical testing.
    Builds a table in the style of Supplementary Table 3 (median±range or mean±SE).
    """

    print("\n" + "=" * 60)
    print("SHORT-CHAIN FATTY ACID (SCFA) ANALYSIS")
    print("=" * 60)

    df = df_metadata.copy()
    print("\nOutlier filtering (|Z| > 3)...")
    df = remove_scfa_outliers(df, scfa_columns)

    results_A_BC = []
    results_B_C = []

    # ----------------------------------------------------------
    # Main SCFA analysis loop
    # ----------------------------------------------------------
    for scfa in scfa_columns:
        if scfa not in df.columns:
            continue

        group_A = df[df[target_col] == 'A'][scfa].dropna()
        group_B = df[df[target_col] == 'B'][scfa].dropna()
        group_C = df[df[target_col] == 'C'][scfa].dropna()
        group_BC = df[df[target_col].isin(['B', 'C'])][scfa].dropna()

        # --- A vs BC ---
        if len(group_A) > 2 and len(group_BC) > 2:
            _, pA = shapiro(group_A)
            _, pBC = shapiro(group_BC)
            normal = (pA >= 0.05) and (pBC >= 0.05)
            if normal:
                test = "t-test"
                stat, pval = ttest_ind(group_A, group_BC, equal_var=False)
                A_stat = f"{group_A.mean():.3f} ± {group_A.std(ddof=1)/np.sqrt(len(group_A)):.3f}"
                BC_stat = f"{group_BC.mean():.3f} ± {group_BC.std(ddof=1)/np.sqrt(len(group_BC)):.3f}"
            else:
                test = "Mann–Whitney"
                stat, pval = mannwhitneyu(group_A, group_BC)
                A_stat = f"{np.median(group_A):.3f} ({np.min(group_A):.3f}, {np.max(group_A):.3f})"
                BC_stat = f"{np.median(group_BC):.3f} ({np.min(group_BC):.3f}, {np.max(group_BC):.3f})"

            results_A_BC.append({
                "SCFA": scfa,
                "A": A_stat,
                "BC": BC_stat,
                "p_value": pval,
                "Test": test,
                "N_A": len(group_A),
                "N_BC": len(group_BC)
            })

        # --- B vs C ---
        if len(group_B) > 2 and len(group_C) > 2:
            _, pB = shapiro(group_B)
            _, pC = shapiro(group_C)
            normal = (pB >= 0.05) and (pC >= 0.05)
            if normal:
                test = "t-test"
                stat, pval = ttest_ind(group_B, group_C, equal_var=False)
                B_stat = f"{group_B.mean():.3f} ± {group_B.std(ddof=1)/np.sqrt(len(group_B)):.3f}"
                C_stat = f"{group_C.mean():.3f} ± {group_C.std(ddof=1)/np.sqrt(len(group_C)):.3f}"
            else:
                test = "Mann–Whitney"
                stat, pval = mannwhitneyu(group_B, group_C)
                B_stat = f"{np.median(group_B):.3f} ({np.min(group_B):.3f}, {np.max(group_B):.3f})"
                C_stat = f"{np.median(group_C):.3f} ({np.min(group_C):.3f}, {np.max(group_C):.3f})"

            results_B_C.append({
                "SCFA": scfa,
                "B": B_stat,
                "C": C_stat,
                "p_value": pval,
                "Test": test,
                "N_B": len(group_B),
                "N_C": len(group_C)
            })

    # ----------------------------------------------------------
    # Merge to final display table (and fix column naming)
    # ----------------------------------------------------------
    df_A_BC = pd.DataFrame(results_A_BC)
    df_B_C = pd.DataFrame(results_B_C)

    # Merge on SCFA only
    df_final = pd.merge(df_A_BC, df_B_C, on="SCFA", how="outer", suffixes=("_A_BC", "_B_C"))

    # Keep only the relevant columns in the right order
    df_display = df_final[[
        "SCFA",
        "A",
        "BC",
        "p_value_A_BC",
        "B",
        "C",
        "p_value_B_C"
    ]].copy()

    # Rename columns to match Table 3 format
    df_display.columns = [
        "SCFA",
        "A - Non-smokers",
        "B,C - Smokers",
        "p-value (A vs BC)",
        "B - Smokers (non-cessation)",
        "C - Smokers (cessation)",
        "p-value (B vs C)"
    ]

    # Add N row at top
    N_row = pd.DataFrame({
        "SCFA": ["N"],
        "A - Non-smokers": [len(df[df[target_col] == 'A'])],
        "B,C - Smokers": [len(df[df[target_col].isin(['B', 'C'])])],
        "p-value (A vs BC)": [""],
        "B - Smokers (non-cessation)": [len(df[df[target_col] == 'B'])],
        "C - Smokers (cessation)": [len(df[df[target_col] == 'C'])],
        "p-value (B vs C)": [""]
    })

    final_table = pd.concat([N_row, df_display], ignore_index=True)

    # Save
    output_file = get_output_path("Supp_Table3_SCFA_results.csv")
    final_table.to_csv(output_file, index=False)
    print(f"\n✓ Table formatted and saved: {output_file}")


    # Visualizations
    plot_scfa_heatmap(df, scfa_columns, target_col)
    plot_scfa_pca(df, scfa_columns, target_col)

    return final_table


def plot_scfa_heatmap(df, scfa_columns, target_col):
    """Create a z-scored SCFA heatmap across participants."""
    df_plot = df.dropna(subset=scfa_columns)
    X = df_plot[scfa_columns]
    X_z = (X - X.mean()) / X.std(ddof=0)
    X_z[target_col] = df_plot[target_col].values

    ordered = X_z.sort_values(by=target_col)
    sns.clustermap(
        ordered[scfa_columns],
        cmap="vlag",
        row_cluster=False,
        col_cluster=True,
        figsize=(8, 6)
    )
    #plt.suptitle("SCFA Heatmap (Z-scored)", fontsize=14, fontweight='bold')
    filename = "scfa_heatmap.png"
    output_file = get_output_path(filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SCFA heatmap saved: {output_file}")


def plot_scfa_pca(df, scfa_columns, target_col):
    """Perform PCA on SCFA data and plot group separation."""
    df_clean = df.dropna(subset=scfa_columns + [target_col])
    X = df_clean[scfa_columns]
    y = df_clean[target_col]

    X_z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_z)

    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df[target_col] = y.values

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=target_col, s=80, palette="coolwarm", edgecolor="black")
    plt.title(f"SCFA PCA (Variance explained: PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    filename = "scfa_pca.png"
    output_file = get_output_path(filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SCFA PCA plot saved: {output_file}")

# ============================================================================
# MACHINE LEARNING: SMOKING CESSATION PREDICTION
# ============================================================================

def build_preprocessor(numeric_columns, categorical_columns=None, binary_columns=None):
    """
    Build preprocessing pipeline for machine learning.

    Parameters:
    -----------
    numeric_columns : list
        Numeric feature names
    categorical_columns : list, optional
        Categorical feature names
    binary_columns : list, optional
        Binary feature names

    Returns:
    --------
    ColumnTransformer: Preprocessing pipeline
    """
    transformers = []

    if numeric_columns:
        transformers.append((
            'numeric',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]),
            numeric_columns
        ))

    if categorical_columns:
        transformers.append((
            'categorical',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]),
            categorical_columns
        ))

    if binary_columns:
        transformers.append((
            'binary',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]),
            binary_columns
        ))

    return ColumnTransformer(transformers=transformers, remainder='drop')


def predict_smoking_cessation(
    df_data,
    feature_columns,
    target_column='Tag',
    title="Model",
    use_pca=False,
    variance_threshold=0.90,
    preprocessor=None,
    plot_roc=True,
    plot_shap=True,
    plot_decision=True
):
    """
    Predict smoking cessation using SVM with Leave-One-Out Cross-Validation.

    Integrates optional preprocessing (ColumnTransformer), PCA (auto component
    selection), ROC, SHAP (summary + SE bar), and 2D SVM decision boundary visualization.
    """

    print(f"\n{'=' * 60}")
    print(f"SMOKING CESSATION PREDICTION - {title}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(df_data)}")
    print(f"  Class 0: {(df_data[target_column] == 0).sum()}")
    print(f"  Class 1: {(df_data[target_column] == 1).sum()}")

    # Separate features and target
    X = df_data[feature_columns].copy()
    y = df_data[target_column].copy()

    # ===== PCA (automatic number of components) =====
    # ===== PCA (with log transformation + automatic component selection) =====
    # ===== PCA (log + Z-score normalization + automatic component selection) =====
    if use_pca:
        print(
            f"\nApplying log(x + ε) transformation, Z-score normalization, and PCA to capture ≥ {variance_threshold * 100:.1f}% variance...")

        # Fill missing values if any
        if X.isnull().values.any():
            print(f"  → Detected {X.isnull().sum().sum()} missing values. Imputing with small pseudocount (1e-6).")
            X = X.fillna(1e-6)

        # Log-transform (stabilize variance, handle zeros)
        epsilon = 1e-6
        X_log = np.log(X + epsilon)
        print("  → Applied log(x + 1e-6) transformation.")

        # Z-score normalization (feature-wise standardization)
        X_z = (X_log - X_log.mean()) / X_log.std(ddof=0)
        print("  → Applied Z-score normalization (mean=0, std=1 per feature).")

        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_z)
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1

        print(f"  → Selected {n_components} components (cover {cumulative_var[n_components - 1]:.3f} variance).")

        X = pd.DataFrame(
            X_pca[:, :n_components],
            index=X.index,
            columns=[f'PC{i + 1}' for i in range(n_components)]
        )

    # ===== Leave-One-Out Cross-Validation =====
    loo = LeaveOneOut()
    y_true, y_pred_prob = [], []
    trained_models = []

    print("\nPerforming Leave-One-Out Cross-Validation...")

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply preprocessor if provided
        if preprocessor is not None:
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)
        else:
            X_train_t, X_test_t = X_train, X_test

        model = SVC(probability=True, kernel='rbf', gamma=1, random_state=42)
        model.fit(X_train_t, y_train)

        y_prob = model.predict_proba(X_test_t)[:, 1]
        y_true.append(y_test.values[0])
        y_pred_prob.append(y_prob[0])
        trained_models.append(model)

        if (fold + 1) % 5 == 0 or fold == len(X) - 1:
            print(f"  Processed {fold + 1}/{len(X)} folds")

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    # ===== Compute ROC / AUC =====
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC: {roc_auc:.3f}")

    if plot_roc:
        plot_roc_curve(fpr, tpr, roc_auc, title)

    # ===== Decision Boundary (2D only) =====
    if plot_decision and X.shape[1] == 2:
        print("\nPlotting SVM decision boundary (2D space)...")
        plot_svm_decision_surface(X, y, trained_models[-1], title)
    elif plot_decision:
        print("\nSkipping decision boundary: requires exactly 2 dimensions.")

    # ===== SHAP Explanation =====
    if plot_shap and len(trained_models) > 0:
        # If a preprocessor exists, transform full X for SHAP consistency
        if preprocessor is not None:
            X_transformed = preprocessor.fit_transform(X)
            feature_names = getattr(preprocessor, "get_feature_names_out", lambda: X.columns)()
            X_df = pd.DataFrame(X_transformed, columns=feature_names)
        else:
            X_df = X
        explain_model_shap(trained_models[-1], X_df, title)

    return {
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'models': trained_models,
        'features': X.columns.tolist()
    }


# === Plotting & SHAP utilities ===

def plot_roc_curve(fpr, tpr, roc_auc, title):
    """Simple ROC plot in black aesthetic."""
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='black', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title(f"ROC Curve - {title}", fontsize=15, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"roc_curve_{title.replace(' ', '_')}.png"
    output_file = get_output_path(filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved: {output_file}")


def plot_svm_decision_surface(X, y, model, title):
    """Plots the 2D SVM RBF decision boundary and data points."""
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=80)
    plt.title(f"SVM Decision Boundary - {title}", fontsize=15, fontweight='bold')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.tight_layout()

    filename = f"decision_boundary_{title.replace(' ', '_')}.png"
    output_file = get_output_path(filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Decision boundary saved: {output_file}")


def explain_model_shap(model, X, title):
    """SHAP summary plot and quantitative mean ± SE bar chart."""
    print(f"\nGenerating SHAP explanations for {title}...")
    try:
        explainer = shap.KernelExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_class1 = shap_values[1]
        else:
            shap_class1 = shap_values[:, :, 1]

        # ===== Summary plot =====
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_class1, X, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {title}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f"shap_summary_{title.replace(' ', '_')}.png"
        output_file = get_output_path(filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP summary saved: {output_file}")

        # ===== Mean ± SE plot =====
        shap_values_x = np.array(shap_class1)
        mean_shap = np.mean(shap_values_x, axis=0)
        std_shap = np.std(shap_values_x, axis=0)
        se_shap = std_shap / np.sqrt(shap_values_x.shape[0])

        sorted_idx = np.argsort(np.abs(mean_shap))
        sorted_features = X.columns[sorted_idx]
        sorted_means = mean_shap[sorted_idx]
        sorted_se = se_shap[sorted_idx]

        plt.figure(figsize=(10, 7))
        plt.barh(sorted_features, sorted_means, xerr=sorted_se, capsize=5, color='black', alpha=0.7)
        plt.xlabel("Mean SHAP Value ± SE", fontsize=13)
        plt.title(f"Quantitative SHAP Importance - {title}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f"shap_mean_se_{title.replace(' ', '_')}.png"
        output_file = get_output_path(filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP quantitative (mean±SE) saved: {output_file}")

    except Exception as e:
        print(f"  Warning: Could not compute SHAP: {e}")



# ============================================================================
# CORRELATION ANALYSIS: SCFA vs CLINICAL/BEHAVIORAL FEATURES
# ============================================================================
def correlation_matrix(
    df,
    columns_x,
    columns_y=None,
    corr_method=spearmanr,
    alpha=0.05,
    fdr_correct=True,
    figsize=(14, 10)
):
    """
    Compute and visualize correlations between two sets of variables, with optional FDR correction.
    Displays true q-values (FDR-adjusted p-values) for significant correlations.
    """

    if columns_y is None:
        columns_y = columns_x

    corr_matrix = pd.DataFrame(index=columns_x, columns=columns_y, dtype=float)
    pval_matrix = pd.DataFrame(index=columns_x, columns=columns_y, dtype=float)

    print("\n" + "=" * 50)
    print("COMPUTE CORRELATIONS")
    print("=" * 50)

    # --- Compute pairwise correlations
    for i in columns_x:
        for j in columns_y:
            x = df[i].dropna()
            y = df[j].dropna()
            valid_idx = x.index.intersection(y.index)
            if len(valid_idx) > 2:
                stat, pval = corr_method(df.loc[valid_idx, i], df.loc[valid_idx, j])
                corr_matrix.loc[i, j] = stat
                pval_matrix.loc[i, j] = pval
            else:
                corr_matrix.loc[i, j] = np.nan
                pval_matrix.loc[i, j] = np.nan

    # --- FDR correction
    qval_matrix = pval_matrix.copy()
    if fdr_correct:
        pvals_flat = pval_matrix.values.flatten()
        mask_valid = ~np.isnan(pvals_flat)
        adj_pvals = np.full_like(pvals_flat, np.nan)
        _, qvals, _, _ = multipletests(pvals_flat[mask_valid], method='fdr_bh')
        adj_pvals[mask_valid] = qvals
        qval_matrix = pd.DataFrame(
            adj_pvals.reshape(pval_matrix.shape),
            index=pval_matrix.index,
            columns=pval_matrix.columns
        )
    else:
        qval_matrix[:] = pval_matrix.values  # If no correction, q = p

    # --- Significance mask
    significant_mask = qval_matrix < alpha if fdr_correct else pval_matrix < alpha

    # --- Build annotation matrix (show q-values for significant)
    annotations = pd.DataFrame(index=columns_x, columns=columns_y, dtype=str)
    for i in columns_x:
        for j in columns_y:
            r_val = corr_matrix.loc[i, j]
            p_val = pval_matrix.loc[i, j]
            q_val = qval_matrix.loc[i, j]
            if pd.notna(r_val):
                if fdr_correct:
                    if significant_mask.loc[i, j]:
                        label = f"{r_val:.2f}\n(p={p_val:.3f}, q={q_val:.3f})"
                    else:
                        label = f"{r_val:.2f}\n(p={p_val:.3f}, q={q_val:.3f})"
                else:
                    label = f"{r_val:.2f}\n(p={p_val:.3f})"
                annotations.loc[i, j] = label
            else:
                annotations.loc[i, j] = ""

    # --- Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_matrix.astype(float),
        annot=annotations,
        fmt='',
        cmap='coolwarm',
        center=0,
        cbar_kws={'shrink': 0.6},
        annot_kws={'fontsize': 11}
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(
        f"Correlation Matrix ({'FDR-corrected' if fdr_correct else 'uncorrected'}, α={alpha})",
        fontsize=14, fontweight='bold'
    )

    # --- Outline significant cells
    for i, row in enumerate(columns_x):
        for j, col in enumerate(columns_y):
            if significant_mask.loc[row, col]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.tight_layout()

    output_file = get_output_path("Correlations_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlations Heatmap saved as '{output_file}'")

    return corr_matrix, pval_matrix, qval_matrix


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """
    Main analysis pipeline executing all steps.
    """
    print("\n" + "=" * 70)
    print(" MICROBIOME AND SMOKING CESSATION ANALYSIS PIPELINE")
    print("=" * 70)

    # Create output directory
    output_dir = ensure_output_dir()
    config = Config()
    print(f"\n✓ Output directory created/verified: {output_dir}")
    print(f"  All results will be saved to: {config.OUTPUT_DIR}/")
    print(f"  Base folder: {config.BASE_FOLDER}")

    # ========== Step 1: Load Data ==========
    data = load_all_data()
    if data is None:
        print("Error: Could not load data. Exiting.")
        return

    # ========== Step 2: Table 1 - Demographics ==========
    all_numeric = (Config.CLINICAL_COLUMNS + Config.SCFA_COLUMNS +
                   Config.ACTIGRAPHY_COLUMNS + Config.PSG_COLUMNS +
                   Config.QUESTIONNAIRE_COLUMNS)

    table1, pvals = create_table_one(data['metadata_A'], all_numeric)

    # ========== Step 3: SCFA Analysis ==========
    scfa_results = statistical_analyze_scfa(data['metadata_A'], Config.SCFA_COLUMNS)



    # ========== Step 4: Differential Abundance Analysis ==========
    print("\n" + "=" * 70)
    print(" DIFFERENTIAL ABUNDANCE ANALYSIS")
    print("=" * 70)

    # A vs BC Comparison
    mimic_sig_a_bc, mimic_samba_a_bc = run_mimic_test(
        data['microbiome_a_bc'],
        data['metadata_a_bc'],
        comparison='A_vs_BC'
    )

    mw_results_a_bc, mw_sig_a_bc = run_mann_whitney_test(
        data['microbiome_a_bc'],
        data['metadata_a_bc'],
        comparison='A_vs_BC',
        fdr_alpha=0.05
    )

    # plot_combined_abundance_heatmap(
    #     data['microbiome_mean'],
    #     data['metadata_mean'],
    #     mw_sig_a_bc,
    #     mimic_folder="Datas/OK195/analysis_result/mimic_output_A_vs_BC",
    #     comparison="A_vs_BC",
    #     renamed=True
    # )

    # B vs C Comparison
    mimic_sig_b_c, mimic_samba_b_c = run_mimic_test(
        data['microbiome_bc'],
        data['metadata_bc'],
        comparison='B_vs_C'
    )

    mw_results_b_c, mw_sig_b_c = run_mann_whitney_test(
        data['microbiome_bc'],
        data['metadata_bc'],
        comparison='B_vs_C',
        fdr_alpha=0.05
    )

    # plot_combined_abundance_heatmap(
    #     data['microbiome_mean'],
    #     data['metadata_mean'],
    #     mw_sig_b_c,
    #     mimic_folder="Datas/OK195/analysis_result/mimic_output_B_vs_C",
    #     comparison="B_vs_C",
    #     renamed=True
    # )

    # Plot Combined Abundance heatmap
    plot_combined_abundance_heatmap_dual(
        data['microbiome_a_bc'],
        data['metadata_a_bc'],
        mw_sig_a_bc, mw_sig_b_c,
        mimic_folder_a_bc="Datas/OK195/analysis_result/mimic_output_A_vs_BC",
        mimic_folder_b_c="Datas/OK195/analysis_result/mimic_output_B_vs_C",
        renamed=True
    )

    # Summary
    print("\n" + "=" * 70)
    print(" DIFFERENTIAL ABUNDANCE SUMMARY")
    print("=" * 70)
    print(f"\nA vs BC:")
    print(f"  Mann-Whitney significant taxa: {len(mw_sig_a_bc)}")

    print(f"\nB vs C:")
    print(f"  Mann-Whitney significant taxa: {len(mw_sig_b_c)}")

    # ========== Step 5: Machine Learning - Microbiome =========
    # Merge microbiome with metadata for prediction

    # Filter to B and C groups only
    df_microbiome_B_C = data['microbiome_bc'][
        data['metadata_bc'][Config.TARGET_COLUMN].isin(['B', 'C'])]

    df_metadata_B_C = data['metadata_bc'][
        data['metadata_bc'][Config.TARGET_COLUMN].isin(['B', 'C'])]

    # Merge metadata Target column into the microbiome dataframe
    df_microbiome_with_meta = pd.concat([df_microbiome_B_C, df_metadata_B_C[Config.TARGET_COLUMN]],
        axis=1)
    print(df_microbiome_with_meta[Config.TARGET_COLUMN])

    # Encode target: B -> 0, C -> 1
    df_microbiome_with_meta[Config.TARGET_COLUMN] = df_microbiome_with_meta[Config.TARGET_COLUMN].apply(
        lambda x: 0 if x == 'B' else 1
    )

    microbiome_results = predict_smoking_cessation(
        df_microbiome_with_meta,
        data['microbiome_bc'].columns.tolist(),
        target_column=Config.TARGET_COLUMN,
        title="Microbiome_B_vs_C",
        use_pca=False,
        variance_threshold=0.95
    )

    # ========== Step 6: Machine Learning - SCFA B-C ==========
    df_scfa_clean = data['metadata_A'][
        data['metadata_A'][Config.TARGET_COLUMN].isin(['B', 'C'])
    ].dropna(subset=Config.SCFA_COLUMNS).copy()

    df_scfa_clean[Config.TARGET_COLUMN] = df_scfa_clean[Config.TARGET_COLUMN].map({'B': 0, 'C': 1})

    # Preprocessor

    numeric_columns = Config.SCFA_COLUMNS
    binary_columns = []
    categorical_columns = []

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_columns),
            ('binary', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ]), binary_columns)
        ],
        remainder='drop'
    )

    scfa_ml_results = predict_smoking_cessation(
        df_scfa_clean,
        feature_columns=Config.SCFA_COLUMNS,
        target_column=Config.TARGET_COLUMN,
        title="SCFA_B_vs_C",
        use_pca=False,
        preprocessor=preprocessor
    )

    # ========== Step 7: Machine Learning - Clinical Features ==========

    clinical_features = (
            Config.CLINICAL_COLUMNS
            + Config.ACTIGRAPHY_COLUMNS
            + Config.PSG_COLUMNS
            + Config.QUESTIONNAIRE_COLUMNS
    )

    # Filter to B and C groups only
    df_clinical_clean = data['metadata_A'][
        data['metadata_A'][Config.TARGET_COLUMN].isin(['B', 'C'])
    ].dropna(subset=clinical_features).copy()

    # Encode target: B -> 0, C -> 1
    df_clinical_clean[Config.TARGET_COLUMN] = df_clinical_clean[Config.TARGET_COLUMN].map({'B': 0, 'C': 1})

    # ===== Define Preprocessor =====
    # Separate by feature type
    numeric_columns = clinical_features  # Assuming all are numeric; adjust if some are categorical/binary
    binary_columns = []  # Add specific binary columns if you have them
    categorical_columns = []  # Add specific categorical columns if needed

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_columns),
            ('binary', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ]), binary_columns)
        ],
        remainder='drop'
    )

    # ===== Run model =====
    clinical_ml_results = predict_smoking_cessation(
        df_data=df_clinical_clean,
        feature_columns=clinical_features,
        target_column=Config.TARGET_COLUMN,
        title="Clinical_Features",
        use_pca=True,
        variance_threshold=0.90,
        preprocessor=preprocessor
    )

    # ========== Step 8: Machine Learning - A vs BC ==========

    # SCFA

    # Use the full metadata_A (no filtering to B/C)
    df_scfa_A_vs_BC = data['metadata_A'].dropna(subset=Config.SCFA_COLUMNS).copy()

    # Encode target: A -> 0, B or C -> 1
    df_scfa_A_vs_BC[Config.TARGET_COLUMN] = df_scfa_A_vs_BC[Config.TARGET_COLUMN].map(
        lambda x: 0 if x == 'A' else 1
    )

    # Compute Z-scores only for SCFA columns
    z_scores = np.abs(zscore(df_scfa_A_vs_BC[Config.SCFA_COLUMNS], nan_policy='omit'))

    # Keep rows where all SCFAs have |Z| ≤ 3
    mask = (z_scores <= 3).all(axis=1)
    df_scfa_A_vs_BC = df_scfa_A_vs_BC[mask]

    print(f"Removed {np.sum(~mask)} outlier samples, remaining {df_scfa_A_vs_BC.shape[0]} samples.")

    # ===== Define Preprocessor =====
    numeric_columns = Config.SCFA_COLUMNS
    binary_columns = []
    categorical_columns = []

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_columns),
            ('binary', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ]), binary_columns)
        ],
        remainder='drop'
    )

    # ===== Run prediction =====
    scfa_A_vs_BC_results = predict_smoking_cessation(
        df_data=df_scfa_A_vs_BC,
        feature_columns=Config.SCFA_COLUMNS,
        target_column=Config.TARGET_COLUMN,
        title="SCFA_A_vs_BC",
        use_pca=False,
        preprocessor=preprocessor
    )





    # ========== Step 9: SCFA-Clinical Correlations ==========

    # Compute and Add SCFA_sum
    data['metadata_A']['SCFA_sum'] = data['metadata_A'][Config.SCFA_COLUMNS].sum(axis=1)
    Config.SCFA_COLUMNS = Config.SCFA_COLUMNS + ['SCFA_sum']

    # Prepare Data
    corr_matrix, pval_matrix, qval_matrix = correlation_matrix(
                                    data['metadata_A'].dropna(subset=Config.SCFA_COLUMNS).copy(),
                                    columns_x=Config.COLUMNS_FOR_CORRELATIONS,
                                    columns_y=Config.SCFA_COLUMNS,
                                    corr_method=spearmanr,
                                    alpha=0.05,
                                    fdr_correct=True
                                )

    # ======== Step 10: SCFA-General Analyses ===========

    scfa_results = analyze_scfa(
        data['metadata_A'].dropna(subset=Config.SCFA_COLUMNS).copy(),
        Config.SCFA_COLUMNS,
        target_col=Config.TARGET_COLUMN,
        alpha=Config.ALPHA
    )

    # ========== Summary of Results ==========
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE - SUMMARY OF RESULTS")
    print("=" * 70)
    print("\nMachine Learning AUC Scores:")
    print(f"  Microbiome Model: {microbiome_results['auc']:.3f}")
    print(f"  SCFA Model: {scfa_ml_results['auc']:.3f}")
    print(f"  Clinical Features Model: {clinical_ml_results['auc']:.3f}")
    print(f"\nAll results have been saved to: {config.OUTPUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()