# ============================================================================
# 0. GLOBAL PATCH FOR MATPLOTLIB LAYOUT ENGINE
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl

_original_tight_layout = plt.tight_layout


def _patched_tight_layout(*args, **kwargs):
    """Redirect plt.tight_layout() to the current figure to avoid colorbar layout crashes."""
    fig = plt.gcf()
    if fig is not None:
        try:
            fig.tight_layout(*args, **kwargs)
        except RuntimeError as e:
            if 'Colorbar layout' in str(e):
                try:
                    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
                except Exception:
                    pass
            else:
                raise e


plt.tight_layout = _patched_tight_layout

import matplotlib.pyplot as plt_orig
if hasattr(plt_orig, 'tight_layout'):
    plt_orig.tight_layout = _patched_tight_layout

mpl.rcParams['figure.constrained_layout.use'] = False
mpl.rcParams['figure.autolayout'] = False


# ============================================================================
# 1. IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import seaborn as sns

from scipy import stats
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.covariance import GraphicalLasso

import pingouin as pg
from scipy.stats import spearmanr, pearsonr

import networkx as nx
from adjustText import adjust_text
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

st.set_page_config(
    page_title="Proton-Conducting Perovskites Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# 2. CONSTANTS AND REFERENCE DATA
# ============================================================================

# Shannon ionic radii (12-coord for A, 6-coord for B)
IONIC_RADII = {
    'Ba': 1.61, 'Ba2+': 1.61,
    'Sr': 1.44, 'Sr2+': 1.44,
    'Ca': 1.34, 'Ca2+': 1.34,
    'La': 1.36, 'La3+': 1.36,
    'Gd': 1.27, 'Gd3+': 1.27,
    'Nd': 1.27, 'Nd3+': 1.27,
    'Sm': 1.24, 'Sm3+': 1.24,
    'Dy': 1.22, 'Dy3+': 1.22,
    'Ho': 1.20, 'Ho3+': 1.20,
    'Yb': 1.20, 'Yb3+': 1.20,
    'Y': 1.19, 'Y3+': 1.19,
    'Eu': 1.20, 'Eu3+': 1.20,
    'Pr': 1.30, 'Pr3+': 1.30,
    'Tb': 1.23, 'Tb3+': 1.23,
    'Tm': 1.19, 'Tm3+': 1.19,
    'Sc': 1.00, 'Sc3+': 1.00,
    'Fe': 0.92, 'Fe3+': 0.92,
    'Zn': 0.88, 'Zn2+': 0.88,
    'In': 0.94, 'In3+': 0.94,
    'Sn': 0.97, 'Sn4+': 0.97,
    'Pb': 1.29, 'Pb2+': 1.29,
    'Bi': 1.17, 'Bi3+': 1.17,
    'Al': 0.67, 'Al3+': 0.67,
    'Ga': 0.76, 'Ga3+': 0.76,
    'Ge': 0.67, 'Ge4+': 0.67,
    'Si': 0.54, 'Si4+': 0.54,
    'Hf': 0.97, 'Hf4+': 0.97,
    'Ti': 0.74, 'Ti4+': 0.74,
    'Zr': 0.86, 'Zr4+': 0.86,
    'Ce': 0.97, 'Ce4+': 0.97,
    'O': 1.40,
}

ELECTRONEGATIVITY = {
    'Ba': 0.89, 'Sr': 0.95, 'Ca': 1.00, 'Mg': 1.31,
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.20,
    'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25,
    'Yb': 1.10, 'Lu': 1.27, 'Y': 1.22, 'Sc': 1.36,
    'Zr': 1.33, 'Hf': 1.30, 'Ti': 1.54, 'Sn': 1.96,
    'In': 1.78, 'Fe': 1.83, 'Zn': 1.65, 'Pb': 2.33,
    'Bi': 2.02, 'Al': 1.61, 'Ga': 1.81, 'Ge': 2.01,
    'Si': 1.90, 'O': 3.44
}

VALENCES = {
    'Ba': 2, 'Sr': 2, 'Ca': 2, 'Mg': 2,
    'La': 3, 'Ce': 4, 'Pr': 3, 'Nd': 3,
    'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3,
    'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3,
    'Yb': 3, 'Lu': 3, 'Y': 3, 'Sc': 3,
    'Zr': 4, 'Hf': 4, 'Ti': 4, 'Sn': 4,
    'In': 3, 'Fe': 3, 'Zn': 2, 'Pb': 2,
    'Bi': 3, 'Al': 3, 'Ga': 3, 'Ge': 4,
    'Si': 4
}

MOLAR_MASS = {
    'Ba': 137.33, 'Sr': 87.62, 'Ca': 40.08, 'Mg': 24.31,
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93,
    'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93,
    'Yb': 173.05, 'Lu': 174.97, 'Y': 88.91, 'Sc': 44.96,
    'Zr': 91.22, 'Hf': 178.49, 'Ti': 47.87, 'Sn': 118.71,
    'In': 114.82, 'Fe': 55.85, 'Zn': 65.38, 'Pb': 207.20,
    'Bi': 208.98, 'Al': 26.98, 'Ga': 69.72, 'Ge': 72.63,
    'Si': 28.09, 'O': 16.00
}

COLOR_PALETTES = {
    'method': {'dilatometry': '#2C3E50', 'HT XRD': '#E67E22', 'HT ND': '#8E44AD'},
    'B_cation': {'Ce': '#E74C3C', 'Zr': '#3498DB', 'Sn': '#2ECC71', 'Ti': '#F39C12'},
    'A_cation': {'Ba': '#1A5276', 'Sr': '#2471A3', 'Ca': '#5DADE2', 'La': '#F39C12'},
    'continuous': 'viridis',
    'diverging': 'coolwarm',
    'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}

TARGET_VARIABLES = [
    'α·106 (K-1)',
    'β',
    'αav·106 (K-1)',
    'T(bends), °C',
    'α/β',
    'αav_LT/α',
    'αav_HT/α'
]

ALL_CATIONS = list(IONIC_RADII.keys())
ALL_CATIONS.remove('O')

REQUIRED_COLUMNS = ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', '[A\']', '[B\']', '[D1]', '[D2]',
                    'δ', 'method', 'β', '∆T, °C', 'α·106 (K-1)', 'T(bends), °C',
                    'αav·106 (K-1)', 'pH2O', 'Ref']

TOP_DESCRIPTORS = [
    'δ', '[B\']', 'D_total', 'B\'_conc', 'D_B_ratio',
    'rAav', 'rBav', 't', 'D_t', 'octahedral_factor', 'Δr_AB', 'σ²_rB', 'r_ratio_AB',
    'χAav', 'χBav', 'Δχ_AB', 'χ_ratio_AB', 'ionicity_AO', 'ionicity_BO',
    'V_Bav', 'Vo_proxy', 'E_BO',
    'M_Aav', 'M_Bav', 'M_total',
    'Z_eff_B', 'proton_affinity'
]

ADDITIONAL_DESCRIPTORS = [
    'δ_actual', 'χ_total', 'acidity_AO', 'acidity_BO', 'Δacidity',
    'S_config_A', 'S_config_B', 'ΔH_hydr', 'ρ',
    'M_ratio_AB', 'M_rA', 'M_χA',
    'total_dopant', 'δ_χB', 'δ_rB',
    'alpha_beta_ratio', 'T_stab', 'σ²_rA', 'Δr_AB_norm',
    'V_cell', 'octahedral_factor'
]

ALL_DESCRIPTORS_FOR_SELECT = TOP_DESCRIPTORS + ADDITIONAL_DESCRIPTORS
seen = set()
ALL_DESCRIPTORS_FOR_SELECT = [x for x in ALL_DESCRIPTORS_FOR_SELECT if not (x in seen or seen.add(x))]

TERNARY_VERTEX_OPTIONS = [
    '[B\']', '[D1]', '[D2]',
    'D_total', 'δ',
    'rAav', 'rBav', 'χAav', 'χBav',
    'M_Aav', 'M_Bav', 'M_total',
    'V_Bav', 'Vo_proxy', 'E_BO',
    'Z_eff_B', 'proton_affinity'
]

# Full descriptive terms for the "Use entire terms" sidebar option.
FULL_TERMS = {
    'δ': 'Oxygen vacancy concentration',
    'δ_actual': 'Oxygen vacancy concentration (calculated)',
    'δ_χB': 'Vacancy–electronegativity product',
    'δ_rB': 'Vacancy–radius product',
    '[B\']': 'Isovalent B-site dopant concentration',
    '[D1]': 'Acceptor dopant D1 concentration',
    '[D2]': 'Acceptor dopant D2 concentration',
    'B\'_conc': 'Isovalent B-site dopant concentration',
    'D_total': 'Total acceptor dopant concentration',
    'D_B_ratio': 'Acceptor-to-isovalent dopant ratio',
    'total_dopant': 'Total B-site dopant concentration',

    'rAav': 'Average A-site ionic radius',
    'rBav': 'Average B-site ionic radius',
    'r_ratio_AB': 'A-to-B ionic radius ratio',
    't': 'Goldschmidt tolerance factor',
    'D_t': 'Deviation of tolerance factor from unity',
    'octahedral_factor': 'Octahedral factor (rB/rO)',
    'Δr_AB': 'Absolute A–B ionic radius difference',
    'Δr_AB_norm': 'Normalized A–B ionic radius difference',
    'σ²_rA': 'A-site ionic radius variance',
    'σ²_rB': 'B-site ionic radius variance',
    'V_cell': 'Pseudo-cubic unit cell volume',

    'χAav': 'Average A-site electronegativity',
    'χBav': 'Average B-site electronegativity',
    'χ_total': 'Average total electronegativity',
    'Δχ_AB': 'Absolute A–B electronegativity difference',
    'χ_ratio_AB': 'A-to-B electronegativity ratio',
    'χ_ratio_t': 'Electronegativity ratio × tolerance factor',
    'ionicity_AO': 'A–O bond ionicity',
    'ionicity_BO': 'B–O bond ionicity',
    'acidity_AO': 'A–O bond acidity',
    'acidity_BO': 'B–O bond acidity',
    'Δacidity': 'Difference of A–O and B–O bond acidities',

    'V_Bav': 'Average B-site valence',
    'Vo_proxy': 'Oxygen vacancy proxy',
    'E_BO': 'B–O bond Coulomb energy',
    'E_vac': 'Oxygen vacancy formation energy',
    'ΔH_hydr': 'Hydration enthalpy',
    'T_stab': 'Proton thermal stability temperature',
    'S_config_A': 'A-site configurational entropy',
    'S_config_B': 'B-site configurational entropy',

    'M_Aav': 'Average A-site molar mass',
    'M_Bav': 'Average B-site molar mass',
    'M_total': 'Total molar mass',
    'M_ratio_AB': 'A-to-B molar mass ratio',
    'M_rA': 'Molar mass × ionic radius product',
    'M_χA': 'Molar mass × electronegativity product',
    'ρ': 'Relative mass density',

    'Z_eff_B': 'Effective B-site charge',
    'proton_affinity': 'Proton affinity',
    'alpha_beta_ratio': 'α-to-β ratio',

    'α·106 (K-1)': 'Thermal expansion coefficient α (10⁻⁶ K⁻¹)',
    'β': 'Chemical expansion coefficient β',
    'αav·106 (K-1)': 'Average thermal expansion coefficient (10⁻⁶ K⁻¹)',
    'αav_LT': 'Average thermal expansion coefficient (low-T)',
    'αav_HT': 'Average thermal expansion coefficient (high-T)',
    'α/β': 'Thermal-to-chemical expansion ratio',
    'αav_LT/α': 'Average low-T α / α ratio',
    'αav_HT/α': 'Average high-T α / α ratio',
    'T(bends), °C': 'Bend temperatures (°C)',
    'T_bends_1': 'First bend temperature (°C)',
    'T_bends_2': 'Second bend temperature (°C)',
    'T_bends_3': 'Third bend temperature (°C)',
    'T_bends_count': 'Number of bend temperatures',
    'pH2O': 'Water partial pressure',
    'T_min': 'Minimum temperature (°C)',
    'T_max': 'Maximum temperature (°C)',
}


# ============================================================================
# 3. SCIENTIFIC STYLE AND VISUAL HELPERS
# ============================================================================

def apply_scientific_style():
    """Apply scientific publication style with full boxed axes and larger fonts."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',

        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        # Full box around the plot area (all four spines)
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,

        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 7,
        'xtick.major.width': 1.2,
        'ytick.major.size': 7,
        'ytick.major.width': 1.2,
        'xtick.minor.size': 4,
        'xtick.minor.width': 1.0,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1.0,

        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.8,

        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': False,
        'figure.figsize': (8, 6),

        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.8,
        'errorbar.capsize': 3,

        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def label_for(name, use_entire_terms=False):
    """Return either the technical descriptor name or its full descriptive term."""
    if use_entire_terms and name in FULL_TERMS:
        return FULL_TERMS[name]
    return name


def style_axes(ax, minorticks=True):
    """Ensure a full boxed frame and enable minor ticks on standard cartesian axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    if minorticks:
        try:
            ax.minorticks_on()
        except Exception:
            pass


def get_stable_color_map(categories, palette=None):
    """Return a deterministic {category: color} mapping regardless of call order."""
    if palette is None:
        palette = COLOR_PALETTES['qualitative']

    clean_categories = sorted(
        set(str(c) for c in categories if pd.notna(c)),
        key=lambda x: x
    )

    if set(clean_categories).issubset(set(COLOR_PALETTES['B_cation'].keys())):
        base = COLOR_PALETTES['B_cation']
    elif set(clean_categories).issubset(set(COLOR_PALETTES['A_cation'].keys())):
        base = COLOR_PALETTES['A_cation']
    elif set(clean_categories).issubset(set(COLOR_PALETTES['method'].keys())):
        base = COLOR_PALETTES['method']
    else:
        base = {}

    color_map = {}
    for i, cat in enumerate(clean_categories):
        if cat in base:
            color_map[cat] = base[cat]
        else:
            color_map[cat] = palette[i % len(palette)]

    return color_map


def make_legend_safe(ax, handles, **legend_kwargs):
    """Add a Legend via add_artist so it does not overwrite the main legend."""
    legend = Legend(ax, handles,
                    [h.get_label() for h in handles] if handles else [],
                    **legend_kwargs)
    ax.add_artist(legend)
    return legend


def create_bubble_legend(ax, data, size_feature, min_val, max_val, mean_val,
                         position='upper left', use_entire_terms=False):
    """Create a size legend for bubble plots; adds itself via add_artist."""
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label=f'Min: {min_val:.4f}',
               markersize=4, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='black', label=f'Mean: {mean_val:.4f}',
               markersize=8, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='black', label=f'Max: {max_val:.4f}',
               markersize=12, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5)
    ]

    title = f'Size: {label_for(size_feature, use_entire_terms)}'
    legend = Legend(
        ax,
        legend_elements,
        [f'Min: {min_val:.4f}', f'Mean: {mean_val:.4f}', f'Max: {max_val:.4f}'],
        title=title,
        loc=position,
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        fontsize=10,
        title_fontsize=10
    )
    ax.add_artist(legend)
    return legend


def _sanitize_for_ml(df, features):
    """Strip NaN/inf so sklearn estimators (PCA, t-SNE, UMAP, KMeans, DBSCAN) accept the matrix."""
    if df is None or len(df) == 0 or features is None or len(features) < 2:
        return None, []

    existing_features = [f for f in features if f in df.columns]
    if len(existing_features) < 2:
        return None, []

    data = df[existing_features].copy()

    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.replace([np.inf, -np.inf], np.nan)

    min_valid_count = max(3, int(0.3 * len(data)))
    data = data.dropna(axis=1, thresh=min_valid_count)

    if data.shape[1] < 2:
        return None, []

    data = data.dropna()

    if data.shape[0] < 3 or data.shape[1] < 2:
        return None, []

    return data, list(data.columns)


def make_qcut_groups(series, n_bins=10):
    """Quantile-binning helper: numeric series with > n_bins unique values get binned into n_bins deciles."""
    if series is None or len(series) == 0:
        return series, [], False

    s = series.copy()
    is_numeric = pd.api.types.is_numeric_dtype(s)

    if not is_numeric:
        cats = [str(c) for c in s.dropna().unique()]
        return s, sorted(cats), False

    n_unique = s.dropna().nunique()

    if n_unique <= n_bins:
        cats = sorted([c for c in s.dropna().unique()])
        return s, cats, False

    try:
        binned = pd.qcut(s, q=n_bins, duplicates='drop')
    except (ValueError, IndexError):
        cats = sorted([c for c in s.dropna().unique()])
        return s, cats, False

    if binned.dropna().nunique() < 2:
        cats = sorted([c for c in s.dropna().unique()])
        return s, cats, False

    def _fmt_interval(interval):
        if pd.isna(interval):
            return None
        try:
            lo = interval.left
            hi = interval.right
            if abs(hi - lo) >= 1:
                return f"{lo:.2f}–{hi:.2f}"
            elif abs(hi - lo) >= 0.01:
                return f"{lo:.3f}–{hi:.3f}"
            else:
                return f"{lo:.4f}–{hi:.4f}"
        except Exception:
            return str(interval)

    binned_str = binned.astype(object).apply(
        lambda v: _fmt_interval(v) if pd.notna(v) else np.nan
    )

    ordered = []
    for cat in binned.cat.categories:
        label = _fmt_interval(cat)
        if label is not None:
            ordered.append(label)

    return binned_str, ordered, True


# ============================================================================
# 4. DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Text parsing and cleaning of the uploaded dataset."""

    def __init__(self):
        self.df = None
        self.raw_data = None

    @staticmethod
    def parse_text_data(text):
        if not text or text.strip() == '':
            return None

        lines = text.strip().split('\n')

        header = lines[0]
        if '\t' in header:
            delimiter = '\t'
        elif ';' in header:
            delimiter = ';'
        else:
            delimiter = r'\s+'

        data = []
        for line in lines:
            if delimiter == r'\s+':
                row = re.split(r'\s+', line.strip())
            else:
                row = line.split(delimiter)
            data.append(row)

        df = pd.DataFrame(data[1:], columns=data[0])
        df = df.replace('-', np.nan)

        numeric_cols = ['№', '[A\']', '[B\']', '[D1]', '[D2]', 'δ', 'β', 'pH2O']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '∆T, °C' in df.columns:
            df['T_min'] = df['∆T, °C'].apply(
                lambda x: float(str(x).split('-')[0]) if pd.notna(x) and '-' in str(x) else np.nan
            )
            df['T_max'] = df['∆T, °C'].apply(
                lambda x: float(str(x).split('-')[1]) if pd.notna(x) and '-' in str(x) else np.nan
            )

        if 'α·106 (K-1)' in df.columns:
            df['α·106 (K-1)'] = pd.to_numeric(df['α·106 (K-1)'], errors='coerce')

        if 'T(bends), °C' in df.columns:
            df['T_bends_1'] = df['T(bends), °C'].apply(
                lambda x: float(str(x).split(';')[0].strip()) if pd.notna(x) and str(x).strip() != '' else np.nan
            )
            df['T_bends_2'] = df['T(bends), °C'].apply(
                lambda x: float(str(x).split(';')[1].strip()) if pd.notna(x) and len(str(x).split(';')) > 1 else np.nan
            )
            df['T_bends_3'] = df['T(bends), °C'].apply(
                lambda x: float(str(x).split(';')[2].strip()) if pd.notna(x) and len(str(x).split(';')) > 2 else np.nan
            )
            df['T_bends_count'] = df['T(bends), °C'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) and str(x).strip() != '' else 0
            )

        if 'αav·106 (K-1)' in df.columns:
            df['αav_1'] = df['αav·106 (K-1)'].apply(
                lambda x: float(str(x).split(';')[0].strip()) if pd.notna(x) and str(x).strip() != '' else np.nan
            )
            df['αav_2'] = df['αav·106 (K-1)'].apply(
                lambda x: float(str(x).split(';')[1].strip()) if pd.notna(x) and len(str(x).split(';')) > 1 else np.nan
            )
            df['αav_3'] = df['αav·106 (K-1)'].apply(
                lambda x: float(str(x).split(';')[2].strip()) if pd.notna(x) and len(str(x).split(';')) > 2 else np.nan
            )
            df['αav_count'] = df['αav·106 (K-1)'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) and str(x).strip() != '' else 0
            )

            df['αav_LT'] = df['αav·106 (K-1)'].apply(
                lambda x: float(str(x).split(';')[0].strip()) if pd.notna(x) and str(x).strip() != '' else np.nan
            )
            df['αav_HT'] = df['αav·106 (K-1)'].apply(
                lambda x: float(str(x).split(';')[-1].strip()) if pd.notna(x) and str(x).strip() != '' else np.nan
            )
            df['αav_LT'] = df['αav_LT'].fillna(df['αav_HT'])
            df['αav_HT'] = df['αav_HT'].fillna(df['αav_LT'])

        if 'α·106 (K-1)' in df.columns and 'αav_LT' in df.columns:
            if 'β' in df.columns:
                df['α/β'] = df['α·106 (K-1)'] / df['β']

            df['αav_LT/α'] = df['αav_LT'] / df['α·106 (K-1)']
            df['αav_HT/α'] = df['αav_HT'] / df['α·106 (K-1)']

        return df

    def clean_data(self, df):
        if df is None:
            return None

        df_clean = df.copy()

        required_for_analysis = ['A', 'B', 'method']
        df_clean = df_clean.dropna(subset=required_for_analysis)

        if 'α·106 (K-1)' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['α·106 (K-1)'].isna()) |
                ((df_clean['α·106 (K-1)'] >= 5) & (df_clean['α·106 (K-1)'] <= 25))
            ]

        if 'β' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['β'].isna()) |
                ((df_clean['β'] >= 0) & (df_clean['β'] <= 1))
            ]

        return df_clean


class DescriptorEngine:
    """Compute all composition-based descriptors."""

    def __init__(self, df):
        self.df = df.copy()
        self.ionic_radii = IONIC_RADII
        self.electronegativity = ELECTRONEGATIVITY
        self.valences = VALENCES
        self.molar_mass = MOLAR_MASS
        self.descriptor_names = []

    def get_radius(self, element, site='B'):
        if element == '-' or pd.isna(element):
            return np.nan
        key = element
        if key in self.ionic_radii:
            return self.ionic_radii[key]
        return np.nan

    def get_electronegativity(self, element):
        if element == '-' or pd.isna(element):
            return np.nan
        if element in self.electronegativity:
            return self.electronegativity[element]
        return np.nan

    def get_valence(self, element):
        if element == '-' or pd.isna(element):
            return np.nan
        if element in self.valences:
            return self.valences[element]
        return np.nan

    def get_molar_mass(self, element):
        if element == '-' or pd.isna(element):
            return np.nan
        if element in self.molar_mass:
            return self.molar_mass[element]
        return np.nan

    def calculate_all(self):
        self.calculate_geometric()
        self.calculate_electronegativity()
        self.calculate_mass()
        self.calculate_thermodynamic()
        self.calculate_defect()
        self.calculate_specific()
        self.calculate_compositional()
        return self.df

    def calculate_geometric(self):
        self.df['rO_est'] = self.ionic_radii['O']

        def calc_rAav(row):
            A = row['A']
            A_prime = row['A\'']
            A_prime_conc = row['[A\']'] if pd.notna(row['[A\']']) else 0

            rA = self.get_radius(A, site='A')
            rA_prime = self.get_radius(A_prime, site='A')

            if pd.isna(rA):
                return np.nan

            if pd.isna(rA_prime):
                return rA * (1 - A_prime_conc) + rA * A_prime_conc
            return rA * (1 - A_prime_conc) + rA_prime * A_prime_conc

        self.df['rAav'] = self.df.apply(calc_rAav, axis=1)

        def calc_rBav(row):
            B = row['B']
            B_prime = row['B\'']
            D1 = row['D1']
            D2 = row['D2']

            B_prime_conc = row['[B\']'] if pd.notna(row['[B\']']) else 0
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0

            total_conc = B_prime_conc + D1_conc + D2_conc

            rB = self.get_radius(B)
            rB_prime = self.get_radius(B_prime)
            rD1 = self.get_radius(D1)
            rD2 = self.get_radius(D2)

            if pd.isna(rB):
                return np.nan

            result = rB * (1 - total_conc)

            if not pd.isna(rB_prime):
                result += rB_prime * B_prime_conc
            else:
                result += rB * B_prime_conc

            if not pd.isna(rD1):
                result += rD1 * D1_conc
            else:
                result += rB * D1_conc

            if not pd.isna(rD2):
                result += rD2 * D2_conc
            else:
                result += rB * D2_conc

            return result

        self.df['rBav'] = self.df.apply(calc_rBav, axis=1)

        rO = self.ionic_radii['O']
        self.df['t'] = (self.df['rAav'] + rO) / (np.sqrt(2) * (self.df['rBav'] + rO))
        self.df['D_t'] = np.abs(1 - self.df['t'])
        self.df['octahedral_factor'] = self.df['rBav'] / rO
        self.df['Δr_AB'] = np.abs(self.df['rAav'] - self.df['rBav'])
        self.df['Δr_AB_norm'] = self.df['Δr_AB'] / rO

        def calc_sigma2_rA(row):
            A = row['A']
            A_prime = row['A\'']
            A_prime_conc = row['[A\']'] if pd.notna(row['[A\']']) else 0

            rA = self.get_radius(A, site='A')
            rA_prime = self.get_radius(A_prime, site='A')

            if pd.isna(rA):
                return np.nan

            rAav = row['rAav']

            if pd.isna(rA_prime) or A_prime == '-' or pd.isna(A_prime):
                return 0

            return (1 - A_prime_conc) * (rA - rAav)**2 + A_prime_conc * (rA_prime - rAav)**2

        self.df['σ²_rA'] = self.df.apply(calc_sigma2_rA, axis=1)

        def calc_sigma2_rB(row):
            B = row['B']
            B_prime = row['B\'']
            D1 = row['D1']
            D2 = row['D2']

            B_prime_conc = row['[B\']'] if pd.notna(row['[B\']']) else 0
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0

            total_conc = B_prime_conc + D1_conc + D2_conc

            rB = self.get_radius(B)
            rB_prime = self.get_radius(B_prime)
            rD1 = self.get_radius(D1)
            rD2 = self.get_radius(D2)

            if pd.isna(rB):
                return np.nan

            rBav = row['rBav']

            result = (1 - total_conc) * (rB - rBav)**2

            if not pd.isna(rB_prime):
                result += B_prime_conc * (rB_prime - rBav)**2
            else:
                result += B_prime_conc * (rB - rBav)**2

            if not pd.isna(rD1):
                result += D1_conc * (rD1 - rBav)**2
            else:
                result += D1_conc * (rB - rBav)**2

            if not pd.isna(rD2):
                result += D2_conc * (rD2 - rBav)**2
            else:
                result += D2_conc * (rB - rBav)**2

            return result

        self.df['σ²_rB'] = self.df.apply(calc_sigma2_rB, axis=1)

        self.df['V_cell'] = (2 * (self.df['rBav'] + self.df['rO_est']))**3

        denom = self.df['rBav']
        self.df['r_ratio_AB'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            self.df['rAav'] / denom,
            np.nan
        )

    def calculate_electronegativity(self):
        rO = self.ionic_radii['O']

        def calc_chiAav(row):
            A = row['A']
            A_prime = row['A\'']
            A_prime_conc = row['[A\']'] if pd.notna(row['[A\']']) else 0

            chiA = self.get_electronegativity(A)
            chiA_prime = self.get_electronegativity(A_prime)

            if pd.isna(chiA):
                return np.nan

            if pd.isna(chiA_prime):
                return chiA * (1 - A_prime_conc) + chiA * A_prime_conc
            return chiA * (1 - A_prime_conc) + chiA_prime * A_prime_conc

        self.df['χAav'] = self.df.apply(calc_chiAav, axis=1)

        def calc_chiBav(row):
            B = row['B']
            B_prime = row['B\'']
            D1 = row['D1']
            D2 = row['D2']

            B_prime_conc = row['[B\']'] if pd.notna(row['[B\']']) else 0
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0

            total_conc = B_prime_conc + D1_conc + D2_conc

            chiB = self.get_electronegativity(B)
            chiB_prime = self.get_electronegativity(B_prime)
            chiD1 = self.get_electronegativity(D1)
            chiD2 = self.get_electronegativity(D2)

            if pd.isna(chiB):
                return np.nan

            result = chiB * (1 - total_conc)

            if not pd.isna(chiB_prime):
                result += chiB_prime * B_prime_conc
            else:
                result += chiB * B_prime_conc

            if not pd.isna(chiD1):
                result += chiD1 * D1_conc
            else:
                result += chiB * D1_conc

            if not pd.isna(chiD2):
                result += chiD2 * D2_conc
            else:
                result += chiB * D2_conc

            return result

        self.df['χBav'] = self.df.apply(calc_chiBav, axis=1)

        self.df['Δχ_AB'] = np.abs(self.df['χAav'] - self.df['χBav'])

        denom = self.df['χBav']
        self.df['χ_ratio_AB'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            self.df['χAav'] / denom,
            np.nan
        )

        chiO = self.electronegativity['O']
        self.df['ionicity_AO'] = 1 - np.exp(-0.25 * (self.df['χAav'] - chiO)**2)
        self.df['ionicity_BO'] = 1 - np.exp(-0.25 * (self.df['χBav'] - chiO)**2)

        denom = self.df['χAav']
        self.df['acidity_AO'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            1 / denom,
            np.nan
        )

        denom = self.df['χBav']
        self.df['acidity_BO'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            1 / denom,
            np.nan
        )

        self.df['Δacidity'] = self.df['acidity_BO'] - self.df['acidity_AO']
        self.df['χ_total'] = (self.df['χAav'] + self.df['χBav']) / 2
        self.df['χ_ratio_t'] = self.df['χ_ratio_AB'] * self.df['t']

    def calculate_thermodynamic(self):
        R = 8.314

        def calc_S_config_A(row):
            A_prime_conc = row['[A\']'] if pd.notna(row['[A\']']) else 0

            if A_prime_conc == 0 or A_prime_conc == 1:
                return 0

            x_A = 1 - A_prime_conc
            x_A_prime = A_prime_conc

            if x_A <= 0 or x_A_prime <= 0:
                return 0

            return -R * (x_A * np.log(x_A) + x_A_prime * np.log(x_A_prime))

        self.df['S_config_A'] = self.df.apply(calc_S_config_A, axis=1)

        def calc_S_config_B(row):
            B_prime_conc = row['[B\']'] if pd.notna(row['[B\']']) else 0
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0

            total_conc = B_prime_conc + D1_conc + D2_conc

            if total_conc == 0 or total_conc == 1:
                return 0

            x_B = 1 - total_conc

            concentrations = [x_B, B_prime_conc, D1_conc, D2_conc]
            concentrations = [x for x in concentrations if x > 0]

            if len(concentrations) <= 1:
                return 0

            entropy = 0
            for x in concentrations:
                entropy -= x * np.log(x)

            return R * entropy

        self.df['S_config_B'] = self.df.apply(calc_S_config_B, axis=1)

        def calc_V_Bav(row):
            B = row['B']
            B_prime = row['B\'']
            D1 = row['D1']
            D2 = row['D2']

            B_prime_conc = row['[B\']'] if pd.notna(row['[B\']']) else 0
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0

            total_conc = B_prime_conc + D1_conc + D2_conc

            vB = self.get_valence(B)
            vB_prime = self.get_valence(B_prime)
            vD1 = self.get_valence(D1)
            vD2 = self.get_valence(D2)

            if pd.isna(vB):
                return np.nan

            result = vB * (1 - total_conc)

            if not pd.isna(vB_prime):
                result += vB_prime * B_prime_conc
            else:
                result += vB * B_prime_conc

            if not pd.isna(vD1):
                result += vD1 * D1_conc
            else:
                result += vB * D1_conc

            if not pd.isna(vD2):
                result += vD2 * D2_conc
            else:
                result += vB * D2_conc

            return result

        self.df['V_Bav'] = self.df.apply(calc_V_Bav, axis=1)
        self.df['Vo_proxy'] = (4 - self.df['V_Bav']) / 2

        chiO = self.electronegativity['O']
        self.df['ΔH_hydr'] = 1 / (self.df['rBav'] + 0.1) * (self.df['χBav'] - chiO)**2
        self.df['E_BO'] = (self.df['V_Bav'] * 2) / (self.df['rBav'] + self.df['rO_est'])

    def calculate_mass(self):
        def calc_MAav(row):
            A = row['A']
            A_prime = row['A\'']
            A_prime_conc = row['[A\']'] if pd.notna(row['[A\']']) else 0

            mA = self.get_molar_mass(A)
            mA_prime = self.get_molar_mass(A_prime)

            if pd.isna(mA):
                return np.nan

            if pd.isna(mA_prime):
                return mA * (1 - A_prime_conc) + mA * A_prime_conc
            return mA * (1 - A_prime_conc) + mA_prime * A_prime_conc

        self.df['M_Aav'] = self.df.apply(calc_MAav, axis=1)

        def calc_MBav(row):
            B = row['B']
            B_prime = row['B\'']
            D1 = row['D1']
            D2 = row['D2']

            B_prime_conc = row['[B\']'] if pd.notna(row['[B\']']) else 0
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0

            total_conc = B_prime_conc + D1_conc + D2_conc

            mB = self.get_molar_mass(B)
            mB_prime = self.get_molar_mass(B_prime)
            mD1 = self.get_molar_mass(D1)
            mD2 = self.get_molar_mass(D2)

            if pd.isna(mB):
                return np.nan

            result = mB * (1 - total_conc)

            if not pd.isna(mB_prime):
                result += mB_prime * B_prime_conc
            else:
                result += mB * B_prime_conc

            if not pd.isna(mD1):
                result += mD1 * D1_conc
            else:
                result += mB * D1_conc

            if not pd.isna(mD2):
                result += mD2 * D2_conc
            else:
                result += mB * D2_conc

            return result

        self.df['M_Bav'] = self.df.apply(calc_MBav, axis=1)
        self.df['M_total'] = self.df['M_Aav'] + self.df['M_Bav'] + 3 * self.molar_mass['O']

        denom = self.df['M_Bav']
        self.df['M_ratio_AB'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            self.df['M_Aav'] / denom,
            np.nan
        )

        self.df['M_rA'] = self.df['M_Aav'] * self.df['rAav']
        self.df['M_χA'] = self.df['M_Aav'] * self.df['χAav']

        denom = self.df['V_cell']
        self.df['ρ'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            self.df['M_Bav'] / denom,
            np.nan
        )

    def calculate_defect(self):
        def calc_delta_actual(row):
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0
            return D1_conc / 2 + D2_conc / 2

        self.df['δ_actual'] = self.df.apply(calc_delta_actual, axis=1)
        self.df['Z_eff_B'] = 4 - 2 * self.df['δ_actual']

        denom = (self.df['rBav'] + 0.1) * self.df['χBav']
        self.df['proton_affinity'] = np.where(
            (denom.notna()) & (np.abs(denom) > 1e-12),
            1 / denom,
            np.nan
        )

        chiO = self.electronegativity['O']
        self.df['E_vac'] = 1 / ((self.df['rBav'] + 0.1)**2) * (self.df['χBav'] - chiO)

    def calculate_specific(self):
        if 'α·106 (K-1)' in self.df.columns and 'β' in self.df.columns:
            denom = self.df['β']
            self.df['alpha_beta_ratio'] = np.where(
                (denom.notna()) & (np.abs(denom) > 1e-12),
                self.df['α·106 (K-1)'] / denom,
                np.nan
            )

        R = 8.314
        self.df['T_stab'] = -self.df['ΔH_hydr'] / R
        self.df['δ_χB'] = self.df['δ_actual'] * self.df['χBav']
        self.df['δ_rB'] = self.df['δ_actual'] * self.df['rBav']

    def calculate_compositional(self):
        self.df['B\'_conc'] = self.df['[B\']']
        self.df['D_total'] = self.df['[D1]'] + self.df['[D2]']

        denom = self.df['B\'_conc'] + 0.001
        self.df['D_B_ratio'] = np.where(
            np.abs(denom) > 1e-12,
            self.df['D_total'] / denom,
            np.nan
        )

        self.df['total_dopant'] = self.df['[B\']'] + self.df['[D1]'] + self.df['[D2]']


class CorrelationAnalyzer:
    """Correlation analysis (Pearson, Spearman, partial, network)."""

    def __init__(self, df):
        self.df = df.copy()
        self.targets = TARGET_VARIABLES
        self.all_descriptors = []
        self.top_features = {}

    def get_numeric_features(self):
        exclude_cols = ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Ref', 'method',
                        '∆T, °C', 'T(bends), °C', 'αav·106 (K-1)']
        exclude_cols.extend(self.targets)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col not in exclude_cols]

        for col in features[:]:
            if self.df[col].isna().sum() / len(self.df) > 0.5:
                features.remove(col)
            elif self.df[col].nunique() < 2:
                features.remove(col)

        self.all_descriptors = features
        return features

    def pearson_correlation(self, features=None, target=None):
        if features is None:
            features = self.get_numeric_features()

        if target is None:
            target = self.targets[0]

        if target not in self.df.columns:
            st.session_state['correlation_debug'] = f"Target variable {target} not found"
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10 and self.df[feature].nunique() > 1:
                    valid_features.append(feature)

        if len(valid_features) == 0:
            st.session_state['correlation_debug'] = f"No valid features for {target}"
            return None

        st.session_state['correlation_debug'] = f"Found {len(valid_features)} valid features"

        results = []
        for feature in valid_features:
            data = self.df[[feature, target]].dropna()
            if len(data) > 10:
                corr, p_value = pearsonr(data[feature], data[target])
                if not np.isnan(corr):
                    results.append({
                        'feature': feature,
                        'correlation': corr,
                        'p_value': p_value,
                        'n': len(data)
                    })

        if len(results) == 0:
            st.session_state['correlation_debug'] = f"No significant correlations for {target}"
            return None

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['significant'] = results_df['p_value'] < 0.05
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)

        return results_df

    def spearman_correlation(self, features=None, target=None):
        if features is None:
            features = self.get_numeric_features()

        if target is None:
            target = self.targets[0]

        if target not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10 and self.df[feature].nunique() > 1:
                    valid_features.append(feature)

        if len(valid_features) == 0:
            return None

        results = []
        for feature in valid_features:
            data = self.df[[feature, target]].dropna()
            if len(data) > 10:
                corr, p_value = spearmanr(data[feature], data[target])
                if not np.isnan(corr):
                    results.append({
                        'feature': feature,
                        'correlation': corr,
                        'p_value': p_value,
                        'n': len(data)
                    })

        if len(results) == 0:
            return None

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['significant'] = results_df['p_value'] < 0.05
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)

        return results_df

    def partial_correlation(self, target, control_vars=['pH2O'], features=None):
        if features is None:
            features = self.get_numeric_features()

        if target not in self.df.columns:
            return None

        cols = features + [target] + control_vars
        cols = [c for c in cols if c in self.df.columns]
        data = self.df[cols].dropna()

        if len(data) < 10:
            return None

        results = []
        for feature in features:
            if feature in data.columns and feature != target:
                try:
                    result = pg.partial_corr(
                        data=data,
                        x=feature,
                        y=target,
                        covar=control_vars
                    )
                    results.append({
                        'feature': feature,
                        'correlation': result['r'].values[0],
                        'p_value': result['p-val'].values[0],
                        'n': len(data)
                    })
                except Exception:
                    continue

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['significant'] = results_df['p_value'] < 0.05
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)

        return results_df

    def get_top_features(self, target=None, n=20):
        if target is None:
            target = self.targets[0]

        if target not in self.df.columns:
            return []

        pearson_results = self.pearson_correlation(target=target)
        spearman_results = self.spearman_correlation(target=target)

        if pearson_results is None and spearman_results is None:
            return []

        if pearson_results is None:
            merged = spearman_results.copy()
            merged['combined_score'] = np.abs(merged['correlation'])
        elif spearman_results is None:
            merged = pearson_results.copy()
            merged['combined_score'] = np.abs(merged['correlation'])
        else:
            merged = pearson_results.merge(
                spearman_results,
                on='feature',
                suffixes=('_pearson', '_spearman')
            )

            if len(merged) == 0:
                merged = pearson_results.copy()
                merged['combined_score'] = np.abs(merged['correlation'])
            else:
                merged['abs_pearson'] = np.abs(merged['correlation_pearson'])
                merged['abs_spearman'] = np.abs(merged['correlation_spearman'])
                merged['combined_score'] = (merged['abs_pearson'] + merged['abs_spearman']) / 2

        merged = merged.sort_values('combined_score', ascending=False)
        top_features = merged['feature'].head(n).tolist()

        self.top_features[target] = top_features
        return top_features[:n]

    def correlation_matrix_with_filter(self, features=None, threshold=0.3):
        if features is None:
            features = self.get_numeric_features()

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        data = self.df[valid_features].dropna()
        if len(data) < 3:
            return None

        corr_matrix = data.corr()
        mask = np.abs(corr_matrix) < threshold
        corr_filtered = corr_matrix.copy()
        corr_filtered[mask] = 0
        np.fill_diagonal(corr_filtered.values, 1)

        return corr_filtered

    def correlation_network(self, features=None, threshold=0.5):
        if features is None:
            features = self.get_numeric_features()

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        data = self.df[valid_features].dropna()
        if len(data) < 3:
            return None

        corr_matrix = data.corr()

        G = nx.Graph()
        for feature in valid_features:
            G.add_node(feature)

        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                if i < j:
                    corr = corr_matrix.loc[feat1, feat2]
                    if abs(corr) >= threshold and not np.isnan(corr):
                        G.add_edge(feat1, feat2, weight=abs(corr), sign=np.sign(corr))

        return G


class VisualizationEngine:
    """All Matplotlib / Seaborn plotting methods."""

    def __init__(self, df, use_entire_terms=False):
        self.df = df.copy()
        self.targets = TARGET_VARIABLES
        self.available_targets = [t for t in self.targets if t in self.df.columns]
        self.compositional = ['[B\']', 'D_total', 'δ']
        self.all_features = []
        self.use_entire_terms = use_entire_terms

    def set_features(self, features):
        self.all_features = features

    def _label(self, name):
        return label_for(name, self.use_entire_terms)

    # ------------------------------------------------------------------
    # CATEGORY 1: OVERVIEW PLOTS
    # ------------------------------------------------------------------

    def plot_distributions(self, features=None):
        if features is None:
            features = self.all_features[:8] if len(self.all_features) > 0 else []

        if len(features) == 0:
            return None

        n_cols = 3
        n_rows = (len(features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, feature in enumerate(features):
            if i < len(axes) and feature in self.df.columns:
                data = self.df[feature].dropna()
                if len(data) > 0:
                    axes[i].hist(data, bins=20, edgecolor='black', alpha=0.7, color='#3498DB')
                    axes[i].set_xlabel(self._label(feature), fontsize=12)
                    axes[i].set_ylabel('Frequency', fontsize=12)
                    axes[i].grid(True, alpha=0.3)
                    style_axes(axes[i])

        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_missing_matrix(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        missing = self.df.isna().sum() / len(self.df) * 100
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            ax.text(0.5, 0.5, 'No missing data', transform=ax.transAxes, ha='center', va='center')
            return fig

        cols = missing.index.tolist()
        data = self.df[cols]

        im = ax.imshow(data.isna().T, aspect='auto', cmap='Reds', interpolation='none')
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=9)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Missing', fontsize=12)

        for i, col in enumerate(cols):
            pct = missing[col]
            ax.text(len(data) + 1, i, f'{pct:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        return fig

    def plot_box_by_method(self, target=None):
        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns or 'method' not in self.df.columns:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        data = self.df[['method', target]].dropna()

        if len(data) == 0:
            return None

        methods = data['method'].unique()
        box_data = [data[data['method'] == m][target].values for m in methods]

        bp = ax.boxplot(box_data, labels=methods, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)

        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(target), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_violin_by_B(self, target=None, group_feature='B', show_stats=True,
                         sort_by_mass=True, color_map=None,
                         bin_numeric=True, n_bins=10):
        """Violin plot grouped by B-cation (or any feature); numeric features get quantile-binned."""
        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns or group_feature not in self.df.columns:
            return None

        data = self.df[[group_feature, target]].dropna()

        if len(data) < 3:
            return None

        was_binned = False
        if bin_numeric:
            grouped_series, ordered_cats, was_binned = make_qcut_groups(
                data[group_feature], n_bins=n_bins
            )
            data = data.copy()
            data['__group__'] = grouped_series.values
            actual_group_col = '__group__'
        else:
            actual_group_col = group_feature

        data = data.dropna(subset=[actual_group_col])

        if len(data) < 3:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_groups = list(data[actual_group_col].unique())

        if sort_by_mass and not was_binned and group_feature in ('B', 'A'):
            unique_groups = sorted(
                unique_groups,
                key=lambda g: MOLAR_MASS.get(str(g).strip(), float('inf'))
            )
        elif was_binned:
            unique_groups = [g for g in ordered_cats if g in unique_groups]

        groups = []
        group_labels = []
        group_stats = []
        group_masses = []

        for g in unique_groups:
            vals = data[data[actual_group_col] == g][target].values
            if len(vals) >= 2:
                groups.append(vals)
                group_labels.append(str(g))
                if group_feature in ('B', 'A') and not was_binned:
                    group_masses.append(MOLAR_MASS.get(str(g).strip(), np.nan))
                else:
                    group_masses.append(np.nan)
                group_stats.append({
                    'label': str(g),
                    'n': len(vals),
                    'mean': np.mean(vals),
                    'median': np.median(vals),
                    'std': np.std(vals, ddof=1) if len(vals) > 1 else 0,
                    'mass': MOLAR_MASS.get(str(g).strip(), np.nan)
                            if (group_feature in ('B', 'A') and not was_binned)
                            else np.nan
                })

        if len(groups) == 0:
            return None

        if color_map is None:
            color_map = get_stable_color_map(group_labels)

        parts = ax.violinplot(
            groups,
            positions=range(len(groups)),
            showmeans=True,
            showmedians=True,
            showextrema=True
        )

        for i, pc in enumerate(parts['bodies']):
            lbl = group_labels[i]
            pc.set_facecolor(color_map.get(lbl, COLOR_PALETTES['qualitative'][i % len(COLOR_PALETTES['qualitative'])]))
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')

        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')
        parts['cmaxes'].set_color('gray')
        parts['cmins'].set_color('gray')
        parts['cbars'].set_color('gray')

        for i, vals in enumerate(groups):
            x_jitter = np.random.normal(i, 0.04, size=len(vals))
            ax.scatter(x_jitter, vals, alpha=0.3, s=15, color='black', zorder=1)

        if show_stats:
            y_min = data[target].min()
            y_max = data[target].max()
            y_offset = (y_max - y_min) * 0.03

            for i, st in enumerate(group_stats):
                ax.text(
                    i, y_max + y_offset,
                    f'n={st["n"]}\nμ={st["mean"]:.2f}\nσ={st["std"]:.2f}',
                    ha='center', va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8)
                )

        xtick_labels = []
        for lbl, mass in zip(group_labels, group_masses):
            if not np.isnan(mass):
                xtick_labels.append(f'{lbl}\n({mass:.1f})')
            else:
                xtick_labels.append(lbl)

        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(
            xtick_labels,
            fontsize=9,
            rotation=45 if was_binned and len(group_labels) > 6 else 0,
            ha='right' if was_binned and len(group_labels) > 6 else 'center'
        )

        if group_feature in ('B', 'A') and not was_binned:
            xlabel = f'{group_feature}-cation (sorted by molar mass, g/mol)' if sort_by_mass \
                     else f'{group_feature}-cation'
        elif was_binned:
            xlabel = f'{self._label(group_feature)} (quantile bins, {len(group_labels)} groups)'
        else:
            xlabel = self._label(group_feature)

        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(target), fontsize=14, fontweight='bold')

        if was_binned:
            title = f'Distribution of {self._label(target)} by {self._label(group_feature)} ({len(group_labels)} quantile bins)'
        else:
            title = f'Distribution of {self._label(target)} by {self._label(group_feature)}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        style_axes(ax)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CATEGORY 2: CORRELATION PLOTS
    # ------------------------------------------------------------------

    def plot_correlation_matrix(self, features=None):
        if features is None:
            features = self.all_features[:15] if len(self.all_features) > 0 else []

        if len(features) < 2:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        data = self.df[valid_features].dropna()
        if len(data) < 3:
            return None

        corr_matrix = data.corr()
        display_labels = [self._label(f) for f in valid_features]
        corr_matrix.columns = display_labels
        corr_matrix.index = display_labels

        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt='.2f',
            annot_kws={'size': 9},
            ax=ax
        )

        ax.set_title('Pearson Correlation Matrix', fontsize=15, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_partial_correlation_matrix(self, target=None, control_vars=['pH2O']):
        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns:
            return None

        features = self.all_features[:10] if len(self.all_features) > 0 else []
        if len(features) < 2:
            return None

        cols = features + [target] + control_vars
        cols = [c for c in cols if c in self.df.columns]
        data = self.df[cols].dropna()

        if len(data) < 10:
            return None

        results = []
        for feature in features:
            if feature in data.columns and feature != target:
                try:
                    result = pg.partial_corr(
                        data=data,
                        x=feature,
                        y=target,
                        covar=control_vars
                    )
                    results.append({
                        'feature': feature,
                        'correlation': result['r'].values[0],
                        'p_value': result['p-val'].values[0]
                    })
                except Exception:
                    continue

        if len(results) == 0:
            return None

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('correlation', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))

        display_feats = [self._label(f) for f in results_df['feature'].values]
        colors = ['red' if c < 0 else 'blue' for c in results_df['correlation'].values]
        ax.barh(display_feats, results_df['correlation'], color=colors, alpha=0.7)

        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax.set_xlabel(f'Partial Correlation with {self._label(target)}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
        ax.set_title(f'Partial Correlation (controlling for {", ".join(control_vars)})',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_correlation_network(self, features=None, threshold=0.5):
        if features is None:
            features = self.all_features[:15] if len(self.all_features) > 0 else []

        if len(features) < 2:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        data = self.df[valid_features].dropna()
        if len(data) < 3:
            return None

        corr_matrix = data.corr()

        G = nx.Graph()
        for feature in valid_features:
            G.add_node(feature)

        edge_list = []
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                if i < j:
                    corr = corr_matrix.loc[feat1, feat2]
                    if abs(corr) >= threshold and not np.isnan(corr):
                        edge_list.append((feat1, feat2, abs(corr), np.sign(corr)))

        edge_list = sorted(edge_list, key=lambda x: x[2], reverse=True)

        for edge in edge_list[:30]:
            G.add_edge(edge[0], edge[1], weight=edge[2], sign=edge[3])

        if len(G.edges()) == 0:
            return None

        fig, ax = plt.subplots(figsize=(12, 10))

        pos = nx.spring_layout(G, k=1, seed=42)

        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('sign', 1) > 0]
        negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('sign', 1) < 0]

        if len(positive_edges) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=positive_edges,
                                   edge_color='blue', alpha=0.6, width=2, ax=ax)
        if len(negative_edges) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=negative_edges,
                                   edge_color='red', alpha=0.6, width=2, ax=ax)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=1000, alpha=0.8, ax=ax)

        node_labels = {n: self._label(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold', ax=ax)

        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='Positive correlation'),
            Patch(facecolor='red', alpha=0.6, label='Negative correlation')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(f'Correlation Network (threshold = {threshold})',
                     fontsize=15, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        return fig

    def plot_pairplot_top5(self, features=None, target=None):
        if features is None:
            features = self.all_features[:4] if len(self.all_features) > 0 else []

        if target is None:
            target = 'α·106 (K-1)'

        if len(features) < 2 or target not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        plot_cols = valid_features[:4] + [target]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()

        if len(data) < 5:
            return None

        renamed = data.rename(columns={c: self._label(c) for c in data.columns})
        display_vars = [self._label(v) for v in valid_features[:4]]

        g = sns.pairplot(
            renamed,
            vars=display_vars,
            hue=self._label(target),
            palette='viridis',
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30},
            diag_kws={'alpha': 0.6}
        )

        g.fig.suptitle('Pairplot of Top Features', fontsize=15, fontweight='bold', y=1.02)
        g.fig.set_size_inches(12, 10)

        return g.fig

    def plot_best_linear_relationships(self, features=None, target=None, n=5):
        if features is None:
            features = self.all_features[:10] if len(self.all_features) > 0 else []

        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                data = self.df[[feature, target]].dropna()
                if len(data) > 10:
                    valid_features.append(feature)

        if len(valid_features) == 0:
            return None

        correlations = []
        for feature in valid_features:
            data = self.df[[feature, target]].dropna()
            if len(data) > 2:
                corr, _ = pearsonr(data[feature], data[target])
                if not np.isnan(corr):
                    correlations.append((feature, abs(corr)))

        correlations = sorted(correlations, key=lambda x: x[1], reverse=True)
        top_features = [c[0] for c in correlations[:n]]

        if len(top_features) == 0:
            return None

        n_cols = 2
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, feature in enumerate(top_features):
            if i < len(axes):
                ax = axes[i]
                data = self.df[[feature, target]].dropna()

                if len(data) > 2:
                    ax.scatter(data[feature], data[target], alpha=0.6, color='#3498DB', s=30)

                    try:
                        X = data[[feature]].values
                        y = data[target].values

                        reg = LinearRegression().fit(X, y)
                        y_pred = reg.predict(X)

                        x_range = np.linspace(X.min(), X.max(), 100)
                        y_range = reg.predict(x_range.reshape(-1, 1))
                        ax.plot(x_range, y_range, color='red', linewidth=2, label='Regression')

                        r2 = reg.score(X, y)
                        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                                fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception:
                        pass

                    ax.set_xlabel(self._label(feature), fontsize=12)
                    ax.set_ylabel(self._label(target), fontsize=12)
                    ax.grid(True, alpha=0.3)
                    style_axes(ax)

        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_residuals(self, feature=None, target=None):
        if feature is None:
            if len(self.all_features) > 0:
                feature = self.all_features[0]
            else:
                return None

        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns or feature not in self.df.columns:
            return None

        data = self.df[[feature, target]].dropna()

        if len(data) < 5:
            return None

        try:
            X = data[[feature]].values
            y = data[target].values

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            residuals = y - y_pred

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(y_pred, residuals, alpha=0.6, color='#3498DB', s=30)
            axes[0].axhline(0, color='red', linewidth=2, linestyle='--')
            axes[0].set_xlabel('Predicted Values', fontsize=14)
            axes[0].set_ylabel('Residuals', fontsize=14)
            axes[0].set_title('Residuals vs Fitted', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            style_axes(axes[0])

            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title('Normal Q-Q Plot', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            style_axes(axes[1])

            plt.tight_layout()
            return fig

        except Exception:
            return None

    def plot_heatmap_top_features(self, features=None):
        if features is None:
            features = self.all_features[:20] if len(self.all_features) > 0 else []

        if len(features) < 2:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        data = self.df[valid_features].dropna()
        if len(data) < 3:
            return None

        corr_matrix = data.corr()
        display_labels = [self._label(f) for f in valid_features]
        corr_matrix.columns = display_labels
        corr_matrix.index = display_labels

        fig, ax = plt.subplots(figsize=(14, 12))

        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8},
            ax=ax
        )

        ax.set_title('Correlation Heatmap of Top Features', fontsize=15, fontweight='bold')
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CATEGORY 3: PCA AND CLUSTERING
    # ------------------------------------------------------------------

    def plot_elbow(self, features=None, max_components=15):
        if features is None:
            features = self.all_features

        if features is None or len(features) < 2:
            return None

        data, valid_features = _sanitize_for_ml(self.df, features)
        if data is None or len(valid_features) < 2 or len(data) < 5:
            return None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        pca = PCA()
        pca.fit(data_scaled)

        cumsum = np.cumsum(pca.explained_variance_ratio_)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                     pca.explained_variance_ratio_, 'o-', color='#3498DB', linewidth=2)
        axes[0].set_xlabel('Principal Component', fontsize=14)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=14)
        axes[0].set_title('Elbow Plot', fontsize=15)
        axes[0].grid(True, alpha=0.3)
        style_axes(axes[0])

        axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'o-', color='#E74C3C', linewidth=2)
        axes[1].axhline(0.95, color='green', linewidth=2, linestyle='--', label='95% variance')
        axes[1].set_xlabel('Number of Components', fontsize=14)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=14)
        axes[1].set_title('Cumulative Variance', fontsize=15)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        style_axes(axes[1])

        plt.tight_layout()
        return fig

    def plot_pca_biplot(self, features=None, n_components=2):
        if features is None:
            features = self.all_features

        if features is None or len(features) < 2:
            return None

        data, valid_features = _sanitize_for_ml(self.df, features)
        if data is None or len(valid_features) < 2 or len(data) < 5:
            return None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(pca_result[:, 0], pca_result[:, 1],
                   alpha=0.6, c='#3498DB', s=30)

        for i, feature in enumerate(valid_features):
            ax.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3,
                     head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
            ax.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2,
                    self._label(feature), fontsize=10, ha='center', va='center', color='red')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        ax.set_title('PCA Biplot', fontsize=15, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_tsne_umap(self, features=None, perplexity=30, n_neighbors=15, min_dist=0.1):
        if features is None:
            features = self.all_features

        if features is None or len(features) < 2:
            return None

        data, valid_features = _sanitize_for_ml(self.df, features)
        if data is None or len(valid_features) < 2 or len(data) < 10:
            return None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        try:
            tsne = TSNE(n_components=2, perplexity=min(perplexity, len(data) - 1), random_state=42)
            tsne_result = tsne.fit_transform(data_scaled)
        except Exception:
            tsne_result = None

        try:
            umap_result = umap.UMAP(
                n_neighbors=min(n_neighbors, len(data) - 1),
                min_dist=min_dist,
                random_state=42
            ).fit_transform(data_scaled)
        except Exception:
            umap_result = None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if tsne_result is not None:
            axes[0].scatter(tsne_result[:, 0], tsne_result[:, 1],
                            alpha=0.6, color='#3498DB', s=30)
            axes[0].set_xlabel('t-SNE 1', fontsize=14)
            axes[0].set_ylabel('t-SNE 2', fontsize=14)
            axes[0].set_title(f't-SNE (perplexity={perplexity})', fontsize=15)
            axes[0].grid(True, alpha=0.3)
            style_axes(axes[0])
        else:
            axes[0].text(0.5, 0.5, 't-SNE failed', transform=axes[0].transAxes,
                         ha='center', va='center')

        if umap_result is not None:
            axes[1].scatter(umap_result[:, 0], umap_result[:, 1],
                            alpha=0.6, color='#E74C3C', s=30)
            axes[1].set_xlabel('UMAP 1', fontsize=14)
            axes[1].set_ylabel('UMAP 2', fontsize=14)
            axes[1].set_title(f'UMAP (n_neighbors={n_neighbors})', fontsize=15)
            axes[1].grid(True, alpha=0.3)
            style_axes(axes[1])
        else:
            axes[1].text(0.5, 0.5, 'UMAP failed', transform=axes[1].transAxes,
                         ha='center', va='center')

        plt.tight_layout()
        return fig

    def plot_silhouette(self, features=None, max_clusters=10):
        if features is None:
            features = self.all_features

        if features is None or len(features) < 2:
            return None

        data, valid_features = _sanitize_for_ml(self.df, features)
        if data is None or len(valid_features) < 2 or len(data) < 10:
            return None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        silhouette_scores = []
        for k in range(2, min(max_clusters, len(data) - 1) + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data_scaled)
                score = silhouette_score(data_scaled, labels)
                silhouette_scores.append((k, score))
            except Exception:
                continue

        if len(silhouette_scores) == 0:
            return None

        silhouette_scores = np.array(silhouette_scores)
        optimal_k = silhouette_scores[np.argmax(silhouette_scores[:, 1]), 0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(silhouette_scores[:, 0], silhouette_scores[:, 1],
                     'o-', color='#3498DB', linewidth=2, markersize=8)
        axes[0].axvline(optimal_k, color='red', linewidth=2, linestyle='--',
                        label=f'Optimal K = {int(optimal_k)}')
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=14)
        axes[0].set_ylabel('Silhouette Score', fontsize=14)
        axes[0].set_title('Silhouette Analysis', fontsize=15)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        style_axes(axes[0])

        try:
            kmeans = KMeans(n_clusters=int(optimal_k), random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            silhouette_vals = silhouette_samples(data_scaled, labels)

            y_lower = 10
            for i in range(int(optimal_k)):
                cluster_vals = silhouette_vals[labels == i]
                cluster_vals.sort()
                size_cluster = len(cluster_vals)
                y_upper = y_lower + size_cluster

                color = plt.cm.viridis(i / int(optimal_k))
                axes[1].fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    cluster_vals,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7
                )
                y_lower = y_upper + 10

            axes[1].axvline(np.mean(silhouette_vals), color='red',
                            linestyle='--', label='Average Score')
            axes[1].set_xlabel('Silhouette Coefficient', fontsize=14)
            axes[1].set_ylabel('Cluster', fontsize=14)
            axes[1].set_title(f'Silhouette Plot (K={int(optimal_k)})', fontsize=15)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            style_axes(axes[1])

        except Exception:
            axes[1].text(0.5, 0.5, 'Could not compute silhouette for optimal K',
                         transform=axes[1].transAxes, ha='center', va='center')

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CATEGORY 4: CONCENTRATION MAPS
    # ------------------------------------------------------------------

    def plot_heatmap_2d(self, x_feature, y_feature, target, grid_resolution=50):
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None

        if target not in self.df.columns:
            return None

        data = self.df[[x_feature, y_feature, target]].dropna(subset=[target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        x = data[x_feature].values
        y = data[y_feature].values
        z = data[target].values

        z_min, z_max = z.min(), z.max()
        st.caption(f"Actual range of {self._label(target)}: [{z_min:.3f}, {z_max:.3f}]")

        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except Exception:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except Exception:
                return None

        zi = np.clip(zi, z_min, z_max)

        fig, ax = plt.subplots(figsize=(10, 8))

        levels = np.linspace(z_min, z_max, 21)
        im = ax.contourf(xi, yi, zi, levels=levels, cmap='viridis',
                         vmin=z_min, vmax=z_max, extend='both')

        ax.scatter(x, y, c='red', s=20, alpha=0.5, edgecolors='black', linewidth=0.5)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self._label(target), fontsize=14, fontweight='bold')

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_feature), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(target)} vs {self._label(x_feature)} and {self._label(y_feature)}',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        return fig

    def plot_contour(self, x_feature, y_feature, target, grid_resolution=50):
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None

        if target not in self.df.columns:
            return None

        data = self.df[[x_feature, y_feature, target]].dropna(subset=[target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        x = data[x_feature].values
        y = data[y_feature].values
        z = data[target].values

        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except Exception:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except Exception:
                return None

        fig, ax = plt.subplots(figsize=(10, 8))

        contour = ax.contour(xi, yi, zi, levels=15, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=11, fmt='%1.2f')

        ax.scatter(x, y, c='red', s=20, alpha=0.5, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_feature), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(target)} Contour Plot', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.2)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_heatmap_with_points(self, x_feature, y_feature, target, grid_resolution=50):
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None

        if target not in self.df.columns:
            return None

        data = self.df[[x_feature, y_feature, target]].dropna(subset=[target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        x = data[x_feature].values
        y = data[y_feature].values
        z = data[target].values

        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except Exception:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except Exception:
                return None

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(zi.T, origin='lower', extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                       aspect='auto', cmap='viridis', alpha=0.8)

        ax.scatter(x, y, c=z, s=50, cmap='viridis',
                   edgecolors='black', linewidth=1, vmin=zi.min(), vmax=zi.max())

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self._label(target), fontsize=14, fontweight='bold')

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_feature), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(target)} Heatmap with Data Points',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CATEGORY 4.5: TERNARY PLOT
    # ------------------------------------------------------------------

    def plot_ternary_advanced(self, a_feature, b_feature, c_feature, target, plot_type='scatter', grid_resolution=50):
        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns:
            return None

        if a_feature not in self.df.columns or b_feature not in self.df.columns or c_feature not in self.df.columns:
            return None

        data = self.df[[a_feature, b_feature, c_feature, target]].dropna(subset=[target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        total = data[a_feature] + data[b_feature] + data[c_feature]
        total = total.replace(0, np.nan)

        data_a = data[a_feature] / total
        data_b = data[b_feature] / total
        data_c = data[c_feature] / total

        x_ternary = 0.5 * (2 * data_b + data_c) / (data_a + data_b + data_c)
        y_ternary = (np.sqrt(3) / 2) * data_c / (data_a + data_b + data_c)

        valid_idx = ~(x_ternary.isna() | y_ternary.isna())
        x_ternary = x_ternary[valid_idx]
        y_ternary = y_ternary[valid_idx]
        z_ternary = data[target][valid_idx]

        if len(x_ternary) < 3:
            return None

        if plot_type == 'scatter':
            return self._plot_ternary_scatter(x_ternary, y_ternary, z_ternary, a_feature, b_feature, c_feature, target)
        elif plot_type == 'heatmap':
            return self._plot_ternary_heatmap(x_ternary, y_ternary, z_ternary, a_feature, b_feature, c_feature, target, grid_resolution)
        elif plot_type == 'contour':
            return self._plot_ternary_contour(x_ternary, y_ternary, z_ternary, a_feature, b_feature, c_feature, target, grid_resolution)
        elif plot_type == 'heatmap_with_points':
            return self._plot_ternary_with_points(x_ternary, y_ternary, z_ternary, a_feature, b_feature, c_feature, target, grid_resolution)
        else:
            return self._plot_ternary_scatter(x_ternary, y_ternary, z_ternary, a_feature, b_feature, c_feature, target)

    def _plot_ternary_scatter(self, x, y, z, a_feature, b_feature, c_feature, target):
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidth=1)

        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)

        ax.text(-0.05, -0.05, self._label(a_feature), fontsize=12, fontweight='bold')
        ax.text(1.05, -0.05, self._label(b_feature), fontsize=12, fontweight='bold')
        ax.text(0.5, np.sqrt(3)/2 + 0.05, self._label(c_feature), fontsize=12, fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(self._label(target), fontsize=14, fontweight='bold')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Ternary Plot: {self._label(a_feature)} - {self._label(b_feature)} - {self._label(c_feature)}',
                     fontsize=15, fontweight='bold')

        plt.tight_layout()
        return fig

    def _plot_ternary_heatmap(self, x, y, z, a_feature, b_feature, c_feature, target, grid_resolution=50):
        xi = np.linspace(0, 1, grid_resolution)
        yi = np.linspace(0, np.sqrt(3)/2, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except Exception:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except Exception:
                return None

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = (yi_grid <= np.sqrt(3) * xi_grid) & (yi_grid <= -np.sqrt(3) * (xi_grid - 1))
        zi_masked = np.ma.masked_where(~mask, zi)

        im = ax.imshow(zi_masked.T, origin='lower', extent=[0, 1, 0, np.sqrt(3)/2],
                       aspect='auto', cmap='viridis', alpha=0.9)

        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)

        ax.text(-0.05, -0.05, self._label(a_feature), fontsize=12, fontweight='bold')
        ax.text(1.05, -0.05, self._label(b_feature), fontsize=12, fontweight='bold')
        ax.text(0.5, np.sqrt(3)/2 + 0.05, self._label(c_feature), fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self._label(target), fontsize=14, fontweight='bold')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Ternary Heatmap: {self._label(a_feature)} - {self._label(b_feature)} - {self._label(c_feature)}',
                     fontsize=15, fontweight='bold')

        plt.tight_layout()
        return fig

    def _plot_ternary_contour(self, x, y, z, a_feature, b_feature, c_feature, target, grid_resolution=50):
        xi = np.linspace(0, 1, grid_resolution)
        yi = np.linspace(0, np.sqrt(3)/2, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except Exception:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except Exception:
                return None

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = (yi_grid <= np.sqrt(3) * xi_grid) & (yi_grid <= -np.sqrt(3) * (xi_grid - 1))
        zi_masked = np.ma.masked_where(~mask, zi)

        contour = ax.contour(xi_grid, yi_grid, zi_masked, levels=15, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=11, fmt='%1.2f')

        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)

        ax.text(-0.05, -0.05, self._label(a_feature), fontsize=12, fontweight='bold')
        ax.text(1.05, -0.05, self._label(b_feature), fontsize=12, fontweight='bold')
        ax.text(0.5, np.sqrt(3)/2 + 0.05, self._label(c_feature), fontsize=12, fontweight='bold')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Ternary Contour: {self._label(a_feature)} - {self._label(b_feature)} - {self._label(c_feature)}',
                     fontsize=15, fontweight='bold')

        plt.tight_layout()
        return fig

    def _plot_ternary_with_points(self, x, y, z, a_feature, b_feature, c_feature, target, grid_resolution=50):
        xi = np.linspace(0, 1, grid_resolution)
        yi = np.linspace(0, np.sqrt(3)/2, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except Exception:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except Exception:
                return None

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = (yi_grid <= np.sqrt(3) * xi_grid) & (yi_grid <= -np.sqrt(3) * (xi_grid - 1))
        zi_masked = np.ma.masked_where(~mask, zi)

        im = ax.imshow(zi_masked.T, origin='lower', extent=[0, 1, 0, np.sqrt(3)/2],
                       aspect='auto', cmap='viridis', alpha=0.7)

        ax.scatter(x, y, c=z, s=40, cmap='viridis',
                   edgecolors='black', linewidth=1, vmin=zi.min(), vmax=zi.max())

        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)

        ax.text(-0.05, -0.05, self._label(a_feature), fontsize=12, fontweight='bold')
        ax.text(1.05, -0.05, self._label(b_feature), fontsize=12, fontweight='bold')
        ax.text(0.5, np.sqrt(3)/2 + 0.05, self._label(c_feature), fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self._label(target), fontsize=14, fontweight='bold')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Ternary Heatmap with Points: {self._label(a_feature)} - {self._label(b_feature)} - {self._label(c_feature)}',
                     fontsize=15, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_temperature_slice(self, feature=None, target=None):
        if target is None:
            target = 'α·106 (K-1)'

        if target not in self.df.columns:
            return None

        if 'T_min' not in self.df.columns or 'T_max' not in self.df.columns:
            return None

        if feature is None:
            feature = 'B'

        if feature not in self.df.columns:
            return None

        data = self.df[[feature, 'T_min', 'T_max', target]].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        groups = data.groupby(feature)

        for name, group in groups:
            if len(group) > 1:
                temp_avg = (group['T_min'] + group['T_max']) / 2
                ax.scatter(temp_avg, group[target], label=name, s=50, alpha=0.7)

        ax.set_xlabel('Average Temperature (°C)', fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(target), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(target)} vs Temperature (colored by {feature})',
                     fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CATEGORY 5: BUBBLE PLOTS
    # ------------------------------------------------------------------

    def plot_bubble_4d(self, x_feature, y_target, color_feature, size_feature, shape_by='method'):
        if x_feature not in self.df.columns or color_feature not in self.df.columns:
            return None

        if size_feature not in self.df.columns:
            return None

        if y_target not in self.df.columns:
            return None

        data = self.df[[x_feature, y_target, color_feature, size_feature]].dropna(subset=[y_target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        fig, ax = plt.subplots(figsize=(10, 8))

        size_min = data[size_feature].min()
        size_max = data[size_feature].max()
        size_norm = (data[size_feature] - size_min) / (size_max - size_min + 1e-10)
        size_scaled = 20 + size_norm * 60

        cmap = plt.cm.viridis
        norm = plt.Normalize(data[color_feature].min(), data[color_feature].max())

        scatter = ax.scatter(
            data[x_feature],
            data[y_target],
            s=size_scaled,
            c=data[color_feature],
            cmap='viridis',
            norm=norm,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(self._label(color_feature), fontsize=14, fontweight='bold')

        create_bubble_legend(
            ax, data, size_feature, size_min, size_max, data[size_feature].mean(),
            use_entire_terms=self.use_entire_terms
        )

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_target), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(y_target)} vs {self._label(x_feature)}',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_compositional_bubble(self, x_feature='[B\']', target='α·106 (K-1)',
                                  size_feature='δ', color_feature='B',
                                  color_map=None,
                                  bin_numeric=True, n_bins=10):
        """Compositional bubble; color variable may be quantile-binned to keep it consistent with violin."""
        if target not in self.df.columns:
            return None
        if x_feature not in self.df.columns:
            return None
        if size_feature not in self.df.columns:
            return None
        if color_feature not in self.df.columns:
            return None

        data = self.df[[x_feature, target, color_feature, size_feature]].dropna(subset=[target])
        data = data.dropna(subset=[x_feature, size_feature])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing data")

        was_binned = False
        if bin_numeric:
            grouped_series, ordered_cats, was_binned = make_qcut_groups(
                data[color_feature], n_bins=n_bins
            )
            data = data.copy()
            data['__color__'] = grouped_series.values
            actual_color_col = '__color__'
            actual_color_name = f'{self._label(color_feature)} ({len(ordered_cats)} quantile bins)' if was_binned else self._label(color_feature)
        else:
            actual_color_col = color_feature
            actual_color_name = self._label(color_feature)

        data = data.dropna(subset=[actual_color_col])

        if len(data) < 3:
            return None

        fig, ax = plt.subplots(figsize=(13, 8))

        size_min = float(data[size_feature].min())
        size_max = float(data[size_feature].max())
        size_range = (size_max - size_min) if (size_max - size_min) > 1e-12 else 1.0
        size_norm = (data[size_feature] - size_min) / size_range
        size_scaled = 20 + size_norm * 60

        if (size_max - size_min) < 1e-12:
            size_scaled = pd.Series([50.0] * len(data), index=data.index)
            st.info(f"All values of '{self._label(size_feature)}' are identical — bubbles have equal size")

        is_categorical = (
            data[actual_color_col].dtype == object or
            isinstance(data[actual_color_col].dtype, pd.CategoricalDtype) or
            str(data[actual_color_col].dtype) == 'string'
        )

        if not is_categorical and not was_binned:
            if pd.api.types.is_numeric_dtype(data[actual_color_col]):
                if data[actual_color_col].nunique() <= 5:
                    is_categorical = True

        if was_binned:
            is_categorical = True

        is_numeric_color = (not is_categorical) and \
                           pd.api.types.is_numeric_dtype(data[actual_color_col])

        if is_numeric_color:
            scatter = ax.scatter(
                data[x_feature], data[target],
                s=size_scaled, c=data[actual_color_col], cmap='viridis',
                alpha=0.7, edgecolors='black', linewidth=0.8
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(actual_color_name, fontsize=14, fontweight='bold')

            size_legend = Legend(
                ax,
                [
                    Line2D([0], [0], marker='o', color='black', label=f'Min: {size_min:.4f}',
                           markersize=4, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5),
                    Line2D([0], [0], marker='o', color='black', label=f'Mean: {data[size_feature].mean():.4f}',
                           markersize=8, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5),
                    Line2D([0], [0], marker='o', color='black', label=f'Max: {size_max:.4f}',
                           markersize=12, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5)
                ],
                [f'Min: {size_min:.4f}',
                 f'Mean: {data[size_feature].mean():.4f}',
                 f'Max: {size_max:.4f}'],
                title=f'Size: {self._label(size_feature)}',
                loc='upper left',
                bbox_to_anchor=(1.25, 0.5),
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                fontsize=10,
                title_fontsize=11
            )
            ax.add_artist(size_legend)

        else:
            if was_binned:
                categories = [str(c) for c in ordered_cats if c in data[actual_color_col].astype(str).unique()]
            else:
                categories = [str(c) for c in data[actual_color_col].dropna().unique()]

            if color_map is None:
                color_map = get_stable_color_map(categories)

            legend_elements = []
            for cat in categories:
                color = color_map.get(str(cat), '#2C3E50')
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='black', label=str(cat),
                           markersize=8, markerfacecolor=color, linestyle='None')
                )

            cat_legend = ax.legend(
                handles=legend_elements,
                title=actual_color_name,
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                fontsize=10,
                title_fontsize=11,
                borderaxespad=0
            )
            ax.add_artist(cat_legend)

            colors = [color_map.get(str(b), '#2C3E50') for b in data[actual_color_col]]

            ax.scatter(
                data[x_feature], data[target],
                s=size_scaled, c=colors,
                alpha=0.7, edgecolors='black', linewidth=0.8
            )

            size_legend = Legend(
                ax,
                [
                    Line2D([0], [0], marker='o', color='black', label=f'Min: {size_min:.4f}',
                           markersize=4, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5),
                    Line2D([0], [0], marker='o', color='black', label=f'Mean: {data[size_feature].mean():.4f}',
                           markersize=8, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5),
                    Line2D([0], [0], marker='o', color='black', label=f'Max: {size_max:.4f}',
                           markersize=12, markerfacecolor='gray', linestyle='None', markeredgewidth=0.5)
                ],
                [f'Min: {size_min:.4f}',
                 f'Mean: {data[size_feature].mean():.4f}',
                 f'Max: {size_max:.4f}'],
                title=f'Size: {self._label(size_feature)}',
                loc='upper left',
                bbox_to_anchor=(1.02, 0.0),
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                fontsize=10,
                title_fontsize=11,
                borderaxespad=0
            )
            ax.add_artist(size_legend)

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(target), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        fig.subplots_adjust(right=0.72)

        return fig

    def plot_bubble_with_trend(self, x_feature, y_target, color_feature=None):
        if x_feature not in self.df.columns or y_target not in self.df.columns:
            return None

        data = self.df[[x_feature, y_target]].dropna(subset=[y_target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        fig, ax = plt.subplots(figsize=(10, 8))

        if color_feature and color_feature in self.df.columns:
            color_data = self.df.loc[data.index, color_feature]
            scatter = ax.scatter(
                data[x_feature], data[y_target],
                c=color_data, cmap='viridis',
                s=50, alpha=0.7, edgecolors='black', linewidth=0.8
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(self._label(color_feature), fontsize=14, fontweight='bold')
        else:
            ax.scatter(data[x_feature], data[y_target],
                       color='#3498DB', s=50, alpha=0.7, edgecolors='black', linewidth=0.8)

        try:
            X = data[[x_feature]].values
            y = data[y_target].values

            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)

            x_range = np.linspace(X.min(), X.max(), 100)
            y_range = reg.predict(x_range.reshape(-1, 1))

            ax.plot(x_range, y_range, color='red', linewidth=2, label=f'Regression (R² = {r2:.3f})')
            ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black', fontsize=11)

        except Exception:
            pass

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_target), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(y_target)} vs {self._label(x_feature)}',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_bubble_filtered_by_ph2o(self, x_feature, y_target, color_feature=None):
        if x_feature not in self.df.columns or y_target not in self.df.columns:
            return None

        if 'pH2O' not in self.df.columns:
            return None

        data = self.df[[x_feature, y_target, 'pH2O']].dropna(subset=[y_target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        fig, ax = plt.subplots(figsize=(10, 8))

        ph2o_groups = pd.cut(data['pH2O'], bins=4, labels=['Very Low', 'Low', 'Medium', 'High'])
        colors = ['#2C3E50', '#3498DB', '#F39C12', '#E74C3C']

        for i, group in enumerate(ph2o_groups.cat.categories):
            group_data = data[ph2o_groups == group]
            if len(group_data) > 0:
                ax.scatter(
                    group_data[x_feature], group_data[y_target],
                    color=colors[i % len(colors)],
                    label=group,
                    s=50, alpha=0.7, edgecolors='black', linewidth=0.8
                )

        ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black', fontsize=11)

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_target), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(y_target)} vs {self._label(x_feature)} (colored by pH₂O)',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_bubble_size_delta(self, x_feature, y_target, color_feature=None):
        if x_feature not in self.df.columns or y_target not in self.df.columns:
            return None

        if 'δ' not in self.df.columns:
            return None

        data = self.df[[x_feature, y_target, 'δ']].dropna(subset=[y_target])

        if len(data) < 3:
            return None

        st.info(f"Points used: {len(data)} of {len(self.df)}")
        if len(data) < len(self.df):
            st.warning(f"Excluded {len(self.df) - len(data)} rows with missing target")

        fig, ax = plt.subplots(figsize=(10, 8))

        delta_min = data['δ'].min()
        delta_max = data['δ'].max()
        delta_norm = (data['δ'] - delta_min) / (delta_max - delta_min + 1e-10)
        size_scaled = 20 + delta_norm * 60

        if color_feature and color_feature in self.df.columns:
            color_data = self.df.loc[data.index, color_feature]
            scatter = ax.scatter(
                data[x_feature], data[y_target],
                s=size_scaled,
                c=color_data,
                cmap='viridis',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.8
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(self._label(color_feature), fontsize=14, fontweight='bold')
        else:
            ax.scatter(
                data[x_feature], data[y_target],
                s=size_scaled,
                c='#3498DB',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.8
            )

        create_bubble_legend(
            ax, data, 'δ', delta_min, delta_max, data['δ'].mean(),
            use_entire_terms=self.use_entire_terms
        )

        ax.set_xlabel(self._label(x_feature), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label(y_target), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label(y_target)} vs {self._label(x_feature)} (Size = {self._label("δ")})',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CATEGORY 6: SPECIALIZED PLOTS
    # ------------------------------------------------------------------

    def plot_t_bends_vs_delta(self):
        if 'δ' not in self.df.columns or 'T_bends_1' not in self.df.columns:
            return None

        data = self.df[['δ', 'T_bends_1', 'B', 'method']].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['δ'], group['T_bends_1'], label=b_cation, s=50, alpha=0.7)

        ax.set_xlabel(self._label('δ'), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label('T_bends_1'), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label("T_bends_1")} vs {self._label("δ")} (colored by B-cation)',
                     fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_alpha_vs_beta(self):
        if 'α·106 (K-1)' not in self.df.columns or 'β' not in self.df.columns:
            return None

        data = self.df[['α·106 (K-1)', 'β', 'B', 'method']].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['α·106 (K-1)'], group['β'], label=b_cation, s=50, alpha=0.7)

        ax.set_xlabel(self._label('α·106 (K-1)'), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label('β'), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label("α·106 (K-1)")} vs {self._label("β")} (colored by B-cation)',
                     fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_beta_vs_ph2o(self):
        if 'β' not in self.df.columns or 'pH2O' not in self.df.columns:
            return None

        data = self.df[['β', 'pH2O', 'B']].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['pH2O'], group['β'], label=b_cation, s=50, alpha=0.7)

        ax.set_xlabel(self._label('pH2O') + ' (log scale)', fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label('β'), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label("β")} vs {self._label("pH2O")} (colored by B-cation)',
                     fontsize=15, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_alpha_vs_rAav(self):
        if 'α·106 (K-1)' not in self.df.columns or 'rAav' not in self.df.columns:
            return None

        data = self.df[['α·106 (K-1)', 'rAav', 'A', 'method']].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        for a_cation in data['A'].unique():
            group = data[data['A'] == a_cation]
            ax.scatter(group['rAav'], group['α·106 (K-1)'], label=a_cation, s=50, alpha=0.7)

        ax.set_xlabel(self._label('rAav'), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label('α·106 (K-1)'), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label("α·106 (K-1)")} vs {self._label("rAav")} (colored by A-cation)',
                     fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_beta_vs_chiBav(self):
        if 'β' not in self.df.columns or 'χBav' not in self.df.columns:
            return None

        data = self.df[['β', 'χBav', 'B']].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['χBav'], group['β'], label=b_cation, s=50, alpha=0.7)

        ax.set_xlabel(self._label('χBav'), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label('β'), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label("β")} vs {self._label("χBav")} (colored by B-cation)',
                     fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    def plot_t_bends_vs_t_stab(self):
        if 'T_bends_1' not in self.df.columns or 'T_stab' not in self.df.columns:
            return None

        data = self.df[['T_bends_1', 'T_stab', 'B']].dropna()

        if len(data) < 5:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['T_stab'], group['T_bends_1'], label=b_cation, s=50, alpha=0.7)

        ax.set_xlabel(self._label('T_stab'), fontsize=14, fontweight='bold')
        ax.set_ylabel(self._label('T_bends_1'), fontsize=14, fontweight='bold')
        ax.set_title(f'{self._label("T_bends_1")} vs {self._label("T_stab")} (colored by B-cation)',
                     fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ADDITIONAL PLOTS
    # ------------------------------------------------------------------

    def plot_pairplot_colored(self, features, hue='method'):
        if len(features) < 2 or len(features) > 5:
            return None

        if hue not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        plot_cols = valid_features + [hue]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()

        if len(data) < 5:
            return None

        if hue == 'method':
            palette = COLOR_PALETTES['method']
        elif hue == 'B':
            palette = COLOR_PALETTES['B_cation']
        elif hue == 'A':
            palette = COLOR_PALETTES['A_cation']
        else:
            palette = 'viridis'

        renamed = data.rename(columns={c: self._label(c) for c in data.columns})
        display_vars = [self._label(v) for v in valid_features]
        display_hue = self._label(hue)

        g = sns.pairplot(
            renamed,
            vars=display_vars,
            hue=display_hue,
            palette=palette,
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30},
            diag_kws={'alpha': 0.6}
        )

        g.fig.suptitle(f'Pairplot of Selected Features (colored by {display_hue})',
                       fontsize=15, fontweight='bold', y=1.02)
        g.fig.set_size_inches(12, 10)

        return g.fig

    def plot_pairplot_with_params(self, features, hue='method', diag_kind='kde', alpha=0.6, marker_size=30):
        if len(features) < 2 or len(features) > 5:
            return None

        if hue not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 2:
            return None

        plot_cols = valid_features + [hue]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()

        if len(data) < 5:
            return None

        if hue == 'method':
            palette = COLOR_PALETTES['method']
        elif hue == 'B':
            palette = COLOR_PALETTES['B_cation']
        elif hue == 'A':
            palette = COLOR_PALETTES['A_cation']
        else:
            palette = 'viridis'

        if diag_kind == 'None':
            diag_kind = None

        renamed = data.rename(columns={c: self._label(c) for c in data.columns})
        display_vars = [self._label(v) for v in valid_features]
        display_hue = self._label(hue)

        g = sns.pairplot(
            renamed,
            vars=display_vars,
            hue=display_hue,
            palette=palette,
            diag_kind=diag_kind,
            plot_kws={'alpha': alpha, 's': marker_size},
            diag_kws={'alpha': alpha} if diag_kind is not None else None
        )

        g.fig.suptitle(f'Pairplot of Selected Features (colored by {display_hue})',
                       fontsize=15, fontweight='bold', y=1.02)
        g.fig.set_size_inches(12, 10)

        return g.fig

    def plot_radar_chart(self, features, group_by='method'):
        if len(features) < 3:
            return None

        if group_by not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 3:
            return None

        groups = self.df.groupby(group_by)

        means = {}
        for name, group in groups:
            if len(group) > 3:
                means[name] = group[valid_features].mean().values

        if len(means) == 0:
            return None

        all_values = np.array(list(means.values()))
        all_values_norm = (all_values - all_values.min(axis=0)) / (all_values.max(axis=0) - all_values.min(axis=0) + 1e-10)

        angles = np.linspace(0, 2 * np.pi, len(valid_features), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']

        for i, (name, values_norm) in enumerate(zip(means.keys(), all_values_norm)):
            values = values_norm.tolist()
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        display_labels = [self._label(f) for f in valid_features]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_labels, fontsize=10)
        ax.set_title(f'Radar Chart: Comparison by {group_by}', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_parallel_coordinates(self, features, target='α·106 (K-1)'):
        if len(features) < 3:
            return None

        if target not in self.df.columns:
            return None

        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)

        if len(valid_features) < 3:
            return None

        plot_cols = valid_features + [target]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()

        if len(data) < 5:
            return None

        scaler = StandardScaler()
        data_scaled = data.copy()
        data_scaled[valid_features] = scaler.fit_transform(data[valid_features])

        fig, ax = plt.subplots(figsize=(12, 6))

        target_norm = (data[target] - data[target].min()) / (data[target].max() - data[target].min() + 1e-10)
        colors = plt.cm.viridis(target_norm)

        for i, row in data_scaled.iterrows():
            ax.plot(range(len(valid_features) + 1),
                    row[valid_features + [target]].values,
                    color=colors[i], alpha=0.5, linewidth=0.8)

        display_labels = [self._label(f) for f in valid_features] + [self._label(target)]
        ax.set_xticks(range(len(valid_features) + 1))
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Standardized Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Parallel Coordinates (colored by {self._label(target)})',
                     fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        plt.tight_layout()
        return fig


# ============================================================================
# 5. USER INTERFACE
# ============================================================================

def main():
    """Streamlit application entry point."""
    apply_scientific_style()

    with st.sidebar:
        st.title("⚡ Perovskite Analyzer")
        st.markdown("---")

        st.subheader("📂 Data Upload")
        data_text = st.text_area(
            "Paste data in TSV/CSV format:",
            height=200,
            placeholder="№\tA\tA'\tB\tB'\tD1\tD2\t[A']\t[B']\t[D1]\t[D2]\tδ\tmethod\tβ\t∆T, °C\tα·106 (K-1)\tT(bends), °C\tαav·106 (K-1)\tpH2O\tRef\n1\tBa\t-\tCe\tZr\tY\tYb\t0\t0.1\t0.1\t0.1\t0.1\tdilatometry\t0.0073\t27-1000\t10.6\t400;600\t10.6;4.73;10.1\t0.0001\t10.15826/chimtech.2024.11.4.22"
        )

        col1, col2 = st.columns(2)
        with col1:
            load_button = st.button("🚀 Load and process", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        if clear_button:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if load_button and data_text:
            with st.spinner("Processing data..."):
                processor = DataProcessor()
                df_raw = processor.parse_text_data(data_text)

                if df_raw is not None and len(df_raw) > 0:
                    df_clean = processor.clean_data(df_raw)

                    st.session_state['df_raw'] = df_raw
                    st.session_state['df_clean'] = df_clean
                    st.session_state['data_loaded'] = True

                    engine = DescriptorEngine(df_clean)
                    df_descriptors = engine.calculate_all()
                    st.session_state['df_descriptors'] = df_descriptors
                    st.session_state['filtered_df'] = df_descriptors.copy()

                    st.success(f"✅ Loaded {len(df_clean)} records")
                    st.info(f"📊 Computed {len(df_descriptors.columns) - len(df_clean.columns)} descriptors")
                else:
                    st.error("❌ Failed to load data. Check the format.")

        st.markdown("---")

        st.subheader("🏷️ Label options")
        use_entire_terms = st.checkbox(
            "Use entire terms",
            value=False,
            help="Show full descriptive terms instead of technical descriptor names on axes and legends."
        )

        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            if 'df_descriptors' in st.session_state:
                st.subheader("🔍 Filters")

                df = st.session_state['df_descriptors']

                st.subheader("Measurement method")
                method_options = df['method'].unique().tolist() if 'method' in df.columns else []
                method_filter = st.multiselect(
                    "Select methods:",
                    options=method_options,
                    default=method_options
                )

                st.subheader("A-cation")
                a_options = df['A'].unique().tolist() if 'A' in df.columns else []
                a_filter = st.multiselect(
                    "Select A-cations:",
                    options=a_options,
                    default=a_options
                )

                st.subheader("B-cation")
                b_options = df['B'].unique().tolist() if 'B' in df.columns else []
                b_filter = st.multiselect(
                    "Select B-cations:",
                    options=b_options,
                    default=b_options
                )

                st.subheader("δ range")
                if 'δ' in df.columns:
                    delta_min = float(df['δ'].min())
                    delta_max = float(df['δ'].max())
                    delta_range = st.slider(
                        "δ:",
                        min_value=delta_min,
                        max_value=delta_max,
                        value=(delta_min, delta_max),
                        step=0.01
                    )

                st.subheader("pH₂O")
                if 'pH2O' in df.columns:
                    ph2o_min = float(df['pH2O'].min())
                    ph2o_max = float(df['pH2O'].max())
                    ph2o_range = st.slider(
                        "pH₂O:",
                        min_value=ph2o_min,
                        max_value=ph2o_max,
                        value=(ph2o_min, ph2o_max),
                        step=0.001
                    )

                with st.expander("Descriptor filters"):
                    if 't' in df.columns:
                        t_min = float(df['t'].min())
                        t_max = float(df['t'].max())
                        t_range = st.slider(
                            "Tolerance factor (t):",
                            min_value=t_min,
                            max_value=t_max,
                            value=(t_min, t_max),
                            step=0.01
                        )

                    if 'rAav' in df.columns:
                        ra_min = float(df['rAav'].min())
                        ra_max = float(df['rAav'].max())
                        ra_range = st.slider(
                            "rAav (Å):",
                            min_value=ra_min,
                            max_value=ra_max,
                            value=(ra_min, ra_max),
                            step=0.05
                        )

                st.markdown("---")

                filtered_df = df.copy()

                if method_filter:
                    filtered_df = filtered_df[filtered_df['method'].isin(method_filter)]

                if a_filter:
                    filtered_df = filtered_df[filtered_df['A'].isin(a_filter)]

                if b_filter:
                    filtered_df = filtered_df[filtered_df['B'].isin(b_filter)]

                if 'δ' in df.columns:
                    filtered_df = filtered_df[(filtered_df['δ'] >= delta_range[0]) & (filtered_df['δ'] <= delta_range[1])]

                if 'pH2O' in df.columns:
                    filtered_df = filtered_df[(filtered_df['pH2O'] >= ph2o_range[0]) & (filtered_df['pH2O'] <= ph2o_range[1])]

                if 't' in df.columns:
                    filtered_df = filtered_df[(filtered_df['t'] >= t_range[0]) & (filtered_df['t'] <= t_range[1])]

                if 'rAav' in df.columns:
                    filtered_df = filtered_df[(filtered_df['rAav'] >= ra_range[0]) & (filtered_df['rAav'] <= ra_range[1])]

                st.session_state['filtered_df'] = filtered_df
                st.caption(f"📊 Rows after filtering: {len(filtered_df)}")
            else:
                st.warning("⚠️ Data loaded but descriptors were not computed. Please click 'Load and process'.")

    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        if 'df_descriptors' in st.session_state and 'filtered_df' in st.session_state:

            df = st.session_state['filtered_df'] if 'filtered_df' in st.session_state else st.session_state['df_descriptors']

            use_entire_terms = st.session_state.get('use_entire_terms', use_entire_terms) if 'use_entire_terms' in st.session_state else use_entire_terms
            use_entire_terms = st.session_state.get('Use entire terms', use_entire_terms) if 'Use entire terms' in st.session_state else use_entire_terms
            use_entire_terms = bool(use_entire_terms)

            viz = VisualizationEngine(df, use_entire_terms=use_entire_terms)

            all_descriptors = [col for col in ALL_DESCRIPTORS_FOR_SELECT if col in df.columns]
            extra_descriptors = [col for col in df.columns if col not in all_descriptors and
                                 col not in ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Ref', 'method', '∆T, °C', 'T(bends), °C', 'αav·106 (K-1)']]
            extra_descriptors = [col for col in extra_descriptors if df[col].dtype in ['float64', 'int64']]
            extra_descriptors = [col for col in extra_descriptors if df[col].isna().sum() / len(df) < 0.5]

            all_descriptors = list(set(all_descriptors + extra_descriptors))
            all_descriptors = [col for col in all_descriptors if col in df.columns]

            viz.set_features(all_descriptors)

            tabs = st.tabs([
                "📊 Overview",
                "🧮 Descriptors",
                "📈 Correlations",
                "🗺️ Maps & Clusters",
                "🎯 Visualization",
                "📤 Export"
            ])

            # ---------------- TAB 1: Overview ----------------
            with tabs[0]:
                st.header("📊 Data Overview")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total records", len(df))
                with col2:
                    unique_compositions = df.groupby(['A', 'B']).ngroups if 'A' in df.columns and 'B' in df.columns else 0
                    st.metric("Unique compositions", unique_compositions)
                with col3:
                    methods = df['method'].nunique() if 'method' in df.columns else 0
                    st.metric("Measurement methods", methods)
                with col4:
                    refs = df['Ref'].nunique() if 'Ref' in df.columns else 0
                    st.metric("Sources (Ref)", refs)

                st.subheader("📄 Data")
                st.dataframe(df, use_container_width=True, height=400)

                st.subheader("📊 Numeric column statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

                st.subheader("📈 Target variable distributions")
                target_cols = [t for t in TARGET_VARIABLES if t in df.columns]
                if target_cols:
                    cols = st.columns(min(len(target_cols), 3))
                    for i, target in enumerate(target_cols[:3]):
                        with cols[i]:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            data = df[target].dropna()
                            if len(data) > 0:
                                ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='#3498DB')
                                ax.set_xlabel(label_for(target, use_entire_terms), fontsize=12, fontweight='bold')
                                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                                ax.grid(True, alpha=0.3)
                                style_axes(ax)
                                st.pyplot(fig)
                            else:
                                st.write("No data")

            # ---------------- TAB 2: Descriptors ----------------
            with tabs[1]:
                st.header("🧮 Computed Descriptors")

                descriptor_cols = [col for col in df.columns if col not in ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Ref', 'method', '∆T, °C']]
                descriptor_cols = [col for col in descriptor_cols if df[col].dtype in ['float64', 'int64']]

                st.info(f"📊 Total computed: {len(descriptor_cols)} descriptors")

                with st.expander("📋 Descriptors by category"):
                    categories = {
                        "Geometric": ['rAav', 'rBav', 't', 'D_t', 'octahedral_factor', 'Δr_AB', 'Δr_AB_norm', 'σ²_rA', 'σ²_rB', 'V_cell', 'r_ratio_AB'],
                        "Electronegativity": ['χAav', 'χBav', 'Δχ_AB', 'χ_ratio_AB', 'ionicity_AO', 'ionicity_BO', 'acidity_AO', 'acidity_BO', 'Δacidity', 'χ_total', 'χ_ratio_t'],
                        "Thermodynamic": ['S_config_A', 'S_config_B', 'V_Bav', 'Vo_proxy', 'ΔH_hydr', 'E_BO', 'ρ'],
                        "Mass": ['M_Aav', 'M_Bav', 'M_total', 'M_ratio_AB', 'M_rA', 'M_χA'],
                        "Defect": ['δ_actual', 'Z_eff_B', 'proton_affinity', 'E_vac'],
                        "Specific": ['alpha_beta_ratio', 'T_stab', 'δ_χB', 'δ_rB'],
                        "Compositional": ['B\'_conc', 'D_total', 'D_B_ratio', 'total_dopant']
                    }

                    for category, descs in categories.items():
                        present = [d for d in descs if d in df.columns]
                        st.write(f"**{category}**: {len(present)}/{len(descs)} descriptors")
                        st.write(", ".join(present))
                        st.write("")

                st.subheader("📊 Descriptor table")
                if descriptor_cols:
                    display_cols = ['№' if '№' in df.columns else None, 'A' if 'A' in df.columns else None, 'B' if 'B' in df.columns else None] + descriptor_cols[:10]
                    display_cols = [c for c in display_cols if c is not None]
                    st.dataframe(df[display_cols], use_container_width=True, height=300)

                st.subheader("📈 Descriptor distribution")
                if descriptor_cols:
                    selected_desc = st.selectbox("Select a descriptor to display:", descriptor_cols)
                    if selected_desc:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        data = df[selected_desc].dropna()
                        if len(data) > 0:
                            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='#2ECC71')
                            ax.set_xlabel(label_for(selected_desc, use_entire_terms), fontsize=12, fontweight='bold')
                            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            style_axes(ax)
                            st.pyplot(fig)
                        else:
                            st.warning("No data for the selected descriptor")

            # ---------------- TAB 3: Correlations ----------------
            with tabs[2]:
                st.header("📈 Extended Correlation Analysis")

                if len(all_descriptors) > 1:
                    target_options = [t for t in TARGET_VARIABLES if t in df.columns]
                    if target_options:
                        target = st.selectbox(
                            "Select target variable:",
                            options=target_options
                        )

                        max_available = min(30, len(all_descriptors))
                        min_available = min(5, max_available)

                        n_features = st.slider(
                            "Number of top descriptors:",
                            min_value=min_available,
                            max_value=max_available,
                            value=min(20, max_available),
                            step=1
                        )

                        st.caption(f"ℹ️ Up to {n_features} most significant descriptors will be selected for {target}")

                        if st.button("🔍 Analyze correlations", use_container_width=True):
                            with st.spinner("Analyzing correlations..."):
                                analyzer = CorrelationAnalyzer(df)
                                top_features = analyzer.get_top_features(target, n=n_features)

                                if 'correlation_debug' in st.session_state:
                                    st.info(f"ℹ️ {st.session_state['correlation_debug']}")

                                if top_features:
                                    actual_n = len(top_features)

                                    if actual_n < n_features:
                                        st.warning(
                                            f"⚠️ Requested {n_features} descriptors, "
                                            f"but only {actual_n} were found (insufficient valid data)."
                                        )
                                    else:
                                        st.success(f"✅ Found {actual_n} significant descriptors")

                                    st.session_state['top_features'] = top_features
                                    st.session_state['top_features_n'] = actual_n
                                    st.session_state['top_features_target'] = target

                                    st.subheader(f"🏆 Top-{actual_n} descriptors for {target}")

                                    pearson_results = analyzer.pearson_correlation(target=target)
                                    if pearson_results is not None:
                                        top_df = pearson_results[pearson_results['feature'].isin(top_features)].copy()
                                        top_df['feature'] = top_df['feature'].apply(lambda f: label_for(f, use_entire_terms))
                                        st.dataframe(
                                            top_df[['feature', 'correlation', 'p_value', 'significant']],
                                            use_container_width=True
                                        )

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.subheader(f"📊 Top-{actual_n} correlations")
                                        if pearson_results is not None and len(pearson_results) > 0:
                                            fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * actual_n)))
                                            top_plot = pearson_results.head(actual_n).copy()
                                            display_names = [label_for(f, use_entire_terms) for f in top_plot['feature']]
                                            colors = ['red' if c < 0 else 'blue' for c in top_plot['correlation'].values]
                                            ax.barh(display_names, top_plot['correlation'], color=colors, alpha=0.7)
                                            ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
                                            ax.set_xlabel(f'Correlation with {label_for(target, use_entire_terms)}', fontsize=14, fontweight='bold')
                                            ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
                                            ax.set_title(f'Top {actual_n} Correlations with {label_for(target, use_entire_terms)}', fontsize=15, fontweight='bold')
                                            ax.grid(True, alpha=0.3)
                                            style_axes(ax)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                        else:
                                            st.warning("No correlations to display")

                                    with col2:
                                        st.subheader(f"🔗 Network graph (top-{actual_n})")
                                        G = analyzer.correlation_network(features=top_features, threshold=0.4)
                                        if G is not None and len(G.edges()) > 0:
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            pos = nx.spring_layout(G, k=1, seed=42)
                                            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                                                   node_size=500, alpha=0.8, ax=ax)
                                            node_labels = {n: label_for(n, use_entire_terms) for n in G.nodes()}
                                            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', ax=ax)
                                            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)
                                            ax.set_title(f'Correlation Network (top {actual_n})', fontsize=15, fontweight='bold')
                                            ax.axis('off')
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                        else:
                                            st.warning("Not enough connections to build the network")

                                    st.subheader(f"📊 Correlation matrix for top-{actual_n} descriptors")
                                    if actual_n > 20:
                                        st.warning(
                                            f"⚠️ A matrix of {actual_n} features may be hard to read. "
                                            f"Consider reducing the number of top descriptors."
                                        )
                                    corr_fig = viz.plot_correlation_matrix(features=top_features[:actual_n])
                                    if corr_fig is not None:
                                        st.pyplot(corr_fig)
                                    else:
                                        st.warning("Not enough data for the correlation matrix")

                                    pairplot_n = min(actual_n, 5)
                                    st.subheader(f"🎨 Pairplot of top-{pairplot_n} descriptors")
                                    if pairplot_n < 2:
                                        st.warning("Pairplot requires at least 2 features")
                                    else:
                                        pair_fig = viz.plot_pairplot_top5(features=top_features[:pairplot_n], target=target)
                                        if pair_fig is not None:
                                            st.pyplot(pair_fig)
                                        else:
                                            st.warning("Not enough data for Pairplot")
                                else:
                                    st.warning("No significant correlations found. Try another target or increase the number of features.")
                    else:
                        st.warning("No target variables in the data")
                else:
                    st.warning("Not enough descriptors for correlation analysis")

            # ---------------- TAB 4: Maps & Clusters ----------------
            with tabs[3]:
                st.header("🗺️ Concentration Maps and Clustering")

                sub_tabs = st.tabs(["Concentration Maps", "Ternary Plot", "PCA", "Clustering"])

                with sub_tabs[0]:
                    st.subheader("🗺️ Concentration maps")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_options = [d for d in TOP_DESCRIPTORS if d in df.columns]
                        x_axis = st.selectbox("X-axis:", options=x_options, index=0 if x_options else None)
                    with col2:
                        y_options = [d for d in TOP_DESCRIPTORS if d in df.columns and d != x_axis]
                        y_axis = st.selectbox("Y-axis:", options=y_options, index=min(1, len(y_options)-1) if y_options else None)
                    with col3:
                        color_options = [t for t in TARGET_VARIABLES if t in df.columns]
                        color_by = st.selectbox("Color (target variable):", options=color_options, index=0 if color_options else None)

                    map_type = st.radio(
                        "Map type:",
                        options=['Heatmap', 'Contour', 'Heatmap + points'],
                        horizontal=True
                    )

                    if x_axis and y_axis and color_by:
                        if st.button("🗺️ Build map", use_container_width=True):
                            with st.spinner("Building map..."):
                                if map_type == 'Heatmap':
                                    fig = viz.plot_heatmap_2d(x_axis, y_axis, color_by)
                                elif map_type == 'Contour':
                                    fig = viz.plot_contour(x_axis, y_axis, color_by)
                                else:
                                    fig = viz.plot_heatmap_with_points(x_axis, y_axis, color_by)

                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data to build the map")

                with sub_tabs[1]:
                    st.subheader("📐 Extended Ternary Plot")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        a_vertex = st.selectbox(
                            "Vertex A:",
                            options=[d for d in TERNARY_VERTEX_OPTIONS if d in df.columns],
                            index=0
                        )
                    with col2:
                        b_vertex = st.selectbox(
                            "Vertex B:",
                            options=[d for d in TERNARY_VERTEX_OPTIONS if d in df.columns and d != a_vertex],
                            index=min(1, len([d for d in TERNARY_VERTEX_OPTIONS if d in df.columns and d != a_vertex])-1) if len([d for d in TERNARY_VERTEX_OPTIONS if d in df.columns and d != a_vertex]) > 0 else 0
                        )
                    with col3:
                        c_vertex = st.selectbox(
                            "Vertex C:",
                            options=[d for d in TERNARY_VERTEX_OPTIONS if d in df.columns and d not in [a_vertex, b_vertex]],
                            index=0
                        )

                    ternary_target = st.selectbox(
                        "Target variable:",
                        options=[t for t in TARGET_VARIABLES if t in df.columns],
                        key="ternary_target"
                    )

                    ternary_map_type = st.radio(
                        "Ternary map type:",
                        options=['Scatter', 'Heatmap', 'Contour', 'Heatmap + points'],
                        horizontal=True
                    )

                    if a_vertex and b_vertex and c_vertex and ternary_target:
                        if st.button("🔺 Build Ternary Plot", use_container_width=True):
                            with st.spinner("Building Ternary Plot..."):
                                fig = viz.plot_ternary_advanced(
                                    a_vertex, b_vertex, c_vertex,
                                    ternary_target,
                                    plot_type=ternary_map_type.lower().replace(' ', '_')
                                )
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data for Ternary Plot")

                with sub_tabs[2]:
                    st.subheader("📊 PCA analysis")

                    if st.button("📊 Build Biplot", use_container_width=True):
                        with st.spinner("Building Biplot..."):
                            fig = viz.plot_pca_biplot()
                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.warning("Not enough data for PCA")

                    st.subheader("📉 Elbow Plot")
                    max_components = st.slider("Maximum number of components:", 5, 20, 10)
                    if st.button("📉 Build Elbow Plot", use_container_width=True):
                        fig = viz.plot_elbow(max_components=max_components)
                        if fig is not None:
                            st.pyplot(fig)
                        else:
                            st.warning("Not enough data for Elbow Plot")

                with sub_tabs[3]:
                    st.subheader("🔍 Clustering")

                    algo = st.selectbox(
                        "Algorithm:",
                        options=['K-means', 'DBSCAN', 'Hierarchical']
                    )

                    if algo == 'K-means':
                        n_clusters = st.slider("K (number of clusters):", 2, 10, 3)
                        if st.button("🔍 Run K-means clustering", use_container_width=True):
                            with st.spinner("Running K-means clustering..."):
                                features = all_descriptors[:20]
                                data, valid_features = _sanitize_for_ml(df, features)

                                if data is not None and len(valid_features) > 1 and len(data) > 10:
                                    scaler = StandardScaler()
                                    data_scaled = scaler.fit_transform(data)

                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                    labels = kmeans.fit_predict(data_scaled)

                                    score = silhouette_score(data_scaled, labels)
                                    st.success(f"✅ Silhouette Score: {score:.3f}")

                                    pca = PCA(n_components=2)
                                    pca_result = pca.fit_transform(data_scaled)

                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                                                         c=labels, cmap='viridis', s=50, alpha=0.7)
                                    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
                                    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
                                    ax.set_title(f'K-means Clustering (K={n_clusters})', fontsize=15, fontweight='bold')
                                    plt.colorbar(scatter, ax=ax, label='Cluster')
                                    ax.grid(True, alpha=0.3)
                                    style_axes(ax)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data for clustering (after removing NaN/inf)")

                    elif algo == 'DBSCAN':
                        eps = st.slider("eps (maximum distance):", 0.1, 5.0, 0.5, 0.1)
                        min_samples = st.slider("min_samples:", 2, 10, 5)
                        if st.button("🔍 Run DBSCAN clustering", use_container_width=True):
                            with st.spinner("Running DBSCAN clustering..."):
                                features = all_descriptors[:20]
                                data, valid_features = _sanitize_for_ml(df, features)

                                if data is not None and len(valid_features) > 1 and len(data) > 10:
                                    scaler = StandardScaler()
                                    data_scaled = scaler.fit_transform(data)

                                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                                    labels = dbscan.fit_predict(data_scaled)

                                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                    n_noise = list(labels).count(-1)

                                    st.info(f"📊 Clusters found: {n_clusters}, Noise points: {n_noise}")

                                    pca = PCA(n_components=2)
                                    pca_result = pca.fit_transform(data_scaled)

                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                                                         c=labels, cmap='viridis', s=50, alpha=0.7)
                                    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
                                    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
                                    ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})',
                                                 fontsize=15, fontweight='bold')
                                    plt.colorbar(scatter, ax=ax, label='Cluster')
                                    ax.grid(True, alpha=0.3)
                                    style_axes(ax)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data for clustering (after removing NaN/inf)")

                    else:
                        if st.button("🔍 Run hierarchical clustering", use_container_width=True):
                            with st.spinner("Running hierarchical clustering..."):
                                fig = viz.plot_silhouette()
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data for clustering")

            # ---------------- TAB 5: Visualization ----------------
            with tabs[4]:
                st.header("🎯 Interactive Visualization")

                plot_category = st.selectbox(
                    "Select plot category:",
                    options=[
                        "Bubble plots",
                        "Specialized plots",
                        "Radar charts",
                        "Pairplot"
                    ]
                )

                if plot_category == "Bubble plots":
                    st.subheader("💎 Bubble chart")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        x_feature = st.selectbox("X:", options=all_descriptors, index=0 if all_descriptors else None)
                    with col2:
                        y_target = st.selectbox("Y (target):", options=[t for t in TARGET_VARIABLES if t in df.columns], index=0)
                    with col3:
                        color_options = [d for d in all_descriptors if d != x_feature and d != y_target]
                        color_feature = st.selectbox("Color:", options=color_options, index=0 if color_options else None)
                    with col4:
                        size_options = [d for d in all_descriptors if d != x_feature and d != y_target and d != color_feature]
                        size_feature = st.selectbox("Size:", options=size_options, index=0 if size_options else None)

                    if x_feature and y_target and color_feature and size_feature:
                        if st.button("💎 Build bubble chart", use_container_width=True):
                            with st.spinner("Building..."):
                                fig = viz.plot_bubble_4d(x_feature, y_target, color_feature, size_feature)
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data to build the chart")

                    st.subheader("📊 Compositional Bubble + Violin plot")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_comp = st.selectbox(
                            "X descriptor:",
                            options=[d for d in all_descriptors if d in df.columns],
                            index=0,
                            key="comp_x"
                        )
                    with col2:
                        size_comp = st.selectbox(
                            "Bubble size:",
                            options=[d for d in all_descriptors if d in df.columns],
                            index=0,
                            key="comp_size"
                        )
                    with col3:
                        target_comp = st.selectbox(
                            "Target variable:",
                            options=[t for t in TARGET_VARIABLES if t in df.columns],
                            index=0,
                            key="comp_target"
                        )

                    color_comp = st.selectbox(
                        "Color (category):",
                        options=['B', 'A', 'method'] + [d for d in all_descriptors if d in df.columns and d not in [x_comp, size_comp, target_comp]],
                        index=0,
                        key="comp_color"
                    )

                    n_bins_comp = st.slider(
                        "Maximum number of groups (numeric variables get quantile-binned):",
                        min_value=5,
                        max_value=20,
                        value=10,
                        step=1,
                        key="comp_n_bins"
                    )

                    if st.button("📊 Build Bubble + Violin", use_container_width=True):
                        with st.spinner("Building..."):
                            try:
                                _color_series = df[color_comp].dropna()
                                _grouped, _ordered_cats, _was_binned = make_qcut_groups(
                                    _color_series, n_bins=n_bins_comp
                                )
                                color_categories = [str(c) for c in _ordered_cats]
                            except Exception:
                                color_categories = []
                                _was_binned = False

                            shared_color_map = get_stable_color_map(color_categories)

                            fig_bubble = viz.plot_compositional_bubble(
                                x_feature=x_comp,
                                target=target_comp,
                                size_feature=size_comp,
                                color_feature=color_comp,
                                color_map=shared_color_map,
                                bin_numeric=True,
                                n_bins=n_bins_comp
                            )

                            fig_violin = viz.plot_violin_by_B(
                                target=target_comp,
                                group_feature=color_comp,
                                show_stats=True,
                                sort_by_mass=(color_comp in ('B', 'A')),
                                color_map=shared_color_map,
                                bin_numeric=True,
                                n_bins=n_bins_comp
                            )

                            st.session_state['comp_bubble_fig'] = fig_bubble
                            st.session_state['comp_violin_fig'] = fig_violin
                            st.session_state['comp_violin_group'] = color_comp
                            st.session_state['comp_violin_bins'] = n_bins_comp
                            st.session_state['comp_was_binned'] = _was_binned

                    if st.session_state.get('comp_bubble_fig') is not None or \
                       st.session_state.get('comp_violin_fig') is not None:

                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.markdown("**🔵 Compositional Bubble**")
                            if st.session_state.get('comp_bubble_fig') is not None:
                                st.pyplot(st.session_state['comp_bubble_fig'])
                            else:
                                st.warning("Not enough data for Bubble")

                        with col_right:
                            violin_group = st.session_state.get('comp_violin_group', 'B')
                            violin_bins = st.session_state.get('comp_violin_bins', 10)
                            was_binned_state = st.session_state.get('comp_was_binned', False)

                            if was_binned_state:
                                st.markdown(
                                    f"**🎻 Violin plot (grouped by {violin_group}, "
                                    f"{violin_bins} quantile groups)**"
                                )
                            else:
                                st.markdown(f"**🎻 Violin plot (grouped by {violin_group})**")

                            if st.session_state.get('comp_violin_fig') is not None:
                                st.pyplot(st.session_state['comp_violin_fig'])
                            else:
                                st.warning("Not enough data for Violin")

                        with st.expander("📊 Summary statistics by group"):
                            violin_group = st.session_state.get('comp_violin_group', 'B')

                            if violin_group in df.columns:
                                _data_stat = df[[violin_group, target_comp]].dropna()

                                if len(_data_stat) > 0:
                                    _grouped_stat, _ordered_stat, _binned_stat = make_qcut_groups(
                                        _data_stat[violin_group], n_bins=violin_bins
                                    )

                                    _data_stat = _data_stat.copy()
                                    _data_stat['__group__'] = _grouped_stat.values
                                    _data_stat = _data_stat.dropna(subset=['__group__'])

                                    if len(_data_stat) > 0:
                                        stats_df = _data_stat.groupby('__group__')[target_comp].agg(
                                            ['count', 'mean', 'median', 'std', 'min', 'max']
                                        ).round(3)
                                        stats_df.columns = ['N', 'Mean', 'Median', 'Std', 'Min', 'Max']
                                        stats_df.index.name = violin_group
                                        st.dataframe(stats_df, use_container_width=True)
                                    else:
                                        st.info("Not enough data for summary statistics")

                elif plot_category == "Specialized plots":
                    st.subheader("🎯 Specialized plots")

                    spec_plots = [
                        "T(bends) vs δ",
                        "α vs β (compromise)",
                        "β vs pH₂O",
                        "α vs rAav",
                        "β vs χBav",
                        "T(bends) vs T_stab"
                    ]

                    selected_plot = st.selectbox("Select a plot:", spec_plots)

                    if st.button("📊 Build", use_container_width=True):
                        with st.spinner("Building..."):
                            fig = None
                            if selected_plot == "T(bends) vs δ":
                                fig = viz.plot_t_bends_vs_delta()
                            elif selected_plot == "α vs β (compromise)":
                                fig = viz.plot_alpha_vs_beta()
                            elif selected_plot == "β vs pH₂O":
                                fig = viz.plot_beta_vs_ph2o()
                            elif selected_plot == "α vs rAav":
                                fig = viz.plot_alpha_vs_rAav()
                            elif selected_plot == "β vs χBav":
                                fig = viz.plot_beta_vs_chiBav()
                            elif selected_plot == "T(bends) vs T_stab":
                                fig = viz.plot_t_bends_vs_t_stab()

                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.warning("Not enough data to build the plot")

                elif plot_category == "Radar charts":
                    st.subheader("📊 Radar chart for group comparison")

                    radar_features = st.multiselect(
                        "Select features (min 3):",
                        options=all_descriptors,
                        default=all_descriptors[:3] if len(all_descriptors) >= 3 else all_descriptors
                    )

                    group_by = st.selectbox(
                        "Group by:",
                        options=['method', 'B', 'A'],
                        index=0
                    )

                    if len(radar_features) >= 3:
                        if st.button("📊 Build radar chart", use_container_width=True):
                            with st.spinner("Building..."):
                                fig = viz.plot_radar_chart(radar_features, group_by)
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data")
                    else:
                        st.warning("Select at least 3 features")

                elif plot_category == "Pairplot":
                    st.header("🎨 Configurable Pairplot")

                    st.markdown("""
                    Configure the Pairplot to visualize relationships between features.
                    Color encodes group membership by the selected criterion.
                    """)

                    features = st.multiselect(
                        "Select features (2-5):",
                        options=all_descriptors,
                        default=all_descriptors[:3] if len(all_descriptors) >= 3 else all_descriptors
                    )

                    hue_options = ['method', 'B', 'A'] + [t for t in TARGET_VARIABLES if t in df.columns]
                    hue_by = st.selectbox(
                        "Color by (hue):",
                        options=hue_options,
                        index=0
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        diag_kind = st.selectbox(
                            "Diagonal plot type:",
                            options=['kde', 'hist', 'None'],
                            index=0
                        )
                    with col2:
                        alpha = st.slider(
                            "Marker transparency:",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.6,
                            step=0.1
                        )
                    with col3:
                        marker_size = st.slider(
                            "Marker size:",
                            min_value=10,
                            max_value=100,
                            value=30,
                            step=5
                        )

                    if len(features) >= 2:
                        if st.button("🎨 Build Pairplot", use_container_width=True):
                            with st.spinner("Building Pairplot..."):
                                fig = viz.plot_pairplot_with_params(
                                    features=features,
                                    hue=hue_by,
                                    diag_kind=diag_kind if diag_kind != 'None' else None,
                                    alpha=alpha,
                                    marker_size=marker_size
                                )
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Not enough data to build the Pairplot")
                    else:
                        st.warning("Select at least 2 features for Pairplot")

            # ---------------- TAB 6: Export ----------------
            with tabs[5]:
                st.header("📤 Export")

                st.subheader("📊 Data export")

                if st.button("📥 Download data with descriptors (CSV)", use_container_width=True):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="perovskite_descriptors.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.subheader("🖼️ Plot export")

                export_plots = st.multiselect(
                    "Select plot categories to export:",
                    options=[
                        "Correlation plots",
                        "PCA and clustering",
                        "Concentration maps",
                        "Bubble plots",
                        "Specialized plots",
                        "Pairplot and radar charts"
                    ]
                )

                if export_plots:
                    st.info(f"ℹ️ Selected {len(export_plots)} plot categories")

                    if st.button("📦 Generate plots for export", use_container_width=True):
                        with st.spinner("Generating plots..."):
                            st.warning("⚠️ ZIP export is under development")

                st.subheader("📄 Report export")

                if st.button("📄 Generate HTML report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        st.warning("⚠️ HTML report generation is under development")

        else:
            st.warning("⚠️ Data loaded but descriptors not computed. Please click 'Load and process' in the sidebar.")

    else:
        st.info("👈 Upload data via the sidebar to start the analysis")

        st.markdown("""
        ### 📋 Data format

        The data should be provided in TSV (tab-separated) or CSV format with the following columns:

        - `№` - record number
        - `A` - A-site cation
        - `A'` - A-site dopant
        - `B` - B-site cation
        - `B'` - isovalent dopant
        - `D1`, `D2` - acceptor dopants
        - `[A']`, `[B']`, `[D1]`, `[D2]` - concentrations
        - `δ` - vacancy concentration
        - `method` - measurement method
        - `β` - chemical expansion coefficient
        - `∆T, °C` - temperature range
        - `α·106 (K-1)` - thermal expansion coefficient
        - `T(bends), °C` - bend temperatures
        - `αav·106 (K-1)` - average thermal expansion coefficient per segment
        - `pH2O` - water partial pressure
        - `Ref` - source reference

        ### 🚀 Features

        - 📊 63+ descriptors computed from composition
        - 📈 Extended correlation analysis (7 target variables)
        - 🗺️ Concentration maps and clustering
        - 📐 Advanced Ternary Plot (4 visualization types)
        - 🎯 35 interactive plot types (Matplotlib)
        - 🔍 Filtering by composition and measurement conditions
        - 🎨 Configurable Pairplot with a dedicated section
        - 📊 Scientific publication style with full boxed axes
        """)


# ============================================================================
# 6. APP ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
