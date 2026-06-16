"""
Proton-Conducting Perovskites Analysis Dashboard
Streamlit application for analyzing composition-structure-property relationships
in proton-conducting perovskite oxides.

Author: Materials Science Research Group
Version: 1.0.0
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import re
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Data science
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Perovskite Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SECTION 2: BUILT-IN DATABASES
# ============================================================================

# Shannon ionic radii (12-coordination for A-site, 6-coordination for B-site)
IONIC_RADII = {
    # A-site cations (12-coordination)
    'Ba': 1.61, 'Ba2': 1.61,  # Ba2+
    'Sr': 1.44, 'Sr2': 1.44,  # Sr2+
    'Ca': 1.34, 'Ca2': 1.34,  # Ca2+
    'La': 1.36, 'La3': 1.36,  # La3+
    'Pr': 1.30, 'Pr3': 1.30,  # Pr3+
    'Nd': 1.27, 'Nd3': 1.27,  # Nd3+
    'Sm': 1.24, 'Sm3': 1.24,  # Sm3+
    'Eu': 1.20, 'Eu3': 1.20,  # Eu3+
    'Gd': 1.19, 'Gd3': 1.19,  # Gd3+
    'Tb': 1.18, 'Tb3': 1.18,  # Tb3+
    'Dy': 1.17, 'Dy3': 1.17,  # Dy3+
    'Ho': 1.16, 'Ho3': 1.16,  # Ho3+
    'Y': 1.14, 'Y3': 1.14,    # Y3+
    'Yb': 1.13, 'Yb3': 1.13,  # Yb3+
    'Tm': 1.14, 'Tm3': 1.14,  # Tm3+
    'Sc': 1.14, 'Sc3': 1.14,  # Sc3+
    
    # B-site cations (6-coordination)
    'Ce': 0.87, 'Ce4': 0.87,  # Ce4+
    'Zr': 0.72, 'Zr4': 0.72,  # Zr4+
    'Sn': 0.69, 'Sn4': 0.69,  # Sn4+
    'Ti': 0.605, 'Ti4': 0.605,  # Ti4+
    'Hf': 0.71, 'Hf4': 0.71,  # Hf4+
    'Y': 0.90, 'Y3': 0.90,    # Y3+
    'Yb': 0.868, 'Yb3': 0.868,  # Yb3+
    'Sc': 0.745, 'Sc3': 0.745,  # Sc3+
    'In': 0.80, 'In3': 0.80,  # In3+
    'Fe': 0.645, 'Fe3': 0.645,  # Fe3+ (low spin)
    'Dy': 0.912, 'Dy3': 0.912,  # Dy3+
    'Sm': 0.958, 'Sm3': 0.958,  # Sm3+
    'Nd': 0.983, 'Nd3': 0.983,  # Nd3+
    'Gd': 0.938, 'Gd3': 0.938,  # Gd3+
    'Eu': 0.947, 'Eu3': 0.947,  # Eu3+
    'Tm': 0.880, 'Tm3': 0.880,  # Tm3+
    'Tb': 0.923, 'Tb3': 0.923,  # Tb3+
    'Ho': 0.901, 'Ho3': 0.901,  # Ho3+
    'Zn': 0.60, 'Zn2': 0.60,   # Zn2+
    'Al': 0.535, 'Al3': 0.535,  # Al3+
    
    # Oxygen
    'O': 1.40, 'O2': 1.40  # O2- (6-coordination)
}

# Electronegativities (Pauling scale)
ELECTRONEGATIVITIES = {
    'Ba': 0.89, 'Sr': 0.95, 'Ca': 1.00, 'La': 1.10,
    'Ce': 1.12, 'Zr': 1.33, 'Sn': 1.96, 'Ti': 1.54,
    'Y': 1.22, 'Yb': 1.10, 'Sc': 1.36, 'In': 1.78,
    'Fe': 1.83, 'Gd': 1.20, 'Sm': 1.17, 'Nd': 1.14,
    'Eu': 1.20, 'Dy': 1.22, 'Ho': 1.23, 'Tm': 1.25,
    'Tb': 1.20, 'Hf': 1.30, 'Zn': 1.65, 'Al': 1.61,
    'Pr': 1.13, 'O': 3.44
}

# Valences
VALENCES = {
    'Ba': 2, 'Sr': 2, 'Ca': 2, 'La': 3,
    'Ce': 4, 'Zr': 4, 'Sn': 4, 'Ti': 4,
    'Y': 3, 'Yb': 3, 'Sc': 3, 'In': 3,
    'Fe': 3, 'Gd': 3, 'Sm': 3, 'Nd': 3,
    'Eu': 3, 'Dy': 3, 'Ho': 3, 'Tm': 3,
    'Tb': 3, 'Hf': 4, 'Zn': 2, 'Al': 3,
    'Pr': 3
}

# Molar masses (g/mol)
MOLAR_MASSES = {
    'Ba': 137.327, 'Sr': 87.62, 'Ca': 40.078, 'La': 138.905,
    'Ce': 140.116, 'Zr': 91.224, 'Sn': 118.710, 'Ti': 47.867,
    'Y': 88.906, 'Yb': 173.045, 'Sc': 44.956, 'In': 114.818,
    'Fe': 55.845, 'Gd': 157.25, 'Sm': 150.36, 'Nd': 144.242,
    'Eu': 151.964, 'Dy': 162.500, 'Ho': 164.930, 'Tm': 168.934,
    'Tb': 158.925, 'Hf': 178.49, 'Zn': 65.38, 'Al': 26.982,
    'Pr': 140.908, 'O': 15.999
}

# ============================================================================
# SECTION 3: SCIENTIFIC PLOTTING STYLE
# ============================================================================

def apply_scientific_style():
    """
    Apply scientific publication style for matplotlib plots.
    Optimized for materials science journals.
    """
    plt.style.use('seaborn-v0-8-whitegrid')
    plt.rcParams.update({
        # Font settings
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        
        # Axes
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#000000',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Ticks
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 7,
        'xtick.major.width': 1.5,
        'ytick.major.size': 7,
        'ytick.major.width': 1.5,
        
        # Legend
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#000000',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        
        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        
        # Lines and markers
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'errorbar.capsize': 3,
        
        # PDF export
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


# Color palettes
COLOR_PALETTES = {
    'B_cation': {
        'Ce': '#E74C3C', 'Zr': '#3498DB', 'Sn': '#2ECC71',
        'Ti': '#F39C12', 'Hf': '#9B59B6'
    },
    'A_cation': {
        'Ba': '#1A5276', 'Sr': '#2471A3', 'Ca': '#5DADE2',
        'La': '#F39C12', 'Pr': '#E67E22'
    },
    'method': {
        'dilatometry': '#2C3E50', 'HT XRD': '#E67E22',
        'HT ND': '#8E44AD', 'HT-XRD': '#E67E22', 'HT-ND': '#8E44AD'
    },
    'continuous': 'viridis',
    'diverging': 'coolwarm',
    'qualitative': ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', 
                    '#9B59B6', '#1ABC9C', '#E67E22', '#2C3E50']
}

# ============================================================================
# SECTION 4: DATA PARSING AND CLEANING
# ============================================================================

@st.cache_data
def parse_uploaded_data(text: str) -> pd.DataFrame:
    """
    Parse pasted tabular data into DataFrame.
    Handles various delimiters and cleans the data.
    """
    if not text.strip():
        return pd.DataFrame()
    
    # Try different delimiters
    lines = text.strip().split('\n')
    
    # Check if first line contains expected columns
    expected_cols = ['№', 'A', "A'", 'B', "B'", 'D1', 'D2', 
                     "[A']", "[B']", '[D1]', '[D2]', 'δ',
                     'method', 'β', '∆T, °C', 'α·106 (K-1)',
                     'T(bends), °C', 'αav·106 (K-1)', 'pH2O', 'Ref']
    
    # Try tab delimiter first
    for delimiter in ['\t', ',', ';', '  ', ' ']:
        try:
            df = pd.read_csv(StringIO(text), delimiter=delimiter, 
                            dtype=str, keep_default_na=False)
            # Check if we got expected number of columns
            if len(df.columns) >= 10:
                # Clean column names
                df.columns = df.columns.str.strip()
                return clean_dataframe(df)
        except:
            continue
    
    # If all fail, try manual parsing
    st.warning("Auto-detection failed. Please ensure data is tab-separated.")
    return pd.DataFrame()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the DataFrame.
    """
    # Rename columns if needed
    expected_names = ['№', 'A', "A'", 'B', "B'", 'D1', 'D2', 
                      "[A']", "[B']", '[D1]', '[D2]', 'δ',
                      'method', 'β', 'delta_T', 'alpha',
                      'T_bends', 'alpha_av', 'pH2O', 'Ref']
    
    # Map columns if they match expected pattern
    if len(df.columns) >= 19:
        df.columns = expected_names[:len(df.columns)]
    
    # Replace '-' with NaN
    df = df.replace('-', pd.NA)
    df = df.replace('—', pd.NA)
    df = df.replace('–', pd.NA)
    
    # Convert numeric columns
    numeric_cols = ['№', "[A']", "[B']", '[D1]', '[D2]', 'δ', 'β', 
                    'alpha', 'pH2O', 'alpha_av']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert delta_T to numeric range
    if 'delta_T' in df.columns:
        df['delta_T_low'] = df['delta_T'].str.split('-').str[0].astype(float)
        df['delta_T_high'] = df['delta_T'].str.split('-').str[1].astype(float)
    
    # Parse T_bends
    if 'T_bends' in df.columns:
        df['T_bends_list'] = df['T_bends'].apply(parse_t_bends)
        df['T_bends_count'] = df['T_bends_list'].apply(len)
        df['T_bends_first'] = df['T_bends_list'].apply(lambda x: x[0] if x else np.nan)
        df['T_bends_last'] = df['T_bends_list'].apply(lambda x: x[-1] if x else np.nan)
    
    # Parse alpha_av
    if 'alpha_av' in df.columns:
        df['alpha_av_list'] = df['alpha_av'].apply(parse_alpha_av)
        df['alpha_av_count'] = df['alpha_av_list'].apply(len)
        df['alpha_av_first'] = df['alpha_av_list'].apply(lambda x: x[0] if x else np.nan)
        df['alpha_av_last'] = df['alpha_av_list'].apply(lambda x: x[-1] if x else np.nan)
    
    # Calculate delta if missing
    if 'δ' in df.columns and '[D1]' in df.columns and '[D2]' in df.columns:
        df['δ_calc'] = df['[D1]'] / 2 + df['[D2]'] / 2
        # Use calculated delta where original is NaN
        if df['δ'].isna().any():
            df['δ'] = df['δ'].fillna(df['δ_calc'])
    
    return df


def parse_t_bends(value: Any) -> List[float]:
    """
    Parse semicolon-separated temperature bend values.
    Example: "400;600" -> [400.0, 600.0]
    """
    if pd.isna(value) or value == '-' or value == '—' or value == '':
        return []
    try:
        if isinstance(value, (int, float)):
            return [float(value)]
        # Remove any whitespace and split
        value_str = str(value).strip()
        if ';' in value_str:
            parts = value_str.split(';')
        elif ',' in value_str:
            parts = value_str.split(',')
        else:
            return [float(value_str)]
        
        return [float(p.strip()) for p in parts if p.strip()]
    except:
        return []


def parse_alpha_av(value: Any) -> List[float]:
    """
    Parse semicolon-separated alpha_av values.
    Example: "10.6;4.73;10.1" -> [10.6, 4.73, 10.1]
    """
    if pd.isna(value) or value == '-' or value == '—' or value == '':
        return []
    try:
        if isinstance(value, (int, float)):
            return [float(value)]
        value_str = str(value).strip()
        if ';' in value_str:
            parts = value_str.split(';')
        elif ',' in value_str:
            parts = value_str.split(',')
        else:
            return [float(value_str)]
        
        return [float(p.strip()) for p in parts if p.strip()]
    except:
        return []


def get_cation_radius(element: str, site: str = 'B') -> float:
    """
    Get Shannon ionic radius for a cation.
    """
    if pd.isna(element) or element == '':
        return np.nan
    
    # Handle simple element symbols
    element = str(element).strip()
    
    # Try exact match first
    if element in IONIC_RADII:
        return IONIC_RADII[element]
    
    # Try with valence
    if site == 'A':
        variants = [f'{element}2', f'{element}3']
    else:
        variants = [f'{element}4', f'{element}3', f'{element}2']
    
    for v in variants:
        if v in IONIC_RADII:
            return IONIC_RADII[v]
    
    return np.nan


def get_electronegativity(element: str) -> float:
    """
    Get Pauling electronegativity for an element.
    """
    if pd.isna(element) or element == '':
        return np.nan
    
    element = str(element).strip()
    # Remove valence suffix if present
    element = re.sub(r'[234]\+?', '', element)
    
    if element in ELECTRONEGATIVITIES:
        return ELECTRONEGATIVITIES[element]
    
    return np.nan


def get_valence(element: str) -> float:
    """
    Get valence state for an element.
    """
    if pd.isna(element) or element == '':
        return np.nan
    
    element = str(element).strip()
    
    if element in VALENCES:
        return VALENCES[element]
    
    return np.nan


def get_molar_mass(element: str) -> float:
    """
    Get molar mass for an element.
    """
    if pd.isna(element) or element == '':
        return np.nan
    
    element = str(element).strip()
    element = re.sub(r'[234]\+?', '', element)
    
    if element in MOLAR_MASSES:
        return MOLAR_MASSES[element]
    
    return np.nan

# ============================================================================
# SECTION 5: DESCRIPTOR ENGINE
# ============================================================================

@st.cache_data
def calculate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 63+ descriptors for the perovskite compositions.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    df_desc = df.copy()
    
    # Get concentrations
    A_conc = 1 - df_desc["[A']"].fillna(0)
    B_conc = 1 - df_desc["[B']"].fillna(0) - df_desc['[D1]'].fillna(0) - df_desc['[D2]'].fillna(0)
    
    # ========================================================================
    # Group 1: Geometric descriptors
    # ========================================================================
    
    # Average A-site radius
    rA = df_desc['A'].apply(lambda x: get_cation_radius(x, 'A'))
    rA_prime = df_desc["A'"].apply(lambda x: get_cation_radius(x, 'A'))
    df_desc['rAav'] = rA * A_conc + rA_prime * df_desc["[A']"].fillna(0)
    
    # Average B-site radius
    rB = df_desc['B'].apply(lambda x: get_cation_radius(x, 'B'))
    rB_prime = df_desc["B'"].apply(lambda x: get_cation_radius(x, 'B'))
    rD1 = df_desc['D1'].apply(lambda x: get_cation_radius(x, 'B'))
    rD2 = df_desc['D2'].apply(lambda x: get_cation_radius(x, 'B'))
    
    df_desc['rBav'] = (rB * B_conc + 
                       rB_prime * df_desc["[B']"].fillna(0) +
                       rD1 * df_desc['[D1]'].fillna(0) +
                       rD2 * df_desc['[D2]'].fillna(0))
    
    # Oxygen radius
    rO = IONIC_RADII['O']
    
    # Goldschmidt tolerance factor
    df_desc['t'] = (df_desc['rAav'] + rO) / (np.sqrt(2) * (df_desc['rBav'] + rO))
    
    # Tolerance factor deviation
    df_desc['D_t'] = np.abs(1 - df_desc['t'])
    
    # Octahedral factor
    df_desc['octahedral_factor'] = df_desc['rBav'] / rO
    
    # Radius difference A-B
    df_desc['delta_r_AB'] = np.abs(df_desc['rAav'] - df_desc['rBav'])
    df_desc['delta_r_AB_norm'] = df_desc['delta_r_AB'] / rO
    
    # Radius ratio
    df_desc['r_ratio_AB'] = df_desc['rAav'] / df_desc['rBav']
    
    # Variance of B-site radii
    df_desc['sigma2_rB'] = (B_conc * (rB - df_desc['rBav'])**2 +
                            df_desc["[B']"].fillna(0) * (rB_prime - df_desc['rBav'])**2 +
                            df_desc['[D1]'].fillna(0) * (rD1 - df_desc['rBav'])**2 +
                            df_desc['[D2]'].fillna(0) * (rD2 - df_desc['rBav'])**2)
    
    # Variance of A-site radii
    df_desc['sigma2_rA'] = (A_conc * (rA - df_desc['rAav'])**2 +
                            df_desc["[A']"].fillna(0) * (rA_prime - df_desc['rAav'])**2)
    
    # Unit cell volume proxy (pseudocubic)
    df_desc['V_cell'] = (df_desc['rAav'] + rO) * (df_desc['rBav'] + rO)**3
    
    # Octahedral distortion proxy
    df_desc['oct_dist'] = (np.abs(rB - df_desc['rBav']) + 
                           np.abs(rB_prime - df_desc['rBav']) +
                           np.abs(rD1 - df_desc['rBav']) +
                           np.abs(rD2 - df_desc['rBav'])) / 4
    
    # ========================================================================
    # Group 2: Electronegativity descriptors
    # ========================================================================
    
    # Get electronegativities
    chiA = df_desc['A'].apply(lambda x: get_electronegativity(x))
    chiA_prime = df_desc["A'"].apply(lambda x: get_electronegativity(x))
    chiB = df_desc['B'].apply(lambda x: get_electronegativity(x))
    chiB_prime = df_desc["B'"].apply(lambda x: get_electronegativity(x))
    chiD1 = df_desc['D1'].apply(lambda x: get_electronegativity(x))
    chiD2 = df_desc['D2'].apply(lambda x: get_electronegativity(x))
    
    # Average electronegativities
    df_desc['chiAav'] = chiA * A_conc + chiA_prime * df_desc["[A']"].fillna(0)
    df_desc['chiBav'] = (chiB * B_conc + 
                         chiB_prime * df_desc["[B']"].fillna(0) +
                         chiD1 * df_desc['[D1]'].fillna(0) +
                         chiD2 * df_desc['[D2]'].fillna(0))
    
    # Difference and ratio
    df_desc['delta_chi_AB'] = np.abs(df_desc['chiAav'] - df_desc['chiBav'])
    df_desc['chi_ratio_AB'] = df_desc['chiAav'] / df_desc['chiBav']
    
    # Ionicity (Pauling formula)
    chiO = ELECTRONEGATIVITIES['O']
    df_desc['ionicity_AO'] = 1 - np.exp(-0.25 * (df_desc['chiAav'] - chiO)**2)
    df_desc['ionicity_BO'] = 1 - np.exp(-0.25 * (df_desc['chiBav'] - chiO)**2)
    
    # Acidity (inverse electronegativity)
    df_desc['acidity_AO'] = 1 / df_desc['chiAav']
    df_desc['acidity_BO'] = 1 / df_desc['chiBav']
    df_desc['delta_acidity'] = df_desc['acidity_BO'] - df_desc['acidity_AO']
    
    # ========================================================================
    # Group 3: Thermodynamic descriptors
    # ========================================================================
    
    # Configurational entropy
    R = 8.314  # J/(mol*K)
    
    # A-site entropy
    df_desc['S_config_A'] = -R * (A_conc * np.log(A_conc + 1e-10) +
                                   df_desc["[A']"].fillna(0) * np.log(df_desc["[A']"].fillna(0) + 1e-10))
    
    # B-site entropy
    df_desc['S_config_B'] = -R * (B_conc * np.log(B_conc + 1e-10) +
                                   df_desc["[B']"].fillna(0) * np.log(df_desc["[B']"].fillna(0) + 1e-10) +
                                   df_desc['[D1]'].fillna(0) * np.log(df_desc['[D1]'].fillna(0) + 1e-10) +
                                   df_desc['[D2]'].fillna(0) * np.log(df_desc['[D2]'].fillna(0) + 1e-10))
    
    # Average B-site valence
    vB = df_desc['B'].apply(lambda x: get_valence(x))
    vB_prime = df_desc["B'"].apply(lambda x: get_valence(x))
    vD1 = df_desc['D1'].apply(lambda x: get_valence(x))
    vD2 = df_desc['D2'].apply(lambda x: get_valence(x))
    
    df_desc['V_Bav'] = (vB * B_conc + 
                        vB_prime * df_desc["[B']"].fillna(0) +
                        vD1 * df_desc['[D1]'].fillna(0) +
                        vD2 * df_desc['[D2]'].fillna(0))
    
    # Oxygen vacancy proxy (for Ce4+/Zr4+ systems)
    df_desc['Vo_proxy'] = (4 - df_desc['V_Bav']) / 2
    
    # Hydration enthalpy proxy
    df_desc['delta_H_hydr'] = 1 / (df_desc['rBav'] + 1e-10) * (df_desc['chiBav'] - chiO)**2
    
    # B-O bond energy proxy (Coulombic)
    df_desc['E_BO'] = (df_desc['V_Bav'] * 2) / (df_desc['rBav'] + 1e-10)
    
    # Mass density proxy
    df_desc['rho'] = df_desc['M_total'] / df_desc['V_cell']
    
    # ========================================================================
    # Group 4: Mass descriptors
    # ========================================================================
    
    # Get molar masses
    mA = df_desc['A'].apply(lambda x: get_molar_mass(x))
    mA_prime = df_desc["A'"].apply(lambda x: get_molar_mass(x))
    mB = df_desc['B'].apply(lambda x: get_molar_mass(x))
    mB_prime = df_desc["B'"].apply(lambda x: get_molar_mass(x))
    mD1 = df_desc['D1'].apply(lambda x: get_molar_mass(x))
    mD2 = df_desc['D2'].apply(lambda x: get_molar_mass(x))
    
    df_desc['M_Aav'] = mA * A_conc + mA_prime * df_desc["[A']"].fillna(0)
    df_desc['M_Bav'] = (mB * B_conc + 
                        mB_prime * df_desc["[B']"].fillna(0) +
                        mD1 * df_desc['[D1]'].fillna(0) +
                        mD2 * df_desc['[D2]'].fillna(0))
    
    df_desc['M_total'] = df_desc['M_Aav'] + df_desc['M_Bav'] + 3 * MOLAR_MASSES['O']
    df_desc['M_ratio_AB'] = df_desc['M_Aav'] / (df_desc['M_Bav'] + 1e-10)
    df_desc['M_rA'] = df_desc['M_Aav'] * df_desc['rAav']
    df_desc['M_chiA'] = df_desc['M_Aav'] * df_desc['chiAav']
    
    # ========================================================================
    # Group 5: Defect descriptors
    # ========================================================================
    
    # Effective B-site charge
    df_desc['Z_eff_B'] = 4 - 2 * df_desc['δ']
    
    # Proton affinity proxy
    df_desc['proton_affinity'] = 1 / ((df_desc['rBav'] + 1e-10) * (df_desc['chiBav'] + 1e-10))
    
    # Vacancy formation energy proxy
    df_desc['E_vac'] = 1 / (df_desc['rBav']**2 + 1e-10) * (df_desc['chiBav'] - chiO)
    
    # ========================================================================
    # Group 6: T(bends) specific descriptors
    # ========================================================================
    
    # Alpha/beta ratio (if both available)
    df_desc['alpha_beta_ratio'] = df_desc['alpha'] / (df_desc['β'] + 1e-10)
    
    # Proton stability temperature proxy
    df_desc['T_stab'] = -df_desc['delta_H_hydr'] / R
    
    # Combined descriptors
    df_desc['delta_chiB'] = df_desc['δ'] * df_desc['chiBav']
    df_desc['delta_rB'] = df_desc['δ'] * df_desc['rBav']
    
    # ========================================================================
    # Group 7: Compositional descriptors
    # ========================================================================
    
    df_desc["B'_conc"] = df_desc["[B']"].fillna(0)
    df_desc['D_total'] = df_desc['[D1]'].fillna(0) + df_desc['[D2]'].fillna(0)
    df_desc['D_ratio'] = df_desc['[D1]'].fillna(0) / (df_desc['[D2]'].fillna(0) + 1e-10)
    
    # ========================================================================
    # Group 8: Combined (physics-inspired) descriptors
    # ========================================================================
    
    df_desc['delta_chi_div_t'] = df_desc['delta_chi_AB'] / (df_desc['t'] + 1e-10)
    df_desc['delta_chi_mul_t'] = df_desc['delta_chi_AB'] * df_desc['t']
    df_desc['disorder_over_distortion'] = df_desc['sigma2_rB'] / (df_desc['D_t'] + 1e-10)
    df_desc['ionic_x_octa'] = df_desc['ionicity_BO'] * df_desc['octahedral_factor']
    df_desc['chi_ratio_t'] = df_desc['chi_ratio_AB'] * df_desc['t']
    df_desc['rBav_x_chiBav'] = df_desc['rBav'] * df_desc['chiBav']
    
    return df_desc

# ============================================================================
# SECTION 6: CORRELATION ANALYSIS
# ============================================================================

@st.cache_data
def calculate_correlations(df: pd.DataFrame, target_cols: List[str]) -> Dict:
    """
    Calculate various correlation matrices and statistics.
    """
    if df.empty:
        return {}
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with too many NaN
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > 10]
    
    if len(valid_cols) < 3:
        return {}
    
    # Fill NaN with median
    df_filled = df[valid_cols].copy()
    for col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    
    # Pearson correlation
    pearson_corr = df_filled.corr(method='pearson')
    
    # Spearman correlation
    spearman_corr = df_filled.corr(method='spearman')
    
    # Calculate p-values for Pearson
    p_values = pd.DataFrame(np.ones_like(pearson_corr), 
                           index=pearson_corr.index, 
                           columns=pearson_corr.columns)
    
    for i in range(len(pearson_corr.columns)):
        for j in range(len(pearson_corr.columns)):
            if i != j:
                col1 = pearson_corr.columns[i]
                col2 = pearson_corr.columns[j]
                # Remove NaN for this pair
                mask = df_filled[col1].notna() & df_filled[col2].notna()
                if mask.sum() > 3:
                    corr, p_val = stats.pearsonr(df_filled[col1][mask], df_filled[col2][mask])
                    p_values.iloc[i, j] = p_val
    
    # Partial correlation (controlling for pH2O)
    partial_corr = df_filled.copy()
    if 'pH2O' in partial_corr.columns:
        # Remove pH2O and compute partial correlations
        pH2O = partial_corr['pH2O']
        for col in partial_corr.columns:
            if col != 'pH2O':
                # Regress out pH2O
                model = LinearRegression()
                model.fit(pH2O.values.reshape(-1, 1), partial_corr[col].values)
                partial_corr[col] = partial_corr[col] - model.predict(pH2O.values.reshape(-1, 1))
        
        # Compute correlations of residuals
        partial_corr_matrix = partial_corr.drop('pH2O', axis=1).corr()
    else:
        partial_corr_matrix = pd.DataFrame()
    
    # Distance correlation
    from scipy.spatial.distance import pdist, squareform
    
    def distance_correlation(x, y):
        n = len(x)
        # Compute distance matrices
        dx = squareform(pdist(x.reshape(-1, 1)))
        dy = squareform(pdist(y.reshape(-1, 1)))
        # Center matrices
        dx = dx - dx.mean(axis=0) - dx.mean(axis=1)[:, np.newaxis] + dx.mean()
        dy = dy - dy.mean(axis=0) - dy.mean(axis=1)[:, np.newaxis] + dy.mean()
        # Compute distance correlation
        if np.sum(dx * dy) > 0:
            dcorr = np.sqrt(np.sum(dx * dy) / (np.sqrt(np.sum(dx**2)) * np.sqrt(np.sum(dy**2)) + 1e-10))
        else:
            dcorr = 0
        return dcorr
    
    # Calculate distance correlation for target columns
    dcorr_dict = {}
    for target in target_cols:
        if target in df_filled.columns:
            dcorr_dict[target] = {}
            for col in valid_cols:
                if col != target:
                    x = df_filled[target].values
                    y = df_filled[col].values
                    # Remove NaN
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    if mask.sum() > 3:
                        dcorr_dict[target][col] = distance_correlation(x[mask], y[mask])
    
    # Feature importance using Random Forest
    rf_importance = {}
    for target in target_cols:
        if target in df_filled.columns:
            # Prepare data
            X = df_filled.drop(target, axis=1)
            y = df_filled[target]
            
            # Remove rows with NaN in target
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            
            if len(X) > 10 and len(X.columns) > 0:
                # Fill NaN
                X = X.fillna(X.median())
                
                try:
                    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X, y)
                    rf_importance[target] = pd.Series(rf.feature_importances_, 
                                                     index=X.columns).sort_values(ascending=False)
                except:
                    rf_importance[target] = pd.Series()
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'p_values': p_values,
        'partial': partial_corr_matrix,
        'distance': dcorr_dict,
        'rf_importance': rf_importance,
        'valid_cols': valid_cols,
        'filled_data': df_filled
    }


def find_top_descriptors(correlations: Dict, target_cols: List[str], n_top: int = 20) -> List[str]:
    """
    Find top N descriptors based on correlation with target variables.
    """
    if not correlations or 'pearson' not in correlations:
        return []
    
    pearson = correlations['pearson']
    rf_importance = correlations.get('rf_importance', {})
    
    # Score each descriptor
    scores = {}
    for col in pearson.columns:
        if col in target_cols:
            continue
        
        score = 0
        count = 0
        
        # Add Pearson correlation with each target
        for target in target_cols:
            if target in pearson.index:
                corr_val = abs(pearson.loc[target, col])
                if not np.isnan(corr_val):
                    score += corr_val
                    count += 1
        
        # Add Random Forest importance if available
        for target in target_cols:
            if target in rf_importance and col in rf_importance[target].index:
                imp_val = rf_importance[target][col]
                if not np.isnan(imp_val):
                    score += imp_val * 0.5  # Weighted less than correlation
        
        if count > 0:
            scores[col] = score / count
    
    # Sort and get top N
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_descriptors = [item[0] for item in sorted_scores[:n_top]]
    
    # Ensure target columns are included
    for target in target_cols:
        if target not in top_descriptors and target in pearson.columns:
            top_descriptors.append(target)
    
    return top_descriptors

# ============================================================================
# SECTION 7: PCA AND CLUSTERING
# ============================================================================

@st.cache_data
def perform_pca_analysis(df: pd.DataFrame, descriptors: List[str]) -> Dict:
    """
    Perform PCA analysis on selected descriptors.
    """
    if df.empty or len(descriptors) < 3:
        return {}
    
    # Select and clean data
    X = df[descriptors].copy()
    
    # Fill NaN with median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Remove constant columns
    variance_threshold = VarianceThreshold(threshold=0.01)
    X_filtered = variance_threshold.fit_transform(X)
    selected_cols = X.columns[variance_threshold.get_support()].tolist()
    
    if len(selected_cols) < 2:
        return {}
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Determine optimal number of components (elbow)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find elbow point
    n_components_optimal = np.argmax(cumulative_variance > 0.85) + 1
    
    # Get loadings
    loadings = pd.DataFrame(pca.components_[:n_components_optimal].T,
                           index=selected_cols,
                           columns=[f'PC{i+1}' for i in range(n_components_optimal)])
    
    return {
        'pca': pca,
        'scaler': scaler,
        'X_scaled': X_scaled,
        'pca_result': pca_result,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'n_components_optimal': n_components_optimal,
        'loadings': loadings,
        'selected_cols': selected_cols,
        'original_cols': descriptors
    }


@st.cache_data
def perform_clustering(pca_result: np.ndarray, n_clusters: int = None) -> Dict:
    """
    Perform K-means clustering with optimal cluster number detection.
    """
    if pca_result is None or len(pca_result) < 3:
        return {}
    
    X = pca_result[:, :min(3, pca_result.shape[1])]
    
    # Determine optimal number of clusters using silhouette
    if n_clusters is None:
        max_clusters = min(10, len(X) // 3)
        if max_clusters < 2:
            return {}
        
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        optimal_k = np.argmax(silhouette_scores) + 2 if silhouette_scores else 2
    else:
        optimal_k = n_clusters
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate silhouette
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X, labels)
        silhouette_per_sample = silhouette_samples(X, labels)
    else:
        silhouette_avg = -1
        silhouette_per_sample = np.ones(len(X))
    
    return {
        'labels': labels,
        'centers': kmeans.cluster_centers_,
        'optimal_k': optimal_k,
        'silhouette_avg': silhouette_avg,
        'silhouette_samples': silhouette_per_sample,
        'kmeans': kmeans,
        'X_used': X
    }

# ============================================================================
# SECTION 8: VISUALIZATION FUNCTIONS
# ============================================================================

# Plotting functions will be added here
# Each function returns a matplotlib figure or plotly figure

def create_distribution_plots(df: pd.DataFrame, columns: List[str]) -> plt.Figure:
    """
    Create distribution histograms for selected columns.
    """
    apply_scientific_style()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns[:len(axes)]):
        ax = axes[idx]
        data = df[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=20, edgecolor='black', color='#3498DB', alpha=0.7)
            ax.set_xlabel(col, fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.axvline(data.mean(), color='red', linestyle='--', 
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='--',
                      label=f'Median: {data.median():.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> plt.Figure:
    """
    Create a heatmap of correlation matrix.
    """
    apply_scientific_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'size': 8})
    
    ax.set_title(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig


def create_pairplot(df: pd.DataFrame, features: List[str], hue_col: str = None) -> plt.Figure:
    """
    Create a pairplot of selected features.
    """
    apply_scientific_style()
    
    # Prepare data
    plot_data = df[features].copy()
    
    # Add hue if provided
    if hue_col and hue_col in df.columns:
        plot_data[hue_col] = df[hue_col]
        g = sns.pairplot(plot_data, vars=features, hue=hue_col, 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
    else:
        g = sns.pairplot(plot_data, diag_kind='kde', plot_kws={'alpha': 0.6})
    
    # Adjust figure
    fig = g.figure
    fig.set_size_inches(12, 10)
    plt.tight_layout()
    return fig


def create_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str, 
                                  color_col: str = None) -> plt.Figure:
    """
    Create a scatter plot with regression line.
    """
    apply_scientific_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    valid_mask = df[x_col].notna() & df[y_col].notna()
    x_data = df[x_col][valid_mask]
    y_data = df[y_col][valid_mask]
    
    if len(x_data) < 3:
        ax.text(0.5, 0.5, 'Insufficient data points', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create scatter
    if color_col and color_col in df.columns:
        scatter = ax.scatter(x_data, y_data, c=df[color_col][valid_mask],
                           cmap='viridis', alpha=0.7, s=50)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_col, fontweight='bold')
    else:
        ax.scatter(x_data, y_data, color='#3498DB', alpha=0.7, s=50)
    
    # Add regression line
    if len(x_data) > 2:
        model = LinearRegression()
        model.fit(x_data.values.reshape(-1, 1), y_data.values)
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regression')
        
        # Add statistics
        r2 = model.score(x_data.values.reshape(-1, 1), y_data.values)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add equation
        eq_text = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
        ax.text(0.05, 0.88, eq_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_pca_biplot(pca_result: np.ndarray, loadings: pd.DataFrame, 
                     labels: np.ndarray = None, title: str = "PCA Biplot") -> plt.Figure:
    """
    Create a PCA biplot with loadings and projections.
    """
    apply_scientific_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Projections
    if labels is not None:
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                           c=labels, cmap='tab10', alpha=0.7, s=50)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontweight='bold')
    else:
        ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                  color='#3498DB', alpha=0.7, s=50)
    
    # Loadings (vectors)
    n_loadings = min(10, len(loadings))
    for i in range(n_loadings):
        vector = loadings.iloc[i, :2].values
        if np.linalg.norm(vector) > 0.1:
            ax.arrow(0, 0, vector[0] * 3, vector[1] * 3, 
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
            ax.text(vector[0] * 3.2, vector[1] * 3.2, 
                   loadings.index[i], fontsize=9, color='red')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Principal Component 1', fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_plotly_concentration_heatmap(df: pd.DataFrame, x_col: str, y_col: str, 
                                       color_col: str, filters: Dict = None) -> go.Figure:
    """
    Create an interactive 2D concentration heatmap using Plotly.
    """
    # Prepare data
    plot_data = df[[x_col, y_col, color_col]].dropna()
    
    if filters:
        for key, value in filters.items():
            if key in df.columns:
                plot_data = plot_data[plot_data[key] == value]
    
    if len(plot_data) < 4:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data points for heatmap", 
                          showarrow=False, font=dict(size=16))
        return fig
    
    # Create heatmap using scatter with interpolation
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=plot_data[x_col],
        y=plot_data[y_col],
        mode='markers',
        marker=dict(
            size=10,
            color=plot_data[color_col],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_col),
            line=dict(width=1, color='black')
        ),
        text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}' 
              for x, y, c in zip(plot_data[x_col], plot_data[y_col], plot_data[color_col])],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f'Concentration Map: {color_col} vs {x_col} and {y_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=600,
        width=800
    )
    
    return fig


def create_plotly_bubble_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                              color_col: str, size_col: str, shape_col: str = None,
                              filters: Dict = None) -> go.Figure:
    """
    Create an interactive 4D bubble chart using Plotly.
    """
    # Prepare data
    plot_data = df[[x_col, y_col, color_col, size_col]].dropna()
    
    if shape_col and shape_col in df.columns:
        plot_data[shape_col] = df[shape_col]
    
    if filters:
        for key, value in filters.items():
            if key in df.columns:
                plot_data = plot_data[plot_data[key] == value]
    
    if len(plot_data) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data points", 
                          showarrow=False, font=dict(size=16))
        return fig
    
    # Define shapes
    shape_map = {
        'circle': 'circle',
        'square': 'square',
        'diamond': 'diamond',
        'triangle-up': 'triangle-up',
        'star': 'star',
        'pentagon': 'pentagon',
        'hexagon': 'hexagon',
        'cross': 'x'
    }
    
    # Create bubble chart
    fig = go.Figure()
    
    if shape_col and shape_col in plot_data.columns:
        # Separate by shape category
        for category in plot_data[shape_col].unique():
            data = plot_data[plot_data[shape_col] == category]
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(
                    size=data[size_col] * 30 + 5,
                    color=data[color_col],
                    colorscale='Viridis',
                    showscale=True if category == plot_data[shape_col].unique()[0] else False,
                    symbol=shape_map.get(str(category).lower(), 'circle'),
                    line=dict(width=1, color='black'),
                    sizemode='diameter',
                    sizeref=2. * max(plot_data[size_col]) / (40. ** 2),
                    sizemin=4
                ),
                name=f'{shape_col}: {category}',
                text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}<br>{size_col}: {s:.3f}' 
                      for x, y, c, s in zip(data[x_col], data[y_col], data[color_col], data[size_col])],
                hoverinfo='text'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_data[x_col],
            y=plot_data[y_col],
            mode='markers',
            marker=dict(
                size=plot_data[size_col] * 30 + 5,
                color=plot_data[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col),
                line=dict(width=1, color='black'),
                sizemode='diameter',
                sizeref=2. * max(plot_data[size_col]) / (40. ** 2),
                sizemin=4
            ),
            text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}<br>{size_col}: {s:.3f}' 
                  for x, y, c, s in zip(plot_data[x_col], plot_data[y_col], 
                                       plot_data[color_col], plot_data[size_col])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f'Bubble Chart: {y_col} vs {x_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=600,
        width=900,
        legend=dict(x=1.05, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def create_plotly_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                            color_col: str = None, size_col: str = None,
                            filters: Dict = None) -> go.Figure:
    """
    Create an interactive 3D scatter plot using Plotly.
    """
    plot_data = df[[x_col, y_col, z_col]].dropna()
    
    if color_col and color_col in df.columns:
        plot_data[color_col] = df[color_col]
    if size_col and size_col in df.columns:
        plot_data[size_col] = df[size_col]
    
    if filters:
        for key, value in filters.items():
            if key in df.columns:
                plot_data = plot_data[plot_data[key] == value]
    
    if len(plot_data) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data points", 
                          showarrow=False, font=dict(size=16))
        return fig
    
    # Create 3D scatter
    fig = go.Figure()
    
    marker_size = plot_data[size_col] * 10 + 5 if size_col else 8
    
    fig.add_trace(go.Scatter3d(
        x=plot_data[x_col],
        y=plot_data[y_col],
        z=plot_data[z_col],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=plot_data[color_col] if color_col else 'blue',
            colorscale='Viridis',
            showscale=True if color_col else False,
            colorbar=dict(title=color_col) if color_col else None,
            line=dict(width=0.5, color='black')
        ),
        text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{z_col}: {z:.3f}' 
              for x, y, z in zip(plot_data[x_col], plot_data[y_col], plot_data[z_col])],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f'3D Scatter: {z_col} vs {x_col} and {y_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        template='plotly_white',
        height=700,
        width=900
    )
    
    return fig


def create_network_correlation(df: pd.DataFrame, correlation_matrix: pd.DataFrame, 
                              threshold: float = 0.5) -> plt.Figure:
    """
    Create a network graph of correlations.
    """
    apply_scientific_style()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for node in correlation_matrix.columns:
        G.add_node(node)
    
    # Add edges based on correlation threshold
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(col1, col2, weight=abs(corr))
    
    if len(G.edges) == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'No correlations above threshold {threshold}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate edge colors based on correlation sign
    edge_colors = []
    edge_widths = []
    for (u, v, d) in G.edges(data=True):
        corr = correlation_matrix.loc[u, v]
        edge_colors.append('red' if corr > 0 else 'blue')
        edge_widths.append(d['weight'] * 3)
    
    # Draw nodes
    node_sizes = [500 for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                          width=edge_widths, alpha=0.6, ax=ax,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(f'Correlation Network (|corr| > {threshold})', fontweight='bold', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_cluster_profiles(df: pd.DataFrame, clusters: np.ndarray, 
                           features: List[str]) -> plt.Figure:
    """
    Create cluster profile heatmap.
    """
    apply_scientific_style()
    
    # Add cluster labels to dataframe
    df_clust = df[features].copy()
    df_clust['Cluster'] = clusters
    
    # Calculate means by cluster
    cluster_means = df_clust.groupby('Cluster')[features].mean()
    cluster_stds = df_clust.groupby('Cluster')[features].std()
    
    # Standardize means across clusters
    means_scaled = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-10)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(cluster_means) * 0.5)))
    
    im = ax.imshow(means_scaled.values, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(cluster_means.index)))
    ax.set_yticklabels([f'Cluster {i}' for i in cluster_means.index], fontsize=10)
    
    # Add text annotations
    for i in range(len(cluster_means.index)):
        for j in range(len(features)):
            value = cluster_means.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color='white' if means_scaled.iloc[i, j] > 0.5 else 'black',
                   fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value', fontweight='bold')
    
    ax.set_title('Cluster Profiles (Mean Values)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig


def create_alpha_beta_compromise(df: pd.DataFrame, alpha_col: str = 'alpha', 
                                beta_col: str = 'β', color_col: str = None) -> plt.Figure:
    """
    Create α vs β compromise diagram.
    """
    apply_scientific_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    valid_mask = df[alpha_col].notna() & df[beta_col].notna()
    x_data = df[alpha_col][valid_mask]
    y_data = df[beta_col][valid_mask]
    
    if len(x_data) < 3:
        ax.text(0.5, 0.5, 'Insufficient data points', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create scatter
    if color_col and color_col in df.columns:
        scatter = ax.scatter(x_data, y_data, c=df[color_col][valid_mask],
                           cmap='tab10', alpha=0.7, s=80)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_col, fontweight='bold')
    else:
        ax.scatter(x_data, y_data, color='#3498DB', alpha=0.7, s=80)
    
    # Add ideal zone (low α, low β)
    # ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='β = 0.01')
    # ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='α = 10 × 10⁻⁶ K⁻¹')
    
    # Add quadrants
    ax.axhline(y=y_data.median(), color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=x_data.median(), color='gray', linestyle='--', alpha=0.3)
    
    # Highlight optimal region (low α, low β)
    # ax.text(0.05, 0.05, 'Optimal region', transform=ax.transAxes,
    #        fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
    
    ax.set_xlabel(f'{alpha_col} (×10⁻⁶ K⁻¹)', fontweight='bold')
    ax.set_ylabel(f'{beta_col}', fontweight='bold')
    ax.set_title('α vs β: Compromise Diagram', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_feature_importance_plot(importance_dict: Dict, target_col: str, 
                                  n_top: int = 15) -> plt.Figure:
    """
    Create feature importance bar plot from Random Forest.
    """
    apply_scientific_style()
    
    if target_col not in importance_dict or importance_dict[target_col].empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'No importance data for {target_col}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    importance = importance_dict[target_col].head(n_top)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance) * 0.4)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))[::-1]
    ax.barh(importance.index, importance.values, color=colors, edgecolor='black')
    
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    ax.set_title(f'Top {n_top} Features for {target_col}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def create_cluster_radar_chart(cluster_means: pd.DataFrame, features: List[str], 
                              max_clusters: int = 6) -> plt.Figure:
    """
    Create radar chart for cluster comparison.
    """
    apply_scientific_style()
    
    n_clusters = min(len(cluster_means), max_clusters)
    if n_clusters < 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, 'Need at least 2 clusters', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Normalize features
    cluster_means_norm = cluster_means.iloc[:n_clusters].copy()
    for col in cluster_means_norm.columns:
        if col in features:
            cluster_means_norm[col] = (cluster_means_norm[col] - cluster_means_norm[col].min()) / \
                                     (cluster_means_norm[col].max() - cluster_means_norm[col].min() + 1e-10)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        values = cluster_means_norm.iloc[i, :len(features)].values
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Cluster Radar Chart', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    return fig

# ============================================================================
# SECTION 9: FILTERING SYSTEM
# ============================================================================

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply multiple filters to the dataframe.
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    for key, value in filters.items():
        if value is None or value == 'All' or (isinstance(value, list) and not value):
            continue
        
        if key in filtered_df.columns:
            if isinstance(value, (list, tuple)):
                if len(value) == 2 and isinstance(value[0], (int, float)):
                    # Range filter
                    filtered_df = filtered_df[(filtered_df[key] >= value[0]) & 
                                             (filtered_df[key] <= value[1])]
                else:
                    # List filter
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
            else:
                # Single value filter
                filtered_df = filtered_df[filtered_df[key] == value]
    
    return filtered_df

# ============================================================================
# SECTION 10: MAIN UI
# ============================================================================

def main():
    """
    Main application entry point.
    """
    st.title("🔬 Proton-Conducting Perovskite Analysis Dashboard")
    st.markdown("""
    **Analyze composition-structure-property relationships in proton-conducting perovskite oxides**
    
    This dashboard provides comprehensive analysis of thermal and chemical expansion coefficients
    for perovskite materials used in solid oxide fuel cells (SOFCs).
    """)
    
    # Initialize session state
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = pd.DataFrame()
    if 'df_descriptors' not in st.session_state:
        st.session_state.df_descriptors = pd.DataFrame()
    if 'correlations' not in st.session_state:
        st.session_state.correlations = {}
    if 'top_descriptors' not in st.session_state:
        st.session_state.top_descriptors = []
    if 'pca_results' not in st.session_state:
        st.session_state.pca_results = {}
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = {}
    
    # Sidebar - Filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Data upload section
        st.subheader("📤 Data Input")
        
        data_input = st.text_area(
            "Paste your data here (tab-separated, with headers):",
            height=200,
            help="Paste data in the format shown in the example"
        )
        
        if st.button("🔄 Load Data", type="primary"):
            if data_input:
                with st.spinner("Loading and processing data..."):
                    st.session_state.df_raw = parse_uploaded_data(data_input)
                    if not st.session_state.df_raw.empty:
                        st.session_state.df_descriptors = calculate_descriptors(st.session_state.df_raw)
                        st.success(f"Loaded {len(st.session_state.df_descriptors)} rows")
                    else:
                        st.error("Failed to parse data. Please check format.")
            else:
                st.warning("Please paste data first.")
        
        # Show status
        if not st.session_state.df_raw.empty:
            st.info(f"📊 Data loaded: {len(st.session_state.df_raw)} rows, {len(st.session_state.df_raw.columns)} columns")
        
        st.divider()
        
        # Filter section (only if data loaded)
        if not st.session_state.df_descriptors.empty:
            st.subheader("📊 Data Filters")
            
            df = st.session_state.df_descriptors
            
            # Basic filters
            with st.expander("Basic Filters", expanded=True):
                # Method filter
                if 'method' in df.columns:
                    methods = ['All'] + sorted(df['method'].dropna().unique().tolist())
                    selected_method = st.selectbox("Method", methods)
                else:
                    selected_method = 'All'
                
                # A-cation filter
                if 'A' in df.columns:
                    a_cations = ['All'] + sorted(df['A'].dropna().unique().tolist())
                    selected_a = st.selectbox("A-cation", a_cations)
                else:
                    selected_a = 'All'
                
                # B-cation filter
                if 'B' in df.columns:
                    b_cations = ['All'] + sorted(df['B'].dropna().unique().tolist())
                    selected_b = st.selectbox("B-cation", b_cations)
                else:
                    selected_b = 'All'
            
            # Advanced filters
            with st.expander("Advanced Filters"):
                # Delta range
                if 'δ' in df.columns:
                    delta_min = float(df['δ'].min())
                    delta_max = float(df['δ'].max())
                    delta_range = st.slider("δ range", delta_min, delta_max, (delta_min, delta_max))
                else:
                    delta_range = (0.0, 1.0)
                
                # pH2O range
                if 'pH2O' in df.columns:
                    pH_min = float(df['pH2O'].min())
                    pH_max = float(df['pH2O'].max())
                    pH_range = st.slider("pH₂O range", pH_min, pH_max, (pH_min, pH_max))
                else:
                    pH_range = (0.0, 1.0)
                
                # Temperature range
                if 'delta_T_low' in df.columns and 'delta_T_high' in df.columns:
                    t_min = float(df['delta_T_low'].min())
                    t_max = float(df['delta_T_high'].max())
                    t_range = st.slider("Temperature range (°C)", t_min, t_max, (t_min, t_max))
                else:
                    t_range = (0.0, 1200.0)
                
                # Has T_bends
                has_bends = st.checkbox("Only with T(bends) data")
            
            # Descriptor filters
            with st.expander("Descriptor Filters"):
                # rAav range
                if 'rAav' in df.columns:
                    rA_min = float(df['rAav'].min())
                    rA_max = float(df['rAav'].max())
                    rA_range = st.slider("rAav range (Å)", rA_min, rA_max, (rA_min, rA_max))
                else:
                    rA_range = (0.0, 3.0)
                
                # Tolerance factor
                if 't' in df.columns:
                    t_min = float(df['t'].min())
                    t_max = float(df['t'].max())
                    t_range = st.slider("Tolerance factor (t)", t_min, t_max, (t_min, t_max))
                else:
                    t_range = (0.7, 1.1)
                
                # chiBav range
                if 'chiBav' in df.columns:
                    chi_min = float(df['chiBav'].min())
                    chi_max = float(df['chiBav'].max())
                    chi_range = st.slider("χBav range", chi_min, chi_max, (chi_min, chi_max))
                else:
                    chi_range = (0.0, 3.0)
            
            # Apply filters button
            if st.button("Apply Filters", type="primary"):
                # Build filter dictionary
                filter_dict = {}
                if selected_method != 'All':
                    filter_dict['method'] = selected_method
                if selected_a != 'All':
                    filter_dict['A'] = selected_a
                if selected_b != 'All':
                    filter_dict['B'] = selected_b
                if 'δ' in df.columns:
                    filter_dict['δ'] = delta_range
                if 'pH2O' in df.columns:
                    filter_dict['pH2O'] = pH_range
                if 'rAav' in df.columns:
                    filter_dict['rAav'] = rA_range
                if 't' in df.columns:
                    filter_dict['t'] = t_range
                if 'chiBav' in df.columns:
                    filter_dict['chiBav'] = chi_range
                if has_bends:
                    filter_dict['T_bends_count'] = (1, 100)
                
                st.session_state.filtered_data = apply_filters(df, filter_dict)
                st.success(f"Filtered to {len(st.session_state.filtered_data)} rows")
    
    # Main content area
    if st.session_state.df_descriptors.empty:
        st.info("👈 Please load data using the sidebar")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔬 Descriptors",
        "📈 Correlations",
        "🧬 PCA & Clustering",
        "📉 Visualizations",
        "💾 Export"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.df_raw))
        with col2:
            st.metric("Total Columns", len(st.session_state.df_raw.columns))
        with col3:
            if 'method' in st.session_state.df_raw.columns:
                methods = st.session_state.df_raw['method'].value_counts()
                st.metric("Methods", len(methods))
        
        st.subheader("Sample Data")
        st.dataframe(st.session_state.df_raw.head(10))
        
        st.subheader("Data Statistics")
        st.dataframe(st.session_state.df_raw.describe())
    
    # Tab 2: Descriptors
    with tab2:
        st.header("Descriptor Analysis")
        
        # Show calculated descriptors
        st.subheader("Calculated Descriptors")
        descriptor_cols = [col for col in st.session_state.df_descriptors.columns 
                          if col in ['rAav', 'rBav', 't', 'D_t', 'chiAav', 'chiBav',
                                    'delta_chi_AB', 'V_cell', 'S_config_A', 'S_config_B',
                                    'M_total', 'Vo_proxy', 'proton_affinity', 'T_stab',
                                    "B'_conc", 'D_total']]
        
        if descriptor_cols:
            st.dataframe(st.session_state.df_descriptors[descriptor_cols].head(10))
        else:
            st.warning("No descriptor columns available")
        
        # Distribution plots
        st.subheader("Descriptor Distributions")
        selected_descriptors = st.multiselect(
            "Select descriptors to visualize",
            descriptor_cols,
            default=descriptor_cols[:min(6, len(descriptor_cols))]
        )
        
        if selected_descriptors:
            fig = create_distribution_plots(st.session_state.df_descriptors, selected_descriptors)
            st.pyplot(fig)
    
    # Tab 3: Correlations
    with tab3:
        st.header("Correlation Analysis")
        
        # Target variables
        target_cols = ['alpha', 'β', 'alpha_av', 'T_bends_first']
        available_targets = [col for col in target_cols if col in st.session_state.df_descriptors.columns]
        
        if available_targets:
            with st.spinner("Calculating correlations..."):
                st.session_state.correlations = calculate_correlations(
                    st.session_state.df_descriptors, 
                    available_targets
                )
        
        if st.session_state.correlations:
            # Correlation matrix
            st.subheader("Correlation Matrix")
            if 'pearson' in st.session_state.correlations:
                corr_matrix = st.session_state.correlations['pearson']
                fig = create_correlation_heatmap(corr_matrix, "Pearson Correlation Matrix")
                st.pyplot(fig)
            
            # Top descriptors
            st.subheader("Top Descriptors")
            
            # Find top descriptors if not already
            if not st.session_state.top_descriptors:
                st.session_state.top_descriptors = find_top_descriptors(
                    st.session_state.correlations,
                    available_targets,
                    n_top=20
                )
            
            if st.session_state.top_descriptors:
                st.write("Top 20 descriptors by correlation with target variables:")
                st.dataframe(pd.DataFrame({
                    'Descriptor': st.session_state.top_descriptors,
                    'Rank': range(1, len(st.session_state.top_descriptors) + 1)
                }))
                
                # Feature importance
                if 'rf_importance' in st.session_state.correlations:
                    st.subheader("Feature Importance (Random Forest)")
                    for target in available_targets:
                        if target in st.session_state.correlations['rf_importance']:
                            fig = create_feature_importance_plot(
                                st.session_state.correlations['rf_importance'],
                                target
                            )
                            st.pyplot(fig)
            
            # Network correlation
            st.subheader("Correlation Network")
            threshold = st.slider("Correlation threshold", 0.3, 0.8, 0.5, 0.05)
            if 'pearson' in st.session_state.correlations:
                fig = create_network_correlation(
                    st.session_state.df_descriptors,
                    st.session_state.correlations['pearson'],
                    threshold
                )
                st.pyplot(fig)
        else:
            st.warning("Insufficient data for correlation analysis")
    
    # Tab 4: PCA & Clustering
    with tab4:
        st.header("PCA and Clustering Analysis")
        
        # Ensure we have top descriptors
        if not st.session_state.top_descriptors and st.session_state.correlations:
            st.session_state.top_descriptors = find_top_descriptors(
                st.session_state.correlations,
                available_targets,
                n_top=20
            )
        
        if st.session_state.top_descriptors:
            # PCA
            st.subheader("Principal Component Analysis")
            
            # Select descriptors for PCA
            pca_descriptors = st.multiselect(
                "Select descriptors for PCA",
                st.session_state.top_descriptors,
                default=st.session_state.top_descriptors[:min(10, len(st.session_state.top_descriptors))]
            )
            
            if len(pca_descriptors) >= 3:
                with st.spinner("Performing PCA..."):
                    st.session_state.pca_results = perform_pca_analysis(
                        st.session_state.df_descriptors,
                        pca_descriptors
                    )
                
                if st.session_state.pca_results:
                    # Show explained variance
                    fig, ax = plt.subplots(figsize=(8, 5))
                    explained = st.session_state.pca_results['explained_variance']
                    cumulative = st.session_state.pca_results['cumulative_variance']
                    
                    ax.bar(range(1, len(explained) + 1), explained * 100, 
                          alpha=0.7, color='#3498DB', label='Individual')
                    ax.plot(range(1, len(cumulative) + 1), cumulative * 100, 
                           'ro-', linewidth=2, label='Cumulative')
                    ax.axhline(y=85, color='red', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Principal Component', fontweight='bold')
                    ax.set_ylabel('Variance Explained (%)', fontweight='bold')
                    ax.set_title('PCA Explained Variance', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Biplot
                    st.subheader("PCA Biplot")
                    fig = create_pca_biplot(
                        st.session_state.pca_results['pca_result'],
                        st.session_state.pca_results['loadings']
                    )
                    st.pyplot(fig)
            
            # Clustering
            st.subheader("Clustering Analysis")
            
            if st.session_state.pca_results and 'pca_result' in st.session_state.pca_results:
                n_clusters = st.number_input("Number of clusters (0 for automatic)", 
                                            min_value=0, max_value=10, value=0)
                
                with st.spinner("Performing clustering..."):
                    st.session_state.clustering_results = perform_clustering(
                        st.session_state.pca_results['pca_result'],
                        n_clusters if n_clusters > 0 else None
                    )
                
                if st.session_state.clustering_results:
                    # Show silhouette
                    st.metric("Silhouette Score", 
                             f"{st.session_state.clustering_results['silhouette_avg']:.3f}")
                    st.metric("Optimal Clusters", 
                             st.session_state.clustering_results['optimal_k'])
                    
                    # Cluster profiles
                    if 'cluster_means' not in locals():
                        df_clust = st.session_state.df_descriptors[pca_descriptors].copy()
                        df_clust['Cluster'] = st.session_state.clustering_results['labels']
                        cluster_means = df_clust.groupby('Cluster')[pca_descriptors].mean()
                    
                    fig = create_cluster_profiles(
                        st.session_state.df_descriptors,
                        st.session_state.clustering_results['labels'],
                        pca_descriptors[:min(10, len(pca_descriptors))]
                    )
                    st.pyplot(fig)
        
        else:
            st.warning("Not enough descriptors for PCA. Please load data and calculate correlations first.")
    
    # Tab 5: Visualizations
    with tab5:
        st.header("Advanced Visualizations")
        
        # Get filtered data or use full dataset
        df_viz = getattr(st.session_state, 'filtered_data', st.session_state.df_descriptors)
        
        # Target variables for visualization
        target_vars = ['alpha', 'β', 'alpha_av', 'T_bends_first']
        available_targets = [col for col in target_vars if col in df_viz.columns]
        
        if not available_targets:
            st.warning("No target variables available for visualization")
            return
        
        # Select target
        selected_target = st.selectbox("Select target variable", available_targets)
        
        # Get top descriptors
        if not st.session_state.top_descriptors and st.session_state.correlations:
            st.session_state.top_descriptors = find_top_descriptors(
                st.session_state.correlations,
                available_targets,
                n_top=20
            )
        
        # Add compositional descriptors
        comp_descriptors = ["B'_conc", 'D_total', 'δ', 'D_ratio']
        all_descriptors = comp_descriptors + st.session_state.top_descriptors[:15]
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select visualization type",
            ["Concentration Heatmap", "Bubble Chart", "3D Scatter", 
             "Pairplot", "Scatter with Regression", "α vs β Compromise"]
        )
        
        if viz_type == "Concentration Heatmap":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", all_descriptors, key='heat_x')
            with col2:
                y_col = st.selectbox("Y-axis", all_descriptors, key='heat_y')
            
            if x_col and y_col and x_col != y_col:
                fig = create_plotly_concentration_heatmap(
                    df_viz, x_col, y_col, selected_target
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bubble Chart":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", all_descriptors, key='bub_x')
            with col2:
                color_col = st.selectbox("Color by", all_descriptors, key='bub_color')
            with col3:
                size_col = st.selectbox("Size by", all_descriptors, key='bub_size')
            
            # Shape options
            shape_col = st.selectbox("Shape by (optional)", ['None'] + 
                                    ['B', 'A', 'method'], key='bub_shape')
            
            if x_col and color_col and size_col:
                fig = create_plotly_bubble_chart(
                    df_viz, x_col, selected_target, color_col, size_col,
                    shape_col if shape_col != 'None' else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "3D Scatter":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", all_descriptors, key='3d_x')
                y_col = st.selectbox("Y-axis", all_descriptors, key='3d_y')
            with col2:
                z_col = st.selectbox("Z-axis", ['alpha', 'β', 'alpha_av'], key='3d_z')
                color_3d = st.selectbox("Color by (optional)", ['None'] + all_descriptors, key='3d_color')
            
            if x_col and y_col and z_col:
                fig = create_plotly_3d_scatter(
                    df_viz, x_col, y_col, z_col,
                    color_3d if color_3d != 'None' else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pairplot":
            selected_features = st.multiselect(
                "Select features for pairplot",
                all_descriptors,
                default=all_descriptors[:min(4, len(all_descriptors))]
            )
            
            hue_opt = st.selectbox("Hue (optional)", ['None'] + ['A', 'B', 'method'], key='pair_hue')
            
            if len(selected_features) >= 2:
                fig = create_pairplot(
                    df_viz, selected_features,
                    hue_opt if hue_opt != 'None' else None
                )
                st.pyplot(fig)
        
        elif viz_type == "Scatter with Regression":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", all_descriptors, key='reg_x')
            with col2:
                color_reg = st.selectbox("Color by (optional)", ['None'] + all_descriptors, key='reg_color')
            
            if x_col:
                fig = create_scatter_with_regression(
                    df_viz, x_col, selected_target,
                    color_reg if color_reg != 'None' else None
                )
                st.pyplot(fig)
        
        elif viz_type == "α vs β Compromise":
            alpha_col = st.selectbox("Alpha column", ['alpha', 'alpha_av'], key='ab_alpha')
            beta_col = st.selectbox("Beta column", ['β'], key='ab_beta')
            color_ab = st.selectbox("Color by", ['None', 'A', 'B', 'method'], key='ab_color')
            
            fig = create_alpha_beta_compromise(
                df_viz, alpha_col, beta_col,
                color_ab if color_ab != 'None' else None
            )
            st.pyplot(fig)
    
    # Tab 6: Export
    with tab6:
        st.header("Export Results")
        
        st.subheader("Export Data")
        
        # Export options
        export_type = st.radio("Export type", ["Raw Data", "Descriptors", "Filtered Data"])
        
        if export_type == "Raw Data":
            export_df = st.session_state.df_raw
        elif export_type == "Descriptors":
            export_df = st.session_state.df_descriptors
        else:
            export_df = getattr(st.session_state, 'filtered_data', st.session_state.df_descriptors)
        
        if not export_df.empty:
            # Download as CSV
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{export_type.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            # Download as Excel
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Data')
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"{export_type.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.subheader("Export Plots")
        st.info("Plots can be saved individually by clicking the download icon on each plot.")
        
        st.subheader("Session Summary")
        st.write(f"Data rows: {len(st.session_state.df_raw)}")
        st.write(f"Descriptors calculated: {len(st.session_state.df_descriptors.columns)}")
        if st.session_state.top_descriptors:
            st.write(f"Top descriptors identified: {len(st.session_state.top_descriptors)}")
        if st.session_state.correlations:
            st.write("Correlation analysis: Completed")
        if st.session_state.pca_results:
            st.write("PCA analysis: Completed")
        if st.session_state.clustering_results:
            st.write("Clustering analysis: Completed")
        
        # Reset option
        if st.button("🔄 Reset All Data", type="primary"):
            st.session_state.df_raw = pd.DataFrame()
            st.session_state.df_descriptors = pd.DataFrame()
            st.session_state.correlations = {}
            st.session_state.top_descriptors = []
            st.session_state.pca_results = {}
            st.session_state.clustering_results = {}
            st.rerun()


if __name__ == "__main__":
    main()
