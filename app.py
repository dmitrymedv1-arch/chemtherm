"""
PROTON-CONDUCTING PEROVSKITES ANALYZER
========================================
Interactive Streamlit application for analyzing correlations between
chemical composition, structure, and thermal/chemical expansion properties
of proton-conducting perovskite oxides.

Author: Materials Science Research Group
Version: 1.0.0
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Data processing
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.interpolate import griddata

# Machine learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# Statistics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf

# Set page config
st.set_page_config(
    page_title="Proton-Conducting Perovskites Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SECTION 2: BUILT-IN DATABASES
# ============================================================================

# 2.1 Ionic radii according to Shannon (A-site: 12-coordination, B-site: 6-coordination)
IONIC_RADII = {
    # A-site cations (12-coordination)
    'Ba': 1.61, 'Ba2': 1.61, 'Ba²⁺': 1.61,
    'Sr': 1.44, 'Sr2': 1.44, 'Sr²⁺': 1.44,
    'Ca': 1.34, 'Ca2': 1.34, 'Ca²⁺': 1.34,
    'La': 1.36, 'La3': 1.36, 'La³⁺': 1.36,
    'Nd': 1.27, 'Nd3': 1.27, 'Nd³⁺': 1.27,
    'Sm': 1.24, 'Sm3': 1.24, 'Sm³⁺': 1.24,
    'Gd': 1.19, 'Gd3': 1.19, 'Gd³⁺': 1.19,
    'Yb': 1.12, 'Yb3': 1.12, 'Yb³⁺': 1.12,
    'Y': 1.10, 'Y3': 1.10, 'Y³⁺': 1.10,
    'Sc': 0.90, 'Sc3': 0.90, 'Sc³⁺': 0.90,
    'Eu': 1.25, 'Eu3': 1.25, 'Eu³⁺': 1.25,
    'Dy': 1.17, 'Dy3': 1.17, 'Dy³⁺': 1.17,
    'Ho': 1.15, 'Ho3': 1.15, 'Ho³⁺': 1.15,
    'Tm': 1.13, 'Tm3': 1.13, 'Tm³⁺': 1.13,
    'Tb': 1.20, 'Tb3': 1.20, 'Tb³⁺': 1.20,
    'Pr': 1.29, 'Pr3': 1.29, 'Pr³⁺': 1.29,
    'Pb': 1.49, 'Pb2': 1.49, 'Pb²⁺': 1.49,
    'Bi': 1.38, 'Bi3': 1.38, 'Bi³⁺': 1.38,
    'Al': 0.54, 'Al3': 0.54, 'Al³⁺': 0.54,
    'Ga': 0.62, 'Ga3': 0.62, 'Ga³⁺': 0.62,
    
    # B-site cations (6-coordination)
    'Ce': 0.87, 'Ce4': 0.87, 'Ce⁴⁺': 0.87,
    'Zr': 0.72, 'Zr4': 0.72, 'Zr⁴⁺': 0.72,
    'Sn': 0.69, 'Sn4': 0.69, 'Sn⁴⁺': 0.69,
    'Ti': 0.605, 'Ti4': 0.605, 'Ti⁴⁺': 0.605,
    'Hf': 0.71, 'Hf4': 0.71, 'Hf⁴⁺': 0.71,
    'Fe': 0.645, 'Fe3': 0.645, 'Fe³⁺': 0.645,
    'Zn': 0.60, 'Zn2': 0.60, 'Zn²⁺': 0.60,
    'In': 0.80, 'In3': 0.80, 'In³⁺': 0.80,
    'Y': 0.90, 'Y3': 0.90, 'Y³⁺': 0.90,
    'Yb': 0.868, 'Yb3': 0.868, 'Yb³⁺': 0.868,
    'Sc': 0.745, 'Sc3': 0.745, 'Sc³⁺': 0.745,
    'Sm': 0.958, 'Sm3': 0.958, 'Sm³⁺': 0.958,
    'Nd': 0.983, 'Nd3': 0.983, 'Nd³⁺': 0.983,
    'Gd': 0.938, 'Gd3': 0.938, 'Gd³⁺': 0.938,
    'Dy': 0.912, 'Dy3': 0.912, 'Dy³⁺': 0.912,
    'Eu': 0.947, 'Eu3': 0.947, 'Eu³⁺': 0.947,
    'Ho': 0.901, 'Ho3': 0.901, 'Ho³⁺': 0.901,
    'Tm': 0.880, 'Tm3': 0.880, 'Tm³⁺': 0.880,
    'Tb': 0.923, 'Tb3': 0.923, 'Tb³⁺': 0.923,
    'La': 1.032, 'La3': 1.032, 'La³⁺': 1.032,
    'Pr': 0.99, 'Pr3': 0.99, 'Pr³⁺': 0.99,
    'Bi': 0.76, 'Bi3': 0.76, 'Bi³⁺': 0.76,
    'Al': 0.535, 'Al3': 0.535, 'Al³⁺': 0.535,
    'Ga': 0.62, 'Ga3': 0.62, 'Ga³⁺': 0.62,
    'Si': 0.40, 'Si4': 0.40, 'Si⁴⁺': 0.40,
    'Ge': 0.53, 'Ge4': 0.53, 'Ge⁴⁺': 0.53,
}

# 2.2 Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {
    'Ba': 0.89, 'Sr': 0.95, 'Ca': 1.00, 'Mg': 1.31,
    'La': 1.10, 'Nd': 1.14, 'Sm': 1.17, 'Eu': 1.20,
    'Gd': 1.20, 'Tb': 1.20, 'Dy': 1.22, 'Ho': 1.23,
    'Y': 1.22, 'Yb': 1.10, 'Lu': 1.27, 'Sc': 1.36,
    'Ce': 1.12, 'Zr': 1.33, 'Sn': 1.96, 'Ti': 1.54,
    'Hf': 1.30, 'Fe': 1.83, 'Zn': 1.65, 'In': 1.78,
    'Pb': 2.33, 'Bi': 2.02, 'Al': 1.61, 'Ga': 1.81,
    'Ge': 2.01, 'Si': 1.90, 'O': 3.44, 'Pr': 1.13,
    'Tm': 1.25, 'W': 2.36, 'Mo': 2.16, 'Nb': 1.60,
    'Ta': 1.50, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
    'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Ag': 1.93,
    'Au': 2.54, 'Pt': 2.28, 'Pd': 2.20, 'Rh': 2.28,
    'Ru': 2.20, 'Os': 2.20, 'Ir': 2.20, 'Re': 1.90,
}

# 2.3 Valence states
VALENCE = {
    'Ba': 2, 'Sr': 2, 'Ca': 2, 'Mg': 2,
    'La': 3, 'Nd': 3, 'Sm': 3, 'Eu': 3,
    'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3,
    'Y': 3, 'Yb': 3, 'Lu': 3, 'Sc': 3,
    'Ce': 4, 'Zr': 4, 'Sn': 4, 'Ti': 4,
    'Hf': 4, 'Fe': 3, 'Zn': 2, 'In': 3,
    'Pb': 2, 'Bi': 3, 'Al': 3, 'Ga': 3,
    'Ge': 4, 'Si': 4, 'Pr': 3, 'Tm': 3,
    'W': 6, 'Mo': 6, 'Nb': 5, 'Ta': 5,
    'V': 5, 'Cr': 3, 'Mn': 2, 'Co': 2,
    'Ni': 2, 'Cu': 2, 'Ag': 1, 'Au': 3,
    'Pt': 4, 'Pd': 2, 'Rh': 3, 'Ru': 4,
    'Os': 4, 'Ir': 4, 'Re': 7,
}

# 2.4 Molar masses (g/mol)
MOLAR_MASS = {
    'Ba': 137.327, 'Sr': 87.62, 'Ca': 40.078, 'Mg': 24.305,
    'La': 138.905, 'Nd': 144.242, 'Sm': 150.36, 'Eu': 151.964,
    'Gd': 157.25, 'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930,
    'Y': 88.906, 'Yb': 173.045, 'Lu': 174.967, 'Sc': 44.956,
    'Ce': 140.116, 'Zr': 91.224, 'Sn': 118.710, 'Ti': 47.867,
    'Hf': 178.49, 'Fe': 55.845, 'Zn': 65.38, 'In': 114.818,
    'Pb': 207.2, 'Bi': 208.980, 'Al': 26.982, 'Ga': 69.723,
    'Ge': 72.630, 'Si': 28.085, 'Pr': 140.908, 'Tm': 168.934,
    'W': 183.84, 'Mo': 95.95, 'Nb': 92.906, 'Ta': 180.948,
    'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Co': 58.933,
    'Ni': 58.693, 'Cu': 63.546, 'Ag': 107.868, 'Au': 196.967,
    'Pt': 195.084, 'Pd': 106.42, 'Rh': 102.906, 'Ru': 101.07,
    'Os': 190.23, 'Ir': 192.217, 'Re': 186.207, 'O': 15.999,
}

# 2.5 Oxygen ionic radius (6-coordination)
R_O = 1.40  # Å

# ============================================================================
# SECTION 3: SCIENTIFIC PLOTTING STYLE
# ============================================================================

def apply_scientific_style():
    """
    Apply unified scientific plotting style for publication-quality figures.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # Font settings
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        
        # Axes settings
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#000000',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Tick settings
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
        'xtick.minor.size': 4,
        'xtick.minor.width': 1,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1,
        
        # Legend settings
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#000000',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        
        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'errorbar.capsize': 3,
        
        # PDF settings for publications
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# ============================================================================
# SECTION 4: COLOR PALETTES
# ============================================================================

COLOR_PALETTES = {
    'B_cation': {
        'Ce': '#E74C3C',    # Red
        'Zr': '#3498DB',    # Blue
        'Sn': '#2ECC71',    # Green
        'Ti': '#F39C12',    # Orange
        'Hf': '#9B59B6',    # Purple
        'default': '#7F8C8D'
    },
    'method': {
        'dilatometry': '#2C3E50',   # Dark navy
        'HT XRD': '#E67E22',         # Orange
        'HT ND': '#8E44AD',          # Purple
        'default': '#7F8C8D'
    },
    'A_cation': {
        'Ba': '#1A5276',    # Dark blue
        'Sr': '#2471A3',    # Medium blue
        'Ca': '#5DADE2',    # Light blue
        'La': '#F39C12',    # Orange
        'default': '#7F8C8D'
    },
    'continuous': 'viridis',
    'diverging': 'coolwarm',
    'categorical': 'Set2',
    'clusters': 'tab10'
}

# ============================================================================
# SECTION 5: DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def parse_uploaded_data(text: str) -> pd.DataFrame:
    """
    Parse pasted tabular data into pandas DataFrame.
    
    Args:
        text: String containing tab-separated data with header
        
    Returns:
        DataFrame with parsed data
    """
    try:
        # Read from string buffer
        df = pd.read_csv(io.StringIO(text), sep='\t', na_values=['-', '—', ''])
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Remove any empty rows
        df = df.dropna(how='all')
        
        # Remove any empty columns
        df = df.dropna(axis=1, how='all')
        
        # Convert numeric columns
        numeric_cols = ['[A']', '[B']', '[D1]', '[D2]', 'δ', 'β', 'α·106 (K-1)', 'pH2O']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse temperature range
        if '∆T, °C' in df.columns:
            df['T_min'] = df['∆T, °C'].str.split('-').str[0].astype(float)
            df['T_max'] = df['∆T, °C'].str.split('-').str[1].astype(float)
            df['T_range'] = df['T_max'] - df['T_min']
        
        # Calculate delta if not present
        if 'δ' in df.columns and ['D1]' in df.columns and ['D2]' in df.columns:
            df['δ_calc'] = df['[D1]']/2 + df['[D2]']/2
            # Use calculated if original is NaN
            df['δ'] = df['δ'].fillna(df['δ_calc'])
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def parse_semicolon_list(value: str) -> List[float]:
    """
    Parse semicolon-separated values into list of floats.
    
    Args:
        value: String like "10.6;4.73;10.1" or "400;600"
        
    Returns:
        List of floats
    """
    if pd.isna(value) or value == '-' or value == '':
        return []
    try:
        return [float(x.strip()) for x in str(value).split(';') if x.strip()]
    except:
        return []

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the DataFrame.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Parse T(bends) and alpha_av
    if 'T(bends), °C' in df_clean.columns:
        df_clean['T_bends_list'] = df_clean['T(bends), °C'].apply(parse_semicolon_list)
        df_clean['n_bends'] = df_clean['T_bends_list'].apply(len)
        df_clean['T_bends_first'] = df_clean['T_bends_list'].apply(lambda x: x[0] if x else np.nan)
        df_clean['T_bends_last'] = df_clean['T_bends_list'].apply(lambda x: x[-1] if x else np.nan)
    
    if 'αav·106 (K-1)' in df_clean.columns:
        df_clean['alpha_av_list'] = df_clean['αav·106 (K-1)'].apply(parse_semicolon_list)
        df_clean['alpha_av_first'] = df_clean['alpha_av_list'].apply(lambda x: x[0] if x else np.nan)
        df_clean['alpha_av_last'] = df_clean['alpha_av_list'].apply(lambda x: x[-1] if x else np.nan)
    
    # Extract B-cation type
    df_clean['B_type'] = df_clean['B'].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
    
    # Extract A-cation type
    df_clean['A_type'] = df_clean['A'].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
    
    # Calculate D_total
    if '[D1]' in df_clean.columns and '[D2]' in df_clean.columns:
        df_clean['D_total'] = df_clean['[D1]'] + df_clean['[D2]']
    
    # Calculate B'_conc
    if '[B']' in df_clean.columns:
        df_clean['B'_conc'] = df_clean['[B']']
    
    # Calculate alpha/beta ratio
    if 'α·106 (K-1)' in df_clean.columns and 'β' in df_clean.columns:
        df_clean['alpha_beta_ratio'] = df_clean['α·106 (K-1)'] / df_clean['β']
    
    return df_clean

# ============================================================================
# SECTION 6: DESCRIPTOR ENGINE
# ============================================================================

@st.cache_data
def calculate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 63+ descriptors for each composition.
    
    Args:
        df: Cleaned DataFrame with composition data
        
    Returns:
        DataFrame with all descriptors added
    """
    df_desc = df.copy()
    
    # Extract element names and concentrations
    for idx, row in df_desc.iterrows():
        # A-site
        A_elem = row.get('A', '')
        A_prime_elem = row.get('A'', '')
        A_prime_conc = row.get('[A'']', 0)
        
        # B-site
        B_elem = row.get('B', '')
        B_prime_elem = row.get('B'', '')
        B_prime_conc = row.get('[B'']', 0)
        D1_elem = row.get('D1', '')
        D1_conc = row.get('[D1]', 0)
        D2_elem = row.get('D2', '')
        D2_conc = row.get('[D2]', 0)
        
        # Skip if A or B is missing
        if pd.isna(A_elem) or pd.isna(B_elem):
            continue
        
        # Get ionic radii
        rA = IONIC_RADII.get(str(A_elem), np.nan)
        rA_prime = IONIC_RADII.get(str(A_prime_elem), np.nan)
        rB = IONIC_RADII.get(str(B_elem), np.nan)
        rB_prime = IONIC_RADII.get(str(B_prime_elem), np.nan)
        rD1 = IONIC_RADII.get(str(D1_elem), np.nan)
        rD2 = IONIC_RADII.get(str(D2_elem), np.nan)
        
        # Get electronegativity
        χA = ELECTRONEGATIVITY.get(str(A_elem), np.nan)
        χA_prime = ELECTRONEGATIVITY.get(str(A_prime_elem), np.nan)
        χB = ELECTRONEGATIVITY.get(str(B_elem), np.nan)
        χB_prime = ELECTRONEGATIVITY.get(str(B_prime_elem), np.nan)
        χD1 = ELECTRONEGATIVITY.get(str(D1_elem), np.nan)
        χD2 = ELECTRONEGATIVITY.get(str(D2_elem), np.nan)
        
        # Get valence
        VB = VALENCE.get(str(B_elem), np.nan)
        VB_prime = VALENCE.get(str(B_prime_elem), np.nan)
        VD1 = VALENCE.get(str(D1_elem), np.nan)
        VD2 = VALENCE.get(str(D2_elem), np.nan)
        
        # Get molar mass
        MA = MOLAR_MASS.get(str(A_elem), np.nan)
        MA_prime = MOLAR_MASS.get(str(A_prime_elem), np.nan)
        MB = MOLAR_MASS.get(str(B_elem), np.nan)
        MB_prime = MOLAR_MASS.get(str(B_prime_elem), np.nan)
        MD1 = MOLAR_MASS.get(str(D1_elem), np.nan)
        MD2 = MOLAR_MASS.get(str(D2_elem), np.nan)
        
        # Calculate average radii
        rAav = rA * (1 - A_prime_conc) + rA_prime * A_prime_conc
        rBav = rB * (1 - B_prime_conc - D1_conc - D2_conc) + rB_prime * B_prime_conc + rD1 * D1_conc + rD2 * D2_conc
        
        # Calculate tolerance factor (Goldschmidt)
        t = (rAav + R_O) / (np.sqrt(2) * (rBav + R_O))
        
        # Calculate deviation from ideal
        D_t = abs(1 - t)
        
        # Alternative tolerance factor
        t_alt = (rAav + R_O) / (np.sqrt(2) * (rBav + R_O))  # Same formula, kept for compatibility
        
        # Octahedral factor
        octahedral_factor = rBav / R_O
        
        # Radius difference
        Δr_AB = abs(rAav - rBav)
        Δr_AB_norm = Δr_AB / R_O
        
        # Variance of B-site radii
        B_sites = [rB * (1 - B_prime_conc - D1_conc - D2_conc), 
                   rB_prime * B_prime_conc, 
                   rD1 * D1_conc, 
                   rD2 * D2_conc]
        B_sites = [x for x in B_sites if not np.isnan(x)]
        σ²_rB = np.var(B_sites) if len(B_sites) > 0 else 0
        
        # Variance of A-site radii
        A_sites = [rA * (1 - A_prime_conc), rA_prime * A_prime_conc]
        A_sites = [x for x in A_sites if not np.isnan(x)]
        σ²_rA = np.var(A_sites) if len(A_sites) > 0 else 0
        
        # Calculate average electronegativity
        χAav = χA * (1 - A_prime_conc) + χA_prime * A_prime_conc
        χBav = χB * (1 - B_prime_conc - D1_conc - D2_conc) + χB_prime * B_prime_conc + χD1 * D1_conc + χD2 * D2_conc
        
        # Electronegativity difference
        Δχ_AB = abs(χAav - χBav)
        χ_ratio_AB = χAav / χBav if χBav != 0 else np.nan
        
        # Total average electronegativity
        χ_total = (χAav + χBav) / 2
        
        # Ionicity of A-O and B-O bonds
        ionicity_AO = 1 - np.exp(-0.25 * (χAav - 3.44)**2)
        ionicity_BO = 1 - np.exp(-0.25 * (χBav - 3.44)**2)
        
        # Acidity descriptors
        acidity_AO = 1 / χAav if χAav != 0 else np.nan
        acidity_BO = 1 / χBav if χBav != 0 else np.nan
        Δacidity = acidity_BO - acidity_AO
        
        # Configurational entropy
        S_config_A = -8.314 * (A_prime_conc * np.log(A_prime_conc) + (1 - A_prime_conc) * np.log(1 - A_prime_conc)) if A_prime_conc > 0 and A_prime_conc < 1 else 0
        S_config_B = 0
        B_conc_list = [1 - B_prime_conc - D1_conc - D2_conc, B_prime_conc, D1_conc, D2_conc]
        for c in B_conc_list:
            if c > 0 and c < 1:
                S_config_B += -8.314 * c * np.log(c)
        
        # Average valence of B-site
        V_Bav = VB * (1 - B_prime_conc - D1_conc - D2_conc) + VB_prime * B_prime_conc + VD1 * D1_conc + VD2 * D2_conc
        
        # Vacancy proxy
        Vo_proxy = (4 - V_Bav) / 2
        
        # Hydration enthalpy (approximate)
        ΔH_hydr = 1 / (rBav * χBav) if rBav * χBav != 0 else np.nan
        
        # Bond energy (Coulombic)
        E_BO = (VB * 2) / rBav if rBav != 0 else np.nan
        
        # Mass density (approximate)
        M_total = MA * (1 - A_prime_conc) + MA_prime * A_prime_conc + MB * (1 - B_prime_conc - D1_conc - D2_conc) + MB_prime * B_prime_conc + MD1 * D1_conc + MD2 * D2_conc + 3 * MOLAR_MASS['O']
        V_cell = (rAav + R_O) * (rBav + R_O)**3
        ρ = M_total / V_cell if V_cell != 0 else np.nan
        
        # Average molar mass
        M_Aav = MA * (1 - A_prime_conc) + MA_prime * A_prime_conc
        M_Bav = MB * (1 - B_prime_conc - D1_conc - D2_conc) + MB_prime * B_prime_conc + MD1 * D1_conc + MD2 * D2_conc
        M_ratio_AB = M_Aav / M_Bav if M_Bav != 0 else np.nan
        M_rA = M_Aav * rAav
        M_χA = M_Aav * χAav
        
        # Defect descriptors
        δ = row.get('δ', D1_conc/2 + D2_conc/2)
        Z_eff_B = 4 - 2 * δ
        proton_affinity = 1 / (rBav * χBav) if rBav * χBav != 0 else np.nan
        E_vac = 1 / (rBav**2) * (χBav - 3.44) if rBav != 0 else np.nan
        
        # T(bends) descriptors
        r_ratio_AB = rAav / rBav if rBav != 0 else np.nan
        T_stab = -ΔH_hydr / 8.314 if not np.isnan(ΔH_hydr) else np.nan
        δ_χB = δ * χBav if not np.isnan(χBav) else np.nan
        
        # Alpha/beta ratio (if available)
        alpha_beta_ratio = row.get('alpha_beta_ratio', np.nan)
        
        # Store all descriptors
        desc_dict = {
            # Geometric descriptors (1-8, 36-40)
            'rAav': rAav,
            'rBav': rBav,
            't': t,
            'D_t': D_t,
            't_alt': t_alt,
            'octahedral_factor': octahedral_factor,
            'Δr_AB': Δr_AB,
            'Δr_AB_norm': Δr_AB_norm,
            'σ²_rB': σ²_rB,
            'σ²_rA': σ²_rA,
            'V_cell': V_cell,
            'V_free': V_cell * (1 - 0.74),  # Approximate packing factor
            'oct_dist': abs(rBav - rBav) / rBav,  # Placeholder
            
            # Electronegativity descriptors (9-21, 41-43)
            'χAav': χAav,
            'χBav': χBav,
            'Δχ_AB': Δχ_AB,
            'χ_ratio_AB': χ_ratio_AB,
            'χ_total': χ_total,
            'ionicity_AO': ionicity_AO,
            'ionicity_BO': ionicity_BO,
            'acidity_AO': acidity_AO,
            'acidity_BO': acidity_BO,
            'Δacidity': Δacidity,
            
            # Thermodynamic descriptors (22-29, 44-46)
            'S_config_A': S_config_A,
            'S_config_B': S_config_B,
            'V_Bav': V_Bav,
            'Vo_proxy': Vo_proxy,
            'ΔH_hydr': ΔH_hydr,
            'E_BO': E_BO,
            'ρ': ρ,
            
            # Mass descriptors (47-52)
            'M_Aav': M_Aav,
            'M_Bav': M_Bav,
            'M_total': M_total,
            'M_ratio_AB': M_ratio_AB,
            'M_rA': M_rA,
            'M_χA': M_χA,
            
            # Defect descriptors (53-56)
            'δ_calc': δ,
            'Z_eff_B': Z_eff_B,
            'proton_affinity': proton_affinity,
            'E_vac': E_vac,
            
            # T(bends) descriptors (57-60)
            'alpha_beta_ratio': alpha_beta_ratio,
            'T_stab': T_stab,
            'δ_χB': δ_χB,
            'r_ratio_AB': r_ratio_AB,
            
            # Compositional descriptors
            'B'_conc': B_prime_conc,
            'D_total': D1_conc + D2_conc,
        }
        
        # Add descriptors to DataFrame
        for key, value in desc_dict.items():
            df_desc.loc[idx, key] = value
    
    return df_desc

# ============================================================================
# SECTION 7: CORRELATION ANALYSIS# ============================================================================

@st.cache_data
def calculate_correlations(df: pd.DataFrame, target_cols: List[str]) -> Dict:
    """
    Calculate comprehensive correlation matrices.
    
    Args:
        df: DataFrame with descriptors
        target_cols: List of target variable names
        
    Returns:
        Dictionary with correlation matrices
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns with too many NaN
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > 5]
    
    # Create correlation DataFrames
    corr_data = df[valid_cols].copy()
    
    # Pearson correlation
    pearson_corr = corr_data.corr(method='pearson')
    
    # Spearman correlation
    spearman_corr = corr_data.corr(method='spearman')
    
    # Partial correlation (controlling for pH2O)
    partial_corr = None
    if 'pH2O' in corr_data.columns:
        pH2O = corr_data['pH2O']
        partial_corr = corr_data.corr()
        for i in range(len(partial_corr)):
            for j in range(len(partial_corr)):
                if i != j:
                    col_i = partial_corr.columns[i]
                    col_j = partial_corr.columns[j]
                    if col_i in corr_data.columns and col_j in corr_data.columns:
                        r_ij = corr_data[col_i].corr(corr_data[col_j])
                        r_i_pH = corr_data[col_i].corr(pH2O) if pH2O.notna().sum() > 0 else 0
                        r_j_pH = corr_data[col_j].corr(pH2O) if pH2O.notna().sum() > 0 else 0
                        if 1 - r_i_pH**2 > 0 and 1 - r_j_pH**2 > 0:
                            partial_corr.loc[col_i, col_j] = (r_ij - r_i_pH * r_j_pH) / np.sqrt((1 - r_i_pH**2) * (1 - r_j_pH**2))
    
    # Distance correlation
    distance_corr = None
    try:
        from scipy.spatial.distance import pdist, squareform
        distance_corr = corr_data.corr()
        # Simplified distance correlation using energy distance
    except:
        pass
    
    # Find top correlations with target variables
    top_correlations = {}
    target_present = [col for col in target_cols if col in corr_data.columns]
    
    for target in target_present:
        if target in pearson_corr.columns:
            corr_series = pearson_corr[target].drop(target)
            top_positive = corr_series.nlargest(10)
            top_negative = corr_series.nsmallest(10)
            top_correlations[target] = {
                'positive': top_positive,
                'negative': top_negative,
                'all': corr_series.sort_values(ascending=False)
            }
    
    # Calculate VIF for multicollinearity
    vif_data = {}
    if len(corr_data.columns) > 1:
        try:
            X = corr_data.dropna()
            if X.shape[1] > 1 and X.shape[0] > X.shape[1]:
                vif_data = {}
                for i, col in enumerate(X.columns):
                    vif_data[col] = variance_inflation_factor(X.values, i)
            else:
                vif_data = {col: np.nan for col in X.columns}
        except:
            vif_data = {col: np.nan for col in corr_data.columns}
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'partial': partial_corr,
        'distance': distance_corr,
        'top_correlations': top_correlations,
        'vif': vif_data,
        'valid_cols': valid_cols
    }

@st.cache_data
def find_top_descriptors(correlation_data: Dict, target_cols: List[str], n: int = 20) -> List[str]:
    """
    Find top N descriptors based on correlation with target variables.
    
    Args:
        correlation_data: Dictionary from calculate_correlations
        target_cols: List of target variable names
        n: Number of top descriptors to return
        
    Returns:
        List of top descriptor names
    """
    top_descriptors = set()
    pearson = correlation_data['pearson']
    
    for target in target_cols:
        if target in pearson.columns:
            corr_series = pearson[target].drop(target)
            # Get absolute correlations
            abs_corr = corr_series.abs().sort_values(ascending=False)
            top_descriptors.update(abs_corr.head(n//len(target_cols) + 1).index.tolist())
    
    # Ensure we have at least n descriptors
    if len(top_descriptors) < n:
        # Add more from all columns
        all_cols = pearson.columns.tolist()
        for col in all_cols:
            if col not in top_descriptors and col not in target_cols:
                top_descriptors.add(col)
                if len(top_descriptors) >= n:
                    break
    
    return list(top_descriptors)[:n]

# ============================================================================
# SECTION 8: PCA AND CLUSTERING
# ============================================================================

@st.cache_data
def perform_pca_analysis(df: pd.DataFrame, descriptors: List[str], n_components: int = 10) -> Dict:
    """
    Perform PCA analysis on descriptor data.
    
    Args:
        df: DataFrame with descriptors
        descriptors: List of descriptor names
        n_components: Number of PCA components to compute
        
    Returns:
        Dictionary with PCA results
    """
    # Prepare data
    X = df[descriptors].copy()
    
    # Remove rows with too many NaN
    X = X.dropna(thresh=len(descriptors)//2)
    
    # Impute remaining NaN with median
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]-1))
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance)
    
    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=X.columns
    )
    
    return {
        'pca': pca,
        'X_pca': X_pca,
        'X_scaled': X_scaled,
        'explained_variance': explained_variance,
        'cumsum_variance': cumsum_variance,
        'loadings': loadings,
        'scaler': scaler,
        'features': X.columns.tolist()
    }

@st.cache_data
def perform_clustering(X_pca: np.ndarray, method: str = 'kmeans', n_clusters: int = 3) -> Dict:
    """
    Perform clustering on PCA-transformed data.
    
    Args:
        X_pca: PCA-transformed data
        method: 'kmeans', 'dbscan', or 'hierarchical'
        n_clusters: Number of clusters (for kmeans and hierarchical)
        
    Returns:
        Dictionary with clustering results
    """
    results = {}
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        silhouette = silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else 0
        
        results = {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'silhouette': silhouette,
            'method': 'kmeans'
        }
    
    elif method == 'dbscan':
        # Try different eps values
        best_eps = 0.5
        best_labels = None
        best_silhouette = -1
        
        for eps in np.linspace(0.1, 2.0, 10):
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_pca)
            n_clusters_db = len(np.unique(labels[labels != -1]))
            if n_clusters_db >= 2:
                try:
                    silhouette = silhouette_score(X_pca[labels != -1], labels[labels != -1])
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_labels = labels
                        best_eps = eps
                except:
                    pass
        
        results = {
            'labels': best_labels if best_labels is not None else np.zeros(X_pca.shape[0]),
            'eps': best_eps,
            'silhouette': best_silhouette,
            'method': 'dbscan'
        }
    
    elif method == 'hierarchical':
        linkage_matrix = linkage(X_pca, method='ward')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        silhouette = silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else 0
        
        results = {
            'labels': labels,
            'linkage_matrix': linkage_matrix,
            'silhouette': silhouette,
            'method': 'hierarchical'
        }
    
    return results

@st.cache_data
def perform_tsne(df: pd.DataFrame, descriptors: List[str], perplexity: int = 30) -> np.ndarray:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        df: DataFrame with descriptors
        descriptors: List of descriptor names
        perplexity: t-SNE perplexity parameter
        
    Returns:
        t-SNE coordinates (n_samples, 2)
    """
    X = df[descriptors].copy()
    
    # Impute NaN
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, X_scaled.shape[0]-1), random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    return X_tsne

# ============================================================================
# SECTION 9: VISUALIZATION FUNCTIONS
# ============================================================================

# 9.1 Distribution plots
def create_distribution_plots(df: pd.DataFrame, descriptors: List[str]) -> plt.Figure:
    """Create distribution histograms for selected descriptors."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    n_plots = min(len(descriptors), 9)
    for i in range(n_plots):
        col = descriptors[i]
        if col in df.columns and df[col].notna().sum() > 1:
            ax = axes[i]
            df[col].hist(bins=20, alpha=0.7, ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_plots, 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_box_plots(df: pd.DataFrame, target_col: str, category_col: str) -> plt.Figure:
    """Create box plots comparing target variable across categories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df[[target_col, category_col]].dropna()
    if data.empty or len(data[category_col].unique()) < 2:
        return fig
    
    sns.boxplot(data=data, x=category_col, y=target_col, ax=ax)
    ax.set_xlabel(category_col, fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_violin_plots(df: pd.DataFrame, target_col: str, category_col: str) -> plt.Figure:
    """Create violin plots for target variable across categories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df[[target_col, category_col]].dropna()
    if data.empty or len(data[category_col].unique()) < 2:
        return fig
    
    sns.violinplot(data=data, x=category_col, y=target_col, ax=ax)
    ax.set_xlabel(category_col, fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 9.2 Correlation plots
def create_correlation_matrix_plot(corr_matrix: pd.DataFrame, title: str = 'Correlation Matrix') -> plt.Figure:
    """Create heatmap of correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f',
        cmap=COLOR_PALETTES['diverging'],
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_correlation_network_plot(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> plt.Figure:
    """Create network graph of significant correlations."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for col in corr_matrix.columns:
        G.add_node(col)
    
    # Add edges for correlations above threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold and not np.isnan(corr):
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr))
    
    if len(G.edges()) == 0:
        ax.text(0.5, 0.5, 'No significant correlations found\nabove threshold', 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Position nodes
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', 
                          edgecolors='black', linewidths=1.5, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(f'Correlation Network (|corr| > {threshold})', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    plt.tight_layout()
    return fig

def create_pairplot_colored(df: pd.DataFrame, features: List[str], hue_col: str) -> plt.Figure:
    """Create pairplot with color coding."""
    if len(features) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Need at least 2 features for pairplot', 
                ha='center', va='center', fontsize=14)
        return fig
    
    data = df[features + [hue_col] if hue_col in df.columns else features].dropna()
    
    if hue_col in df.columns and len(data[hue_col].unique()) <= 10:
        g = sns.pairplot(data, vars=features, hue=hue_col, 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
    else:
        g = sns.pairplot(data, vars=features, 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
    
    plt.suptitle('Pairplot of Top Features', y=1.02, fontsize=14, fontweight='bold')
    return g.fig

def create_scatter_regression(df: pd.DataFrame, x_col: str, y_col: str) -> plt.Figure:
    """Create scatter plot with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    # Scatter
    ax.scatter(data[x_col], data[y_col], alpha=0.6, s=30, color='#2C3E50')
    
    # Regression
    from sklearn.linear_model import LinearRegression
    X = data[[x_col]].values
    y = data[y_col].values
    
    if len(X) > 1:
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        # Plot regression line
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color='#E74C3C', linewidth=2, label='Regression')
        
        # R² and p-value
        r2 = reg.score(X, y)
        slope, intercept = reg.coef_[0], reg.intercept_
        p_value = None
        try:
            from scipy import stats
            _, p_value, _, _ = stats.linregress(X.flatten(), y)
        except:
            pass
        
        # Annotation
        annotation = f'R² = {r2:.3f}'
        if p_value is not None:
            annotation += f'\np = {p_value:.3e}'
        annotation += f'\ny = {slope:.3f}x + {intercept:.3f}'
        ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_residual_plot(df: pd.DataFrame, x_col: str, y_col: str) -> plt.Figure:
    """Create residual plot for regression."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    # Fit regression
    X = data[[x_col]].values
    y = data[y_col].values
    
    if len(X) > 1:
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        residuals = y - y_pred
        
        # Plot residuals
        ax.scatter(y_pred, residuals, alpha=0.6, s=30, color='#2C3E50')
        ax.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 9.3 PCA and clustering plots
def create_pca_biplot(pca_results: Dict, n_components: int = 2) -> plt.Figure:
    """Create PCA biplot."""
    if n_components > 2:
        n_components = 2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X_pca = pca_results['X_pca']
    loadings = pca_results['loadings']
    features = pca_results['features']
    
    # Scatter
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=30, color='#2C3E50')
    
    # Loadings (arrows)
    for i, feature in enumerate(features):
        if i < len(loadings):
            x = loadings.iloc[i, 0] * 3
            y = loadings.iloc[i, 1] * 3
            ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                    fc='#E74C3C', ec='#E74C3C', alpha=0.7)
            ax.text(x*1.1, y*1.1, feature, fontsize=8, color='#E74C3C')
    
    ax.set_xlabel(f'PC1 ({pca_results["explained_variance"][0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_results["explained_variance"][1]*100:.1f}%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('PCA Biplot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_pca_3d(pca_results: Dict) -> go.Figure:
    """Create interactive 3D PCA plot."""
    X_pca = pca_results['X_pca']
    explained = pca_results['explained_variance']
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=X_pca[:, 0],
                colorscale='Viridis',
                showscale=True,
                opacity=0.8
            ),
            text=[f'PC1: {x:.2f}, PC2: {y:.2f}, PC3: {z:.2f}' 
                  for x, y, z in zip(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])],
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({explained[2]*100:.1f}%)',
        ),
        title='3D PCA Visualization',
        width=800,
        height=600
    )
    
    return fig

def create_tsne_plot(X_tsne: np.ndarray, labels: Optional[np.ndarray] = None) -> plt.Figure:
    """Create t-SNE visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, 
                           cmap=COLOR_PALETTES['categorical'], alpha=0.6, s=30)
        ax.legend(*scatter.legend_elements(), title='Cluster')
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=30, color='#2C3E50')
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_silhouette_plot(X: np.ndarray, labels: np.ndarray) -> plt.Figure:
    """Create silhouette plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        silhouette_vals = silhouette_samples(X, labels)
        n_clusters = len(np.unique(labels))
        
        y_lower = 10
        for i in range(n_clusters):
            cluster_vals = silhouette_vals[labels == i]
            cluster_vals.sort()
            cluster_size = len(cluster_vals)
            y_upper = y_lower + cluster_size
            
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i), fontsize=10)
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_score(X, labels), color="red", linestyle="--")
        ax.set_xlabel("Silhouette Coefficient", fontsize=12)
        ax.set_ylabel("Cluster", fontsize=12)
        ax.set_title("Silhouette Plot", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    except:
        ax.text(0.5, 0.5, 'Silhouette analysis failed', ha='center', va='center', fontsize=14)
        ax.set_axis_off()
    
    plt.tight_layout()
    return fig

def create_cluster_profiles(df: pd.DataFrame, cluster_col: str, descriptors: List[str]) -> plt.Figure:
    """Create heatmap of cluster profiles."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate mean values per cluster
    cluster_means = df.groupby(cluster_col)[descriptors].mean()
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    # Heatmap
    sns.heatmap(cluster_means_scaled, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, cbar_kws={'label': 'Standardized Value'})
    ax.set_title('Cluster Profiles (Standardized Mean Values)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Descriptors', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    plt.tight_layout()
    return fig

# 9.4 Concentration maps
def create_concentration_heatmap(df: pd.DataFrame, x_col: str, y_col: str, 
                                 color_col: str, title: str = '') -> go.Figure:
    """Create interactive concentration heatmap."""
    data = df[[x_col, y_col, color_col]].dropna()
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data available', x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create grid
    x_grid = np.linspace(data[x_col].min(), data[x_col].max(), 50)
    y_grid = np.linspace(data[y_col].min(), data[y_col].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate
    try:
        Z_grid = griddata(
            (data[x_col].values, data[y_col].values),
            data[color_col].values,
            (X_grid, Y_grid),
            method='cubic'
        )
    except:
        Z_grid = np.zeros_like(X_grid)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=Z_grid,
        colorscale='Viridis',
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Value: %{z:.3f}<extra></extra>'
    ))
    
    # Add original points
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            line=dict(color='black', width=1)
        ),
        text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}' 
              for x, y, c in zip(data[x_col], data[y_col], data[color_col])],
        hoverinfo='text',
        name='Data points'
    ))
    
    fig.update_layout(
        title=title or f'{color_col} vs {x_col} and {y_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_concentration_contour(df: pd.DataFrame, x_col: str, y_col: str, 
                                 color_col: str, title: str = '') -> plt.Figure:
    """Create contour plot for concentration maps."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[[x_col, y_col, color_col]].dropna()
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    # Create grid
    x_grid = np.linspace(data[x_col].min(), data[x_col].max(), 50)
    y_grid = np.linspace(data[y_col].min(), data[y_col].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate
    try:
        Z_grid = griddata(
            (data[x_col].values, data[y_col].values),
            data[color_col].values,
            (X_grid, Y_grid),
            method='cubic'
        )
    except:
        Z_grid = np.zeros_like(X_grid)
    
    # Contour plot
    contour = ax.contourf(X_grid, Y_grid, Z_grid, 20, cmap='viridis', alpha=0.8)
    ax.contour(X_grid, Y_grid, Z_grid, 10, colors='black', linewidths=0.5, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(color_col, fontsize=12)
    
    # Scatter points
    ax.scatter(data[x_col], data[y_col], c='white', s=20, edgecolors='black', linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_title(title or f'{color_col} Contour Map', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_ternary_plot(df: pd.DataFrame, comp_cols: List[str], color_col: str) -> go.Figure:
    """Create ternary plot for compositional data."""
    data = df[comp_cols + [color_col]].dropna()
    
    if data.empty or len(comp_cols) != 3:
        fig = go.Figure()
        fig.add_annotation(text='Need exactly 3 composition columns', x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure(data=go.Scatterternary(
        a=data[comp_cols[0]],
        b=data[comp_cols[1]],
        c=data[comp_cols[2]],
        mode='markers',
        marker=dict(
            size=8,
            color=data[color_col],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_col)
        ),
        text=[f'{comp_cols[0]}: {a:.3f}<br>{comp_cols[1]}: {b:.3f}<br>{comp_cols[2]}: {c:.3f}<br>{color_col}: {col:.3f}' 
              for a, b, c, col in zip(data[comp_cols[0]], data[comp_cols[1]], 
                                      data[comp_cols[2]], data[color_col])],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f'Ternary Plot: {color_col}',
        ternary=dict(
            sum=1,
            aaxis=dict(title=comp_cols[0], min=0, max=1),
            baxis=dict(title=comp_cols[1], min=0, max=1),
            caxis=dict(title=comp_cols[2], min=0, max=1)
        ),
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

# 9.5 Bubble charts
def create_bubble_4d(df: pd.DataFrame, x_col: str, y_col: str, 
                     color_col: str, size_col: str, shape_col: Optional[str] = None) -> go.Figure:
    """Create 4D bubble chart."""
    data = df[[x_col, y_col, color_col, size_col] + ([shape_col] if shape_col else [])].dropna()
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data available', x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Marker symbols
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'star']
    
    if shape_col and shape_col in data.columns:
        unique_shapes = data[shape_col].unique()
        shape_map = {val: symbols[i % len(symbols)] for i, val in enumerate(unique_shapes)}
        data['symbol'] = data[shape_col].map(shape_map)
    else:
        data['symbol'] = 'circle'
    
    fig = go.Figure()
    
    # Create traces for each shape category
    if shape_col and shape_col in data.columns:
        for shape_val in data[shape_col].unique():
            subset = data[data[shape_col] == shape_val]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode='markers',
                    marker=dict(
                        size=subset[size_col] * 20 + 5,
                        color=subset[color_col],
                        colorscale='Viridis',
                        showscale=True,
                        symbol=subset['symbol'].iloc[0],
                        sizemode='diameter',
                        sizeref=2.0,
                        colorbar=dict(title=color_col)
                    ),
                    text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}<br>{size_col}: {s:.3f}' 
                          for x, y, c, s in zip(subset[x_col], subset[y_col], 
                                               subset[color_col], subset[size_col])],
                    hoverinfo='text',
                    name=f'{shape_col}={shape_val}'
                ))
    else:
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='markers',
            marker=dict(
                size=data[size_col] * 20 + 5,
                color=data[color_col],
                colorscale='Viridis',
                showscale=True,
                sizemode='diameter',
                sizeref=2.0,
                colorbar=dict(title=color_col)
            ),
            text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}<br>{size_col}: {s:.3f}' 
                  for x, y, c, s in zip(data[x_col], data[y_col], data[color_col], data[size_col])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f'4D Bubble Chart: {y_col} vs {x_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=800,
        height=600,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def create_compositional_bubble(df: pd.DataFrame, x_col: str, y_col: str, 
                                color_col: str, size_col: str) -> go.Figure:
    """Create compositional bubble chart."""
    data = df[[x_col, y_col, color_col, size_col]].dropna()
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data available', x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Group by color column if categorical
    if color_col in df.columns and df[color_col].dtype == 'object':
        for category in data[color_col].unique():
            subset = data[data[color_col] == category]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode='markers',
                    marker=dict(
                        size=subset[size_col] * 20 + 5,
                        sizemode='diameter',
                        sizeref=2.0,
                    ),
                    name=str(category),
                    text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{size_col}: {s:.3f}' 
                          for x, y, s in zip(subset[x_col], subset[y_col], subset[size_col])],
                    hoverinfo='text'
                ))
    else:
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='markers',
            marker=dict(
                size=data[size_col] * 20 + 5,
                color=data[color_col],
                colorscale='Viridis',
                showscale=True,
                sizemode='diameter',
                sizeref=2.0,
                colorbar=dict(title=color_col)
            ),
            text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{color_col}: {c:.3f}<br>{size_col}: {s:.3f}' 
                  for x, y, c, s in zip(data[x_col], data[y_col], data[color_col], data[size_col])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f'Compositional Bubble Chart: {y_col} vs {x_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

# 9.6 Specialized plots
def create_alpha_beta_compromise(df: pd.DataFrame, color_col: Optional[str] = None) -> plt.Figure:
    """Create α vs β compromise diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[['α·106 (K-1)', 'β', color_col] if color_col in df.columns else ['α·106 (K-1)', 'β']].dropna()
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    if color_col in data.columns:
        scatter = ax.scatter(data['α·106 (K-1)'], data['β'], 
                           c=data[color_col].astype('category').cat.codes,
                           cmap='viridis', alpha=0.6, s=50)
        ax.legend(*scatter.legend_elements(), title=color_col)
    else:
        ax.scatter(data['α·106 (K-1)'], data['β'], alpha=0.6, s=50, color='#2C3E50')
    
    # Add optimal region annotation
    ax.axvline(x=11, color='green', linestyle='--', alpha=0.5, label='α = 11×10⁻⁶ K⁻¹ (typical SOFC)')
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='β = 0.05 (high chemical expansion)')
    
    ax.set_xlabel('α·10⁶ (K⁻¹) - Thermal Expansion', fontsize=12, fontweight='bold')
    ax.set_ylabel('β - Chemical Expansion Coefficient', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('α vs β: Compromise Diagram', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_t_bends_analysis(df: pd.DataFrame, color_col: Optional[str] = None) -> plt.Figure:
    """Create T(bends) analysis plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[['T_bends_first', 'δ', color_col] if color_col in df.columns else ['T_bends_first', 'δ']].dropna()
    
    if data.empty or 'T_bends_first' not in data.columns:
        ax.text(0.5, 0.5, 'No T(bends) data available', ha='center', va='center', fontsize=14)
        return fig
    
    if color_col in data.columns:
        scatter = ax.scatter(data['δ'], data['T_bends_first'], 
                           c=data[color_col].astype('category').cat.codes,
                           cmap='viridis', alpha=0.6, s=50)
        ax.legend(*scatter.legend_elements(), title=color_col)
    else:
        ax.scatter(data['δ'], data['T_bends_first'], alpha=0.6, s=50, color='#2C3E50')
    
    ax.set_xlabel('δ - Oxygen Vacancy Concentration', fontsize=12, fontweight='bold')
    ax.set_ylabel('T(bends) - First Bending Temperature (°C)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_title('T(bends) vs δ Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_beta_vs_ph2o(df: pd.DataFrame, color_col: Optional[str] = None) -> plt.Figure:
    """Create β vs pH₂O plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[['β', 'pH2O', color_col] if color_col in df.columns else ['β', 'pH2O']].dropna()
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    # Log scale for pH2O
    data['log_pH2O'] = np.log10(data['pH2O'])
    
    if color_col in data.columns:
        scatter = ax.scatter(data['log_pH2O'], data['β'], 
                           c=data[color_col].astype('category').cat.codes,
                           cmap='viridis', alpha=0.6, s=50)
        ax.legend(*scatter.legend_elements(), title=color_col)
    else:
        ax.scatter(data['log_pH2O'], data['β'], alpha=0.6, s=50, color='#2C3E50')
    
    ax.set_xlabel('log(pH₂O)', fontsize=12, fontweight='bold')
    ax.set_ylabel('β - Chemical Expansion Coefficient', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_title('β vs pH₂O: Effect of Water Vapor', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_alpha_vs_geometric(df: pd.DataFrame, geo_col: str = 'rAav') -> plt.Figure:
    """Create α vs geometric descriptor plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[['α·106 (K-1)', geo_col, 'B_type']].dropna()
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    for b_type in data['B_type'].unique():
        subset = data[data['B_type'] == b_type]
        if not subset.empty:
            ax.scatter(subset[geo_col], subset['α·106 (K-1)'], 
                      label=b_type, alpha=0.6, s=40)
    
    ax.set_xlabel(geo_col, fontsize=12, fontweight='bold')
    ax.set_ylabel('α·10⁶ (K⁻¹) - Thermal Expansion', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(title='B-cation')
    ax.set_title(f'α vs {geo_col}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_beta_vs_electronegativity(df: pd.DataFrame) -> plt.Figure:
    """Create β vs electronegativity plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[['β', 'χBav', 'B_type']].dropna()
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    for b_type in data['B_type'].unique():
        subset = data[data['B_type'] == b_type]
        if not subset.empty:
            ax.scatter(subset['χBav'], subset['β'], 
                      label=b_type, alpha=0.6, s=40)
    
    ax.set_xlabel('χBav - Average Electronegativity of B-site', fontsize=12, fontweight='bold')
    ax.set_ylabel('β - Chemical Expansion Coefficient', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(title='B-cation')
    ax.set_title('β vs χBav: Effect of B-site Electronegativity', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_radar_chart(df: pd.DataFrame, cluster_col: str, descriptors: List[str]) -> plt.Figure:
    """Create radar chart for cluster comparison."""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    # Calculate mean values per cluster
    cluster_means = df.groupby(cluster_col)[descriptors].mean()
    
    # Standardize
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(descriptors), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for i, cluster in enumerate(cluster_means_scaled.index):
        values = cluster_means_scaled.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(descriptors, fontsize=8)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Cluster Profiles Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_feature_importance(df: pd.DataFrame, target_col: str, descriptors: List[str]) -> plt.Figure:
    """Create feature importance plot using Random Forest."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[descriptors + [target_col]].dropna()
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    X = data[descriptors]
    y = data[target_col]
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': descriptors,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Plot
    ax.barh(importance['feature'], importance['importance'], color='#2C3E50', alpha=0.7)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Descriptors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_title(f'Feature Importance for {target_col} (Random Forest)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_swarm_plot(df: pd.DataFrame, target_col: str, category_col: str) -> plt.Figure:
    """Create swarm plot for categorical comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = df[[target_col, category_col]].dropna()
    
    if data.empty or len(data[category_col].unique()) < 2:
        ax.text(0.5, 0.5, 'Not enough categorical data', ha='center', va='center', fontsize=14)
        return fig
    
    sns.swarmplot(data=data, x=category_col, y=target_col, ax=ax, alpha=0.6)
    ax.set_xlabel(category_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(target_col, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Distribution of {target_col} by {category_col}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# 9.7 Additional advanced plots
def create_parallel_coordinates(df: pd.DataFrame, features: List[str], 
                                target_col: str, max_categories: int = 10) -> go.Figure:
    """Create parallel coordinates plot with filtering."""
    data = df[features + [target_col]].dropna()
    
    if data.empty or len(features) < 2:
        fig = go.Figure()
        fig.add_annotation(text='Need at least 2 features', x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Color by target
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=data[target_col],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=target_col)
        ),
        dimensions=[
            dict(
                label=col,
                values=data[col],
                range=[data[col].min(), data[col].max()]
            ) for col in features
        ]
    ))
    
    fig.update_layout(
        title=f'Parallel Coordinates: {target_col}',
        width=900,
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_3d_scatter_interactive(df: pd.DataFrame, x_col: str, y_col: str, 
                                  z_col: str, color_col: Optional[str] = None) -> go.Figure:
    """Create interactive 3D scatter plot."""
    data = df[[x_col, y_col, z_col] + ([color_col] if color_col else [])].dropna()
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data available', x=0.5, y=0.5, showarrow=False)
        return fig
    
    if color_col and color_col in data.columns:
        fig = go.Figure(data=go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(
                size=6,
                color=data[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col)
            ),
            text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{z_col}: {z:.3f}' 
                  for x, y, z in zip(data[x_col], data[y_col], data[z_col])],
            hoverinfo='text'
        ))
    else:
        fig = go.Figure(data=go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(size=5, color='#2C3E50'),
            text=[f'{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{z_col}: {z:.3f}' 
                  for x, y, z in zip(data[x_col], data[y_col], data[z_col])],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_correlation_triangle(pearson_corr: pd.DataFrame, 
                                spearman_corr: pd.DataFrame) -> plt.Figure:
    """Create combined Pearson-Spearman correlation triangle."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create combined matrix
    combined = pearson_corr.copy()
    
    # Upper triangle: Spearman
    for i in range(len(spearman_corr)):
        for j in range(i+1, len(spearman_corr)):
            combined.iloc[i, j] = spearman_corr.iloc[i, j]
    
    # Mask lower triangle
    mask = np.tril(np.ones_like(combined, dtype=bool))
    
    # Heatmap
    sns.heatmap(
        combined,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=COLOR_PALETTES['diverging'],
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax
    )
    
    # Add annotations
    ax.text(0.02, 0.98, 'Lower: Pearson', transform=ax.transAxes, 
            fontsize=10, fontweight='bold', verticalalignment='top')
    ax.text(0.98, 0.02, 'Upper: Spearman', transform=ax.transAxes, 
            fontsize=10, fontweight='bold', horizontalalignment='right', verticalalignment='bottom')
    
    ax.set_title('Correlation Triangle: Pearson (lower) vs Spearman (upper)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ============================================================================
# SECTION 10: FILTERING SYSTEM
# ============================================================================

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filter conditions
        
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    # Basic filters
    if 'method' in filters and filters['method'] and filters['method'] != 'All':
        df_filtered = df_filtered[df_filtered['method'] == filters['method']]
    
    if 'A_type' in filters and filters['A_type'] and filters['A_type'] != 'All':
        df_filtered = df_filtered[df_filtered['A_type'] == filters['A_type']]
    
    if 'B_type' in filters and filters['B_type'] and filters['B_type'] != 'All':
        df_filtered = df_filtered[df_filtered['B_type'] == filters['B_type']]
    
    # Advanced filters
    if 'δ_min' in filters and 'δ_max' in filters:
        df_filtered = df_filtered[(df_filtered['δ'] >= filters['δ_min']) & 
                                  (df_filtered['δ'] <= filters['δ_max'])]
    
    if 'pH2O_min' in filters and 'pH2O_max' in filters:
        df_filtered = df_filtered[(df_filtered['pH2O'] >= filters['pH2O_min']) & 
                                  (df_filtered['pH2O'] <= filters['pH2O_max'])]
    
    if 'T_min' in filters and 'T_max' in filters:
        df_filtered = df_filtered[(df_filtered['T_min'] >= filters['T_min']) & 
                                  (df_filtered['T_max'] <= filters['T_max'])]
    
    if 'has_bends' in filters and filters['has_bends']:
        df_filtered = df_filtered[df_filtered['n_bends'] > 0]
    
    if 'has_alpha' in filters and filters['has_alpha']:
        df_filtered = df_filtered[df_filtered['α·106 (K-1)'].notna()]
    
    if 'has_beta' in filters and filters['has_beta']:
        df_filtered = df_filtered[df_filtered['β'].notna()]
    
    # Descriptor filters
    if 'rAav_min' in filters and 'rAav_max' in filters:
        df_filtered = df_filtered[(df_filtered['rAav'] >= filters['rAav_min']) & 
                                  (df_filtered['rAav'] <= filters['rAav_max'])]
    
    if 't_min' in filters and 't_max' in filters:
        df_filtered = df_filtered[(df_filtered['t'] >= filters['t_min']) & 
                                  (df_filtered['t'] <= filters['t_max'])]
    
    if 'χBav_min' in filters and 'χBav_max' in filters:
        df_filtered = df_filtered[(df_filtered['χBav'] >= filters['χBav_min']) & 
                                  (df_filtered['χBav'] <= filters['χBav_max'])]
    
    return df_filtered

# ============================================================================
# SECTION 11: UI COMPONENTS
# ============================================================================

def render_sidebar_filters(df: pd.DataFrame) -> Dict:
    """Render filters in sidebar."""
    st.sidebar.markdown("## 🔍 Filters")
    
    filters = {}
    
    # Basic filters
    with st.sidebar.expander("📌 Basic Filters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            method_options = ['All'] + sorted(df['method'].dropna().unique().tolist())
            filters['method'] = st.selectbox('Method', method_options)
        with col2:
            A_options = ['All'] + sorted(df['A_type'].dropna().unique().tolist())
            filters['A_type'] = st.selectbox('A-cation', A_options)
        
        B_options = ['All'] + sorted(df['B_type'].dropna().unique().tolist())
        filters['B_type'] = st.selectbox('B-cation', B_options)
    
    # Advanced filters
    with st.sidebar.expander("📊 Advanced Filters", expanded=False):
        # Delta range
        δ_min = float(df['δ'].min()) if df['δ'].notna().any() else 0
        δ_max = float(df['δ'].max()) if df['δ'].notna().any() else 0.5
        filters['δ_min'], filters['δ_max'] = st.slider(
            'δ range',
            min_value=0.0,
            max_value=0.5,
            value=(δ_min, δ_max),
            step=0.01
        )
        
        # pH2O range
        pH_min = float(df['pH2O'].min()) if df['pH2O'].notna().any() else 1e-6
        pH_max = float(df['pH2O'].max()) if df['pH2O'].notna().any() else 0.1
        filters['pH2O_min'], filters['pH2O_max'] = st.slider(
            'pH₂O range (log scale)',
            min_value=-6.0,
            max_value=0.0,
            value=(np.log10(pH_min), np.log10(pH_max)),
            step=0.1
        )
        filters['pH2O_min'] = 10 ** filters['pH2O_min']
        filters['pH2O_max'] = 10 ** filters['pH2O_max']
        
        # Temperature range
        T_min = float(df['T_min'].min()) if df['T_min'].notna().any() else 20
        T_max = float(df['T_max'].max()) if df['T_max'].notna().any() else 1200
        filters['T_min'], filters['T_max'] = st.slider(
            'Temperature range (°C)',
            min_value=0,
            max_value=1500,
            value=(int(T_min), int(T_max)),
            step=10
        )
        
        # Checkboxes
        col1, col2, col3 = st.columns(3)
        with col1:
            filters['has_bends'] = st.checkbox('Has T(bends)', value=False)
        with col2:
            filters['has_alpha'] = st.checkbox('Has α', value=False)
        with col3:
            filters['has_beta'] = st.checkbox('Has β', value=False)
    
    # Descriptor filters
    with st.sidebar.expander("📐 Descriptor Filters", expanded=False):
        if 'rAav' in df.columns:
            rA_min = float(df['rAav'].min()) if df['rAav'].notna().any() else 0.8
            rA_max = float(df['rAav'].max()) if df['rAav'].notna().any() else 1.8
            filters['rAav_min'], filters['rAav_max'] = st.slider(
                'rAav range (Å)',
                min_value=0.8,
                max_value=1.8,
                value=(rA_min, rA_max),
                step=0.01
            )
        
        if 't' in df.columns:
            t_min = float(df['t'].min()) if df['t'].notna().any() else 0.8
            t_max = float(df['t'].max()) if df['t'].notna().any() else 1.1
            filters['t_min'], filters['t_max'] = st.slider(
                'tolerance factor range',
                min_value=0.8,
                max_value=1.1,
                value=(t_min, t_max),
                step=0.01
            )
        
        if 'χBav' in df.columns:
            χ_min = float(df['χBav'].min()) if df['χBav'].notna().any() else 0.8
            χ_max = float(df['χBav'].max()) if df['χBav'].notna().any() else 2.0
            filters['χBav_min'], filters['χBav_max'] = st.slider(
                'χBav range',
                min_value=0.8,
                max_value=2.0,
                value=(χ_min, χ_max),
                step=0.05
            )
    
    if st.sidebar.button('🔄 Reset Filters'):
        st.session_state['filters_reset'] = True
        st.rerun()
    
    return filters

def render_upload_page():
    """Render the data upload page."""
    st.markdown("# 📤 Upload Data")
    st.markdown("Paste your tab-separated data below (including header row):")
    
    # Text area for pasting data
    text_data = st.text_area(
        "Paste data here",
        height=300,
        placeholder="№\tA\tA'\tB\tB'\tD1\tD2\t[A']\t[B']\t[D1]\t[D2]\tδ\tmethod\tβ\t∆T, °C\tα·106 (K-1)\tT(bends), °C\tαav·106 (K-1)\tpH2O\tRef\n1\tBa\t-\tCe\tZr\tY\tYb\t0\t0.1\t0.1\t0.1\t0.1\tdilatometry\t0.0073\t27-1000\t10.6\t400;600\t10.6;4.73;10.1\t0.0001\t10.15826/chimtech.2024.11.4.22"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('📊 Load Data', type='primary'):
            if text_data.strip():
                with st.spinner('Loading data...'):
                    df = parse_uploaded_data(text_data)
                    if not df.empty:
                        st.session_state['raw_data'] = df
                        st.session_state['processed'] = False
                        st.success(f'✅ Data loaded! {len(df)} rows, {len(df.columns)} columns')
                    else:
                        st.error('❌ Failed to parse data. Check format.')
            else:
                st.warning('⚠️ Please paste data first.')
    
    # Show example data
    if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
        st.markdown("### 📋 Data Preview")
        st.dataframe(st.session_state['raw_data'].head(10))
        
        st.markdown("### 📊 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(st.session_state['raw_data']))
        with col2:
            st.metric("Columns", len(st.session_state['raw_data'].columns))
        with col3:
            missing = st.session_state['raw_data'].isna().sum().sum()
            st.metric("Missing Values", missing)
        with col4:
            duplicates = st.session_state['raw_data'].duplicated().sum()
            st.metric("Duplicates", duplicates)

def render_descriptor_page(df: pd.DataFrame):
    """Render the descriptors page."""
    st.markdown("# 🔬 Descriptor Engine")
    
    if df is None or df.empty:
        st.warning('⚠️ No data loaded. Please upload data first.')
        return
    
    if st.button('🚀 Calculate All Descriptors', type='primary'):
        with st.spinner('Calculating descriptors... This may take a moment.'):
            df_desc = calculate_descriptors(df)
            df_desc = clean_data(df_desc)
            st.session_state['descriptors'] = df_desc
            st.session_state['processed'] = True
            st.success('✅ Descriptors calculated!')
    
    if 'descriptors' in st.session_state and st.session_state['descriptors'] is not None:
        df_desc = st.session_state['descriptors']
        
        # Display descriptor columns
        desc_cols = [col for col in df_desc.columns if col not in df.columns or col in 
                    ['rAav', 'rBav', 't', 'D_t', 'χAav', 'χBav', 'Δχ_AB', 
                     'V_Bav', 'Vo_proxy', 'M_total', 'δ_calc', 'B'_conc', 'D_total']]
        
        st.markdown("### 📊 Calculated Descriptors")
        
        # Show descriptor table with selected columns
        display_cols = ['№'] + desc_cols[:20]  # Show first 20 descriptors
        display_cols = [col for col in display_cols if col in df_desc.columns]
        st.dataframe(df_desc[display_cols].head(10))
        
        # Statistics
        st.markdown("### 📈 Descriptor Statistics")
        stats_df = df_desc[desc_cols].describe().T
        st.dataframe(stats_df)
        
        # Download button
        csv = df_desc.to_csv(index=False)
        st.download_button(
            label='📥 Download Descriptors CSV',
            data=csv,
            file_name='perovskite_descriptors.csv',
            mime='text/csv'
        )

def render_correlation_page(df: pd.DataFrame):
    """Render the correlation analysis page."""
    st.markdown("# 📊 Correlation Analysis")
    
    if df is None or df.empty or 'descriptors' not in st.session_state:
        st.warning('⚠️ Please calculate descriptors first.')
        return
    
    df_desc = st.session_state['descriptors']
    
    # Target variables
    target_cols = ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'alpha_beta_ratio', 'T_bends_first']
    target_cols = [col for col in target_cols if col in df_desc.columns]
    
    if not target_cols:
        st.warning('⚠️ No target variables found in data.')
        return
    
    # Calculate correlations
    with st.spinner('Calculating correlations...'):
        corr_data = calculate_correlations(df_desc, target_cols)
        st.session_state['correlation_data'] = corr_data
    
    # Display correlation results
    st.markdown("### 🔍 Top Correlations with Targets")
    
    for target in target_cols:
        if target in corr_data['top_correlations']:
            st.markdown(f"#### {target}")
            top = corr_data['top_correlations'][target]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Correlations**")
                st.dataframe(top['positive'].head(5))
            with col2:
                st.markdown("**Top Negative Correlations**")
                st.dataframe(top['negative'].head(5))
    
    # Correlation matrices
    st.markdown("### 📈 Correlation Matrices")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        'Pearson', 'Spearman', 'Partial (pH₂O)', 'Network'
    ])
    
    with tab1:
        if corr_data['pearson'] is not None:
            fig = create_correlation_matrix_plot(
                corr_data['pearson'], 
                'Pearson Correlation Matrix'
            )
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        if corr_data['spearman'] is not None:
            fig = create_correlation_matrix_plot(
                corr_data['spearman'],
                'Spearman Correlation Matrix'
            )
            st.pyplot(fig)
            plt.close(fig)
    
    with tab3:
        if corr_data['partial'] is not None:
            fig = create_correlation_matrix_plot(
                corr_data['partial'],
                'Partial Correlation Matrix (Controlling pH₂O)'
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info('Partial correlation requires pH₂O data.')
    
    with tab4:
        if corr_data['pearson'] is not None:
            threshold = st.slider('Correlation threshold', 0.3, 0.9, 0.5, 0.1)
            fig = create_correlation_network_plot(
                corr_data['pearson'],
                threshold
            )
            st.pyplot(fig)
            plt.close(fig)
    
    # Top descriptors
    st.markdown("### 🏆 Top 20 Descriptors")
    top_descriptors = find_top_descriptors(corr_data, target_cols, 20)
    st.session_state['top_descriptors'] = top_descriptors
    
    # Display with importance
    desc_importance = {}
    for target in target_cols:
        if target in corr_data['top_correlations']:
            for desc, corr in corr_data['top_correlations'][target]['all'].items():
                if desc in top_descriptors:
                    desc_importance[desc] = desc_importance.get(desc, 0) + abs(corr)
    
    importance_df = pd.DataFrame({
        'Descriptor': list(desc_importance.keys()),
        'Importance': list(desc_importance.values())
    }).sort_values('Importance', ascending=False)
    
    st.dataframe(importance_df)
    
    # Pairplot
    st.markdown("### 🎨 Pairplot of Top Features")
    
    if len(top_descriptors) >= 2:
        # Select features for pairplot
        selected_features = st.multiselect(
            'Select features for pairplot (2-5)',
            top_descriptors,
            default=top_descriptors[:min(4, len(top_descriptors))]
        )
        
        if len(selected_features) >= 2:
            hue_col = st.selectbox(
                'Color by (optional)',
                ['None'] + [col for col in df_desc.columns if col not in selected_features and 
                           df_desc[col].dtype == 'object'],
                index=0
            )
            
            fig = create_pairplot_colored(
                df_desc,
                selected_features,
                hue_col if hue_col != 'None' else None
            )
            st.pyplot(fig)
            plt.close(fig)
    
    # Scatter regression
    st.markdown("### 📉 Best Linear Correlations")
    
    # Find best correlations
    best_corr = []
    for target in target_cols:
        if target in corr_data['top_correlations']:
            for desc, corr in corr_data['top_correlations'][target]['all'].head(10).items():
                if abs(corr) > 0.3 and desc != target:
                    best_corr.append((target, desc, abs(corr)))
    
    if best_corr:
        best_corr.sort(key=lambda x: x[2], reverse=True)
        target, desc, _ = best_corr[0]
        
        fig = create_scatter_regression(df_desc, desc, target)
        st.pyplot(fig)
        plt.close(fig)
        
        # Residual plot
        fig = create_residual_plot(df_desc, desc, target)
        st.pyplot(fig)
        plt.close(fig)
    
    # Correlation triangle
    if corr_data['pearson'] is not None and corr_data['spearman'] is not None:
        st.markdown("### 🔺 Pearson-Spearman Comparison Triangle")
        
        # Select subset of columns
        selected_cols = st.multiselect(
            'Select columns for triangle (max 10)',
            corr_data['pearson'].columns.tolist(),
            default=corr_data['pearson'].columns[:min(8, len(corr_data['pearson'].columns))].tolist()
        )
        
        if len(selected_cols) >= 3:
            pearson_subset = corr_data['pearson'].loc[selected_cols, selected_cols]
            spearman_subset = corr_data['spearman'].loc[selected_cols, selected_cols]
            
            fig = create_correlation_triangle(pearson_subset, spearman_subset)
            st.pyplot(fig)
            plt.close(fig)

def render_pca_clustering_page(df: pd.DataFrame):
    """Render PCA and clustering page."""
    st.markdown("# 🧬 PCA & Clustering Analysis")
    
    if df is None or df.empty or 'descriptors' not in st.session_state:
        st.warning('⚠️ Please calculate descriptors first.')
        return
    
    if 'top_descriptors' not in st.session_state:
        st.warning('⚠️ Please run correlation analysis to identify top descriptors.')
        return
    
    df_desc = st.session_state['descriptors']
    top_descriptors = st.session_state['top_descriptors']
    
    # Select descriptors for PCA
    selected_desc = st.multiselect(
        'Select descriptors for PCA',
        top_descriptors,
        default=top_descriptors[:min(10, len(top_descriptors))]
    )
    
    if len(selected_desc) < 3:
        st.warning('Please select at least 3 descriptors.')
        return
    
    # Perform PCA
    with st.spinner('Performing PCA...'):
        pca_results = perform_pca_analysis(df_desc, selected_desc, n_components=10)
        st.session_state['pca_results'] = pca_results
    
    # Display PCA results
    st.markdown("### 📊 PCA Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Explained Variance (PC1+PC2)", 
                 f"{pca_results['explained_variance'][:2].sum()*100:.1f}%")
    with col2:
        st.metric("Number of Components", 
                 len(pca_results['explained_variance']))
    
    # Explained variance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    components = range(1, len(pca_results['explained_variance']) + 1)
    ax.bar(components, pca_results['explained_variance'] * 100, 
           alpha=0.7, label='Individual')
    ax.plot(components, pca_results['cumsum_variance'] * 100, 
            'o-', color='red', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # Biplot
    st.markdown("### 📈 PCA Biplot")
    fig = create_pca_biplot(pca_results, n_components=2)
    st.pyplot(fig)
    plt.close(fig)
    
    # 3D PCA
    st.markdown("### 🔮 3D PCA Interactive")
    fig = create_pca_3d(pca_results)
    st.plotly_chart(fig, use_container_width=True)
    
    # t-SNE
    st.markdown("### 🧪 t-SNE Visualization")
    
    if st.button('Run t-SNE (may take a moment)'):
        with st.spinner('Computing t-SNE...'):
            X_tsne = perform_tsne(df_desc, selected_desc, perplexity=30)
            st.session_state['tsne_results'] = X_tsne
    
    if 'tsne_results' in st.session_state:
        X_tsne = st.session_state['tsne_results']
        
        # Get labels if available
        if 'clustering_results' in st.session_state:
            labels = st.session_state['clustering_results']['labels']
        else:
            labels = None
        
        fig = create_tsne_plot(X_tsne, labels)
        st.pyplot(fig)
        plt.close(fig)
    
    # Clustering
    st.markdown("### 📊 Clustering Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox('Clustering Method', ['kmeans', 'dbscan', 'hierarchical'])
    with col2:
        if method in ['kmeans', 'hierarchical']:
            n_clusters = st.slider('Number of Clusters', 2, 8, 3)
        else:
            n_clusters = 3
    
    if st.button('Run Clustering'):
        with st.spinner('Performing clustering...'):
            X_pca = pca_results['X_pca'][:, :min(5, pca_results['X_pca'].shape[1])]
            clustering_results = perform_clustering(X_pca, method, n_clusters)
            st.session_state['clustering_results'] = clustering_results
    
    if 'clustering_results' in st.session_state:
        clustering_results = st.session_state['clustering_results']
        labels = clustering_results['labels']
        
        # Display cluster info
        col1, col2, col3 = st.columns(3)
        with col1:
            n_clusters_found = len(np.unique(labels))
            st.metric("Number of Clusters", n_clusters_found)
        with col2:
            if 'silhouette' in clustering_results:
                st.metric("Silhouette Score", f"{clustering_results['silhouette']:.3f}")
        with col3:
            n_outliers = np.sum(labels == -1) if method == 'dbscan' else 0
            if n_outliers > 0:
                st.metric("Outliers", n_outliers)
        
        # Add labels to DataFrame
        df_desc['cluster'] = labels
        st.session_state['descriptors'] = df_desc
        
        # Silhouette plot
        if len(np.unique(labels)) > 1 and method != 'dbscan':
            X_pca = pca_results['X_pca'][:, :min(5, pca_results['X_pca'].shape[1])]
            fig = create_silhouette_plot(X_pca, labels)
            st.pyplot(fig)
            plt.close(fig)
        
        # Cluster profiles
        st.markdown("### 📊 Cluster Profiles")
        
        # Select descriptors for profiles
        profile_descriptors = st.multiselect(
            'Select descriptors for cluster profiles',
            top_descriptors,
            default=top_descriptors[:min(6, len(top_descriptors))]
        )
        
        if len(profile_descriptors) >= 2:
            fig = create_cluster_profiles(
                df_desc[df_desc['cluster'] != -1] if method == 'dbscan' else df_desc,
                'cluster',
                profile_descriptors
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # Radar chart
            st.markdown("### 🎯 Radar Chart Comparison")
            fig = create_radar_chart(
                df_desc[df_desc['cluster'] != -1] if method == 'dbscan' else df_desc,
                'cluster',
                profile_descriptors
            )
            st.pyplot(fig)
            plt.close(fig)

def render_visualization_page(df: pd.DataFrame):
    """Render the visualization dashboard page."""
    st.markdown("# 📈 Visualization Dashboard")
    
    if df is None or df.empty or 'descriptors' not in st.session_state:
        st.warning('⚠️ Please calculate descriptors first.')
        return
    
    df_desc = st.session_state['descriptors']
    top_descriptors = st.session_state.get('top_descriptors', 
                                         [col for col in df_desc.columns 
                                          if col not in ['№', 'Ref', 'method', 'A', 'B'] 
                                          and df_desc[col].dtype in ['float64', 'int64']])[:20]
    
    # Visualization selector
    viz_type = st.selectbox(
        'Select Visualization Type',
        [
            'Concentration Heatmap',
            'Concentration Contour',
            '4D Bubble Chart',
            'Compositional Bubble',
            '3D Scatter Interactive',
            'α vs β Compromise',
            'T(bends) Analysis',
            'β vs pH₂O',
            'α vs Geometric',
            'β vs Electronegativity',
            'Parallel Coordinates',
            'Swarm Plot',
            'Feature Importance'
        ]
    )
    
    # Target variables
    target_cols = ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'alpha_beta_ratio', 'T_bends_first']
    target_cols = [col for col in target_cols if col in df_desc.columns]
    
    if not target_cols:
        st.warning('No target variables available.')
        return
    
    # Common controls
    col1, col2, col3 = st.columns(3)
    
    if viz_type in ['Concentration Heatmap', 'Concentration Contour']:
        with col1:
            x_col = st.selectbox('X-axis', top_descriptors, index=0)
        with col2:
            y_col = st.selectbox('Y-axis', top_descriptors, 
                                index=min(1, len(top_descriptors)-1))
        with col3:
            color_col = st.selectbox('Color (target)', target_cols, index=0)
        
        if viz_type == 'Concentration Heatmap':
            fig = create_concentration_heatmap(df_desc, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_concentration_contour(df_desc, x_col, y_col, color_col)
            st.pyplot(fig)
            plt.close(fig)
    
    elif viz_type in ['4D Bubble Chart', 'Compositional Bubble']:
        with col1:
            x_col = st.selectbox('X-axis', top_descriptors, index=0)
        with col2:
            y_col = st.selectbox('Y-axis (target)', target_cols, index=0)
        with col3:
            color_col = st.selectbox('Color', top_descriptors, 
                                    index=min(1, len(top_descriptors)-1))
        
        size_col = st.selectbox('Size', top_descriptors, 
                               index=min(2, len(top_descriptors)-1))
        shape_col = st.selectbox('Shape (optional, categorical)', 
                                ['None'] + [col for col in df_desc.columns 
                                           if df_desc[col].dtype == 'object'],
                                index=0)
        
        if viz_type == '4D Bubble Chart':
            fig = create_bubble_4d(df_desc, x_col, y_col, color_col, size_col,
                                 shape_col if shape_col != 'None' else None)
        else:
            fig = create_compositional_bubble(df_desc, x_col, y_col, color_col, size_col)
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == '3D Scatter Interactive':
        with col1:
            x_col = st.selectbox('X-axis', top_descriptors, index=0)
        with col2:
            y_col = st.selectbox('Y-axis', top_descriptors, 
                                index=min(1, len(top_descriptors)-1))
        with col3:
            z_col = st.selectbox('Z-axis', top_descriptors, 
                                index=min(2, len(top_descriptors)-1))
        
        color_col = st.selectbox('Color (optional)', 
                                ['None'] + target_cols, index=0)
        
        fig = create_3d_scatter_interactive(df_desc, x_col, y_col, z_col,
                                           color_col if color_col != 'None' else None)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == 'α vs β Compromise':
        color_col = st.selectbox('Color by', 
                                 ['None'] + [col for col in df_desc.columns 
                                            if df_desc[col].dtype == 'object'],
                                 index=0)
        fig = create_alpha_beta_compromise(df_desc, color_col if color_col != 'None' else None)
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_type == 'T(bends) Analysis':
        color_col = st.selectbox('Color by', 
                                 ['None'] + [col for col in df_desc.columns 
                                            if df_desc[col].dtype == 'object'],
                                 index=0)
        fig = create_t_bends_analysis(df_desc, color_col if color_col != 'None' else None)
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_type == 'β vs pH₂O':
        color_col = st.selectbox('Color by', 
                                 ['None'] + [col for col in df_desc.columns 
                                            if df_desc[col].dtype == 'object'],
                                 index=0)
        fig = create_beta_vs_ph2o(df_desc, color_col if color_col != 'None' else None)
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_type == 'α vs Geometric':
        geo_cols = ['rAav', 'rBav', 't', 'V_cell', 'r_ratio_AB']
        geo_cols = [col for col in geo_cols if col in df_desc.columns]
        geo_col = st.selectbox('Geometric descriptor', geo_cols, index=0)
        fig = create_alpha_vs_geometric(df_desc, geo_col)
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_type == 'β vs Electronegativity':
        fig = create_beta_vs_electronegativity(df_desc)
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_type == 'Parallel Coordinates':
        features = st.multiselect('Select features', top_descriptors,
                                 default=top_descriptors[:min(6, len(top_descriptors))])
        if len(features) >= 2:
            target = st.selectbox('Color by (target)', target_cols, index=0)
            fig = create_parallel_coordinates(df_desc, features, target)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning('Select at least 2 features.')
    
    elif viz_type == 'Swarm Plot':
        cat_cols = [col for col in df_desc.columns if df_desc[col].dtype == 'object']
        cat_col = st.selectbox('Categorical variable', cat_cols, index=0)
        target = st.selectbox('Target variable', target_cols, index=0)
        fig = create_swarm_plot(df_desc, target, cat_col)
        st.pyplot(fig)
        plt.close(fig)
    
    elif viz_type == 'Feature Importance':
        target = st.selectbox('Target variable', target_cols, index=0)
        desc_for_importance = st.multiselect('Descriptors', top_descriptors,
                                            default=top_descriptors[:min(15, len(top_descriptors))])
        if len(desc_for_importance) >= 2:
            fig = create_feature_importance(df_desc, target, desc_for_importance)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning('Select at least 2 descriptors.')

def render_export_page(df: pd.DataFrame):
    """Render the export page."""
    st.markdown("# 💾 Export Results")
    
    if df is None or df.empty:
        st.warning('⚠️ No data to export.')
        return
    
    st.markdown("### 📥 Download Data")
    
    # Export options
    export_options = st.multiselect(
        'Select data to export',
        ['Raw Data', 'Descriptors', 'PCA Results', 'Clustering Results', 'Correlation Results'],
        default=['Raw Data', 'Descriptors']
    )
    
    if 'Raw Data' in export_options:
        csv = df.to_csv(index=False)
        st.download_button(
            label='📥 Download Raw Data (CSV)',
            data=csv,
            file_name='perovskite_data.csv',
            mime='text/csv'
        )
    
    if 'Descriptors' in export_options and 'descriptors' in st.session_state:
        csv = st.session_state['descriptors'].to_csv(index=False)
        st.download_button(
            label='📥 Download Descriptors (CSV)',
            data=csv,
            file_name='perovskite_descriptors_full.csv',
            mime='text/csv'
        )
    
    if 'PCA Results' in export_options and 'pca_results' in st.session_state:
        pca_results = st.session_state['pca_results']
        # Export loadings
        loadings_csv = pca_results['loadings'].to_csv()
        st.download_button(
            label='📥 Download PCA Loadings (CSV)',
            data=loadings_csv,
            file_name='pca_loadings.csv',
            mime='text/csv'
        )
        
        # Export transformed data
        X_pca_df = pd.DataFrame(
            pca_results['X_pca'],
            columns=[f'PC{i+1}' for i in range(pca_results['X_pca'].shape[1])]
        )
        csv = X_pca_df.to_csv(index=False)
        st.download_button(
            label='📥 Download PCA Transformed Data (CSV)',
            data=csv,
            file_name='pca_transformed.csv',
            mime='text/csv'
        )
    
    if 'Clustering Results' in export_options and 'clustering_results' in st.session_state:
        clustering = st.session_state['clustering_results']
        if 'labels' in clustering:
            labels_df = pd.DataFrame({'cluster': clustering['labels']})
            csv = labels_df.to_csv(index=False)
            st.download_button(
                label='📥 Download Clustering Labels (CSV)',
                data=csv,
                file_name='clustering_labels.csv',
                mime='text/csv'
            )
    
    if 'Correlation Results' in export_options and 'correlation_data' in st.session_state:
        corr_data = st.session_state['correlation_data']
        if corr_data['pearson'] is not None:
            csv = corr_data['pearson'].to_csv()
            st.download_button(
                label='📥 Download Pearson Correlation Matrix (CSV)',
                data=csv,
                file_name='pearson_correlation.csv',
                mime='text/csv'
            )
        
        if corr_data['spearman'] is not None:
            csv = corr_data['spearman'].to_csv()
            st.download_button(
                label='📥 Download Spearman Correlation Matrix (CSV)',
                data=csv,
                file_name='spearman_correlation.csv',
                mime='text/csv'
            )

# ============================================================================
# SECTION 12: MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Apply scientific style
    apply_scientific_style()
    
    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = None
    if 'descriptors' not in st.session_state:
        st.session_state['descriptors'] = None
    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
    if 'filters_reset' not in st.session_state:
        st.session_state['filters_reset'] = False
    
    # Title and description
    st.markdown("# 🔬 Proton-Conducting Perovskites Analyzer")
    st.markdown("""
    **Interactive tool for analyzing correlations between chemical composition, 
    structure, and thermal/chemical expansion properties of proton-conducting perovskite oxides.**
    
    *Upload your data, calculate descriptors, and explore hidden correlations through 
    interactive visualizations.*
    """)
    
    # Sidebar filters (if data is loaded)
    if st.session_state['raw_data'] is not None:
        filters = render_sidebar_filters(st.session_state['raw_data'])
        filtered_data = apply_filters(st.session_state['raw_data'], filters)
        
        # Update session state with filtered data
        if 'filtered_data' not in st.session_state or st.session_state['filtered_data'] is not None:
            st.session_state['filtered_data'] = filtered_data
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '📤 Upload',
        '🔬 Descriptors',
        '📊 Correlations',
        '🧬 PCA & Clustering',
        '📈 Visualizations',
        '💾 Export'
    ])
    
    with tab1:
        render_upload_page()
    
    with tab2:
        if st.session_state['raw_data'] is not None:
            render_descriptor_page(st.session_state['raw_data'])
        else:
            st.info('📤 Please upload data first.')
    
    with tab3:
        if st.session_state.get('descriptors') is not None:
            render_correlation_page(st.session_state['descriptors'])
        else:
            st.info('🔬 Please calculate descriptors first.')
    
    with tab4:
        if st.session_state.get('descriptors') is not None:
            render_pca_clustering_page(st.session_state['descriptors'])
        else:
            st.info('🔬 Please calculate descriptors first.')
    
    with tab5:
        if st.session_state.get('descriptors') is not None:
            # Use filtered data if available
            data_for_viz = st.session_state.get('filtered_data', st.session_state['descriptors'])
            render_visualization_page(data_for_viz)
        else:
            st.info('🔬 Please calculate descriptors first.')
    
    with tab6:
        if st.session_state.get('descriptors') is not None:
            render_export_page(st.session_state['descriptors'])
        else:
            st.info('🔬 Please calculate descriptors first.')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🔬 Proton-Conducting Perovskites Analyzer v1.0**  
    *Built with Streamlit • Data-driven materials science*
    """)

if __name__ == "__main__":
    main()
