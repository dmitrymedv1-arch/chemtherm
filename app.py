"""
Proton-Conducting Perovskites Analysis Platform
================================================
Interactive web application for analyzing thermal and chemical expansion
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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import scipy.stats as stats
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr, pearsonr, kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import umap
import shap
import pingouin as pg
import networkx as nx
from adjustText import adjust_text
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Perovskite Analysis Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SECTION 2: SCIENTIFIC STYLE CONFIGURATION
# ============================================================================

def apply_scientific_style():
    """
    Применение улучшенного научного стиля для всех matplotlib графиков.
    Стиль оптимизирован для публикаций в научных журналах.
    """
    try:
    plt.style.use('seaborn-whitegrid')
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
    plt.rcParams.update({
        # Шрифты
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        
        # Оси
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#000000',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Метки
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
        
        # Легенда
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#000000',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        
        # Фигура
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        
        # Линии
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'errorbar.capsize': 3,
        
        # PDF
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

apply_scientific_style()

# Цветовые палитры
COLOR_PALETTES = {
    'B_cation': {'Ce': '#E74C3C', 'Zr': '#3498DB', 'Sn': '#2ECC71', 'Ti': '#F39C12', 'Hf': '#9B59B6'},
    'method': {'dilatometry': '#2C3E50', 'HT XRD': '#E67E22', 'HT ND': '#8E44AD'},
    'A_cation': {'Ba': '#1A5276', 'Sr': '#2471A3', 'Ca': '#5DADE2', 'La': '#F39C12', 'Sr': '#2ECC71'},
    'continuous': 'viridis',
    'diverging': 'coolwarm',
    'qualitative': px.colors.qualitative.Set1,
}

# ============================================================================
# SECTION 3: BUILT-IN DATABASES
# ============================================================================

# 3.1 Ионные радиусы по Шеннону (Å)
# A-позиция: 12-координация, B-позиция: 6-координация
IONIC_RADII = {
    # A-site cations (12-coordination)
    'A_site': {
        'Ba': 1.61, 'Ba2+': 1.61,
        'Sr': 1.44, 'Sr2+': 1.44,
        'Ca': 1.34, 'Ca2+': 1.34,
        'La': 1.36, 'La3+': 1.36,
        'La': 1.36,
        'Pb': 1.49, 'Pb2+': 1.49,
        'Bi': 1.17, 'Bi3+': 1.17,
        'K': 1.64, 'K+': 1.64,
        'Na': 1.39, 'Na+': 1.39,
        'Gd': 1.27, 'Gd3+': 1.27,
        'Nd': 1.27, 'Nd3+': 1.27,
        'Sm': 1.24, 'Sm3+': 1.24,
        'Eu': 1.20, 'Eu3+': 1.20,
        'Y': 1.19, 'Y3+': 1.19,
        'Yb': 1.12, 'Yb3+': 1.12,
        'Dy': 1.19, 'Dy3+': 1.19,
        'Ho': 1.16, 'Ho3+': 1.16,
        'Er': 1.14, 'Er3+': 1.14,
        'Tm': 1.13, 'Tm3+': 1.13,
        'Lu': 1.11, 'Lu3+': 1.11,
        'Sc': 0.87, 'Sc3+': 0.87,
        'In': 1.04, 'In3+': 1.04,
        'Fe': 0.78, 'Fe3+': 0.78,
        'Zn': 0.74, 'Zn2+': 0.74,
        'Al': 0.535, 'Al3+': 0.535,
        'Ga': 0.62, 'Ga3+': 0.62,
        'Ge': 0.53, 'Ge4+': 0.53,
        'Si': 0.40, 'Si4+': 0.40,
        'Ti': 0.605, 'Ti4+': 0.605,
        'Zr': 0.72, 'Zr4+': 0.72,
        'Hf': 0.71, 'Hf4+': 0.71,
        'Sn': 0.69, 'Sn4+': 0.69,
        'Ce': 0.87, 'Ce4+': 0.87,
        'Ce3+': 1.01,
        'Pr': 0.85, 'Pr4+': 0.85,
        'Tb': 0.76, 'Tb4+': 0.76,
        'Tb3+': 0.923,
    },
    # B-site cations (6-coordination)
    'B_site': {
        'Ti': 0.605, 'Ti4+': 0.605,
        'Zr': 0.72, 'Zr4+': 0.72,
        'Hf': 0.71, 'Hf4+': 0.71,
        'Sn': 0.69, 'Sn4+': 0.69,
        'Ce': 0.87, 'Ce4+': 0.87,
        'Ce3+': 1.01,
        'Y': 0.90, 'Y3+': 0.90,
        'Yb': 0.87, 'Yb3+': 0.87,
        'Sc': 0.745, 'Sc3+': 0.745,
        'In': 0.80, 'In3+': 0.80,
        'Fe': 0.645, 'Fe3+': 0.645,
        'Fe2+': 0.61,
        'Zn': 0.60, 'Zn2+': 0.60,
        'Gd': 0.938, 'Gd3+': 0.938,
        'Sm': 0.958, 'Sm3+': 0.958,
        'Nd': 0.983, 'Nd3+': 0.983,
        'Eu': 0.947, 'Eu3+': 0.947,
        'Dy': 0.912, 'Dy3+': 0.912,
        'Ho': 0.901, 'Ho3+': 0.901,
        'Er': 0.890, 'Er3+': 0.890,
        'Tm': 0.880, 'Tm3+': 0.880,
        'Tb': 0.923, 'Tb3+': 0.923,
        'Al': 0.535, 'Al3+': 0.535,
        'Ga': 0.620, 'Ga3+': 0.620,
        'Ge': 0.530, 'Ge4+': 0.530,
        'Si': 0.400, 'Si4+': 0.400,
        'Pb': 0.775, 'Pb4+': 0.775,
        'Bi': 0.760, 'Bi5+': 0.760,
        'Pr': 0.850, 'Pr4+': 0.850,
        'Ru': 0.620, 'Ru4+': 0.620,
        'Mn': 0.530, 'Mn4+': 0.530,
        'Co': 0.545, 'Co3+': 0.545,
        'Ni': 0.560, 'Ni3+': 0.560,
        'Cu': 0.570, 'Cu2+': 0.570,
        'Cr': 0.550, 'Cr3+': 0.550,
    },
    # Кислород
    'O': {
        'O2-': 1.40,  # 6-coordination
        'O2-_12': 1.38,  # 12-coordination
    }
}

# 3.2 Электроотрицательности по Поллингу
ELECTRONEGATIVITY = {
    'Ba': 0.89, 'Sr': 0.95, 'Ca': 1.00, 'Mg': 1.31,
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.20,
    'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25,
    'Yb': 1.10, 'Lu': 1.27, 'Y': 1.22, 'Sc': 1.36,
    'Zr': 1.33, 'Hf': 1.30, 'Ti': 1.54, 'Sn': 1.96,
    'Pb': 2.33, 'Bi': 2.02, 'In': 1.78, 'Ga': 1.81,
    'Al': 1.61, 'Si': 1.90, 'Ge': 2.01, 'Fe': 1.83,
    'Zn': 1.65, 'Cu': 1.90, 'Ni': 1.91, 'Co': 1.88,
    'Mn': 1.55, 'Cr': 1.66, 'V': 1.63, 'Mo': 2.16,
    'W': 2.36, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20,
    'Ag': 1.93, 'Cd': 1.69, 'Au': 2.54, 'Pt': 2.28,
    'O': 3.44, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96,
    'I': 2.66, 'S': 2.58, 'Se': 2.55, 'N': 3.04,
    'P': 2.19, 'As': 2.18, 'Sb': 2.05, 'Te': 2.10,
}

# 3.3 Валентности (степени окисления)
VALENCE = {
    'Ba': 2, 'Sr': 2, 'Ca': 2, 'Mg': 2,
    'La': 3, 'Ce': 4, 'Ce3+': 3, 'Pr': 4, 'Nd': 3,
    'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Tb4+': 4,
    'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3,
    'Lu': 3, 'Y': 3, 'Sc': 3, 'Zr': 4, 'Hf': 4,
    'Ti': 4, 'Sn': 4, 'Pb': 4, 'Bi': 5, 'In': 3,
    'Ga': 3, 'Al': 3, 'Si': 4, 'Ge': 4, 'Fe': 3,
    'Zn': 2, 'Cu': 2, 'Ni': 3, 'Co': 3, 'Mn': 4,
    'Cr': 3, 'V': 5, 'Mo': 6, 'W': 6, 'Ru': 4,
    'Rh': 3, 'Pd': 4, 'Ag': 1, 'Cd': 2, 'Au': 3,
    'Pt': 4, 'O': -2,
}

# 3.4 Молярные массы (г/моль)
MOLAR_MASS = {
    'Ba': 137.327, 'Sr': 87.62, 'Ca': 40.078, 'Mg': 24.305,
    'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.243,
    'Sm': 150.362, 'Eu': 151.964, 'Gd': 157.250, 'Tb': 158.925,
    'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259, 'Tm': 168.934,
    'Yb': 173.045, 'Lu': 174.967, 'Y': 88.906, 'Sc': 44.956,
    'Zr': 91.224, 'Hf': 178.490, 'Ti': 47.867, 'Sn': 118.710,
    'Pb': 207.200, 'Bi': 208.980, 'In': 114.818, 'Ga': 69.723,
    'Al': 26.982, 'Si': 28.086, 'Ge': 72.630, 'Fe': 55.845,
    'Zn': 65.380, 'Cu': 63.546, 'Ni': 58.693, 'Co': 58.933,
    'Mn': 54.938, 'Cr': 51.996, 'V': 50.942, 'Mo': 95.950,
    'W': 183.840, 'Ru': 101.070, 'Rh': 102.906, 'Pd': 106.420,
    'Ag': 107.868, 'Cd': 112.414, 'Au': 196.967, 'Pt': 195.084,
    'O': 15.999,
}

# ============================================================================
# SECTION 4: DATA LOADING AND PARSING FUNCTIONS
# ============================================================================

def parse_uploaded_data(text_content):
    """
    Парсинг вставленного текста с данными в DataFrame.
    
    Parameters:
    -----------
    text_content : str
        Текст с данными в формате TSV (табуляция как разделитель)
    
    Returns:
    --------
    pd.DataFrame : Обработанный DataFrame с данными
    """
    try:
        # Чтение данных из текста
        df = pd.read_csv(StringIO(text_content), sep='\t', dtype=str)
        
        # Очистка данных
        df = clean_data(df)
        
        return df
    except Exception as e:
        st.error(f"Ошибка при парсинге данных: {str(e)}")
        return None

def clean_data(df):
    """
    Очистка и преобразование данных.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    
    Returns:
    --------
    pd.DataFrame : Очищенный DataFrame
    """
    # Копия для предотвращения изменений исходного
    df_clean = df.copy()
    
    # Замена '-' на NaN
    df_clean = df_clean.replace('-', pd.NA)
    df_clean = df_clean.replace('—', pd.NA)
    df_clean = df_clean.replace('‑', pd.NA)
    
    # Преобразование числовых колонок
    numeric_cols = ['[A\']', '[B\']', '[D1]', '[D2]', 'δ', 'β', 'α·106 (K-1)', 'pH2O']
    for col in numeric_cols:
        if col in df_clean.columns:
            # Замена запятых на точки (европейский формат)
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Парсинг температурного диапазона
    if '∆T, °C' in df_clean.columns:
        df_clean[['T_min', 'T_max']] = df_clean['∆T, °C'].astype(str).str.split('-', expand=True)
        df_clean['T_min'] = pd.to_numeric(df_clean['T_min'], errors='coerce')
        df_clean['T_max'] = pd.to_numeric(df_clean['T_max'], errors='coerce')
    
    # Парсинг T(bends)
    if 'T(bends), °C' in df_clean.columns:
        df_clean['T_bends_list'] = df_clean['T(bends), °C'].apply(parse_t_bends)
        df_clean['T_bends_count'] = df_clean['T_bends_list'].apply(len)
        df_clean['T_bends_first'] = df_clean['T_bends_list'].apply(lambda x: x[0] if len(x) > 0 else pd.NA)
        df_clean['T_bends_last'] = df_clean['T_bends_list'].apply(lambda x: x[-1] if len(x) > 0 else pd.NA)
    
    # Парсинг αav
    if 'αav·106 (K-1)' in df_clean.columns:
        df_clean['αav_list'] = df_clean['αav·106 (K-1)'].apply(parse_alpha_av)
        df_clean['αav_mean'] = df_clean['αav_list'].apply(lambda x: np.mean(x) if len(x) > 0 else pd.NA)
        df_clean['αav_min'] = df_clean['αav_list'].apply(lambda x: min(x) if len(x) > 0 else pd.NA)
        df_clean['αav_max'] = df_clean['αav_list'].apply(lambda x: max(x) if len(x) > 0 else pd.NA)
    
    return df_clean

def parse_t_bends(value):
    """
    Парсинг строки с температурами изломов (например, "400;600").
    
    Returns:
    --------
    list : Список чисел
    """
    if pd.isna(value) or value == '-' or value == '—' or value == '‑' or value == '':
        return []
    try:
        if isinstance(value, (int, float)):
            return [float(value)]
        # Замена различных разделителей
        value_str = str(value).replace(';', ',').replace(':', ',').replace(' ', '')
        parts = value_str.split(',')
        return [float(p.strip()) for p in parts if p.strip()]
    except:
        return []

def parse_alpha_av(value):
    """
    Парсинг строки со средними значениями КТР (например, "10.6;4.73;10.1").
    
    Returns:
    --------
    list : Список чисел
    """
    if pd.isna(value) or value == '-' or value == '—' or value == '‑' or value == '':
        return []
    try:
        if isinstance(value, (int, float)):
            return [float(value)]
        # Замена различных разделителей
        value_str = str(value).replace(';', ',').replace(':', ',').replace(' ', '')
        parts = value_str.split(',')
        return [float(p.strip()) for p in parts if p.strip()]
    except:
        return []

def parse_composition_formula(row):
    """
    Формирование химической формулы из данных строки.
    
    Returns:
    --------
    str : Химическая формула
    """
    A = row.get('A', '')
    A_prime = row.get("A'", '')
    B = row.get('B', '')
    B_prime = row.get("B'", '')
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    
    # Концентрации
    A_conc = row.get('[A\']', 0)
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    
    formula = f"{A}"
    if not pd.isna(A_prime) and A_prime != '-':
        if A_conc > 0:
            formula += f"_{1-A_conc:.2f}{A_prime}_{A_conc:.2f}"
    
    formula += f"{B}"
    if not pd.isna(B_prime) and B_prime != '-':
        if B_conc > 0:
            formula += f"_{1-B_conc-D1_conc-D2_conc:.2f}{B_prime}_{B_conc:.2f}"
    
    if not pd.isna(D1) and D1 != '-':
        if D1_conc > 0:
            formula += f"{D1}_{D1_conc:.2f}"
    
    if not pd.isna(D2) and D2 != '-':
        if D2_conc > 0:
            formula += f"{D2}_{D2_conc:.2f}"
    
    return formula

# ============================================================================
# SECTION 5: DESCRIPTOR ENGINE
# ============================================================================

def calculate_rAav(row):
    """
    Расчёт среднего радиуса A-позиции.
    
    Returns:
    --------
    float : Средний радиус A-позиции (Å)
    """
    A = row.get('A', '')
    A_prime = row.get("A'", '')
    A_conc = row.get('[A\']', 0)
    
    rA = IONIC_RADII['A_site'].get(A, np.nan)
    rA_prime = IONIC_RADII['A_site'].get(A_prime, np.nan)
    
    if pd.isna(rA):
        return np.nan
    
    if pd.isna(rA_prime) or A_prime == '-' or pd.isna(A_conc):
        return rA
    
    rAav = rA * (1 - A_conc) + rA_prime * A_conc
    return rAav

def calculate_rBav(row):
    """
    Расчёт среднего радиуса B-позиции.
    
    Returns:
    --------
    float : Средний радиус B-позиции (Å)
    """
    B = row.get('B', '')
    B_prime = row.get("B'", '')
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    
    rB = IONIC_RADII['B_site'].get(B, np.nan)
    rB_prime = IONIC_RADII['B_site'].get(B_prime, np.nan)
    rD1 = IONIC_RADII['B_site'].get(D1, np.nan)
    rD2 = IONIC_RADII['B_site'].get(D2, np.nan)
    
    if pd.isna(rB):
        return np.nan
    
    # Суммарная концентрация всех допантов в B-позиции
    total_conc = B_conc + D1_conc + D2_conc
    
    # Проверка на валидность концентраций
    if total_conc > 1.0:
        # Корректировка
        B_conc = B_conc / (total_conc) if total_conc > 0 else 0
        D1_conc = D1_conc / (total_conc) if total_conc > 0 else 0
        D2_conc = D2_conc / (total_conc) if total_conc > 0 else 0
        total_conc = 1.0
    
    rBav = rB * (1 - total_conc)
    
    if not pd.isna(rB_prime) and B_prime != '-' and B_prime != '':
        rBav += rB_prime * B_conc
    
    if not pd.isna(rD1) and D1 != '-' and D1 != '':
        rBav += rD1 * D1_conc
    
    if not pd.isna(rD2) and D2 != '-' and D2 != '':
        rBav += rD2 * D2_conc
    
    return rBav

def calculate_geometric_descriptors(row):
    """
    Расчёт всех геометрических дескрипторов.
    
    Returns:
    --------
    dict : Словарь с геометрическими дескрипторами
    """
    rO = IONIC_RADII['O']['O2-']
    rAav = calculate_rAav(row)
    rBav = calculate_rBav(row)
    
    descriptors = {}
    
    # 1. Толерант-фактор Гольдшмидта
    if not pd.isna(rAav) and not pd.isna(rBav):
        t = (rAav + rO) / (np.sqrt(2) * (rBav + rO))
        descriptors['t'] = t
        descriptors['D_t'] = abs(1 - t)
    else:
        descriptors['t'] = np.nan
        descriptors['D_t'] = np.nan
    
    # 2. Октаэдрический фактор
    if not pd.isna(rBav):
        descriptors['octahedral_factor'] = rBav / rO
    else:
        descriptors['octahedral_factor'] = np.nan
    
    # 3. Разница радиусов A и B
    if not pd.isna(rAav) and not pd.isna(rBav):
        descriptors['Δr_AB'] = abs(rAav - rBav)
        descriptors['Δr_AB_norm'] = abs(rAav - rBav) / rO
    else:
        descriptors['Δr_AB'] = np.nan
        descriptors['Δr_AB_norm'] = np.nan
    
    # 4. Средние радиусы
    descriptors['rAav'] = rAav
    descriptors['rBav'] = rBav
    
    # 5. Отношение радиусов
    if not pd.isna(rAav) and not pd.isna(rBav):
        descriptors['r_ratio_AB'] = rAav / rBav
    else:
        descriptors['r_ratio_AB'] = np.nan
    
    # 6. Дисперсия радиусов B-site (если есть допанты)
    B = row.get('B', '')
    B_prime = row.get("B'", '')
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    
    radii = []
    fractions = []
    
    rB = IONIC_RADII['B_site'].get(B, np.nan)
    if not pd.isna(rB):
        total_conc = B_conc + D1_conc + D2_conc
        if total_conc < 1:
            radii.append(rB)
            fractions.append(1 - total_conc)
    
    rB_prime = IONIC_RADII['B_site'].get(B_prime, np.nan)
    if not pd.isna(rB_prime) and B_prime != '-' and B_prime != '' and B_conc > 0:
        radii.append(rB_prime)
        fractions.append(B_conc)
    
    rD1 = IONIC_RADII['B_site'].get(D1, np.nan)
    if not pd.isna(rD1) and D1 != '-' and D1 != '' and D1_conc > 0:
        radii.append(rD1)
        fractions.append(D1_conc)
    
    rD2 = IONIC_RADII['B_site'].get(D2, np.nan)
    if not pd.isna(rD2) and D2 != '-' and D2 != '' and D2_conc > 0:
        radii.append(rD2)
        fractions.append(D2_conc)
    
    if len(radii) > 1:
        # Нормализация фракций
        fractions = np.array(fractions) / sum(fractions)
        rBav_calc = sum([r * f for r, f in zip(radii, fractions)])
        variance = sum([f * (r - rBav_calc)**2 for r, f in zip(radii, fractions)])
        descriptors['σ²_rB'] = variance
    else:
        descriptors['σ²_rB'] = 0
    
    # 7. Объём элементарной ячейки (расчётный)
    if not pd.isna(rAav) and not pd.isna(rBav):
        # Для кубического перовскита
        a = np.sqrt(2) * (rBav + rO)
        descriptors['V_cell'] = a**3
        # Свободный объём (грубая оценка)
        descriptors['V_free'] = a**3 - (4/3) * np.pi * (rAav**3 + rBav**3 + 3*rO**3)
    else:
        descriptors['V_cell'] = np.nan
        descriptors['V_free'] = np.nan
    
    return descriptors

def calculate_electronegativity_descriptors(row):
    """
    Расчёт всех электроотрицательных дескрипторов.
    
    Returns:
    --------
    dict : Словарь с электроотрицательными дескрипторами
    """
    A = row.get('A', '')
    A_prime = row.get("A'", '')
    B = row.get('B', '')
    B_prime = row.get("B'", '')
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    A_conc = row.get('[A\']', 0)
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    
    descriptors = {}
    
    # Получение электроотрицательностей
    χA = ELECTRONEGATIVITY.get(A, np.nan)
    χA_prime = ELECTRONEGATIVITY.get(A_prime, np.nan)
    χB = ELECTRONEGATIVITY.get(B, np.nan)
    χB_prime = ELECTRONEGATIVITY.get(B_prime, np.nan)
    χD1 = ELECTRONEGATIVITY.get(D1, np.nan)
    χD2 = ELECTRONEGATIVITY.get(D2, np.nan)
    χO = ELECTRONEGATIVITY.get('O', 3.44)
    
    descriptors['χA'] = χA
    descriptors['χA_prime'] = χA_prime
    descriptors['χB'] = χB
    descriptors['χB_prime'] = χB_prime
    descriptors['χD1'] = χD1
    descriptors['χD2'] = χD2
    
    # Средняя электроотрицательность A-site
    if not pd.isna(χA):
        if not pd.isna(χA_prime) and A_prime != '-' and A_prime != '':
            χAav = χA * (1 - A_conc) + χA_prime * A_conc
        else:
            χAav = χA
        descriptors['χAav'] = χAav
    else:
        descriptors['χAav'] = np.nan
    
    # Средняя электроотрицательность B-site
    if not pd.isna(χB):
        total_conc = B_conc + D1_conc + D2_conc
        if total_conc > 1.0:
            total_conc = 1.0
            B_conc = B_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
            D1_conc = D1_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
            D2_conc = D2_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
        
        χBav = χB * (1 - total_conc)
        
        if not pd.isna(χB_prime) and B_prime != '-' and B_prime != '':
            χBav += χB_prime * B_conc
        
        if not pd.isna(χD1) and D1 != '-' and D1 != '':
            χBav += χD1 * D1_conc
        
        if not pd.isna(χD2) and D2 != '-' and D2 != '':
            χBav += χD2 * D2_conc
        
        descriptors['χBav'] = χBav
    else:
        descriptors['χBav'] = np.nan
    
    # Разница и отношение электроотрицательностей
    if not pd.isna(descriptors['χAav']) and not pd.isna(descriptors['χBav']):
        descriptors['Δχ_AB'] = abs(descriptors['χAav'] - descriptors['χBav'])
        descriptors['χ_ratio_AB'] = descriptors['χAav'] / descriptors['χBav']
    else:
        descriptors['Δχ_AB'] = np.nan
        descriptors['χ_ratio_AB'] = np.nan
    
    # Ионность связи A-O и B-O
    if not pd.isna(descriptors['χAav']):
        descriptors['ionicity_AO'] = 1 - np.exp(-0.25 * (descriptors['χAav'] - χO)**2)
    else:
        descriptors['ionicity_AO'] = np.nan
    
    if not pd.isna(descriptors['χBav']):
        descriptors['ionicity_BO'] = 1 - np.exp(-0.25 * (descriptors['χBav'] - χO)**2)
    else:
        descriptors['ionicity_BO'] = np.nan
    
    # Кислотность
    if not pd.isna(descriptors['χAav']):
        descriptors['acidity_AO'] = 1 / descriptors['χAav']
    else:
        descriptors['acidity_AO'] = np.nan
    
    if not pd.isna(descriptors['χBav']):
        descriptors['acidity_BO'] = 1 / descriptors['χBav']
    else:
        descriptors['acidity_BO'] = np.nan
    
    # Разница кислотностей
    if not pd.isna(descriptors['acidity_AO']) and not pd.isna(descriptors['acidity_BO']):
        descriptors['Δacidity'] = descriptors['acidity_BO'] - descriptors['acidity_AO']
    else:
        descriptors['Δacidity'] = np.nan
    
    return descriptors

def calculate_thermodynamic_descriptors(row):
    """
    Расчёт термодинамических дескрипторов.
    
    Returns:
    --------
    dict : Словарь с термодинамическими дескрипторами
    """
    B = row.get('B', '')
    B_prime = row.get("B'", '')
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    delta = row.get('δ', 0)
    
    descriptors = {}
    R = 8.314  # Дж/(моль·К)
    
    # Конфигурационная энтропия A-site
    A = row.get('A', '')
    A_prime = row.get("A'", '')
    A_conc_val = row.get('[A\']', 0)
    
    if A != '-' and A != '':
        S_config_A = 0
        if A_conc_val < 1:
            S_config_A -= (1 - A_conc_val) * np.log(1 - A_conc_val) if (1 - A_conc_val) > 0 else 0
        if A_conc_val > 0:
            S_config_A -= A_conc_val * np.log(A_conc_val) if A_conc_val > 0 else 0
        descriptors['S_config_A'] = R * S_config_A
    else:
        descriptors['S_config_A'] = 0
    
    # Конфигурационная энтропия B-site
    total_conc = B_conc + D1_conc + D2_conc
    if total_conc > 0 and total_conc <= 1:
        fractions = [1 - total_conc, B_conc, D1_conc, D2_conc]
        fractions = [f for f in fractions if f > 0]
        S_config_B = -sum([f * np.log(f) for f in fractions])
        descriptors['S_config_B'] = R * S_config_B
    else:
        descriptors['S_config_B'] = 0
    
    # Средняя валентность B-site
    VB = VALENCE.get(B, 4)
    VB_prime = VALENCE.get(B_prime, 4)
    VD1 = VALENCE.get(D1, 3)
    VD2 = VALENCE.get(D2, 3)
    
    if not pd.isna(VB):
        total_conc_norm = B_conc + D1_conc + D2_conc
        if total_conc_norm > 1.0:
            total_conc_norm = 1.0
            B_conc_norm = B_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
            D1_conc_norm = D1_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
            D2_conc_norm = D2_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
        else:
            B_conc_norm = B_conc
            D1_conc_norm = D1_conc
            D2_conc_norm = D2_conc
        
        V_Bav = VB * (1 - total_conc_norm)
        
        if not pd.isna(VB_prime) and B_prime != '-' and B_prime != '':
            V_Bav += VB_prime * B_conc_norm
        
        if not pd.isna(VD1) and D1 != '-' and D1 != '':
            V_Bav += VD1 * D1_conc_norm
        
        if not pd.isna(VD2) and D2 != '-' and D2 != '':
            V_Bav += VD2 * D2_conc_norm
        
        descriptors['V_Bav'] = V_Bav
        descriptors['Vo_proxy'] = (4 - V_Bav) / 2 if not pd.isna(V_Bav) else np.nan
    else:
        descriptors['V_Bav'] = np.nan
        descriptors['Vo_proxy'] = np.nan
    
    # Энергия связи B-O (кулоновская)
    rBav = calculate_rBav(row)
    χBav = descriptors.get('χBav', np.nan)
    
    if not pd.isna(rBav) and not pd.isna(χBav):
        # Упрощённая оценка
        Z_B = descriptors.get('V_Bav', 4)
        Z_O = -2
        descriptors['E_BO'] = (Z_B * abs(Z_O)) / rBav
    else:
        descriptors['E_BO'] = np.nan
    
    # Массовая плотность (грубая оценка)
    M_Aav = calculate_mass_descriptors(row).get('M_Aav', np.nan)
    M_Bav = calculate_mass_descriptors(row).get('M_Bav', np.nan)
    V_cell = calculate_geometric_descriptors(row).get('V_cell', np.nan)
    
    if not pd.isna(M_Aav) and not pd.isna(M_Bav) and not pd.isna(V_cell):
        M_total = M_Aav + M_Bav + 3 * 15.999
        descriptors['ρ'] = M_total / (V_cell * 1e-24 * 6.022e23)  # г/см³
    else:
        descriptors['ρ'] = np.nan
    
    # Энтальпия гидратации (расчётная)
    if not pd.isna(rBav) and not pd.isna(χBav):
        χO = 3.44
        descriptors['ΔH_hydr'] = (χBav - χO)**2 / rBav
    else:
        descriptors['ΔH_hydr'] = np.nan
    
    return descriptors

def calculate_mass_descriptors(row):
    """
    Расчёт массовых дескрипторов.
    
    Returns:
    --------
    dict : Словарь с массовыми дескрипторами
    """
    A = row.get('A', '')
    A_prime = row.get("A'", '')
    B = row.get('B', '')
    B_prime = row.get("B'", '')
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    A_conc = row.get('[A\']', 0)
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    
    descriptors = {}
    
    # Молярные массы
    MA = MOLAR_MASS.get(A, np.nan)
    MA_prime = MOLAR_MASS.get(A_prime, np.nan)
    MB = MOLAR_MASS.get(B, np.nan)
    MB_prime = MOLAR_MASS.get(B_prime, np.nan)
    MD1 = MOLAR_MASS.get(D1, np.nan)
    MD2 = MOLAR_MASS.get(D2, np.nan)
    
    descriptors['M_A'] = MA
    descriptors['M_A_prime'] = MA_prime
    descriptors['M_B'] = MB
    descriptors['M_B_prime'] = MB_prime
    descriptors['M_D1'] = MD1
    descriptors['M_D2'] = MD2
    
    # Средняя молярная масса A-site
    if not pd.isna(MA):
        if not pd.isna(MA_prime) and A_prime != '-' and A_prime != '':
            M_Aav = MA * (1 - A_conc) + MA_prime * A_conc
        else:
            M_Aav = MA
        descriptors['M_Aav'] = M_Aav
    else:
        descriptors['M_Aav'] = np.nan
    
    # Средняя молярная масса B-site
    if not pd.isna(MB):
        total_conc = B_conc + D1_conc + D2_conc
        if total_conc > 1.0:
            total_conc = 1.0
            B_conc = B_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
            D1_conc = D1_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
            D2_conc = D2_conc / (B_conc + D1_conc + D2_conc) if (B_conc + D1_conc + D2_conc) > 0 else 0
        
        M_Bav = MB * (1 - total_conc)
        
        if not pd.isna(MB_prime) and B_prime != '-' and B_prime != '':
            M_Bav += MB_prime * B_conc
        
        if not pd.isna(MD1) and D1 != '-' and D1 != '':
            M_Bav += MD1 * D1_conc
        
        if not pd.isna(MD2) and D2 != '-' and D2 != '':
            M_Bav += MD2 * D2_conc
        
        descriptors['M_Bav'] = M_Bav
    else:
        descriptors['M_Bav'] = np.nan
    
    # Общая молярная масса
    if not pd.isna(descriptors['M_Aav']) and not pd.isna(descriptors['M_Bav']):
        descriptors['M_total'] = descriptors['M_Aav'] + descriptors['M_Bav'] + 3 * 15.999
    else:
        descriptors['M_total'] = np.nan
    
    # Отношение масс A/B
    if not pd.isna(descriptors['M_Aav']) and not pd.isna(descriptors['M_Bav']):
        descriptors['M_ratio_AB'] = descriptors['M_Aav'] / descriptors['M_Bav']
    else:
        descriptors['M_ratio_AB'] = np.nan
    
    # Комбинированные дескрипторы
    rAav = calculate_rAav(row)
    rBav = calculate_rBav(row)
    χAav = calculate_electronegativity_descriptors(row).get('χAav', np.nan)
    χBav = calculate_electronegativity_descriptors(row).get('χBav', np.nan)
    
    if not pd.isna(descriptors['M_Aav']) and not pd.isna(rAav):
        descriptors['M_rA'] = descriptors['M_Aav'] * rAav
    else:
        descriptors['M_rA'] = np.nan
    
    if not pd.isna(descriptors['M_Aav']) and not pd.isna(χAav):
        descriptors['M_χA'] = descriptors['M_Aav'] * χAav
    else:
        descriptors['M_χA'] = np.nan
    
    return descriptors

def calculate_defect_descriptors(row):
    """
    Расчёт дефектных дескрипторов.
    
    Returns:
    --------
    dict : Словарь с дефектными дескрипторами
    """
    D1 = row.get('D1', '')
    D2 = row.get('D2', '')
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    delta = row.get('δ', 0)
    
    descriptors = {}
    
    # Проверка и пересчёт δ
    if not pd.isna(D1_conc) and not pd.isna(D2_conc):
        delta_calc = D1_conc/2 + D2_conc/2
        descriptors['δ_calc'] = delta_calc
        descriptors['δ'] = delta if not pd.isna(delta) else delta_calc
    else:
        descriptors['δ_calc'] = np.nan
        descriptors['δ'] = delta
    
    # Эффективный заряд B-site
    V_Bav = calculate_thermodynamic_descriptors(row).get('V_Bav', 4)
    if not pd.isna(V_Bav):
        descriptors['Z_eff_B'] = V_Bav
    else:
        descriptors['Z_eff_B'] = np.nan
    
    # Сродство к протону
    rBav = calculate_rBav(row)
    χBav = calculate_electronegativity_descriptors(row).get('χBav', np.nan)
    
    if not pd.isna(rBav) and not pd.isna(χBav):
        descriptors['proton_affinity'] = 1 / (rBav * χBav)
    else:
        descriptors['proton_affinity'] = np.nan
    
    # Энергия образования вакансии
    if not pd.isna(rBav) and not pd.isna(χBav):
        χO = 3.44
        descriptors['E_vac'] = (χBav - χO) / rBav**2
    else:
        descriptors['E_vac'] = np.nan
    
    # Комбинированный дескриптор δ * χB
    if not pd.isna(descriptors['δ']) and not pd.isna(χBav):
        descriptors['δ_χB'] = descriptors['δ'] * χBav
    else:
        descriptors['δ_χB'] = np.nan
    
    return descriptors

def calculate_t_bends_descriptors(row):
    """
    Расчёт дескрипторов, специфических для T(bends).
    
    Returns:
    --------
    dict : Словарь с дескрипторами для T(bends)
    """
    descriptors = {}
    
    # Отношение α/β (если есть оба)
    alpha = row.get('α·106 (K-1)', np.nan)
    beta = row.get('β', np.nan)
    
    if not pd.isna(alpha) and not pd.isna(beta) and beta != 0:
        descriptors['alpha_beta_ratio'] = alpha / beta
    else:
        descriptors['alpha_beta_ratio'] = np.nan
    
    # Температурная стабильность протона
    ΔH_hydr = calculate_thermodynamic_descriptors(row).get('ΔH_hydr', np.nan)
    if not pd.isna(ΔH_hydr):
        descriptors['T_stab'] = -ΔH_hydr / 8.314  # R = 8.314 Дж/(моль·К)
    else:
        descriptors['T_stab'] = np.nan
    
    # r_ratio_AB уже рассчитан в геометрических дескрипторах
    r_ratio = calculate_geometric_descriptors(row).get('r_ratio_AB', np.nan)
    descriptors['r_ratio_AB'] = r_ratio
    
    return descriptors

def calculate_compositional_descriptors(row):
    """
    Расчёт композиционных дескрипторов.
    
    Returns:
    --------
    dict : Словарь с композиционными дескрипторами
    """
    B_conc = row.get('[B\']', 0)
    D1_conc = row.get('[D1]', 0)
    D2_conc = row.get('[D2]', 0)
    
    descriptors = {}
    
    # Концентрация изовалентного допанта
    descriptors['B_prime_conc'] = B_conc if not pd.isna(B_conc) else 0
    
    # Суммарная концентрация акцепторных допантов
    descriptors['D_total'] = (D1_conc if not pd.isna(D1_conc) else 0) + (D2_conc if not pd.isna(D2_conc) else 0)
    
    # Соотношение акцепторных допантов
    if descriptors['D_total'] > 0:
        descriptors['D1_ratio'] = (D1_conc if not pd.isna(D1_conc) else 0) / descriptors['D_total']
        descriptors['D2_ratio'] = (D2_conc if not pd.isna(D2_conc) else 0) / descriptors['D_total']
    else:
        descriptors['D1_ratio'] = 0
        descriptors['D2_ratio'] = 0
    
    return descriptors

def calculate_all_descriptors(df):
    """
    Расчёт всех дескрипторов для всего DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    
    Returns:
    --------
    pd.DataFrame : DataFrame с добавленными дескрипторами
    """
    df_desc = df.copy()
    
    # Список для хранения всех дескрипторов
    all_descriptors = []
    
    for idx, row in df.iterrows():
        descriptors = {}
        
        # Геометрические
        geo = calculate_geometric_descriptors(row)
        descriptors.update(geo)
        
        # Электроотрицательные
        en = calculate_electronegativity_descriptors(row)
        descriptors.update(en)
        
        # Термодинамические
        thermo = calculate_thermodynamic_descriptors(row)
        descriptors.update(thermo)
        
        # Массовые
        mass = calculate_mass_descriptors(row)
        descriptors.update(mass)
        
        # Дефектные
        defect = calculate_defect_descriptors(row)
        descriptors.update(defect)
        
        # T(bends)-специфические
        tb = calculate_t_bends_descriptors(row)
        descriptors.update(tb)
        
        # Композиционные
        comp = calculate_compositional_descriptors(row)
        descriptors.update(comp)
        
        # Химическая формула
        descriptors['formula'] = parse_composition_formula(row)
        
        all_descriptors.append(descriptors)
    
    # Добавление дескрипторов в DataFrame
    desc_df = pd.DataFrame(all_descriptors)
    
    # Объединение с исходным DataFrame
    df_result = pd.concat([df_desc.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    
    return df_result

def get_target_variables(df):
    """
    Извлечение целевых переменных из DataFrame.
    
    Returns:
    --------
    dict : Словарь с целевыми переменными
    """
    targets = {}
    
    if 'α·106 (K-1)' in df.columns:
        targets['alpha'] = df['α·106 (K-1)']
    
    if 'β' in df.columns:
        targets['beta'] = df['β']
    
    if 'αav_mean' in df.columns:
        targets['alpha_av'] = df['αav_mean']
    
    if 'T_bends_first' in df.columns:
        targets['T_bends'] = df['T_bends_first']
    
    if 'α·106 (K-1)' in df.columns and 'β' in df.columns:
        targets['alpha_beta_ratio'] = df['α·106 (K-1)'] / df['β']
    
    return targets

# ============================================================================
# SECTION 6: CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation_matrices(df, descriptors, targets):
    """
    Расчёт различных корреляционных матриц.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными
    descriptors : list
        Список дескрипторов для корреляции
    targets : dict
        Словарь с целевыми переменными
    
    Returns:
    --------
    dict : Словарь с корреляционными матрицами
    """
    results = {}
    
    # Выборка данных
    all_vars = descriptors + list(targets.keys())
    data = df[all_vars].dropna()
    
    if len(data) < 3:
        return results
    
    # 1. Корреляция Пирсона
    pearson_corr = data.corr(method='pearson')
    results['pearson'] = pearson_corr
    
    # 2. Корреляция Спирмена
    spearman_corr = data.corr(method='spearman')
    results['spearman'] = spearman_corr
    
    # 3. p-values для Пирсона
    p_values = pd.DataFrame(np.ones_like(pearson_corr), 
                           index=pearson_corr.index, 
                           columns=pearson_corr.columns)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i != j:
                _, p_val = pearsonr(data.iloc[:, i], data.iloc[:, j])
                p_values.iloc[i, j] = p_val
    results['p_values'] = p_values
    
    # 4. Частичная корреляция (с контролем pH2O)
    if 'pH2O' in df.columns and len(data) > 5:
        try:
            partial_corr = pg.partial_corr(data=data, 
                                          x=descriptors, 
                                          y=list(targets.keys()), 
                                          covar=['pH2O'])
            results['partial'] = partial_corr
        except:
            pass
    
    # 5. Дистанционная корреляция
    try:
        from scipy.spatial.distance import pdist, squareform
        distance_corr = pd.DataFrame(index=data.columns, columns=data.columns)
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i != j:
                    dcorr = distance_correlation(data[col1].values, data[col2].values)
                    distance_corr.iloc[i, j] = dcorr
                else:
                    distance_corr.iloc[i, j] = 1.0
        results['distance'] = distance_corr
    except:
        pass
    
    return results

def distance_correlation(x, y):
    """
    Расчёт дистанционной корреляции.
    """
    n = len(x)
    a = squareform(pdist(x.reshape(-1, 1)))
    b = squareform(pdist(y.reshape(-1, 1)))
    
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
    
    dCov = np.sqrt((A * B).sum() / (n * n))
    dVarX = np.sqrt((A * A).sum() / (n * n))
    dVarY = np.sqrt((B * B).sum() / (n * n))
    
    if dVarX > 0 and dVarY > 0:
        dCor = dCov / np.sqrt(dVarX * dVarY)
    else:
        dCor = 0
    
    return dCor

def find_top_descriptors(df, descriptors, targets, n_top=20):
    """
    Отбор топ-N дескрипторов по корреляции с целевыми переменными.
    
    Returns:
    --------
    dict : Словарь с топ-дескрипторами для каждой цели
    """
    top_descriptors = {}
    
    for target_name, target_col in targets.items():
        correlations = {}
        for desc in descriptors:
            if desc in df.columns and target_col.name in df.columns:
                data = df[[desc, target_col.name]].dropna()
                if len(data) > 2:
                    corr, _ = pearsonr(data[desc].values, data[target_col.name].values)
                    correlations[desc] = abs(corr)
        
        # Сортировка по абсолютной корреляции
        sorted_descs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_descriptors[target_name] = sorted_descs[:n_top]
    
    return top_descriptors

def calculate_vif(df, descriptors):
    """
    Расчёт VIF (Variance Inflation Factor) для проверки мультиколлинеарности.
    """
    data = df[descriptors].dropna()
    if len(data) < 3:
        return {}
    
    vif_data = pd.DataFrame()
    vif_data['Variable'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    
    return vif_data

# ============================================================================
# SECTION 7: PCA AND CLUSTERING
# ============================================================================

def perform_pca_analysis(df, descriptors, n_components=None):
    """
    Выполнение PCA-анализа.
    
    Returns:
    --------
    dict : Результаты PCA
    """
    # Подготовка данных
    data = df[descriptors].dropna()
    
    if len(data) < 3:
        return None
    
    # Стандартизация
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Определение оптимального числа компонент
    if n_components is None:
        n_components = min(10, len(data.columns), len(data))
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Создание DataFrame с результатами
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=data.index
    )
    
    # Объяснённая дисперсия
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    results = {
        'pca': pca,
        'pca_df': pca_df,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'loadings': pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
            index=descriptors
        ),
        'scaler': scaler,
        'data_scaled': data_scaled,
        'n_components': pca_result.shape[1]
    }
    
    return results

def perform_clustering(df, descriptors, method='kmeans', n_clusters=None):
    """
    Выполнение кластеризации.
    
    Returns:
    --------
    dict : Результаты кластеризации
    """
    # Подготовка данных
    data = df[descriptors].dropna()
    
    if len(data) < 3:
        return None
    
    # Стандартизация
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Определение оптимального числа кластеров для K-means
    if n_clusters is None and method == 'kmeans':
        max_clusters = min(10, len(data) // 2)
        silhouette_scores = []
        davies_scores = []
        calinski_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            
            silhouette_scores.append(silhouette_score(data_scaled, labels))
            davies_scores.append(davies_bouldin_score(data_scaled, labels))
            calinski_scores.append(calinski_harabasz_score(data_scaled, labels))
        
        # Выбор оптимального K по силуэту
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
        else:
            optimal_k = 3
    else:
        optimal_k = n_clusters if n_clusters else 3
    
    # Выполнение кластеризации
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(data_scaled)
        results = {
            'method': 'kmeans',
            'labels': labels,
            'n_clusters': optimal_k,
            'clusterer': clusterer,
            'silhouette': silhouette_score(data_scaled, labels),
            'davies_bouldin': davies_bouldin_score(data_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(data_scaled, labels),
        }
    
    elif method == 'dbscan':
        # Поиск оптимальных параметров DBSCAN
        eps_values = np.linspace(0.1, 2.0, 20)
        best_score = -1
        best_eps = 0.5
        
        for eps in eps_values:
            clusterer = DBSCAN(eps=eps, min_samples=3)
            labels = clusterer.fit_predict(data_scaled)
            n_clusters_db = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters_db > 1:
                try:
                    score = silhouette_score(data_scaled[labels != -1], labels[labels != -1])
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                except:
                    pass
        
        clusterer = DBSCAN(eps=best_eps, min_samples=3)
        labels = clusterer.fit_predict(data_scaled)
        
        results = {
            'method': 'dbscan',
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'clusterer': clusterer,
            'eps': best_eps,
            'silhouette': best_score if best_score > 0 else None,
        }
    
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=optimal_k)
        labels = clusterer.fit_predict(data_scaled)
        
        # Линковка для дендрограммы
        linkage_matrix = linkage(data_scaled, method='ward')
        
        results = {
            'method': 'hierarchical',
            'labels': labels,
            'n_clusters': optimal_k,
            'clusterer': clusterer,
            'linkage_matrix': linkage_matrix,
            'silhouette': silhouette_score(data_scaled, labels),
        }
    
    results['data_scaled'] = data_scaled
    results['data'] = data
    results['scaler'] = scaler
    
    return results

def perform_tsne_analysis(df, descriptors, n_components=2, perplexity=30):
    """
    Выполнение t-SNE анализа.
    """
    data = df[descriptors].dropna()
    
    if len(data) < 5:
        return None
    
    # Стандартизация
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # t-SNE
    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(data) // 4), random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)
    
    results = {
        'tsne': tsne,
        'tsne_df': pd.DataFrame(tsne_result, columns=['t-SNE 1', 't-SNE 2'], index=data.index),
        'perplexity': perplexity,
    }
    
    return results

def perform_umap_analysis(df, descriptors, n_components=2):
    """
    Выполнение UMAP анализа.
    """
    try:
        data = df[descriptors].dropna()
        
        if len(data) < 5:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # UMAP
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        umap_result = reducer.fit_transform(data_scaled)
        
        results = {
            'umap': reducer,
            'umap_df': pd.DataFrame(umap_result, columns=['UMAP 1', 'UMAP 2'], index=data.index),
        }
        
        return results
    except:
        return None

# ============================================================================
# SECTION 8: VISUALIZATION FUNCTIONS
# ============================================================================

def create_distribution_plots(df, variables, plot_type='histogram'):
    """
    Создание графиков распределений.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(variables[:6]):
        if var in df.columns:
            data = df[var].dropna()
            if len(data) > 0:
                if plot_type == 'histogram':
                    axes[i].hist(data, bins=20, edgecolor='black', alpha=0.7)
                    axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
                elif plot_type == 'boxplot':
                    axes[i].boxplot(data)
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frequency' if plot_type == 'histogram' else 'Value')
                axes[i].set_title(f'{var} Distribution')
                axes[i].legend()
    
    # Удаление пустых подграфиков
    for j in range(i+1, 6):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig

def create_correlation_matrix_plot(corr_matrix, title='Correlation Matrix'):
    """
    Создание тепловой карты корреляционной матрицы.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=True, fmt='.2f', annot_kws={'size': 8},
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_correlation_network(corr_matrix, threshold=0.5):
    """
    Создание сетевого графа корреляций.
    """
    G = nx.Graph()
    
    # Добавление узлов
    for node in corr_matrix.columns:
        G.add_node(node)
    
    # Добавление рёбер
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    pos = nx.spring_layout(G, k=2, seed=42)
    
    # Рисование узлов и рёбер
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                          edge_color=weights, edge_cmap=plt.cm.RdYlGn,
                          ax=ax)
    
    ax.set_title(f'Correlation Network (|corr| > {threshold})', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

def create_pairplot(df, features, hue=None):
    """
    Создание Pairplot с цветовой кодировкой.
    """
    if len(features) < 2:
        return None
    
    data = df[features].dropna()
    
    if hue and hue in df.columns:
        hue_data = df.loc[data.index, hue]
        data['hue'] = hue_data
        
        fig = sns.pairplot(data, vars=features, hue='hue', 
                          diag_kind='kde', plot_kws={'alpha': 0.6})
    else:
        fig = sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6})
    
    return fig

def create_scatter_regression(df, x_var, y_var, color_var=None):
    """
    Создание scatter plot с регрессионной линией.
    """
    data = df[[x_var, y_var]].dropna()
    
    if len(data) < 3:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    if color_var and color_var in df.columns:
        colors = df.loc[data.index, color_var]
        scatter = ax.scatter(data[x_var], data[y_var], c=colors, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=color_var)
    else:
        ax.scatter(data[x_var], data[y_var], alpha=0.7, color='#2C3E50')
    
    # Регрессия
    X = data[x_var].values.reshape(-1, 1)
    y = data[y_var].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    x_range = np.linspace(data[x_var].min(), data[x_var].max(), 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    
    ax.plot(x_range, y_pred, color='red', linewidth=2, label='Regression')
    
    # Статистика
    r2 = reg.score(X, y)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_var, fontweight='bold')
    ax.set_ylabel(y_var, fontweight='bold')
    ax.set_title(f'{y_var} vs {x_var}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pca_biplot(pca_results, target_var=None):
    """
    Создание PCA Biplot.
    """
    if pca_results is None:
        return None
    
    pca_df = pca_results['pca_df']
    loadings = pca_results['loadings']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Проекция образцов
    if target_var and target_var in pca_df.index:
        colors = pca_df[target_var]
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=colors, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=target_var)
    else:
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, color='#2C3E50')
    
    # Нагрузки переменных
    for i, var in enumerate(loadings.index):
        ax.arrow(0, 0, loadings.iloc[i, 0]*2, loadings.iloc[i, 1]*2,
                head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.5)
        ax.text(loadings.iloc[i, 0]*2.2, loadings.iloc[i, 1]*2.2, var,
               fontsize=9, color='red')
    
    ax.set_xlabel(f'PC1 ({pca_results["explained_variance"][0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca_results["explained_variance"][1]*100:.1f}%)', fontweight='bold')
    ax.set_title('PCA Biplot', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pca_3d_scatter(pca_results, target_var=None):
    """
    Создание 3D PCA scatter plot с использованием Plotly.
    """
    if pca_results is None:
        return None
    
    pca_df = pca_results['pca_df']
    
    if pca_df.shape[1] < 3:
        return None
    
    # Создание фигуры
    fig = go.Figure()
    
    # Добавление точек
    if target_var and target_var in pca_df.columns:
        colors = pca_df[target_var]
        fig.add_trace(go.Scatter3d(
            x=pca_df['PC1'], y=pca_df['PC2'], z=pca_df['PC3'],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=target_var)
            ),
            text=pca_df.index,
            hoverinfo='text'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=pca_df['PC1'], y=pca_df['PC2'], z=pca_df['PC3'],
            mode='markers',
            marker=dict(size=8, color='#2C3E50'),
            text=pca_df.index,
            hoverinfo='text'
        ))
    
    # Настройки
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({pca_results["explained_variance"][0]*100:.1f}%)',
            yaxis_title=f'PC2 ({pca_results["explained_variance"][1]*100:.1f}%)',
            zaxis_title=f'PC3 ({pca_results["explained_variance"][2]*100:.1f}%)'
        ),
        title='PCA 3D Projection',
        width=800,
        height=700
    )
    
    return fig

def create_tsne_plot(tsne_results, target_var=None):
    """
    Создание t-SNE проекции.
    """
    if tsne_results is None:
        return None
    
    tsne_df = tsne_results['tsne_df']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if target_var and target_var in tsne_df.index:
        colors = tsne_df[target_var]
        scatter = ax.scatter(tsne_df['t-SNE 1'], tsne_df['t-SNE 2'], 
                            c=colors, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=target_var)
    else:
        ax.scatter(tsne_df['t-SNE 1'], tsne_df['t-SNE 2'], 
                  alpha=0.7, color='#2C3E50')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE Projection (perplexity={tsne_results["perplexity"]})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_concentration_heatmap(df, x_var, y_var, target_var, filter_data=None):
    """
    Создание концентрационной тепловой карты.
    """
    data = df[[x_var, y_var, target_var]].dropna()
    
    if filter_data is not None:
        data = data[data.index.isin(filter_data.index)]
    
    if len(data) < 10:
        return None
    
    # Создание сетки
    x_min, x_max = data[x_var].min(), data[x_var].max()
    y_min, y_max = data[y_var].min(), data[y_var].max()
    
    # Использование RBF интерполяции
    try:
        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        rbf = RBFInterpolator(data[[x_var, y_var]].values, data[target_var].values)
        Z_grid = rbf(np.column_stack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label=target_var)
        
        # Добавление точек
        ax.scatter(data[x_var], data[y_var], c='white', s=20, alpha=0.5, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_var, fontweight='bold')
        ax.set_ylabel(y_var, fontweight='bold')
        ax.set_title(f'{target_var} Concentration Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    except:
        return None

def create_bubble_chart(df, x_var, y_var, color_var=None, size_var=None, shape_var=None):
    """
    Создание многомерной пузырьковой диаграммы с Plotly.
    """
    data = df[[x_var, y_var]].dropna()
    
    if len(data) < 3:
        return None
    
    fig = go.Figure()
    
    # Определение цвета
    if color_var and color_var in df.columns:
        colors = df.loc[data.index, color_var]
    else:
        colors = '#2C3E50'
    
    # Определение размера
    if size_var and size_var in df.columns:
        sizes = df.loc[data.index, size_var]
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 30 + 10
    else:
        sizes = 15
    
    # Определение формы
    if shape_var and shape_var in df.columns:
        shapes = df.loc[data.index, shape_var]
        # Преобразование категорий в символы маркеров
        shape_map = {cat: ['circle', 'square', 'diamond', 'cross', 'star'][i % 5] 
                    for i, cat in enumerate(shapes.unique())}
        marker_symbols = [shape_map[cat] for cat in shapes]
    else:
        marker_symbols = 'circle'
    
    # Создание графика
    fig.add_trace(go.Scatter(
        x=data[x_var],
        y=data[y_var],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale='Viridis' if isinstance(colors, pd.Series) else None,
            showscale=True if isinstance(colors, pd.Series) else False,
            colorbar=dict(title=color_var) if isinstance(colors, pd.Series) else None,
            symbol=marker_symbols,
            line=dict(width=1, color='black'),
        ),
        text=[f'<b>Sample {i}</b><br>' + 
              f'{x_var}: {data[x_var].iloc[i]:.3f}<br>' +
              f'{y_var}: {data[y_var].iloc[i]:.3f}' +
              (f'<br>{color_var}: {colors.iloc[i]:.3f}' if isinstance(colors, pd.Series) else '') +
              (f'<br>{size_var}: {df.loc[data.index[i], size_var]:.3f}' if size_var else '')
              for i in range(len(data))],
        hoverinfo='text'
    ))
    
    # Настройки
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title=y_var,
        title=f'{y_var} vs {x_var}',
        width=900,
        height=700,
        hovermode='closest',
        legend=dict(x=1.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
    )
    
    return fig

def create_alpha_beta_compromise(df, alpha_var='α·106 (K-1)', beta_var='β'):
    """
    Создание диаграммы компромисса α vs β.
    """
    data = df[[alpha_var, beta_var]].dropna()
    
    if len(data) < 3:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot с цветом по B-катиону
    if 'B' in df.columns:
        colors = df.loc[data.index, 'B']
        unique_b = colors.unique()
        color_map = {b: COLOR_PALETTES['B_cation'].get(b, '#95A5A6') for b in unique_b}
        colors_hex = [color_map.get(b, '#95A5A6') for b in colors]
        
        for b in unique_b:
            mask = colors == b
            ax.scatter(data.loc[mask, alpha_var], data.loc[mask, beta_var],
                      label=b, s=50, alpha=0.7, color=color_map.get(b, '#95A5A6'))
    else:
        ax.scatter(data[alpha_var], data[beta_var], s=50, alpha=0.7, color='#2C3E50')
    
    # Целевая область
    ax.axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='β = 0.02')
    ax.axvline(x=11, color='blue', linestyle='--', alpha=0.5, label='α = 11 (10⁻⁶ K⁻¹)')
    
    ax.set_xlabel('α (10⁻⁶ K⁻¹)', fontweight='bold')
    ax.set_ylabel('β', fontweight='bold')
    ax.set_title('Thermal vs Chemical Expansion', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_cluster_profiles(cluster_results, descriptors):
    """
    Создание профилей кластеров.
    """
    if cluster_results is None:
        return None
    
    labels = cluster_results['labels']
    data = cluster_results['data']
    
    # Создание DataFrame с кластерами
    cluster_df = data.copy()
    cluster_df['Cluster'] = labels
    
    # Средние значения по кластерам
    cluster_means = cluster_df.groupby('Cluster').mean()
    
    # Стандартизация для визуализации
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means.T).T,
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Тепловая карта
    sns.heatmap(cluster_means_scaled.T, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, cbar_kws={'label': 'Standardized Value'},
                ax=ax)
    
    ax.set_title('Cluster Profiles', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Descriptor')
    
    plt.tight_layout()
    return fig

def create_feature_importance(df, descriptors, target_var):
    """
    Создание графика важности дескрипторов с использованием Random Forest.
    """
    data = df[descriptors + [target_var]].dropna()
    
    if len(data) < 10:
        return None
    
    X = data[descriptors].values
    y = data[target_var].values
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Важность признаков
    importance = rf.feature_importances_
    
    # Сортировка
    sorted_idx = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(sorted_idx))
    ax.barh(y_pos, importance[sorted_idx], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([descriptors[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance for {target_var}')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def create_radar_chart(cluster_results, descriptors):
    """
    Создание радарной диаграммы для кластеров.
    """
    if cluster_results is None:
        return None
    
    labels = cluster_results['labels']
    data = cluster_results['data']
    
    # Создание DataFrame с кластерами
    cluster_df = data.copy()
    cluster_df['Cluster'] = labels
    
    # Средние значения по кластерам
    cluster_means = cluster_df.groupby('Cluster').mean()
    
    # Стандартизация
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means.T).T,
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    # Выбор топ-8 дескрипторов
    selected_descriptors = descriptors[:8]
    cluster_means_selected = cluster_means_scaled[selected_descriptors]
    
    fig = go.Figure()
    
    for cluster in cluster_means_selected.index:
        values = cluster_means_selected.loc[cluster].values.tolist()
        # Добавление первого значения для закрытия полигона
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=selected_descriptors + [selected_descriptors[0]],
            name=f'Cluster {cluster}',
            fill='toself',
            opacity=0.5
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-2, 2]
            )),
        showlegend=True,
        title='Cluster Radar Chart',
        width=800,
        height=700
    )
    
    return fig

def create_parallel_coordinates(df, descriptors, target_var):
    """
    Создание параллельных координат с фильтрацией.
    """
    data = df[descriptors + [target_var]].dropna()
    
    if len(data) < 3:
        return None
    
    # Выбор ограниченного числа дескрипторов для читаемости
    selected_descriptors = descriptors[:8] + [target_var]
    data_selected = data[selected_descriptors]
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=data_selected[target_var],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=target_var)
        ),
        dimensions=[dict(
            label=col,
            values=data_selected[col],
            range=[data_selected[col].min(), data_selected[col].max()]
        ) for col in data_selected.columns]
    ))
    
    fig.update_layout(
        title='Parallel Coordinates Plot',
        width=1000,
        height=600,
    )
    
    return fig

def create_swarm_plot(df, x_var, y_var):
    """
    Создание swarm plot.
    """
    if x_var not in df.columns or y_var not in df.columns:
        return None
    
    data = df[[x_var, y_var]].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.swarmplot(data=data, x=x_var, y=y_var, size=5, alpha=0.7, ax=ax)
    sns.violinplot(data=data, x=x_var, y=y_var, inner=None, alpha=0.3, ax=ax)
    
    ax.set_xlabel(x_var, fontweight='bold')
    ax.set_ylabel(y_var, fontweight='bold')
    ax.set_title(f'{y_var} vs {x_var}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# SECTION 9: FILTERING SYSTEM
# ============================================================================

def apply_filters(df, filters):
    """
    Применение системы фильтрации к данным.
    """
    df_filtered = df.copy()
    
    # Базовые фильтры
    if 'method' in filters and filters['method']:
        df_filtered = df_filtered[df_filtered['method'].isin(filters['method'])]
    
    if 'A_cation' in filters and filters['A_cation']:
        df_filtered = df_filtered[df_filtered['A'].isin(filters['A_cation'])]
    
    if 'B_cation' in filters and filters['B_cation']:
        df_filtered = df_filtered[df_filtered['B'].isin(filters['B_cation'])]
    
    # Расширенные фильтры
    if 'delta_range' in filters:
        min_d, max_d = filters['delta_range']
        df_filtered = df_filtered[(df_filtered['δ'] >= min_d) & (df_filtered['δ'] <= max_d)]
    
    if 'ph2o_range' in filters:
        min_p, max_p = filters['ph2o_range']
        df_filtered = df_filtered[(df_filtered['pH2O'] >= min_p) & (df_filtered['pH2O'] <= max_p)]
    
    if 'temp_range' in filters:
        min_t, max_t = filters['temp_range']
        if 'T_min' in df_filtered.columns and 'T_max' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['T_min'] >= min_t) & (df_filtered['T_max'] <= max_t)]
    
    if 'has_t_bends' in filters and filters['has_t_bends']:
        df_filtered = df_filtered[df_filtered['T_bends_count'] > 0]
    
    # Дескрипторные фильтры
    if 'rAav_range' in filters:
        min_r, max_r = filters['rAav_range']
        if 'rAav' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['rAav'] >= min_r) & (df_filtered['rAav'] <= max_r)]
    
    if 't_range' in filters:
        min_t, max_t = filters['t_range']
        if 't' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['t'] >= min_t) & (df_filtered['t'] <= max_t)]
    
    if 'chiBav_range' in filters:
        min_c, max_c = filters['chiBav_range']
        if 'χBav' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['χBav'] >= min_c) & (df_filtered['χBav'] <= max_c)]
    
    return df_filtered

# ============================================================================
# SECTION 10: MAIN APPLICATION
# ============================================================================

def main():
    """
    Основная функция приложения.
    """
    st.title("🧪 Proton-Conducting Perovskites Analysis Platform")
    st.markdown("""
    **Interactive analysis of thermal and chemical expansion of proton-conducting perovskite oxides**
    
    *Upload your data in the specified format, calculate descriptors, and perform comprehensive analysis*
    """)
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Global Filters")
        
        # Фильтры будут добавлены после загрузки данных
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload Data",
        "🔬 Descriptors",
        "📊 Correlations",
        "🧬 PCA & Clustering",
        "📈 Visualizations",
        "💾 Export"
    ])
    
    # ========================================================================
    # TAB 1: UPLOAD DATA
    # ========================================================================
    with tab1:
        st.header("📤 Upload Data")
        st.markdown("""
        **Paste your data in the following format:**
        
        Columns: `№ A A' B B' D1 D2 [A'] [B'] [D1] [D2] δ method β ∆T, °C α·106 (K-1) T(bends), °C αav·106 (K-1) pH2O Ref`
        
        Use tab-separated values (TSV format). Replace missing values with `-`.
        """)
        
        # Text input area
        text_data = st.text_area(
            "Paste your data here:",
            height=300,
            placeholder="№\tA\tA'\tB\tB'\tD1\tD2\t[A']\t[B']\t[D1]\t[D2]\tδ\tmethod\tβ\t∆T, °C\tα·106 (K-1)\tT(bends), °C\tαav·106 (K-1)\tpH2O\tRef\n1\tBa\t-\tCe\tZr\tY\tYb\t0\t0.1\t0.1\t0.1\t0.1\tdilatometry\t0.0073\t27-1000\t10.6\t400;600\t10.6;4.73;10.1\t0.0001\t10.15826/chimtech.2024.11.4.22"
        )
        
        # Load button
        if st.button("🚀 Load Data", type="primary"):
            if text_data:
                df = parse_uploaded_data(text_data)
                if df is not None and not df.empty:
                    st.session_state['df_raw'] = df
                    st.session_state['data_loaded'] = True
                    
                    # Calculate descriptors
                    with st.spinner("Calculating descriptors..."):
                        df_desc = calculate_all_descriptors(df)
                        st.session_state['df'] = df_desc
                        st.session_state['descriptors_calculated'] = True
                    
                    st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df_desc.columns)} columns.")
                    
                    # Show preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df_desc.head(10), use_container_width=True)
                    
                    # Show statistics
                    st.subheader("📊 Basic Statistics")
                    numeric_cols = df_desc.select_dtypes(include=[np.number]).columns[:10]
                    st.dataframe(df_desc[numeric_cols].describe(), use_container_width=True)
                else:
                    st.error("❌ Failed to parse data. Please check the format.")
            else:
                st.warning("⚠️ Please paste your data first.")
        
        # Example data button
        if st.button("📝 Load Example Data"):
            example_data = """№	A	A'	B	B'	D1	D2	[A']	[B']	[D1]	[D2]	δ	method	β	∆T, °C	α·106 (K-1)	T(bends), °C	αav·106 (K-1)	pH2O	Ref
1	Ba	-	Ce	Zr	Y	Yb	0	0.1	0.1	0.1	0.1	dilatometry	0.0073	27-1000	10.6	400;600	10.6;4.73;10.1	0.0001	10.15826/chimtech.2024.11.4.22
2	Ba	-	Ce	Zr	Y	Yb	0	0.1	0.1	0.1	0.1	HT XRD	0.0317	27-1000	10.6	300	10.7;8.7	0.02	10.15826/chimtech.2024.11.4.22
3	Ba	-	Ce	Zr	Y	-	0	0.1	0.1	0	0.05	HT ND	-	20-900	11.2	-	-	0.00106	10.1021/acs.jpcc.1c08334
4	Ba	-	Ce	Zr	Y	-	0	0	0.1	0	0.05	dilatometry	0.019	430-630	-	450	-	0.00106	10.1021/acs.jpcc.1c08334
5	Ba	-	Ce	Zr	Y	-	0	0.3	0.1	0	0.05	dilatometry	0.019	430-631	-	-	-	0.00106	10.1021/acs.jpcc.1c08334
6	Ba	-	Ce	Zr	Y	-	0	0.6	0.1	0	0.05	dilatometry	0.023	430-632	-	-	-	0.00106	10.1021/acs.jpcc.1c08334
7	Ba	-	Ce	Zr	Y	-	0	0.9	0.1	0	0.05	dilatometry	0.49	430-633	-	-	-	0.00106	10.1021/acs.jpcc.1c08334
8	Ba	-	Ce	-	Sm	-	0	0	0.2	0	0.1	dilatometry	0.097	100-900	11.5	620	11.5;10.3	-	10.1016/j.jpowsour.2012.07.120
9	Ba	-	Ce	-	Nd	-	0	0	0.2	0	0.1	dilatometry	0.0165	60-900	14	700	14;8.2	0.018	10.1016/j.jpowsour.2014.05.070
10	Ba	-	Ce	Zr	Y	-	0	0	0.2	0	0.1	dilatometry	0.0091	100-900	11.6	620	11.6;8.3	0.018	10.1016/j.jpowsour.2014.12.024
11	Ba	-	Ce	Zr	Y	-	0	0.1	0.2	0	0.1	dilatometry	-	100-900	11.3	630	11.3;8.4	0.018	10.1016/j.jpowsour.2014.12.024
12	Ba	-	Ce	Zr	Y	-	0	0.2	0.2	0	0.1	dilatometry	-	100-900	11.32	620	11.32;8.4	0.018	10.1016/j.jpowsour.2014.12.024
13	Ba	-	Ce	Zr	Y	-	0	0.3	0.2	0	0.1	dilatometry	-	100-900	10.8	575	10.8;8.5	0.018	10.1016/j.jpowsour.2014.12.024
14	Ba	-	Ce	Zr	Y	-	0	0.4	0.2	0	0.1	dilatometry	-	100-900	10.9	630	10.9;8.5	0.018	10.1016/j.jpowsour.2014.12.024
15	Ba	-	Ce	Zr	Y	-	0	0.5	0.2	0	0.1	dilatometry	-	100-900	9.32	-	9.32	0.018	10.1016/j.jpowsour.2014.12.024"""
            
            # Заполняем текстовое поле через session_state
            st.session_state['text_input'] = example_data
            st.rerun()
    
    # ========================================================================
    # TAB 2: DESCRIPTORS
    # ========================================================================
    with tab2:
        st.header("🔬 Descriptor Engine")
        
        if 'df' not in st.session_state or not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please upload data first in the 'Upload Data' tab.")
        else:
            df = st.session_state['df']
            
            # Descriptor statistics
            st.subheader("📊 Descriptor Statistics")
            
            # Select descriptor groups
            desc_groups = {
                'Geometric': ['rAav', 'rBav', 't', 'D_t', 'octahedral_factor', 'Δr_AB', 'Δr_AB_norm', 'σ²_rB', 'V_cell', 'V_free'],
                'Electronegativity': ['χAav', 'χBav', 'Δχ_AB', 'χ_ratio_AB', 'ionicity_AO', 'ionicity_BO', 'acidity_AO', 'acidity_BO', 'Δacidity'],
                'Thermodynamic': ['S_config_A', 'S_config_B', 'V_Bav', 'Vo_proxy', 'E_BO', 'ρ', 'ΔH_hydr'],
                'Mass': ['M_Aav', 'M_Bav', 'M_total', 'M_ratio_AB', 'M_rA', 'M_χA'],
                'Defect': ['δ', 'δ_calc', 'Z_eff_B', 'proton_affinity', 'E_vac', 'δ_χB'],
                'T_bends': ['alpha_beta_ratio', 'T_stab', 'r_ratio_AB'],
                'Compositional': ['B_prime_conc', 'D_total', 'D1_ratio', 'D2_ratio']
            }
            
            # Show descriptor table
            desc_cols = []
            for group, cols in desc_groups.items():
                desc_cols.extend([c for c in cols if c in df.columns])
            
            if desc_cols:
                st.dataframe(df[desc_cols].describe(), use_container_width=True)
            
            # Show full descriptor table
            with st.expander("📋 Full Descriptor Table"):
                st.dataframe(df[desc_cols], use_container_width=True)
            
            # Export descriptors
            if st.button("💾 Export Descriptors"):
                csv = df[desc_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="descriptors.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # TAB 3: CORRELATIONS
    # ========================================================================
    with tab3:
        st.header("📊 Correlation Analysis")
        
        if 'df' not in st.session_state or not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please upload data first in the 'Upload Data' tab.")
        else:
            df = st.session_state['df']
            
            # Select descriptors for correlation
            desc_cols = [c for c in df.columns if c not in ['№', 'method', 'Ref', 'formula', 
                                                           'A', "A'", 'B', "B'", 'D1', 'D2',
                                                           'T_bends_list', 'αav_list', 'T_bends_count',
                                                           'T_bends_first', 'T_bends_last', 'αav_mean',
                                                           'αav_min', 'αav_max', 'T_min', 'T_max']]
            desc_cols = [c for c in desc_cols if df[c].dtype in ['float64', 'int64']]
            
            # Targets
            targets = ['α·106 (K-1)', 'β', 'αav_mean', 'T_bends_first']
            targets = [t for t in targets if t in df.columns]
            
            if len(desc_cols) < 5:
                st.warning("Not enough numeric columns for correlation analysis.")
            else:
                # Select descriptors for analysis
                selected_descs = st.multiselect(
                    "Select descriptors for correlation analysis:",
                    options=desc_cols,
                    default=desc_cols[:10] if len(desc_cols) > 10 else desc_cols
                )
                
                if selected_descs:
                    # Correlation analysis
                    with st.spinner("Calculating correlations..."):
                        corr_results = calculate_correlation_matrices(df, selected_descs, targets)
                    
                    if corr_results:
                        # Display correlation matrices
                        st.subheader("📊 Correlation Matrices")
                        
                        # Pearson
                        if 'pearson' in corr_results:
                            st.write("**Pearson Correlation**")
                            fig = create_correlation_matrix_plot(corr_results['pearson'], 'Pearson Correlation Matrix')
                            st.pyplot(fig)
                            plt.close()
                        
                        # Spearman
                        if 'spearman' in corr_results:
                            st.write("**Spearman Correlation**")
                            fig = create_correlation_matrix_plot(corr_results['spearman'], 'Spearman Correlation Matrix')
                            st.pyplot(fig)
                            plt.close()
                        
                        # Correlation network
                        st.subheader("🌐 Correlation Network")
                        threshold = st.slider("Correlation threshold:", 0.1, 0.9, 0.5, 0.1)
                        
                        if 'pearson' in corr_results:
                            fig = create_correlation_network(corr_results['pearson'], threshold)
                            st.pyplot(fig)
                            plt.close()
                        
                        # Top descriptors
                        st.subheader("🏆 Top Descriptors")
                        
                        for target in targets:
                            if target in df.columns:
                                with st.expander(f"Top descriptors for {target}"):
                                    # Calculate correlations with target
                                    corr_data = {}
                                    for desc in selected_descs:
                                        if desc in df.columns and target in df.columns:
                                            data = df[[desc, target]].dropna()
                                            if len(data) > 2:
                                                corr, p_val = pearsonr(data[desc].values, data[target].values)
                                                corr_data[desc] = {'correlation': corr, 'p_value': p_val}
                                    
                                    # Sort by absolute correlation
                                    sorted_corr = sorted(corr_data.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
                                    
                                    top_df = pd.DataFrame([{'Descriptor': k, 'Correlation': v['correlation'], 
                                                          'p-value': v['p_value']} for k, v in sorted_corr[:15]])
                                    st.dataframe(top_df, use_container_width=True)
                        
                        # VIF analysis
                        st.subheader("📈 Multicollinearity Check (VIF)")
                        if st.button("Calculate VIF"):
                            vif_results = calculate_vif(df, selected_descs[:10])
                            if vif_results:
                                st.dataframe(vif_results, use_container_width=True)
                                
                                # Highlight high VIF
                                high_vif = vif_results[vif_results['VIF'] > 5]
                                if not high_vif.empty:
                                    st.warning(f"⚠️ {len(high_vif)} variables have VIF > 5 (high multicollinearity)")
                                    st.dataframe(high_vif, use_container_width=True)
                    else:
                        st.warning("Not enough data for correlation analysis.")
    
    # ========================================================================
    # TAB 4: PCA & CLUSTERING
    # ========================================================================
    with tab4:
        st.header("🧬 PCA & Clustering Analysis")
        
        if 'df' not in st.session_state or not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please upload data first in the 'Upload Data' tab.")
        else:
            df = st.session_state['df']
            
            # Select descriptors for PCA
            desc_cols = [c for c in df.columns if c not in ['№', 'method', 'Ref', 'formula', 
                                                           'A', "A'", 'B', "B'", 'D1', 'D2',
                                                           'T_bends_list', 'αav_list', 'T_bends_count',
                                                           'T_bends_first', 'T_bends_last', 'αav_mean',
                                                           'αav_min', 'αav_max', 'T_min', 'T_max']]
            desc_cols = [c for c in desc_cols if df[c].dtype in ['float64', 'int64']]
            
            if len(desc_cols) < 3:
                st.warning("Not enough numeric columns for PCA analysis.")
            else:
                # Select descriptors for PCA
                pca_descs = st.multiselect(
                    "Select descriptors for PCA:",
                    options=desc_cols,
                    default=desc_cols[:10] if len(desc_cols) > 10 else desc_cols
                )
                
                if pca_descs and len(pca_descs) >= 2:
                    # PCA Analysis
                    with st.spinner("Performing PCA..."):
                        pca_results = perform_pca_analysis(df, pca_descs)
                    
                    if pca_results:
                        st.subheader("📊 PCA Results")
                        
                        # Explained variance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        components = range(1, len(pca_results['explained_variance']) + 1)
                        ax.bar(components, pca_results['explained_variance'] * 100, alpha=0.7, label='Individual')
                        ax.plot(components, pca_results['cumulative_variance'] * 100, 'o-', color='red', label='Cumulative')
                        ax.set_xlabel('Principal Component')
                        ax.set_ylabel('Explained Variance (%)')
                        ax.set_title('PCA Explained Variance')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        # PCA Biplot
                        st.subheader("🎯 PCA Biplot")
                        target_var = st.selectbox("Color by:", ['None'] + pca_descs)
                        target_var = None if target_var == 'None' else target_var
                        
                        fig = create_pca_biplot(pca_results, target_var)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                        
                        # 3D PCA
                        st.subheader("🎲 3D PCA Projection")
                        if pca_results['n_components'] >= 3:
                            fig = create_pca_3d_scatter(pca_results, target_var)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 3 principal components for 3D visualization.")
                        
                        # Clustering
                        st.subheader("🧩 Clustering Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            cluster_method = st.selectbox("Clustering method:", 
                                                         ['kmeans', 'hierarchical', 'dbscan'])
                        with col2:
                            if cluster_method == 'kmeans':
                                n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                            else:
                                n_clusters = None
                        
                        if st.button("Run Clustering"):
                            with st.spinner("Performing clustering..."):
                                cluster_results = perform_clustering(df, pca_descs, cluster_method, n_clusters)
                            
                            if cluster_results:
                                st.success(f"✅ Clustering complete! {cluster_results['n_clusters']} clusters found.")
                                
                                # Add labels to DataFrame
                                if 'labels' in cluster_results:
                                    df_cluster = df.copy()
                                    df_cluster['Cluster'] = cluster_results['labels']
                                    st.session_state['cluster_labels'] = cluster_results['labels']
                                    
                                    # Show cluster statistics
                                    st.write("**Cluster Statistics:**")
                                    cluster_stats = df_cluster.groupby('Cluster').size().reset_index(name='Count')
                                    if 'silhouette' in cluster_results and cluster_results['silhouette']:
                                        cluster_stats['Silhouette'] = cluster_results['silhouette']
                                    st.dataframe(cluster_stats, use_container_width=True)
                                    
                                    # Cluster profiles
                                    fig = create_cluster_profiles(cluster_results, pca_descs[:8])
                                    if fig:
                                        st.pyplot(fig)
                                        plt.close()
                                    
                                    # Radar chart
                                    fig = create_radar_chart(cluster_results, pca_descs[:8])
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                    
                    # t-SNE
                    st.subheader("🔬 t-SNE Projection")
                    if st.button("Run t-SNE"):
                        with st.spinner("Performing t-SNE..."):
                            tsne_results = perform_tsne_analysis(df, pca_descs)
                        
                        if tsne_results:
                            target_var = st.selectbox("Color by (t-SNE):", ['None'] + pca_descs, key='tsne_color')
                            target_var = None if target_var == 'None' else target_var
                            
                            fig = create_tsne_plot(tsne_results, target_var)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                    
                    # UMAP
                    try:
                        st.subheader("🌐 UMAP Projection")
                        if st.button("Run UMAP"):
                            with st.spinner("Performing UMAP..."):
                                umap_results = perform_umap_analysis(df, pca_descs)
                            
                            if umap_results:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.scatter(umap_results['umap_df']['UMAP 1'], 
                                         umap_results['umap_df']['UMAP 2'],
                                         alpha=0.7, color='#2C3E50')
                                ax.set_xlabel('UMAP 1')
                                ax.set_ylabel('UMAP 2')
                                ax.set_title('UMAP Projection')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close()
                    except:
                        pass
    
    # ========================================================================
    # TAB 5: VISUALIZATIONS
    # ========================================================================
    with tab5:
        st.header("📈 Advanced Visualizations")
        
        if 'df' not in st.session_state or not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please upload data first in the 'Upload Data' tab.")
        else:
            df = st.session_state['df']
            
            # Get descriptor columns
            desc_cols = [c for c in df.columns if c not in ['№', 'method', 'Ref', 'formula', 
                                                           'A', "A'", 'B', "B'", 'D1', 'D2',
                                                           'T_bends_list', 'αav_list', 'T_bends_count',
                                                           'T_bends_first', 'T_bends_last', 'αav_mean',
                                                           'αav_min', 'αav_max', 'T_min', 'T_max']]
            desc_cols = [c for c in desc_cols if df[c].dtype in ['float64', 'int64']]
            
            # Target variables
            targets = ['α·106 (K-1)', 'β', 'αav_mean', 'T_bends_first']
            targets = [t for t in targets if t in df.columns]
            targets = [t for t in targets if not df[t].isna().all()]
            
            if not desc_cols or not targets:
                st.warning("Not enough data for visualizations.")
            else:
                # Create visualization type selector
                viz_type = st.selectbox(
                    "Select visualization type:",
                    [
                        "Distribution Plots",
                        "Pairplot",
                        "Scatter + Regression",
                        "Concentration Heatmap",
                        "Bubble Chart",
                        "α vs β Compromise",
                        "Feature Importance",
                        "Swarm Plot",
                        "Parallel Coordinates",
                        "Concentration Contour"
                    ]
                )
                
                # Filter controls
                st.subheader("🔍 Filters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'method' in df.columns:
                        methods = ['All'] + list(df['method'].dropna().unique())
                        selected_method = st.selectbox("Method:", methods)
                    else:
                        selected_method = 'All'
                
                with col2:
                    if 'B' in df.columns:
                        b_cations = ['All'] + list(df['B'].dropna().unique())
                        selected_b = st.selectbox("B-cation:", b_cations)
                    else:
                        selected_b = 'All'
                
                with col3:
                    if 'A' in df.columns:
                        a_cations = ['All'] + list(df['A'].dropna().unique())
                        selected_a = st.selectbox("A-cation:", a_cations)
                    else:
                        selected_a = 'All'
                
                # Apply filters
                df_filtered = df.copy()
                if selected_method != 'All':
                    df_filtered = df_filtered[df_filtered['method'] == selected_method]
                if selected_b != 'All':
                    df_filtered = df_filtered[df_filtered['B'] == selected_b]
                if selected_a != 'All':
                    df_filtered = df_filtered[df_filtered['A'] == selected_a]
                
                # Visualization based on type
                if viz_type == "Distribution Plots":
                    variables = st.multiselect("Select variables:", desc_cols, default=desc_cols[:6])
                    if variables:
                        fig = create_distribution_plots(df_filtered, variables, 'histogram')
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                
                elif viz_type == "Pairplot":
                    features = st.multiselect("Select features (2-5):", desc_cols, default=desc_cols[:4])
                    if len(features) >= 2:
                        hue = st.selectbox("Hue (optional):", ['None'] + desc_cols + ['method', 'B'])
                        hue = None if hue == 'None' else hue
                        fig = create_pairplot(df_filtered, features, hue)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                
                elif viz_type == "Scatter + Regression":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X-axis:", desc_cols)
                    with col2:
                        y_var = st.selectbox("Y-axis (target):", targets)
                    
                    if x_var and y_var:
                        color_var = st.selectbox("Color by:", ['None'] + desc_cols)
                        color_var = None if color_var == 'None' else color_var
                        
                        fig = create_scatter_regression(df_filtered, x_var, y_var, color_var)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                
                elif viz_type == "Concentration Heatmap":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_var = st.selectbox("X-axis:", desc_cols[:20])
                    with col2:
                        y_var = st.selectbox("Y-axis:", desc_cols[:20])
                    with col3:
                        target_var = st.selectbox("Color (target):", targets)
                    
                    if x_var and y_var and target_var:
                        fig = create_concentration_heatmap(df_filtered, x_var, y_var, target_var)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                
                elif viz_type == "Bubble Chart":
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        x_var = st.selectbox("X-axis:", desc_cols[:20])
                    with col2:
                        y_var = st.selectbox("Y-axis (target):", targets)
                    with col3:
                        color_var = st.selectbox("Color:", ['None'] + desc_cols[:20])
                        color_var = None if color_var == 'None' else color_var
                    with col4:
                        size_var = st.selectbox("Size:", ['None'] + desc_cols[:20])
                        size_var = None if size_var == 'None' else size_var
                    
                    if x_var and y_var:
                        shape_var = st.selectbox("Shape:", ['None', 'method', 'B'])
                        shape_var = None if shape_var == 'None' else shape_var
                        
                        fig = create_bubble_chart(df_filtered, x_var, y_var, color_var, size_var, shape_var)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "α vs β Compromise":
                    fig = create_alpha_beta_compromise(df_filtered)
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                
                elif viz_type == "Feature Importance":
                    target_var = st.selectbox("Target variable:", targets)
                    if target_var:
                        fig = create_feature_importance(df_filtered, desc_cols, target_var)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                
                elif viz_type == "Swarm Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X-axis (categorical):", ['method', 'B', 'A'] + desc_cols[:5])
                    with col2:
                        y_var = st.selectbox("Y-axis:", targets)
                    
                    if x_var and y_var:
                        fig = create_swarm_plot(df_filtered, x_var, y_var)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                
                elif viz_type == "Parallel Coordinates":
                    selected_descs = st.multiselect("Select descriptors (max 8):", desc_cols, default=desc_cols[:5])
                    target_var = st.selectbox("Target (color):", targets)
                    
                    if selected_descs and target_var:
                        fig = create_parallel_coordinates(df_filtered, selected_descs, target_var)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Concentration Contour":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_var = st.selectbox("X-axis (contour):", desc_cols[:20])
                    with col2:
                        y_var = st.selectbox("Y-axis (contour):", desc_cols[:20])
                    with col3:
                        target_var = st.selectbox("Color (target contour):", targets)
                    
                    if x_var and y_var and target_var:
                        fig = create_concentration_heatmap(df_filtered, x_var, y_var, target_var)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
    
    # ========================================================================
    # TAB 6: EXPORT
    # ========================================================================
    with tab6:
        st.header("💾 Export Results")
        
        if 'df' not in st.session_state or not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please upload data first in the 'Upload Data' tab.")
        else:
            df = st.session_state['df']
            
            st.subheader("📥 Export Data")
            
            # Export options
            export_type = st.radio("Export type:", ["Full data with descriptors", "Only descriptors", "Selected columns"])
            
            if export_type == "Full data with descriptors":
                export_df = df
            elif export_type == "Only descriptors":
                desc_cols = [c for c in df.columns if c not in ['№', 'method', 'Ref', 'formula', 
                                                               'A', "A'", 'B', "B'", 'D1', 'D2']]
                export_df = df[desc_cols]
            else:
                selected_cols = st.multiselect("Select columns:", df.columns.tolist())
                if selected_cols:
                    export_df = df[selected_cols]
                else:
                    export_df = df
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="perovskite_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                excel = export_df.to_excel(index=False, engine='openpyxl')
                st.download_button(
                    label="📥 Download Excel",
                    data=excel,
                    file_name="perovskite_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.subheader("📊 Data Summary")
            st.write(f"Total rows: {len(export_df)}")
            st.write(f"Total columns: {len(export_df.columns)}")
            
            # Show data preview
            with st.expander("📋 Preview export data"):
                st.dataframe(export_df.head(10), use_container_width=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
