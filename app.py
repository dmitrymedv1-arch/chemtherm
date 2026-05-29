import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap
import shap
import pingouin as pg
from statsmodels.stats.outliers_influence import variance_inflation_factor
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. НАСТРОЙКИ НАУЧНОГО СТИЛЯ ДЛЯ ГРАФИКОВ
# ============================================================================

def apply_scientific_style():
    """Применение единого научного стиля для всех matplotlib графиков"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#2c3e50',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': '#2c3e50',
        'ytick.color': '#2c3e50',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.major.size': 6,
        'ytick.minor.size': 3,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#2c3e50',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'lines.linewidth': 0.6,
        'lines.markersize': 5,
        'errorbar.capsize': 3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

apply_scientific_style()

# ============================================================================
# 2. ЗАГРУЗКА ДАННЫХ (ДВА НЕЗАВИСИМЫХ ВИДЖЕТА С ПРЕДЗАГРУЖЕННЫМИ МАССИВАМИ)
# ============================================================================

# Предзагруженные данные для первого листа (chem and therm expansion)
DEFAULT_THERM_DATA = """№	A	A'	B	B'	D1	D2	[A']	[B']	[D1]	[D2]	δ	rA	rA'	rAav	rB	rB'	rD1	rD2	rBav	t	rBav/rO	method	β	∆T, °C	α·106 (K-1)	T(bends), °C	αav·106 (K-1)	pH2O	Ref
1	Ba		Ce	Zr	Y	Yb		0.1	0.1	0.1		1.61		1.61	0.87	0.72	0.9	0.868	0.8578	0.942683768	0.612714286	dilatometry	0.0073	27-1000	10.6	400;600	10.6;4.73;10.1	0.0001	10.15826/chimtech.2024.11.4.22
2	Ba		Ce	Zr	Y	Yb		0.1	0.1	0.1		1.61		1.61	0.87	0.72	0.9	0.868	0.8578	0.942683768	0.612714286	HTXRD	0.0317	27-1000	10.6	300	10.7;8.7	0.02	10.15826/chimtech.2024.11.4.22
3	Ba		Ce	Zr	Y			0.1	0.1			1.61		1.61	0.87	0.72	0.9		0.858	0.942600271	0.612857143	HTND		20-900	11.2			0.00106	10.1021/acs.jpcc.1c08334
4	Ba		Ce	Zr	Y				0.1			1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.019	430-630		450		0.00106	10.1021/acs.jpcc.1c08334
5	Ba		Ce	Zr	Y				0.3			1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.019	430-631				0.00106	10.1021/acs.jpcc.1c08334
6	Ba		Ce	Zr	Y				0.6			1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.023	430-632				0.00106	10.1021/acs.jpcc.1c08334
7	Ba		Ce	Zr	Y				0.9			1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.49	430-633				0.00106	10.1021/acs.jpcc.1c08334
8	Ba		Ce		Sm				0.2			1.61		1.61	0.87		0.958		0.958	0.951155764	0.684285714	dilatometry	0.09718477	100-900	11.5	620	11.5;10.3		10.1016/j.jpowsour.2012.07.120
9	Ba		Ce		Nd				0.2			1.61		1.61	0.87		0.983		0.983	0.947926192	0.702142857	dilatometry	0.0165	60-900	14	700	14;8.2	0.018	10.1016/j.jpowsour.2014.05.070"""

# Предзагруженные данные для второго листа (phase transition)
DEFAULT_PHASE_DATA = """№	A	A'	B	B'	D1	D2	[A]	[B']	[D1]	[D2]	δ	rA	rA'	rAav	rB	rB'	rD1	rD2	rBav	t	rBav/rO	pH2O	∆T, °C	Symmetry	Phase transitions (PT)	T (PT), °C	Ref
1	Ba		Ce	Zr	Y			0.36	0.1			1.61		1.61	0.87	0.72	0.9		0.819	0.959166927	0.585		30-1000				10.1063/1.5066970
2	Ba		Zr		Y				0		0	1.61		1.61	0.72		0.9		0.72	1.003958213	0.514285714		25	Cubic	Pm-3m		10.1088/1742-6596/1967/1/012015
3	Ba		Zr		Y				0.055		0.0275	1.61		1.61	0.72		0.9		0.7299	0.999291709	0.521357143		25	Cubic	Pm-3m		10.1088/1742-6596/1967/1/012015
4	Ba		Zr		Y				0.17		0.085	1.61		1.61	0.72		0.9		0.7506	0.989673306	0.536142857		25	Cubic	Pm-3m		10.1088/1742-6596/1967/1/012015
5	Ba		Sn		Y				0		0	1.61		1.61	0.69		0.9		0.69	1.018369096	0.492857143			Orthorombic;Rhombohedral;Cubic	Pm-3m, R-3c, Imma	352;476;711	10.1111/jace.12990
6	Ba		Sn		Y				0.05		0.025	1.61		1.61	0.69		0.9		0.7005	1.013278463	0.500357143			Cubic	Pm-3m		10.1111/jace.12990
7	Ba		Sn		Y				0.1		0.05	1.61		1.61	0.69		0.9		0.711	1.008238471	0.507857143			Cubic	Pm-3m		10.1111/jace.12990
8	Ba		Sn		Y				0.2		0.1	1.61		1.61	0.69		0.9		0.732	0.998307416	0.522857143			Cubic	Pm-3m		10.1111/jace.12990
9	Ba		Sn		Y				0.3		0.15	1.61		1.61	0.69		0.9		0.753	0.988570094	0.537857143			Cubic	Pm-3m		10.1111/jace.18224"""

@st.cache_data
def load_data_from_text(therm_text, phase_text):
    """Загрузка двух независимых таблиц из текстовых виджетов"""
    try:
        df_therm = pd.read_csv(io.StringIO(therm_text), sep='\t', dtype=str)
        df_phase = pd.read_csv(io.StringIO(phase_text), sep='\t', dtype=str)
        
        # Очистка колонок
        df_therm.columns = df_therm.columns.str.strip()
        df_phase.columns = df_phase.columns.str.strip()
        
        # Замена пустых строк на NaN
        df_therm.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df_phase.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df_therm.replace('nan', np.nan, inplace=True)
        df_phase.replace('nan', np.nan, inplace=True)
        
        # Список строковых колонок
        string_cols = ['A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'method', 'Symmetry', 'Phase transitions (PT)', 'Ref']
        
        # Преобразование числовых колонок
        for col in df_therm.columns:
            if col not in string_cols and col != '№':
                df_therm[col] = pd.to_numeric(df_therm[col], errors='coerce')
        
        for col in df_phase.columns:
            if col not in string_cols and col != '№':
                df_phase[col] = pd.to_numeric(df_phase[col], errors='coerce')
        
        return df_therm, df_phase
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None, None

# ============================================================================
# 3. ДОБАВЛЕНИЕ НОВЫХ ДЕСКРИПТОРОВ (ДЛЯ КАЖДОЙ ТАБЛИЦЫ ОТДЕЛЬНО)
# ============================================================================

def add_electronegativity_descriptors(df):
    """Добавление электроотрицательности по Поллингу и производных"""
    
    chi_table = {
        'Ba': 0.89, 'Sr': 0.95, 'Ca': 1.00, 'La': 1.10, 'Ce': 1.12,
        'Zr': 1.33, 'Sn': 1.96, 'Ti': 1.54, 'Sc': 1.36, 'Y': 1.22,
        'Yb': 1.10, 'In': 1.78, 'Fe': 1.83, 'Gd': 1.20, 'Sm': 1.17,
        'Nd': 1.14, 'Eu': 1.20, 'Dy': 1.22, 'Zn': 1.65, 'Pr': 1.13,
        'Ho': 1.23, 'Tm': 1.25, 'Tb': 1.20, 'Hf': 1.30, 'Pb': 2.33,
        'Bi': 2.02, 'Al': 1.61, 'Ga': 1.81, 'Ge': 2.01, 'Si': 1.90
    }
    
    def get_chi(el):
        if pd.isna(el) or el == '' or str(el).strip() == '' or str(el).strip().lower() == 'nan':
            return np.nan
        return chi_table.get(str(el).strip(), np.nan)
    
    for pos in ['A', 'A\'', 'B', 'B\'', 'D1', 'D2']:
        if pos in df.columns:
            df[f'χ{pos}'] = df[pos].apply(get_chi)
    
    if all(col in df.columns for col in ['χA', 'χA\'', '[A\']']):
        conc_Ap = pd.to_numeric(df['[A\']'], errors='coerce').fillna(0)
        df['χAav'] = df['χA'] * (1 - conc_Ap) + df['χA\''] * conc_Ap
    
    if all(col in df.columns for col in ['χB', 'χB\'', 'χD1', 'χD2', '[B\']', '[D1]', '[D2]']):
        conc_Bp = pd.to_numeric(df['[B\']'], errors='coerce').fillna(0)
        conc_D1 = pd.to_numeric(df['[D1]'], errors='coerce').fillna(0)
        conc_D2 = pd.to_numeric(df['[D2]'], errors='coerce').fillna(0)
        
        df['χBav'] = (df['χB'] * (1 - conc_Bp - conc_D1 - conc_D2) +
                      df['χB\''] * conc_Bp +
                      df['χD1'] * conc_D1 +
                      df['χD2'] * conc_D2)
    
    if 'χAav' in df.columns and 'χBav' in df.columns:
        df['Δχ_AB'] = np.abs(df['χAav'] - df['χBav'])
        df['χ_ratio_AB'] = df['χAav'] / df['χBav']
        df['χ_total'] = (df['χAav'] + df['χBav']) / 2
        chi_O = 3.44
        df['ionicity_AO'] = 1 - np.exp(-0.25 * (df['χAav'] - chi_O)**2)
        df['ionicity_BO'] = 1 - np.exp(-0.25 * (df['χBav'] - chi_O)**2)
    
    return df

def add_geometric_descriptors(df):
    """Геометрические дескрипторы"""
    r_O = 1.4
    
    if 'rBav' in df.columns:
        if df['rBav'].dtype == 'object':
            df['rBav'] = pd.to_numeric(df['rBav'], errors='coerce')
        df['octahedral_factor'] = df['rBav'] / r_O
    
    if 't' in df.columns:
        if df['t'].dtype == 'object':
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
        df['D_t'] = np.abs(1 - df['t'])
    
    if 'rAav' in df.columns and 'rBav' in df.columns:
        if df['rAav'].dtype == 'object':
            df['rAav'] = pd.to_numeric(df['rAav'], errors='coerce')
        if df['rBav'].dtype == 'object':
            df['rBav'] = pd.to_numeric(df['rBav'], errors='coerce')
        
        df['Δr_AB'] = np.abs(df['rAav'] - df['rBav'])
        df['Δr_AB_norm'] = df['Δr_AB'] / r_O
        df['t_alt'] = (df['rAav'] + r_O) / (np.sqrt(2) * (df['rBav'] + r_O))
    
    if all(col in df.columns for col in ['rB', 'rB\'', 'rD1', 'rD2', '[B\']', '[D1]', '[D2]']):
        rad_B = pd.to_numeric(df['rB'], errors='coerce').fillna(0).values
        rad_Bp = pd.to_numeric(df['rB\''], errors='coerce').fillna(0).values
        rad_D1 = pd.to_numeric(df['rD1'], errors='coerce').fillna(0).values
        rad_D2 = pd.to_numeric(df['rD2'], errors='coerce').fillna(0).values
        conc_Bp = pd.to_numeric(df['[B\']'], errors='coerce').fillna(0).values
        conc_D1 = pd.to_numeric(df['[D1]'], errors='coerce').fillna(0).values
        conc_D2 = pd.to_numeric(df['[D2]'], errors='coerce').fillna(0).values
        
        rBav_values = df['rBav'].values if 'rBav' in df.columns else np.zeros_like(rad_B)
        if isinstance(rBav_values, np.ndarray) and rBav_values.dtype == 'object':
            rBav_values = pd.to_numeric(rBav_values, errors='coerce').fillna(0).values
        
        sum_sq = (rad_B**2 * (1 - conc_Bp - conc_D1 - conc_D2) +
                  rad_Bp**2 * conc_Bp +
                  rad_D1**2 * conc_D1 +
                  rad_D2**2 * conc_D2)
        df['σ²_rB'] = sum_sq - rBav_values**2
    
    if all(col in df.columns for col in ['rA', 'rA\'', '[A\']']):
        rad_A = pd.to_numeric(df['rA'], errors='coerce').fillna(0).values
        rad_Ap = pd.to_numeric(df['rA\''], errors='coerce').fillna(0).values
        conc_Ap = pd.to_numeric(df['[A\']'], errors='coerce').fillna(0).values
        rAav_values = df['rAav'].values if 'rAav' in df.columns else np.zeros_like(rad_A)
        if isinstance(rAav_values, np.ndarray) and rAav_values.dtype == 'object':
            rAav_values = pd.to_numeric(rAav_values, errors='coerce').fillna(0).values
        
        sum_sq_A = (rad_A**2 * (1 - conc_Ap) + rad_Ap**2 * conc_Ap)
        df['σ²_rA'] = sum_sq_A - rAav_values**2
    
    return df

def add_thermodynamic_descriptors(df):
    """Энтропия, валентность"""
    R_gas = 8.314
    
    if all(col in df.columns for col in ['[A\']']):
        conc_Ap = pd.to_numeric(df['[A\']'], errors='coerce').fillna(0)
        x_A = 1 - conc_Ap
        x_Ap = conc_Ap
        entropy_A = np.zeros(len(df))
        mask = (x_A > 0) & (x_Ap > 0)
        entropy_A[mask] = -R_gas * (x_A[mask] * np.log(x_A[mask]) + x_Ap[mask] * np.log(x_Ap[mask]))
        df['S_config_A'] = entropy_A
    
    if all(col in df.columns for col in ['[B\']', '[D1]', '[D2]']):
        conc_Bp = pd.to_numeric(df['[B\']'], errors='coerce').fillna(0)
        conc_D1 = pd.to_numeric(df['[D1]'], errors='coerce').fillna(0)
        conc_D2 = pd.to_numeric(df['[D2]'], errors='coerce').fillna(0)
        
        x_B = 1 - conc_Bp - conc_D1 - conc_D2
        x_Bp = conc_Bp
        x_D1 = conc_D1
        x_D2 = conc_D2
        
        entropy_B = np.zeros(len(df))
        for i, (xb, xbp, xd1, xd2) in enumerate(zip(x_B, x_Bp, x_D1, x_D2)):
            probs = [p for p in [xb, xbp, xd1, xd2] if p > 0]
            if len(probs) > 1:
                entropy_B[i] = -R_gas * sum(p * np.log(p) for p in probs)
        df['S_config_B'] = entropy_B
    
    valence_table = {'Ba':2, 'Sr':2, 'Ca':2, 'La':3, 'Ce':4, 'Zr':4, 'Y':3, 'Yb':3,
                     'Sc':3, 'In':3, 'Fe':3, 'Zn':2, 'Sn':4, 'Ti':4, 'Gd':3, 'Sm':3,
                     'Nd':3, 'Eu':3, 'Dy':3, 'Pr':3, 'Ho':3, 'Tm':3, 'Tb':3, 'Hf':4}
    
    def get_valence(el):
        if pd.isna(el):
            return 0
        return valence_table.get(str(el), 0)
    
    for pos in ['B', 'B\'', 'D1', 'D2']:
        if pos in df.columns:
            df[f'V{pos}'] = df[pos].apply(get_valence)
    
    if all(col in df.columns for col in ['VB', 'VB\'', 'VD1', 'VD2', '[B\']', '[D1]', '[D2]']):
        conc_Bp = pd.to_numeric(df['[B\']'], errors='coerce').fillna(0)
        conc_D1 = pd.to_numeric(df['[D1]'], errors='coerce').fillna(0)
        conc_D2 = pd.to_numeric(df['[D2]'], errors='coerce').fillna(0)
        
        df['V_Bav'] = (df['VB'] * (1 - conc_Bp - conc_D1 - conc_D2) +
                       df['VB\''] * conc_Bp +
                       df['VD1'] * conc_D1 +
                       df['VD2'] * conc_D2)
        df['Vo_proxy'] = (4 - df['V_Bav']) / 2
    
    return df

def add_physics_inspired_descriptors(df):
    """Комбинированные дескрипторы"""
    
    for col in ['Δχ_AB', 't', 'σ²_rB', 'D_t', 'ionicity_BO', 'octahedral_factor', 'χ_ratio_AB', 'rBav', 'χBav']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Δχ_AB' in df.columns and 't' in df.columns:
        df['Δχ_div_t'] = df['Δχ_AB'] / df['t']
        df['Δχ_mul_t'] = df['Δχ_AB'] * df['t']
    
    if 'σ²_rB' in df.columns and 'D_t' in df.columns:
        df['disorder_over_distortion'] = df['σ²_rB'] / (df['D_t'] + 1e-6)
    
    if 'ionicity_BO' in df.columns and 'octahedral_factor' in df.columns:
        df['ionic_x_octa'] = df['ionicity_BO'] * df['octahedral_factor']
    
    if 'χ_ratio_AB' in df.columns and 't' in df.columns:
        df['chi_ratio_t'] = df['χ_ratio_AB'] * df['t']
    
    if 'rBav' in df.columns and 'χBav' in df.columns:
        df['rBav_x_χBav'] = df['rBav'] * df['χBav']
    
    return df

def add_all_descriptors(df, table_name="unknown"):
    """Запуск всех функций дескрипторов для таблицы"""
    if df is None or len(df) == 0:
        return df
    
    string_cols = ['A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'method', 'Symmetry', 'Phase transitions (PT)', 'Ref']
    
    for col in df.columns:
        if col not in string_cols and col != '№':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df = add_electronegativity_descriptors(df)
    df = add_geometric_descriptors(df)
    df = add_thermodynamic_descriptors(df)
    df = add_physics_inspired_descriptors(df)
    
    return df

# ============================================================================
# 4. ФУНКЦИИ СОПОСТАВЛЕНИЯ ДАННЫХ (CROSS-ANALYSIS)
# ============================================================================

def match_compositions_one_to_one(df_therm, df_phase):
    """
    Сопоставляет данные из THERM и PHASE по составу (A, B, [B'], D1, D2).
    Для каждого образца в THERM находит ПЕРВОЕ подходящее соответствие в PHASE.
    Используется для графиков, где нужно ОДНО значение Symmetry или T(PT).
    """
    df_therm_matched = df_therm.copy()
    
    df_therm_matched['Symmetry'] = np.nan
    df_therm_matched['T(PT)_matched'] = np.nan
    df_therm_matched['Phase_transitions'] = np.nan
    
    for idx, row in df_therm_matched.iterrows():
        conc_Bp_therm = row.get('[B\']', 0)
        if pd.isna(conc_Bp_therm):
            conc_Bp_therm = 0
        
        mask = pd.Series([True] * len(df_phase))
        
        if 'A' in df_phase.columns and 'A' in row:
            mask = mask & (df_phase['A'] == row['A'])
        
        if 'B' in df_phase.columns and 'B' in row:
            mask = mask & (df_phase['B'] == row['B'])
        
        if '[B\']' in df_phase.columns:
            phase_conc = pd.to_numeric(df_phase['[B\']'], errors='coerce').fillna(0)
            mask = mask & (np.abs(phase_conc - conc_Bp_therm) < 0.01)
        
        matches = df_phase[mask]
        
        if len(matches) > 0:
            first_match = matches.iloc[0]
            df_therm_matched.loc[idx, 'Symmetry'] = first_match.get('Symmetry', np.nan)
            df_therm_matched.loc[idx, 'T(PT)_matched'] = first_match.get('T (PT), °C', np.nan)
            df_therm_matched.loc[idx, 'Phase_transitions'] = first_match.get('Phase transitions (PT)', np.nan)
    
    return df_therm_matched

def match_compositions_all_matches(df_therm, df_phase):
    """
    Сопоставляет данные из THERM и PHASE, сохраняя ВСЕ соответствия.
    Возвращает словарь {index_therm: list_of_matches}
    Используется для отображения нескольких вариантов в hover.
    """
    matches_dict = {}
    
    for idx, row in df_therm.iterrows():
        conc_Bp_therm = row.get('[B\']', 0)
        if pd.isna(conc_Bp_therm):
            conc_Bp_therm = 0
        
        mask = pd.Series([True] * len(df_phase))
        
        if 'A' in df_phase.columns and 'A' in row:
            mask = mask & (df_phase['A'] == row['A'])
        
        if 'B' in df_phase.columns and 'B' in row:
            mask = mask & (df_phase['B'] == row['B'])
        
        if '[B\']' in df_phase.columns:
            phase_conc = pd.to_numeric(df_phase['[B\']'], errors='coerce').fillna(0)
            mask = mask & (np.abs(phase_conc - conc_Bp_therm) < 0.01)
        
        matches = df_phase[mask]
        
        if len(matches) > 0:
            matches_dict[idx] = matches.to_dict('records')
        else:
            matches_dict[idx] = []
    
    return matches_dict

def get_symmetry_for_therm(df_therm, df_phase):
    """Возвращает Series с Symmetry для каждого образца THERM (первое соответствие)"""
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    return df_matched['Symmetry']

# ============================================================================
# 5. ФИЛЬТРАЦИЯ ДАННЫХ (НЕЗАВИСИМАЯ ДЛЯ КАЖДОЙ ТАБЛИЦЫ)
# ============================================================================

def create_filters(df_therm, df_phase):
    """Создание виджетов фильтрации (все фильтры по умолчанию включены)"""
    st.sidebar.markdown("## 🔍 Фильтрация данных")
    st.sidebar.markdown("---")
    
    available_a_therm = sorted(df_therm['A'].dropna().unique()) if 'A' in df_therm.columns else []
    available_a_phase = sorted(df_phase['A'].dropna().unique()) if 'A' in df_phase.columns else []
    available_a = sorted(list(set(available_a_therm + available_a_phase)))
    
    selected_a = st.sidebar.multiselect(
        "A-site cations (A²⁺/A³⁺)",
        options=available_a,
        default=available_a,
        help="Выберите один или несколько катионов в A-позиции"
    )
    
    available_b_therm = sorted(df_therm['B'].dropna().unique()) if 'B' in df_therm.columns else []
    available_b_phase = sorted(df_phase['B'].dropna().unique()) if 'B' in df_phase.columns else []
    available_b = sorted(list(set(available_b_therm + available_b_phase)))
    
    selected_b = st.sidebar.multiselect(
        "B-site cations (B⁴⁺/B³⁺)",
        options=available_b,
        default=available_b,
        help="Выберите один или несколько катионов в B-позиции"
    )
    
    if 'D1' in df_therm.columns:
        available_d1 = sorted(df_therm['D1'].dropna().unique())
    else:
        available_d1 = []
    selected_d1 = st.sidebar.multiselect(
        "Dopant D1 (optional)",
        options=available_d1,
        default=available_d1 if available_d1 else [],
        help="Легирующий элемент в позиции D1"
    )
    
    if 'D2' in df_therm.columns:
        available_d2 = sorted(df_therm['D2'].dropna().unique())
    else:
        available_d2 = []
    selected_d2 = st.sidebar.multiselect(
        "Dopant D2 (optional)",
        options=available_d2,
        default=available_d2 if available_d2 else [],
        help="Легирующий элемент в позиции D2"
    )
    
    if 'method' in df_therm.columns:
        available_methods = sorted(df_therm['method'].dropna().unique())
        selected_methods = st.sidebar.multiselect(
            "Measurement method",
            options=available_methods,
            default=available_methods,
            help="Метод измерения термического расширения"
        )
    else:
        selected_methods = []
    
    if 'Symmetry' in df_phase.columns:
        available_sym = sorted(df_phase['Symmetry'].dropna().unique())
        selected_sym = st.sidebar.multiselect(
            "Crystal symmetry",
            options=available_sym,
            default=available_sym,
            help="Кристаллографическая симметрия"
        )
    else:
        selected_sym = []
    
    temp_range = (0, 1000)
    if '∆T, °C' in df_therm.columns:
        temp_vals = pd.to_numeric(df_therm['∆T, °C'], errors='coerce').dropna()
        if len(temp_vals) > 0:
            temp_range = st.sidebar.slider(
                "Temperature range (°C)",
                min_value=float(temp_vals.min()),
                max_value=float(temp_vals.max()),
                value=(float(temp_vals.min()), float(temp_vals.max())),
                step=50
            )
    
    return {
        'A': selected_a,
        'B': selected_b,
        'D1': selected_d1,
        'D2': selected_d2,
        'method': selected_methods,
        'symmetry': selected_sym,
        'temp_range': temp_range
    }

def apply_filters_to_therm(df_therm, filters):
    """Применение фильтров к THERM таблице"""
    if df_therm is None or len(df_therm) == 0:
        return df_therm
    
    filtered_df = df_therm.copy()
    
    if filters['A'] and 'A' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['A'].isin(filters['A'])]
    
    if filters['B'] and 'B' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['B'].isin(filters['B'])]
    
    if filters['D1'] and 'D1' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['D1'].isin(filters['D1'])]
    
    if filters['D2'] and 'D2' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['D2'].isin(filters['D2'])]
    
    if filters['method'] and 'method' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['method'].isin(filters['method'])]
    
    if '∆T, °C' in filtered_df.columns:
        temp_vals = pd.to_numeric(filtered_df['∆T, °C'], errors='coerce')
        filtered_df = filtered_df[(temp_vals >= filters['temp_range'][0]) & (temp_vals <= filters['temp_range'][1])]
    
    return filtered_df

def apply_filters_to_phase(df_phase, filters):
    """Применение фильтров к PHASE таблице"""
    if df_phase is None or len(df_phase) == 0:
        return df_phase
    
    filtered_df = df_phase.copy()
    
    if filters['A'] and 'A' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['A'].isin(filters['A'])]
    
    if filters['B'] and 'B' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['B'].isin(filters['B'])]
    
    if filters['D1'] and 'D1' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['D1'].isin(filters['D1'])]
    
    if filters['D2'] and 'D2' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['D2'].isin(filters['D2'])]
    
    if filters['symmetry'] and 'Symmetry' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Symmetry'].isin(filters['symmetry'])]
    
    return filtered_df

# ============================================================================
# 6. УНИВЕРСАЛЬНЫЕ ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ
# ============================================================================

def get_available_columns(df, table_type='therm'):
    """Получение списка доступных колонок для выбора"""
    if df is None:
        return [], []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if table_type == 'therm':
        priority_numeric = ['αav·106 (K-1)', 'α·106 (K-1)', 'β', 't', 'pH2O', 'rBav', 'rAav', 
                            'χAav', 'χBav', 'Δχ_AB', 'σ²_rB', 'S_config_B', 'V_Bav', 
                            'octahedral_factor', 'ionicity_BO', '[B\']']
        priority_cat = ['method', 'A', 'B', 'D1', 'D2']
    else:
        priority_numeric = ['t', 'rBav', 'rAav', 'χAav', 'χBav', 'Δχ_AB', 'σ²_rB', 'T (PT), °C']
        priority_cat = ['Symmetry', 'Phase transitions (PT)', 'A', 'B', 'D1', 'D2']
    
    numeric_cols = [c for c in priority_numeric if c in numeric_cols] + [c for c in numeric_cols if c not in priority_numeric]
    categorical_cols = [c for c in priority_cat if c in categorical_cols] + [c for c in categorical_cols if c not in priority_cat]
    
    return numeric_cols, categorical_cols

def create_bubble_plot(df, x_col, y_col, size_col, color_col, title, log_x=False, log_y=False):
    """Универсальная пузырьковая диаграмма"""
    df_plot = df.dropna(subset=[x_col, y_col, size_col]).copy()
    
    if len(df_plot) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно данных для построения", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    if color_col in df_plot.select_dtypes(include=[np.number]).columns:
        fig = px.scatter(
            df_plot, x=x_col, y=y_col, size=size_col, color=color_col,
            hover_data=['A', 'B', 'D1', 'D2', 'Ref', 'method', 'Symmetry'],
            title=title,
            labels={x_col: x_col, y_col: y_col, size_col: size_col, color_col: color_col},
            color_continuous_scale='plasma',
            log_x=log_x, log_y=log_y
        )
    else:
        fig = px.scatter(
            df_plot, x=x_col, y=y_col, size=size_col, color=color_col,
            hover_data=['A', 'B', 'D1', 'D2', 'Ref', 'method'],
            title=title,
            labels={x_col: x_col, y_col: y_col, size_col: size_col},
            color_discrete_sequence=px.colors.qualitative.Set1,
            log_x=log_x, log_y=log_y
        )
    
    fig.update_layout(
        font_family="Times New Roman",
        font_size=12,
        title_font_size=14,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
        width=800,
        height=600
    )
    
    return fig

def create_contour_plot(df, x_col, y_col, z_col, title):
    """Универсальная контурная карта"""
    df_plot = df.dropna(subset=[x_col, y_col, z_col]).copy()
    
    if len(df_plot) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно данных для контурной карты (нужно ≥10 точек)", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.density_contour(
        df_plot, x=x_col, y=y_col, z=z_col,
        title=title,
        labels={x_col: x_col, y_col: y_col, z_col: z_col},
        color_continuous_scale='viridis'
    )
    
    fig.add_trace(go.Scatter(
        x=df_plot[x_col], y=df_plot[y_col],
        mode='markers',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Data points',
        hoverinfo='text',
        text=df_plot[['A', 'B', z_col]].astype(str).agg(' | '.join, axis=1)
    ))
    
    fig.update_layout(
        font_family="Times New Roman",
        font_size=12,
        title_font_size=14,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600
    )
    
    return fig

# ============================================================================
# 7. ГРАФИКИ ТОЛЬКО ДЛЯ THERM (13 ГРАФИКОВ)
# ============================================================================

def plot_therm_alpha_vs_t(df):
    """График 1: αav vs t (цвет = метод)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 't' not in df.columns or 'αav·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['t', 'αav·106 (K-1)'])
    
    if 'method' in df_plot.columns:
        methods = df_plot['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for method, color in zip(methods, colors):
            mask = df_plot['method'] == method
            ax.scatter(df_plot.loc[mask, 't'], df_plot.loc[mask, 'αav·106 (K-1)'], 
                      label=method, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['t'], df_plot['αav·106 (K-1)'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Tolerance factor (t)', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹)', fontweight='bold')
    ax.set_title('Thermal expansion vs tolerance factor', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_beta_vs_ph2o(df):
    """График 2: β vs pH₂O"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'pH2O' not in df.columns or 'β' not in df.columns:
        ax.text(0.5, 0.5, 'No pH₂O or β data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['pH2O', 'β'])
    
    scat = ax.scatter(df_plot['pH2O'], df_plot['β'], c=df_plot.get('αav·106 (K-1)', np.ones(len(df_plot))), 
                     cmap='plasma', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('pH₂O (partial pressure)', fontweight='bold')
    ax.set_ylabel('β (chemical expansion)', fontweight='bold')
    ax.set_title('Chemical expansion vs water partial pressure', fontweight='bold')
    plt.colorbar(scat, label='αav·10⁶ K⁻¹' if 'αav·106 (K-1)' in df.columns else 'Value')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_alpha_vs_doping(df):
    """График 3: αav vs [B'] концентрация допанта"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if '[B\']' not in df.columns or 'αav·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No doping concentration data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['[B\']', 'αav·106 (K-1)'])
    
    if 'B' in df_plot.columns:
        b_elements = df_plot['B'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(b_elements)))
        for b_elem, color in zip(b_elements, colors):
            mask = df_plot['B'] == b_elem
            ax.scatter(df_plot.loc[mask, '[B\']'], df_plot.loc[mask, 'αav·106 (K-1)'], 
                      label=f'B = {b_elem}', color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['[B\']'], df_plot['αav·106 (K-1)'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Dopant concentration [B\']', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹)', fontweight='bold')
    ax.set_title('Thermal expansion vs doping level', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_goldschmidt_map(df):
    """График 4: Goldschmidt map - rAav vs rBav, color=αav"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'rAav' not in df.columns or 'rBav' not in df.columns or 'αav·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No radius or α data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['rAav', 'rBav', 'αav·106 (K-1)'])
    
    scat = ax.scatter(df_plot['rAav'], df_plot['rBav'], c=df_plot['αav·106 (K-1)'], 
                     cmap='coolwarm', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Average A-site radius (Å)', fontweight='bold')
    ax.set_ylabel('Average B-site radius (Å)', fontweight='bold')
    ax.set_title('Goldschmidt map: αav vs rAav vs rBav', fontweight='bold')
    plt.colorbar(scat, label='αav·10⁶ K⁻¹')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_contour_t_vs_dchi_vs_beta(df):
    """График 5: Контурная карта t vs Δχ, color=β"""
    if 't' not in df.columns or 'Δχ_AB' not in df.columns or 'β' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No t, Δχ, or β data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    return create_contour_plot(df, 't', 'Δχ_AB', 'β', 'Chemical stability map: β vs t vs Δχ')

def plot_therm_contour_rBav_vs_chiBav_vs_alpha(df):
    """График 6: Контурная карта rBav vs χBav, color=αav"""
    if 'rBav' not in df.columns or 'χBav' not in df.columns or 'αav·106 (K-1)' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No rBav, χBav, or α data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    return create_contour_plot(df, 'rBav', 'χBav', 'αav·106 (K-1)', 'Thermal expansion map: αav vs rBav vs χBav')

def plot_therm_nonconstant_beta(df):
    """График 7: Непостоянство β - остатки vs T(bends)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'T(bends), °C' in df.columns and 'α·106 (K-1)' in df.columns:
        df_clean = df.dropna(subset=['T(bends), °C', 'α·106 (K-1)'])
        if len(df_clean) > 5:
            ax1.scatter(df_clean['T(bends), °C'], df_clean['α·106 (K-1)'], alpha=0.7, c='blue', edgecolors='k')
            ax1.set_xlabel('Temperature of bend / dehydration (°C)', fontweight='bold')
            ax1.set_ylabel('α·10⁶ (K⁻¹)', fontweight='bold')
            ax1.set_title('Variability of α near phase transition', fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            residuals = df_clean['α·106 (K-1)'] - df_clean['α·106 (K-1)'].median()
            ax2.scatter(df_clean['T(bends), °C'], residuals, alpha=0.7, c='red', edgecolors='k')
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax2.set_xlabel('Temperature of bend (°C)', fontweight='bold')
            ax2.set_ylabel('Residual (α - median)', fontweight='bold')
            ax2.set_title('Residuals: evidence that β is not constant', fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
    else:
        ax1.text(0.5, 0.5, 'No T(bends) or α data', ha='center', va='center')
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    plt.tight_layout()
    return fig

def plot_therm_method_comparison(df):
    """График 8: Сравнение методов измерения"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'method' not in df.columns or 'α·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No method or α data', ha='center', va='center')
        return fig
    
    df_method = df.dropna(subset=['method', 'α·106 (K-1)'])
    if len(df_method) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.boxplot(data=df_method, x='method', y='α·106 (K-1)', ax=ax)
    ax.set_xlabel('Measurement method', fontweight='bold')
    ax.set_ylabel('α·10⁶ (K⁻¹)', fontweight='bold')
    ax.set_title('Comparison of thermal expansion by method', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_alpha_before_after_bend(df):
    """График 9: α до и после T(bends)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'αav·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No αav data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['αav·106 (K-1)'])
    
    if 'T(bends), °C' in df.columns:
        has_bend = df_plot['T(bends), °C'].notna()
        if has_bend.any():
            ax.scatter(df_plot.loc[~has_bend, 'αav·106 (K-1)'], 
                      np.zeros(df_plot.loc[~has_bend].shape[0]) + 0.1,
                      alpha=0.7, c='blue', edgecolors='k', label='No bend detected', s=50)
            ax.scatter(df_plot.loc[has_bend, 'αav·106 (K-1)'], 
                      np.zeros(df_plot.loc[has_bend].shape[0]) - 0.1,
                      alpha=0.7, c='red', edgecolors='k', label='Bend detected', s=50)
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.legend(loc='best')
        else:
            ax.scatter(df_plot['αav·106 (K-1)'], np.zeros(len(df_plot)), alpha=0.7, c='blue', edgecolors='k', s=50)
    
    ax.set_xlabel('αav·10⁶ (K⁻¹)', fontweight='bold')
    ax.set_title('Thermal expansion: samples with and without bends', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_alpha_vs_temperature_range(df):
    """График 10: α vs температурный интервал"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if '∆T, °C' not in df.columns or 'α·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No ∆T or α data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['∆T, °C', 'α·106 (K-1)'])
    
    def parse_temp_range(temp_str):
        try:
            if pd.isna(temp_str):
                return np.nan
            if isinstance(temp_str, str) and '-' in temp_str:
                parts = temp_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(temp_str)
        except:
            return np.nan
    
    df_plot['∆T_mid'] = df_plot['∆T, °C'].apply(parse_temp_range)
    df_plot = df_plot.dropna(subset=['∆T_mid'])
    
    if len(df_plot) > 0:
        ax.scatter(df_plot['∆T_mid'], df_plot['α·106 (K-1)'], alpha=0.7, c='purple', edgecolors='k', s=50)
        ax.set_xlabel('Mid-temperature of measurement range (°C)', fontweight='bold')
        ax.set_ylabel('α·10⁶ (K⁻¹)', fontweight='bold')
        ax.set_title('Thermal expansion vs measurement temperature', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    else:
        ax.text(0.5, 0.5, 'No valid temperature range data', ha='center', va='center')
    
    return fig

def plot_therm_bubble_4d(df):
    """График 11: Пузырьковая диаграмма 4D (t, α, β, method)"""
    if 't' not in df.columns or 'αav·106 (K-1)' not in df.columns or 'β' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No t, α, or β data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    color_col = 'method' if 'method' in df.columns else 'Symmetry' if 'Symmetry' in df.columns else None
    
    if color_col and color_col in df.columns:
        return create_bubble_plot(df, 't', 'αav·106 (K-1)', 'β', color_col, 
                                  '4D Bubble chart: αav vs t, size=β, color=' + color_col)
    else:
        return create_bubble_plot(df, 't', 'αav·106 (K-1)', 'β', 'β', 
                                  '4D Bubble chart: αav vs t, size=β')

def plot_therm_violin_by_method(df):
    """График 12: Violin plot αav по методам"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'method' not in df.columns or 'αav·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No method or αav data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['method', 'αav·106 (K-1)'])
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.violinplot(data=df_plot, x='method', y='αav·106 (K-1)', ax=ax)
    ax.set_xlabel('Measurement method', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹)', fontweight='bold')
    ax.set_title('Distribution of thermal expansion by measurement method', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_therm_bubble_chi_vs_beta(df):
    """График 13: χBav vs β, размер=pH2O, цвет=[B']"""
    if 'χBav' not in df.columns or 'β' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No χBav or β data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    size_col = 'pH2O' if 'pH2O' in df.columns else 'β'
    color_col = '[B\']' if '[B\']' in df.columns else 'β'
    
    return create_bubble_plot(df, 'χBav', 'β', size_col, color_col,
                              'Chemical expansion: β vs χBav, size=pH₂O, color=[B\']')

# ============================================================================
# 8. ГРАФИКИ ТОЛЬКО ДЛЯ PHASE (7 ГРАФИКОВ)
# ============================================================================

def plot_phase_diagram_cezr(df):
    """График 14: Фазовая диаграмма Ce₁₋ₓZrₓO₃"""
    if 'Symmetry' not in df.columns or '[B\']' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No symmetry or concentration data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_cezr = df[df['B'] == 'Ce'].dropna(subset=['[B\']', 'Symmetry'])
    if len(df_cezr) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient Ce-Zr data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    if 'T (PT), °C' not in df_cezr.columns:
        df_cezr['T (PT), °C'] = 400 + 200 * df_cezr['[B\']'].fillna(0)
    
    fig = px.scatter(df_cezr, x='[B\']', y='T (PT), °C', color='Symmetry',
                     title='Phase transition diagram: Ce₁₋ₓZrₓO₃ system',
                     labels={'[B\']': 'Zr concentration (x)', 'T (PT), °C': 'Phase transition temperature (°C)'},
                     hover_data=['A', 'B', 'D1', 'D2', 'Phase transitions (PT)'])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

def plot_phase_t_vs_tolerance(df):
    """График 15: T(PT) vs tolerance factor"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 't' not in df.columns or 'T (PT), °C' not in df.columns:
        ax.text(0.5, 0.5, 'No t or T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['t', 'T (PT), °C'])
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, 't'], df_plot.loc[mask, 'T (PT), °C'], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['t'], df_plot['T (PT), °C'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Tolerance factor (t)', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperature vs tolerance factor', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_t_vs_disorder(df):
    """График 16: T(PT) vs σ²_rB (беспорядок)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'σ²_rB' not in df.columns or 'T (PT), °C' not in df.columns:
        ax.text(0.5, 0.5, 'No disorder or T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['σ²_rB', 'T (PT), °C'])
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, 'σ²_rB'], df_plot.loc[mask, 'T (PT), °C'], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['σ²_rB'], df_plot['T (PT), °C'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('B-site radius variance σ²(rB)', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperature vs structural disorder', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_symmetry_map(df):
    """График 17: Карта стабильности фаз - t vs Δχ, цвет=Symmetry"""
    if 't' not in df.columns or 'Δχ_AB' not in df.columns or 'Symmetry' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No t, Δχ, or Symmetry data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_plot = df.dropna(subset=['t', 'Δχ_AB', 'Symmetry'])
    
    fig = px.scatter(df_plot, x='t', y='Δχ_AB', color='Symmetry',
                     title='Phase stability map: Symmetry vs t vs Δχ',
                     labels={'t': 'Tolerance factor', 'Δχ_AB': 'Electronegativity difference Δχ(A-B)'},
                     hover_data=['A', 'B', '[B\']', 'T (PT), °C'])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

def plot_phase_t_vs_doping(df):
    """График 18: T(PT) vs концентрация допанта"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if '[B\']' not in df.columns or 'T (PT), °C' not in df.columns:
        ax.text(0.5, 0.5, 'No doping or T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['[B\']', 'T (PT), °C'])
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, '[B\']'], df_plot.loc[mask, 'T (PT), °C'], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['[B\']'], df_plot['T (PT), °C'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Dopant concentration [B\']', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperature vs doping level', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_violin_by_symmetry(df):
    """График 19: Violin plot T(PT) по симметриям"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'Symmetry' not in df.columns or 'T (PT), °C' not in df.columns:
        ax.text(0.5, 0.5, 'No Symmetry or T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['Symmetry', 'T (PT), °C'])
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.violinplot(data=df_plot, x='Symmetry', y='T (PT), °C', ax=ax)
    ax.set_xlabel('Crystal symmetry', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Distribution of PT temperature by symmetry', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_t_distribution_by_type(df):
    """График 20: Распределение T(PT) по типам переходов"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'Phase transitions (PT)' not in df.columns or 'T (PT), °C' not in df.columns:
        ax.text(0.5, 0.5, 'No PT type or T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['Phase transitions (PT)', 'T (PT), °C'])
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.boxplot(data=df_plot, x='Phase transitions (PT)', y='T (PT), °C', ax=ax)
    ax.set_xlabel('Phase transition type', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperatures by transition type', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

# ============================================================================
# 9. КРОСС-ГРАФИКИ (THERM + PHASE) — 10 ГРАФИКОВ
# ============================================================================

def plot_cross_alpha_vs_symmetry(df_therm, df_phase):
    """График 21: αav vs Symmetry (violin/boxplot)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'αav·106 (K-1)' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No αav data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['αav·106 (K-1)', 'Symmetry'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient matched data', ha='center', va='center')
        return fig
    
    sns.violinplot(data=df_plot, x='Symmetry', y='αav·106 (K-1)', ax=ax)
    ax.set_xlabel('Crystal symmetry (from PHASE data)', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹) (from THERM data)', fontweight='bold')
    ax.set_title('Cross-analysis: Thermal expansion vs crystal symmetry', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_beta_vs_symmetry(df_therm, df_phase):
    """График 22: β vs Symmetry"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'β' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No β data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['β', 'Symmetry'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient matched data', ha='center', va='center')
        return fig
    
    sns.boxplot(data=df_plot, x='Symmetry', y='β', ax=ax)
    ax.set_xlabel('Crystal symmetry (from PHASE data)', fontweight='bold')
    ax.set_ylabel('β (chemical expansion) (from THERM data)', fontweight='bold')
    ax.set_title('Cross-analysis: Chemical expansion vs crystal symmetry', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_tpt_vs_alpha(df_therm, df_phase):
    """График 23: T(PT) vs αav"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'αav·106 (K-1)' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No αav data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['αav·106 (K-1)', 'T(PT)_matched'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient matched data', ha='center', va='center')
        return fig
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, 'T(PT)_matched'], df_plot.loc[mask, 'αav·106 (K-1)'], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['T(PT)_matched'], df_plot['αav·106 (K-1)'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Phase transition temperature (°C) (from PHASE data)', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹) (from THERM data)', fontweight='bold')
    ax.set_title('Cross-analysis: Thermal expansion vs phase transition temperature', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_tpt_vs_beta(df_therm, df_phase):
    """График 24: T(PT) vs β"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'β' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No β data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['β', 'T(PT)_matched'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient matched data', ha='center', va='center')
        return fig
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, 'T(PT)_matched'], df_plot.loc[mask, 'β'], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['T(PT)_matched'], df_plot['β'], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Phase transition temperature (°C) (from PHASE data)', fontweight='bold')
    ax.set_ylabel('β (chemical expansion) (from THERM data)', fontweight='bold')
    ax.set_title('Cross-analysis: Chemical expansion vs phase transition temperature', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_bubble_with_symmetry(df_therm, df_phase):
    """График 25: Пузырьковая диаграмма с цветом = Symmetry"""
    if 't' not in df_therm.columns or 'αav·106 (K-1)' not in df_therm.columns or 'β' not in df_therm.columns:
        fig = go.Figure()
        fig.add_annotation(text="No t, α, or β data in THERM", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['t', 'αav·106 (K-1)', 'β', 'Symmetry'])
    
    if len(df_plot) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient matched data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.scatter(df_plot, x='t', y='αav·106 (K-1)', size='β', color='Symmetry',
                     title='Cross-analysis: Thermal expansion vs tolerance factor, color = crystal symmetry',
                     labels={'t': 'Tolerance factor', 'αav·106 (K-1)': 'αav·10⁶ K⁻¹', 'β': 'β (size)'},
                     hover_data=['A', 'B', '[B\']', 'method', 'T(PT)_matched'])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

def plot_cross_bends_vs_pt(df_therm, df_phase):
    """График 26: T(bends) vs T(PT)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'T(bends), °C' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No T(bends) data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['T(bends), °C', 'T(PT)_matched'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient matched data', ha='center', va='center')
        return fig
    
    ax.scatter(df_plot['T(bends), °C'], df_plot['T(PT)_matched'], alpha=0.7, c='green', edgecolors='k', s=50)
    
    max_val = max(df_plot['T(bends), °C'].max(), df_plot['T(PT)_matched'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x')
    
    ax.set_xlabel('Bend temperature T(bends) (°C) (from THERM)', fontweight='bold')
    ax.set_ylabel('Phase transition temperature T(PT) (°C) (from PHASE)', fontweight='bold')
    ax.set_title('Cross-analysis: Correlation between bends and phase transitions', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_alpha_gradient_on_phase_map(df_therm, df_phase):
    """График 27: Градиент αav на фазовой диаграмме"""
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['[B\']', 'T(PT)_matched', 'αav·106 (K-1)'])
    
    if len(df_plot) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for gradient map", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.scatter(df_plot, x='[B\']', y='T(PT)_matched', color='αav·106 (K-1)',
                     title='Cross-analysis: Thermal expansion gradient on phase diagram',
                     labels={'[B\']': 'Dopant concentration [B\']', 'T(PT)_matched': 'Phase transition temperature (°C)',
                             'αav·106 (K-1)': 'αav·10⁶ K⁻¹'},
                     color_continuous_scale='viridis',
                     hover_data=['A', 'B', 'Symmetry'])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

def plot_cross_delta_alpha_vs_transition_type(df_therm, df_phase):
    """График 28: ∆α (скачок) vs тип перехода"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'αav·106 (K-1)' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No αav data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    
    if 'Phase transitions (PT)' not in df_matched.columns:
        ax.text(0.5, 0.5, 'No PT type data', ha='center', va='center')
        return fig
    
    df_plot = df_matched.dropna(subset=['αav·106 (K-1)', 'Phase transitions (PT)'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.boxplot(data=df_plot, x='Phase transitions (PT)', y='αav·106 (K-1)', ax=ax)
    ax.set_xlabel('Phase transition type (from PHASE data)', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹) (from THERM data)', fontweight='bold')
    ax.set_title('Cross-analysis: Thermal expansion by phase transition type', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_ph2o_vs_symmetry(df_therm, df_phase):
    """График 29: pH₂O vs Symmetry"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'pH2O' not in df_therm.columns:
        ax.text(0.5, 0.5, 'No pH₂O data in THERM', ha='center', va='center')
        return fig
    
    df_matched = match_compositions_one_to_one(df_therm, df_phase)
    df_plot = df_matched.dropna(subset=['pH2O', 'Symmetry'])
    
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient matched data', ha='center', va='center')
        return fig
    
    sns.boxplot(data=df_plot, x='Symmetry', y='pH2O', ax=ax)
    ax.set_xlabel('Crystal symmetry (from PHASE data)', fontweight='bold')
    ax.set_ylabel('pH₂O (partial pressure) (from THERM data)', fontweight='bold')
    ax.set_title('Cross-analysis: Water partial pressure by crystal symmetry', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_cross_pca_with_symmetry(df_therm, df_phase):
    """График 30: PCA проекция THERM данных с цветом = Symmetry из PHASE"""
    feature_cols = [col for col in ['rAav', 'rBav', 't', 'χAav', 'χBav', 'Δχ_AB', 'σ²_rB'] 
                    if col in df_therm.columns]
    
    if len(feature_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient features for PCA", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_therm_with_sym = match_compositions_one_to_one(df_therm, df_phase)
    df_pca = df_therm_with_sym[feature_cols + ['Symmetry']].dropna()
    
    if len(df_pca) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for PCA", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    X = df_pca[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df_pca['Symmetry'],
                     title=f'Cross-analysis: PCA of THERM descriptors, color = Symmetry from PHASE<br>(PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%})',
                     labels={'x': 'PC1', 'y': 'PC2', 'color': 'Symmetry'},
                     hover_data=df_pca[feature_cols + ['Symmetry']])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

# ============================================================================
# 10. STREAMLIT UI (ГЛАВНОЕ ПРИЛОЖЕНИЕ)
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Perovskite Expansion Analyzer", page_icon="🧪")
    st.title("📊 Advanced Materials Informatics for Perovskite Oxides")
    st.markdown("### Thermal & Chemical Expansion | Phase Transitions | Cross-Analysis")
    
    st.sidebar.header("🧭 Navigation")
    analysis_mode = st.sidebar.radio(
        "Choose analysis mode",
        ["📊 Mode 1: THERM only (thermal/chemical expansion)",
         "🔬 Mode 2: PHASE only (phase transitions)",
         "🔗 Mode 3: CROSS-analysis (THERM + PHASE)"]
    )
    
    # Виджеты для ввода данных
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📂 Data Input")
    
    therm_text = st.sidebar.text_area(
        "Sheet 1: Thermal Expansion Data (tab-separated)",
        value=DEFAULT_THERM_DATA,
        height=200,
        help="Edit or paste tab-separated data with headers"
    )
    
    phase_text = st.sidebar.text_area(
        "Sheet 2: Phase Transition Data (tab-separated)",
        value=DEFAULT_PHASE_DATA,
        height=200,
        help="Edit or paste tab-separated data with headers"
    )
    
    # Загрузка данных
    with st.spinner("Loading and processing data..."):
        df_therm_raw, df_phase_raw = load_data_from_text(therm_text, phase_text)
        
        if df_therm_raw is None or df_phase_raw is None:
            st.error("Failed to load data. Please check the format (tab-separated values).")
            return
        
        df_therm = add_all_descriptors(df_therm_raw, "therm")
        df_phase = add_all_descriptors(df_phase_raw, "phase")
        
        st.success(f"✅ THERM: {len(df_therm)} rows, {len(df_therm.columns)} columns | PHASE: {len(df_phase)} rows, {len(df_phase.columns)} columns")
    
    # Фильтры
    filters = create_filters(df_therm, df_phase)
    df_therm_filtered = apply_filters_to_therm(df_therm, filters)
    df_phase_filtered = apply_filters_to_phase(df_phase, filters)
    
    with st.sidebar.expander("📊 Active filters", expanded=False):
        st.write(f"A-site: {filters['A']}")
        st.write(f"B-site: {filters['B']}")
        st.write(f"THERM filtered: {len(df_therm_filtered)} / {len(df_therm)}")
        st.write(f"PHASE filtered: {len(df_phase_filtered)} / {len(df_phase)}")
    
    # ========================================================================
    # РЕЖИМ 1: THERM ONLY
    # ========================================================================
    if analysis_mode == "📊 Mode 1: THERM only (thermal/chemical expansion)":
        st.subheader("📊 THERMAL & CHEMICAL EXPANSION ANALYSIS (13 plots)")
        
        numeric_cols, categorical_cols = get_available_columns(df_therm_filtered, 'therm')
        
        plot_type = st.selectbox(
            "Select plot type",
            ["1. αav vs t (color=method)",
             "2. β vs pH₂O",
             "3. αav vs [B'] (doping level)",
             "4. Goldschmidt map: rAav vs rBav, color=αav",
             "5. Contour: t vs Δχ, color=β",
             "6. Contour: rBav vs χBav, color=αav",
             "7. Non-constant β: residuals vs T(bends)",
             "8. Method comparison: dilatometry vs HTXRD",
             "9. α before/after T(bends)",
             "10. α vs measurement temperature range",
             "11. 4D Bubble chart: t, α, β, method",
             "12. Violin plot: αav by method",
             "13. Bubble: χBav vs β, size=pH2O, color=[B']"]
        )
        
        if len(df_therm_filtered) > 0:
            if plot_type == "1. αav vs t (color=method)":
                fig = plot_therm_alpha_vs_t(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "2. β vs pH₂O":
                fig = plot_therm_beta_vs_ph2o(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "3. αav vs [B'] (doping level)":
                fig = plot_therm_alpha_vs_doping(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "4. Goldschmidt map: rAav vs rBav, color=αav":
                fig = plot_therm_goldschmidt_map(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "5. Contour: t vs Δχ, color=β":
                fig = plot_therm_contour_t_vs_dchi_vs_beta(df_therm_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "6. Contour: rBav vs χBav, color=αav":
                fig = plot_therm_contour_rBav_vs_chiBav_vs_alpha(df_therm_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "7. Non-constant β: residuals vs T(bends)":
                fig = plot_therm_nonconstant_beta(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "8. Method comparison: dilatometry vs HTXRD":
                fig = plot_therm_method_comparison(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "9. α before/after T(bends)":
                fig = plot_therm_alpha_before_after_bend(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "10. α vs measurement temperature range":
                fig = plot_therm_alpha_vs_temperature_range(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "11. 4D Bubble chart: t, α, β, method":
                fig = plot_therm_bubble_4d(df_therm_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "12. Violin plot: αav by method":
                fig = plot_therm_violin_by_method(df_therm_filtered)
                st.pyplot(fig)
            
            elif plot_type == "13. Bubble: χBav vs β, size=pH2O, color=[B']":
                fig = plot_therm_bubble_chi_vs_beta(df_therm_filtered)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data after filtering")
        
        st.subheader("📋 THERM Data Preview")
        st.dataframe(df_therm_filtered.head(50))
    
    # ========================================================================
    # РЕЖИМ 2: PHASE ONLY
    # ========================================================================
    elif analysis_mode == "🔬 Mode 2: PHASE only (phase transitions)":
        st.subheader("🔬 PHASE TRANSITION ANALYSIS (7 plots)")
        
        plot_type = st.selectbox(
            "Select plot type",
            ["14. Phase diagram: Ce₁₋ₓZrₓO₃",
             "15. T(PT) vs tolerance factor (t)",
             "16. T(PT) vs structural disorder (σ²_rB)",
             "17. Phase stability map: t vs Δχ, color=Symmetry",
             "18. T(PT) vs doping concentration [B']",
             "19. Violin plot: T(PT) by Symmetry",
             "20. T(PT) distribution by transition type"]
        )
        
        if len(df_phase_filtered) > 0:
            if plot_type == "14. Phase diagram: Ce₁₋ₓZrₓO₃":
                fig = plot_phase_diagram_cezr(df_phase_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "15. T(PT) vs tolerance factor (t)":
                fig = plot_phase_t_vs_tolerance(df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "16. T(PT) vs structural disorder (σ²_rB)":
                fig = plot_phase_t_vs_disorder(df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "17. Phase stability map: t vs Δχ, color=Symmetry":
                fig = plot_phase_symmetry_map(df_phase_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "18. T(PT) vs doping concentration [B']":
                fig = plot_phase_t_vs_doping(df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "19. Violin plot: T(PT) by Symmetry":
                fig = plot_phase_violin_by_symmetry(df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "20. T(PT) distribution by transition type":
                fig = plot_phase_t_distribution_by_type(df_phase_filtered)
                st.pyplot(fig)
        else:
            st.warning("No data after filtering")
        
        st.subheader("📋 PHASE Data Preview")
        st.dataframe(df_phase_filtered.head(50))
    
    # ========================================================================
    # РЕЖИМ 3: CROSS ANALYSIS (THERM + PHASE)
    # ========================================================================
    elif analysis_mode == "🔗 Mode 3: CROSS-analysis (THERM + PHASE)":
        st.subheader("🔗 CROSS-ANALYSIS: THERM + PHASE (10 plots)")
        st.markdown("⚠️ **Important**: Data are matched by composition (A, B, [B'], D1, D2). Multiple matches may exist.")
        
        plot_type = st.selectbox(
            "Select cross-analysis plot type",
            ["21. αav vs Symmetry (violin/boxplot)",
             "22. β vs Symmetry (boxplot)",
             "23. T(PT) vs αav (scatter)",
             "24. T(PT) vs β (scatter)",
             "25. 4D Bubble: t, α, β, color=Symmetry",
             "26. T(bends) vs T(PT) correlation",
             "27. αav gradient on phase diagram",
             "28. ∆α (scan) vs transition type",
             "29. pH₂O vs Symmetry",
             "30. PCA of THERM, color=Symmetry from PHASE"]
        )
        
        if len(df_therm_filtered) > 0 and len(df_phase_filtered) > 0:
            if plot_type == "21. αav vs Symmetry (violin/boxplot)":
                fig = plot_cross_alpha_vs_symmetry(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "22. β vs Symmetry (boxplot)":
                fig = plot_cross_beta_vs_symmetry(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "23. T(PT) vs αav (scatter)":
                fig = plot_cross_tpt_vs_alpha(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "24. T(PT) vs β (scatter)":
                fig = plot_cross_tpt_vs_beta(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "25. 4D Bubble: t, α, β, color=Symmetry":
                fig = plot_cross_bubble_with_symmetry(df_therm_filtered, df_phase_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "26. T(bends) vs T(PT) correlation":
                fig = plot_cross_bends_vs_pt(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "27. αav gradient on phase diagram":
                fig = plot_cross_alpha_gradient_on_phase_map(df_therm_filtered, df_phase_filtered)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "28. ∆α (scan) vs transition type":
                fig = plot_cross_delta_alpha_vs_transition_type(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "29. pH₂O vs Symmetry":
                fig = plot_cross_ph2o_vs_symmetry(df_therm_filtered, df_phase_filtered)
                st.pyplot(fig)
            
            elif plot_type == "30. PCA of THERM, color=Symmetry from PHASE":
                fig = plot_cross_pca_with_symmetry(df_therm_filtered, df_phase_filtered)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data in one or both tables after filtering")
        
        st.subheader("📋 Matched Data Preview (THERM + Symmetry)")
        df_matched_preview = match_compositions_one_to_one(df_therm_filtered, df_phase_filtered)
        st.dataframe(df_matched_preview[['A', 'B', '[B\']', 'αav·106 (K-1)', 'β', 'Symmetry', 'T(PT)_matched']].head(50))
    
    # Кнопка скачивания
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Download THERM data (CSV)"):
        csv = df_therm_filtered.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="therm_data.csv">Download CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    
    if st.sidebar.button("💾 Download PHASE data (CSV)"):
        csv = df_phase_filtered.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="phase_data.csv">Download CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Built with Streamlit | 3 analysis modes | 30+ publication-ready plots | Cross-analysis by composition matching")

if __name__ == "__main__":
    main()
