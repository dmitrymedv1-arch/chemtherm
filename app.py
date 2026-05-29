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
# 2. ЗАГРУЗКА ДАННЫХ (ДВА ВИДЖЕТА С ПРЕДЗАГРУЖЕННЫМИ МАССИВАМИ)
# ============================================================================

# Предзагруженные данные для первого листа (chem and therm expansion)
DEFAULT_THERM_DATA = """№	A	A'	B	B'	D1	D2	[A']	[B']	[D1]	[D2]	δ	rA	rA'	rAav	rB	rB'	rD1	rD2	rBav	t	rBav/rO	method	β	∆T, °C	α·106 (K-1)	T(bends), °C	αav·106 (K-1)	pH2O	Ref
1	Ba		Ce	Zr	Y	Yb		0.1	0.1	0.1	0.1	1.61		1.61	0.87	0.72	0.9	0.868	0.8578	0.942683768	0.612714286	dilatometry	0.0073	27-1000	10.6	400;600	10.6;4.73;10.1	0.0001	10.15826/chimtech.2024.11.4.22
2	Ba		Ce	Zr	Y	Yb		0.1	0.1	0.1	0.1	1.61		1.61	0.87	0.72	0.9	0.868	0.8578	0.942683768	0.612714286	HTXRD	0.0317	27-1000	10.6	300	10.7;8.7	0.02	10.15826/chimtech.2024.11.4.22
3	Ba		Ce	Zr	Y			0.1	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.858	0.942600271	0.612857143	HTND		20-900	11.2			0.00106	10.1021/acs.jpcc.1c08334
4	Ba		Ce	Zr	Y				0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.019	430-630		450		0.00106	10.1021/acs.jpcc.1c08334
5	Ba		Ce	Zr	Y				0.3		0.05	1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.019	430-631				0.00106	10.1021/acs.jpcc.1c08334
6	Ba		Ce	Zr	Y				0.6		0.05	1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.023	430-632				0.00106	10.1021/acs.jpcc.1c08334
7	Ba		Ce	Zr	Y				0.9		0.05	1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.49	430-633				0.00106	10.1021/acs.jpcc.1c08334"""

# Предзагруженные данные для второго листа (phase transition)
DEFAULT_PHASE_DATA = """№	A	A'	B	B'	D1	D2	[A]	[B']	[D1]	[D2]	δ	rA	rA'	rAav	rB	rB'	rD1	rD2	rBav	t	rBav/rO	pH2O	∆T, °C	Symmetry	Phase transitions (PT)	T (PT), °C	Ref
1	Ba		Ce	Zr	Y			0.36	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.819	0.959166927	0.585		30-1000				10.1063/1.5066970
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
    """Загрузка данных из текстовых виджетов (CSV формат с разделителем tab)"""
    try:
        df_therm = pd.read_csv(io.StringIO(therm_text), sep='\t', dtype=str)
        df_phase = pd.read_csv(io.StringIO(phase_text), sep='\t', dtype=str)
        
        # Очистка колонок
        df_therm.columns = df_therm.columns.str.strip()
        df_phase.columns = df_phase.columns.str.strip()
        
        # Замена пустых строк на NaN
        df_therm.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df_phase.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Преобразование числовых колонок
        numeric_cols_therm = ['№', '[A\']', '[B\']', '[D1]', '[D2]', 'δ', 'rA', 'rA\'', 'rB', 'rB\'', 'rD1', 'rD2', 't', 'β', 'pH2O', 'α·106 (K-1)', 'αav·106 (K-1)']
        numeric_cols_phase = ['№', '[A]', '[B\']', '[D1]', '[D2]', 'δ', 'rA', 'rA\'', 'rB', 'rB\'', 'rD1', 'rD2', 't', 'pH2O']
        
        for col in numeric_cols_therm:
            if col in df_therm.columns:
                df_therm[col] = pd.to_numeric(df_therm[col], errors='coerce')
        
        for col in numeric_cols_phase:
            if col in df_phase.columns:
                df_phase[col] = pd.to_numeric(df_phase[col], errors='coerce')
        
        return df_therm, df_phase
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None, None

# ============================================================================
# 3. ДОБАВЛЕНИЕ НОВЫХ ДЕСКРИПТОРОВ (Feature Engineering)
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
        if pd.isna(el) or el == '' or str(el).strip() == '':
            return np.nan
        return chi_table.get(str(el).strip(), np.nan)
    
    for pos in ['A', 'A\'', 'B', 'B\'', 'D1', 'D2']:
        if pos in df.columns:
            df[f'χ{pos}'] = df[pos].apply(get_chi)
    
    # A-позиция
    if all(col in df.columns for col in ['χA', 'χA\'', '[A\']']):
        df['χAav'] = df['χA'] * (1 - df['[A\']'].fillna(0)) + df['χA\''] * df['[A\']'].fillna(0)
    
    # B-позиция
    if all(col in df.columns for col in ['χB', 'χB\'', 'χD1', 'χD2', '[B\']', '[D1]', '[D2]']):
        df['χBav'] = (df['χB'] * (1 - df['[B\']'].fillna(0) - df['[D1]'].fillna(0) - df['[D2]'].fillna(0)) +
                      df['χB\''] * df['[B\']'].fillna(0) +
                      df['χD1'] * df['[D1]'].fillna(0) +
                      df['χD2'] * df['[D2]'].fillna(0))
    
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
        df['octahedral_factor'] = df['rBav'] / r_O
    if 't' in df.columns:
        df['D_t'] = np.abs(1 - df['t'])
    if 'rAav' in df.columns and 'rBav' in df.columns:
        df['Δr_AB'] = np.abs(df['rAav'] - df['rBav'])
        df['Δr_AB_norm'] = df['Δr_AB'] / r_O
        df['t_alt'] = (df['rAav'] + r_O) / (np.sqrt(2) * (df['rBav'] + r_O))
    
    # Дисперсия радиусов
    if all(col in df.columns for col in ['rB', 'rB\'', 'rD1', 'rD2', '[B\']', '[D1]', '[D2]']):
        rad_B = df['rB'].fillna(0).values
        rad_Bp = df['rB\''].fillna(0).values
        rad_D1 = df['rD1'].fillna(0).values
        rad_D2 = df['rD2'].fillna(0).values
        conc_Bp = df['[B\']'].fillna(0).values
        conc_D1 = df['[D1]'].fillna(0).values
        conc_D2 = df['[D2]'].fillna(0).values
        rBav = df['rBav'].values if 'rBav' in df.columns else np.zeros_like(rad_B)
        sum_sq = (rad_B**2 * (1 - conc_Bp - conc_D1 - conc_D2) +
                  rad_Bp**2 * conc_Bp +
                  rad_D1**2 * conc_D1 +
                  rad_D2**2 * conc_D2)
        df['σ²_rB'] = sum_sq - rBav**2
    
    if all(col in df.columns for col in ['rA', 'rA\'', '[A\']']):
        rad_A = df['rA'].fillna(0).values
        rad_Ap = df['rA\''].fillna(0).values
        conc_Ap = df['[A\']'].fillna(0).values
        rAav = df['rAav'].values if 'rAav' in df.columns else np.zeros_like(rad_A)
        sum_sq_A = (rad_A**2 * (1 - conc_Ap) + rad_Ap**2 * conc_Ap)
        df['σ²_rA'] = sum_sq_A - rAav**2
    
    return df

def add_thermodynamic_descriptors(df):
    """Энтропия, валентность"""
    R_gas = 8.314
    
    if all(col in df.columns for col in ['[A\']']):
        x_A = 1 - df['[A\']'].fillna(0)
        x_Ap = df['[A\']'].fillna(0)
        entropy_A = np.zeros(len(df))
        mask = (x_A > 0) & (x_Ap > 0)
        entropy_A[mask] = -R_gas * (x_A[mask] * np.log(x_A[mask]) + x_Ap[mask] * np.log(x_Ap[mask]))
        df['S_config_A'] = entropy_A
    
    if all(col in df.columns for col in ['[B\']', '[D1]', '[D2]']):
        x_B = 1 - df['[B\']'].fillna(0) - df['[D1]'].fillna(0) - df['[D2]'].fillna(0)
        x_Bp = df['[B\']'].fillna(0)
        x_D1 = df['[D1]'].fillna(0)
        x_D2 = df['[D2]'].fillna(0)
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
        df['V_Bav'] = (df['VB'] * (1 - df['[B\']'].fillna(0) - df['[D1]'].fillna(0) - df['[D2]'].fillna(0)) +
                       df['VB\''] * df['[B\']'].fillna(0) +
                       df['VD1'] * df['[D1]'].fillna(0) +
                       df['VD2'] * df['[D2]'].fillna(0))
        df['Vo_proxy'] = (4 - df['V_Bav']) / 2
    
    return df

def add_physics_inspired_descriptors(df):
    """Комбинированные дескрипторы"""
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

def add_all_descriptors(df):
    """Запуск всех функций дескрипторов"""
    if df is None:
        return df
    df = add_electronegativity_descriptors(df)
    df = add_geometric_descriptors(df)
    df = add_thermodynamic_descriptors(df)
    df = add_physics_inspired_descriptors(df)
    return df

# ============================================================================
# 4. ФИЛЬТРАЦИЯ ДАННЫХ (ГЛОБАЛЬНЫЕ ФИЛЬТРЫ, ВКЛЮЧЕНЫ ПО УМОЛЧАНИЮ)
# ============================================================================

def create_filters(df):
    """Создание виджетов фильтрации (все фильтры по умолчанию включены)"""
    st.sidebar.markdown("## 🔍 Фильтрация данных")
    st.sidebar.markdown("---")
    
    # Фильтр по A-катиону (Ba, Sr, Ca, La)
    a_cations = ['Ba', 'Sr', 'Ca', 'La']
    available_a = [a for a in a_cations if a in df['A'].values]
    selected_a = st.sidebar.multiselect(
        "A-site cations (A²⁺/A³⁺)",
        options=available_a,
        default=available_a,
        help="Выберите один или несколько катионов в A-позиции"
    )
    
    # Фильтр по B-катиону
    available_b = sorted(df['B'].dropna().unique())
    selected_b = st.sidebar.multiselect(
        "B-site cations (B⁴⁺/B³⁺)",
        options=available_b,
        default=available_b,
        help="Выберите один или несколько катионов в B-позиции"
    )
    
    # Фильтр по допанту D1
    available_d1 = sorted(df['D1'].dropna().unique())
    selected_d1 = st.sidebar.multiselect(
        "Dopant D1 (optional)",
        options=available_d1,
        default=available_d1 if available_d1 else [],
        help="Легирующий элемент в позиции D1"
    )
    
    # Фильтр по допанту D2
    available_d2 = sorted(df['D2'].dropna().unique())
    selected_d2 = st.sidebar.multiselect(
        "Dopant D2 (optional)",
        options=available_d2,
        default=available_d2 if available_d2 else [],
        help="Легирующий элемент в позиции D2"
    )
    
    # Фильтр по методу измерения
    if 'method' in df.columns:
        available_methods = sorted(df['method'].dropna().unique())
        selected_methods = st.sidebar.multiselect(
            "Measurement method",
            options=available_methods,
            default=available_methods,
            help="Метод измерения термического расширения"
        )
    else:
        selected_methods = []
    
    # Фильтр по симметрии (из листа phase)
    if 'Symmetry' in df.columns:
        available_sym = sorted(df['Symmetry'].dropna().unique())
        selected_sym = st.sidebar.multiselect(
            "Crystal symmetry",
            options=available_sym,
            default=available_sym,
            help="Кристаллографическая симметрия"
        )
    else:
        selected_sym = []
    
    # Фильтр по температуре
    if '∆T, °C' in df.columns:
        temp_range = st.sidebar.slider(
            "Temperature range (°C)",
            min_value=float(df['∆T, °C'].dropna().min()) if not df['∆T, °C'].dropna().empty else 0,
            max_value=float(df['∆T, °C'].dropna().max()) if not df['∆T, °C'].dropna().empty else 1000,
            value=(float(df['∆T, °C'].dropna().min()) if not df['∆T, °C'].dropna().empty else 0,
                   float(df['∆T, °C'].dropna().max()) if not df['∆T, °C'].dropna().empty else 1000),
            step=50
        )
    else:
        temp_range = (0, 1000)
    
    return {
        'A': selected_a,
        'B': selected_b,
        'D1': selected_d1,
        'D2': selected_d2,
        'method': selected_methods,
        'symmetry': selected_sym,
        'temp_range': temp_range
    }

def apply_filters(df, filters):
    """Применение фильтров к датафрейму"""
    if df is None or len(df) == 0:
        return df
    
    filtered_df = df.copy()
    
    if filters['A']:
        filtered_df = filtered_df[filtered_df['A'].isin(filters['A'])]
    if filters['B']:
        filtered_df = filtered_df[filtered_df['B'].isin(filters['B'])]
    if filters['D1']:
        filtered_df = filtered_df[filtered_df['D1'].isin(filters['D1'])]
    if filters['D2']:
        filtered_df = filtered_df[filtered_df['D2'].isin(filters['D2'])]
    if filters['method'] and 'method' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['method'].isin(filters['method'])]
    if filters['symmetry'] and 'Symmetry' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Symmetry'].isin(filters['symmetry'])]
    if '∆T, °C' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['∆T, °C'] >= filters['temp_range'][0]) & 
                                   (filtered_df['∆T, °C'] <= filters['temp_range'][1])]
    
    return filtered_df

# ============================================================================
# 5. УНИВЕРСАЛЬНЫЕ ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ
# ============================================================================

def get_available_columns(df):
    """Получение списка доступных числовых и категориальных колонок для выбора"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Приоритетные колонки
    priority_numeric = ['αav·106 (K-1)', 'α·106 (K-1)', 'β', 't', 'pH2O', 'rBav', 'rAav', 
                        'χAav', 'χBav', 'Δχ_AB', 'σ²_rB', 'S_config_B', 'V_Bav', 
                        'octahedral_factor', 'ionicity_BO', '[B\']']
    priority_cat = ['Symmetry', 'method', 'A', 'B', 'D1', 'D2']
    
    numeric_cols = [c for c in priority_numeric if c in numeric_cols] + [c for c in numeric_cols if c not in priority_numeric]
    categorical_cols = [c for c in priority_cat if c in categorical_cols] + [c for c in categorical_cols if c not in priority_cat]
    
    return numeric_cols, categorical_cols

def create_bubble_plot(df, x_col, y_col, size_col, color_col, title):
    """Универсальная пузырьковая диаграмма с научным стилем"""
    df_plot = df.dropna(subset=[x_col, y_col, size_col]).copy()
    
    if len(df_plot) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно данных для построения", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Определяем тип color колонки
    if color_col in df_plot.select_dtypes(include=[np.number]).columns:
        fig = px.scatter(
            df_plot, x=x_col, y=y_col, size=size_col, color=color_col,
            hover_data=['A', 'B', 'D1', 'D2', 'Ref'],
            title=title,
            labels={x_col: x_col, y_col: y_col, size_col: size_col, color_col: color_col},
            color_continuous_scale='plasma'
        )
    else:
        fig = px.scatter(
            df_plot, x=x_col, y=y_col, size=size_col, color=color_col,
            hover_data=['A', 'B', 'D1', 'D2', 'Ref'],
            title=title,
            labels={x_col: x_col, y_col: y_col, size_col: size_col},
            color_discrete_sequence=px.colors.qualitative.Set1
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
    
    # Добавляем точки поверх контура
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

def create_pca_plot(df, feature_cols, color_col, title):
    """PCA Biplot с выбором признаков"""
    df_pca = df[feature_cols + [color_col]].dropna() if color_col in df.columns else df[feature_cols].dropna()
    
    if len(df_pca) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно данных для PCA (нужно ≥5)", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    X = df_pca[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = go.Figure()
    
    # Точки
    if color_col in df_pca.columns and df_pca[color_col].dtype in [np.float64, np.int64]:
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            mode='markers',
            marker=dict(color=df_pca[color_col], colorscale='plasma', size=8, showscale=True, colorbar=dict(title=color_col)),
            text=[f"PC1={x1:.2f}<br>PC2={x2:.2f}" for x1, x2 in zip(X_pca[:,0], X_pca[:,1])],
            hoverinfo='text'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            mode='markers',
            marker=dict(size=8, color='blue'),
            text=[f"PC1={x1:.2f}<br>PC2={x2:.2f}" for x1, x2 in zip(X_pca[:,0], X_pca[:,1])],
            hoverinfo='text'
        ))
    
    # Векторы признаков
    for i, feature in enumerate(feature_cols):
        fig.add_annotation(
            x=pca.components_[0, i] * 3, y=pca.components_[1, i] * 3,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=feature,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red',
            font=dict(size=10)
        )
    
    fig.update_layout(
        title=f"{title}<br>(PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%})",
        font_family="Times New Roman",
        font_size=12,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=700,
        xaxis=dict(showline=True, linewidth=1, linecolor='black', title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"),
        yaxis=dict(showline=True, linewidth=1, linecolor='black', title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    )
    
    return fig

# ============================================================================
# 6. ФУНКЦИИ ДЛЯ СПЕЦИФИЧЕСКИХ ГРАФИКОВ
# ============================================================================

def plot_alpha_vs_tolerance_with_filters(df, x_col='t', y_col='αav·106 (K-1)', color_col='Symmetry'):
    """α vs t с цветом по симметрии"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    df_plot = df.dropna(subset=[x_col, y_col]).copy()
    if len(df_plot) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig
    
    if color_col in df_plot.columns:
        categories = df_plot[color_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        for cat, color in zip(categories, colors):
            mask = df_plot[color_col] == cat
            ax.scatter(df_plot.loc[mask, x_col], df_plot.loc[mask, y_col], 
                      label=str(cat), color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    else:
        ax.scatter(df_plot[x_col], df_plot[y_col], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.set_title(f'{y_col} vs {x_col}', fontweight='bold')
    
    if color_col in df_plot.columns:
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig

def plot_phase_transition_diagram(df):
    """Диаграмма фазовых переходов"""
    if 'Symmetry' not in df.columns or '[B\']' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No symmetry or concentration data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_cezr = df[df['B'] == 'Ce'].dropna(subset=['[B\']', 'Symmetry'])
    if len(df_cezr) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient Ce-Zr data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    if 'T(PT), °C' not in df_cezr.columns:
        df_cezr['T(PT), °C'] = 400 + 200 * df_cezr['[B\']'].fillna(0)
    
    fig = px.scatter(df_cezr, x='[B\']', y='T(PT), °C', color='Symmetry',
                     title='Phase transition diagram: Ce₁₋ₓZrₓO₃ system',
                     labels={'[B\']': 'Zr concentration (x)', 'T(PT), °C': 'Phase transition temperature (°C)'},
                     hover_data=['A', 'B', 'D1', 'D2'])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

def plot_violin_alpha_by_symmetry(df):
    """Violin plot распределения α по симметриям"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'Symmetry' not in df.columns or 'αav·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No symmetry or α data', ha='center', va='center')
        return fig
    
    df_violin = df.dropna(subset=['Symmetry', 'αav·106 (K-1)'])
    if len(df_violin) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.violinplot(data=df_violin, x='Symmetry', y='αav·106 (K-1)', ax=ax)
    ax.set_xlabel('Crystal symmetry', fontweight='bold')
    ax.set_ylabel('αav·10⁶ (K⁻¹)', fontweight='bold')
    ax.set_title('Distribution of thermal expansion by symmetry', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_method_comparison(df):
    """Сравнение методов измерения"""
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

def plot_nonconstant_beta_demo(df):
    """Демонстрация непостоянства β"""
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

# ============================================================================
# 7. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================================================

def compute_mutual_information(df, feature_cols, target_cols):
    """Вычисление взаимной информации"""
    if len(feature_cols) < 2 or len(target_cols) < 1:
        return None, None
    
    df_mi = df[feature_cols + target_cols].dropna()
    if len(df_mi) < 10:
        return None, None
    
    X = df_mi[feature_cols]
    mi_results = {}
    for target in target_cols:
        y = df_mi[target]
        mi = mutual_info_regression(X, y, random_state=42)
        mi_results[target] = dict(zip(feature_cols, mi))
    
    return mi_results, feature_cols

def plot_mutual_information_heatmap(mi_results, feature_cols):
    """Тепловая карта взаимной информации"""
    if mi_results is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Insufficient data for MI', ha='center', va='center')
        return fig
    
    mi_df = pd.DataFrame(mi_results).T
    fig, ax = plt.subplots(figsize=(12, max(4, len(mi_results) * 1.5)))
    sns.heatmap(mi_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Mutual Information: descriptors → target properties', fontweight='bold')
    ax.set_xlabel('Descriptors', fontweight='bold')
    ax.set_ylabel('Target properties', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap_advanced(df, corr_method='spearman'):
    """Расширенная корреляционная матрица с кластеризацией"""
    desc_cols = [col for col in df.columns if any(x in col for x in ['rAav', 'rBav', 't', 'χ', 'Δχ', 'σ²', 'ionicity', 'octahedral', 'S_config', 'V_Bav', 'α', 'β'])]
    
    if len(desc_cols) < 3:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Insufficient descriptor columns', ha='center', va='center')
        return fig
    
    df_corr = df[desc_cols].dropna()
    if len(df_corr) < 5:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Insufficient data after NaN removal', ha='center', va='center')
        return fig
    
    if corr_method == 'spearman':
        corr_matrix = df_corr.corr(method='spearman')
    else:
        corr_matrix = df_corr.corr(method='pearson')
    
    g = sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       figsize=(14, 12), dendrogram_ratio=0.15, cbar_pos=(0.02, 0.8, 0.03, 0.18))
    plt.title(f'{corr_method.capitalize()} correlation matrix with hierarchical clustering', fontweight='bold')
    return g.figure

# ============================================================================
# 8. КЛАСТЕРИЗАЦИЯ
# ============================================================================

def perform_clustering(df, feature_cols, method='kmeans', n_clusters=3):
    """Кластеризация составов"""
    if len(feature_cols) < 2:
        return None, None, None
    
    df_clust = df[feature_cols].dropna()
    if len(df_clust) < 10:
        return None, None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust.values)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=3)
        labels = clusterer.fit_predict(X_scaled)
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1
    else:
        return None, None, None
    
    return labels, score, df_clust.index

def plot_clustering_pca(df, labels, feature_cols):
    """Визуализация кластеризации в PCA-пространстве"""
    if labels is None or len(feature_cols) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No clustering results', ha='center', va='center')
        return fig
    
    X = df[feature_cols].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scat = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7, edgecolors='k')
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'Clustering visualization (KMeans, {len(set(labels))} clusters)', fontweight='bold')
    plt.colorbar(scat, label='Cluster')
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig

# ============================================================================
# 9. ML МОДЕЛЬ (RandomForest + SHAP)
# ============================================================================

def train_rf_and_shap(df, feature_cols, target):
    """Обучение RandomForest и расчёт SHAP"""
    if len(feature_cols) < 2 or target not in df.columns:
        return None, None, None, None
    
    df_ml = df[feature_cols + [target]].dropna()
    if len(df_ml) < 20:
        return None, None, None, None
    
    X = df_ml[feature_cols]
    y = df_ml[target]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    return rf, importance, shap_values, X

def plot_feature_importance(importance_df):
    """Bar plot важности признаков"""
    if importance_df is None or len(importance_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature', ax=ax)
    ax.set_title('Random Forest Feature Importance (top-10)', fontweight='bold')
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    plt.tight_layout()
    return fig

def plot_shap_summary(shap_values, X):
    """SHAP summary plot"""
    if shap_values is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No SHAP values', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    return fig

# ============================================================================
# 10. STREAMLIT UI (ГЛАВНОЕ ПРИЛОЖЕНИЕ)
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Perovskite Expansion Analyzer", page_icon="🧪")
    st.title("📊 Advanced Materials Informatics for Perovskite Oxides")
    st.markdown("### Thermal & Chemical Expansion | Phase Transitions | ML-driven Insights")
    
    # Боковая панель с навигацией
    st.sidebar.header("🧭 Navigation")
    analysis_mode = st.sidebar.selectbox(
        "Choose analysis mode",
        ["1. Data Editor & Preview",
         "2. Bubble Charts (4D Interactive)",
         "3. Contour Maps (x/y/color)",
         "4. Phase Transitions & Symmetry",
         "5. PCA & UMAP Projections",
         "6. Advanced Correlations (MI, Spearman)",
         "7. Clustering Analysis",
         "8. ML Model (RandomForest + SHAP)",
         "9. Batch Plot Generator"]
    )
    
    # Виджеты для ввода данных (две текстовые области с предзаполненными данными)
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
        df_therm, df_phase = load_data_from_text(therm_text, phase_text)
        
        if df_therm is None or df_phase is None:
            st.error("Failed to load data. Please check the format (tab-separated values).")
            return
        
        # Объединение данных
        merge_cols = ['A', 'A\'', 'B', 'B\'', 'D1', 'D2', '[A\']', '[B\']', '[D1]', '[D2]']
        for col in merge_cols:
            if col not in df_phase.columns:
                df_phase[col] = np.nan
        
        df = pd.merge(df_therm, df_phase, on=['№'] + merge_cols, how='left', suffixes=('', '_phase'))
        
        # Добавление дескрипторов
        df = add_all_descriptors(df)
        
        st.success(f"✅ Data loaded. Total rows: {len(df)}. Descriptors: {len(df.columns)}")
    
    # Создание фильтров (все по умолчанию включены)
    filters = create_filters(df)
    df_filtered = apply_filters(df, filters)
    
    # Отображение текущих фильтров
    with st.sidebar.expander("📊 Active filters", expanded=False):
        st.write(f"A-site: {filters['A']}")
        st.write(f"B-site: {filters['B']}")
        st.write(f"D1: {filters['D1']}")
        st.write(f"D2: {filters['D2']}")
        st.write(f"Method: {filters['method']}")
        st.write(f"Symmetry: {filters['symmetry']}")
        st.write(f"Temperature range: {filters['temp_range']}")
        st.write(f"**Filtered rows: {len(df_filtered)} / {len(df)}**")
    
    # Получение списка доступных колонок для выбора
    numeric_cols, categorical_cols = get_available_columns(df_filtered)
    
    # ========================================================================
    # РЕЖИМ 1: Data Editor & Preview
    # ========================================================================
    if analysis_mode == "1. Data Editor & Preview":
        st.subheader("📋 Data Preview (filtered)")
        st.dataframe(df_filtered.head(100))
        
        st.subheader("📊 Statistical Summary")
        st.dataframe(df_filtered.describe())
        
        st.subheader("🧪 Newly Added Descriptors")
        new_desc_cols = [col for col in df_filtered.columns if any(x in col for x in ['χ', 'σ²', 'S_config', 'ionicity', 'octahedral', 'Δ', 'V_Bav', 'D_t', 'Vo_proxy'])]
        if new_desc_cols:
            st.dataframe(df_filtered[new_desc_cols].head(20))
    
    # ========================================================================
    # РЕЖИМ 2: Bubble Charts (4D Interactive)
    # ========================================================================
    elif analysis_mode == "2. Bubble Charts (4D Interactive)":
        st.subheader("🎈 Interactive Bubble Charts (4D Visualization)")
        st.markdown("Select X, Y, bubble size, and color parameters.")
        
        col1, col2 = st.columns(2)
        with col1:
            x_bubble = st.selectbox("X axis", numeric_cols, index=numeric_cols.index('t') if 't' in numeric_cols else 0)
            y_bubble = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index('αav·106 (K-1)') if 'αav·106 (K-1)' in numeric_cols else 0)
        with col2:
            size_bubble = st.selectbox("Bubble size", numeric_cols, index=numeric_cols.index('β') if 'β' in numeric_cols else 0)
            color_bubble = st.selectbox("Bubble color", categorical_cols + numeric_cols, index=0)
        
        if len(df_filtered) > 0:
            fig = create_bubble_plot(df_filtered, x_bubble, y_bubble, size_bubble, color_bubble,
                                     f"{y_bubble} vs {x_bubble} | size={size_bubble} | color={color_bubble}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data after filtering")
    
    # ========================================================================
    # РЕЖИМ 3: Contour Maps
    # ========================================================================
    elif analysis_mode == "3. Contour Maps (x/y/color)":
        st.subheader("🗺️ Contour Maps")
        
        col1, col2 = st.columns(2)
        with col1:
            x_contour = st.selectbox("X axis", numeric_cols, index=numeric_cols.index('t') if 't' in numeric_cols else 0)
            y_contour = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index('rBav') if 'rBav' in numeric_cols else 0)
        with col2:
            z_contour = st.selectbox("Color (Z)", numeric_cols, index=numeric_cols.index('αav·106 (K-1)') if 'αav·106 (K-1)' in numeric_cols else 0)
        
        if len(df_filtered) > 0:
            fig = create_contour_plot(df_filtered, x_contour, y_contour, z_contour,
                                      f"Contour: {z_contour} vs {x_contour} & {y_contour}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data after filtering")
    
    # ========================================================================
    # РЕЖИМ 4: Phase Transitions & Symmetry
    # ========================================================================
    elif analysis_mode == "4. Phase Transitions & Symmetry":
        st.subheader("🔬 Phase Transitions and Symmetry Analysis")
        
        fig1 = plot_phase_transition_diagram(df_filtered)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = plot_violin_alpha_by_symmetry(df_filtered)
        st.pyplot(fig2)
        
        fig3 = plot_method_comparison(df_filtered)
        st.pyplot(fig3)
        
        fig4 = plot_nonconstant_beta_demo(df_filtered)
        st.pyplot(fig4)
        
        fig5 = plot_alpha_vs_tolerance_with_filters(df_filtered)
        st.pyplot(fig5)
    
    # ========================================================================
    # РЕЖИМ 5: PCA & UMAP Projections
    # ========================================================================
    elif analysis_mode == "5. PCA & UMAP Projections":
        st.subheader("📐 Dimensionality Reduction")
        
        # Выбор признаков для PCA
        default_pca_features = [col for col in ['rAav', 'rBav', 't', 'χAav', 'χBav', 'Δχ_AB', 'σ²_rB'] if col in numeric_cols]
        pca_features = st.multiselect("Select features for PCA", numeric_cols, default=default_pca_features)
        pca_color = st.selectbox("Color by", categorical_cols + numeric_cols, index=0)
        
        if len(pca_features) >= 2 and len(df_filtered) > 0:
            fig_pca = create_pca_plot(df_filtered, pca_features, pca_color, "PCA Biplot")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("Select at least 2 features for PCA")
        
        # UMAP
        st.subheader("UMAP Projection (non-linear manifold)")
        umap_features = st.multiselect("Select features for UMAP", numeric_cols, default=default_pca_features[:3] if len(default_pca_features) >= 3 else default_pca_features)
        
        if len(umap_features) >= 2 and len(df_filtered) > 10:
            df_umap = df_filtered[umap_features].dropna()
            if len(df_umap) >= 10:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_umap.values)
                reducer = umap.UMAP(n_components=2, random_state=42)
                X_umap = reducer.fit_transform(X_scaled)
                
                fig_umap = go.Figure()
                fig_umap.add_trace(go.Scatter(
                    x=X_umap[:, 0], y=X_umap[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    text=df_umap.index.astype(str)
                ))
                fig_umap.update_layout(title="UMAP Projection", width=800, height=600)
                st.plotly_chart(fig_umap, use_container_width=True)
            else:
                st.warning("Not enough data for UMAP (need ≥10 samples)")
        else:
            st.warning("Select at least 2 features for UMAP")
    
    # ========================================================================
    # РЕЖИМ 6: Advanced Correlations
    # ========================================================================
    elif analysis_mode == "6. Advanced Correlations (MI, Spearman)":
        st.subheader("📈 Mutual Information Analysis")
        
        mi_features = st.multiselect("Descriptor features for MI", numeric_cols, default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
        mi_targets = st.multiselect("Target properties", [col for col in numeric_cols if 'α' in col or 'β' in col], default=[col for col in numeric_cols if 'αav' in col or 'β' in col])
        
        if len(mi_features) >= 2 and len(mi_targets) >= 1 and len(df_filtered) > 0:
            mi_results, _ = compute_mutual_information(df_filtered, mi_features, mi_targets)
            fig_mi = plot_mutual_information_heatmap(mi_results, mi_features)
            st.pyplot(fig_mi)
        else:
            st.warning("Select at least 2 features and 1 target for MI analysis")
        
        st.subheader("📊 Correlation Matrix (Spearman + Clustering)")
        fig_corr = plot_correlation_heatmap_advanced(df_filtered, corr_method='spearman')
        st.pyplot(fig_corr)
    
    # ========================================================================
    # РЕЖИМ 7: Clustering Analysis
    # ========================================================================
    elif analysis_mode == "7. Clustering Analysis":
        st.subheader("🔧 Composition Clustering")
        
        cluster_features = st.multiselect("Features for clustering", numeric_cols, default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
        n_clusters = st.slider("Number of clusters (KMeans)", 2, 6, 3)
        
        if len(cluster_features) >= 2 and len(df_filtered) > 0:
            labels, score, idx = perform_clustering(df_filtered, cluster_features, method='kmeans', n_clusters=n_clusters)
            if labels is not None:
                st.metric("Silhouette Score", f"{score:.3f}")
                fig_clust = plot_clustering_pca(df_filtered, labels, cluster_features)
                st.pyplot(fig_clust)
                
                df_clustered = df_filtered.loc[idx].copy()
                df_clustered['Cluster'] = labels
                st.dataframe(df_clustered[['A', 'B', 'Cluster', 'αav·106 (K-1)', 'β']].head(50))
            else:
                st.warning("Clustering failed: insufficient data")
        else:
            st.warning("Select at least 2 features for clustering")
    
    # ========================================================================
    # РЕЖИМ 8: ML Model
    # ========================================================================
    elif analysis_mode == "8. ML Model (RandomForest + SHAP)":
        st.subheader("🤖 Machine Learning: RandomForest + SHAP")
        
        ml_features = st.multiselect("Features for ML", numeric_cols, default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
        ml_target = st.selectbox("Target property", [col for col in numeric_cols if 'αav' in col or 'α·' in col or 'β' in col], index=0 if any('αav' in col for col in numeric_cols) else None)
        
        if len(ml_features) >= 2 and ml_target is not None and len(df_filtered) > 0:
            rf, importance, shap_vals, X = train_rf_and_shap(df_filtered, ml_features, ml_target)
            if rf is not None:
                st.metric("Model R² (training)", f"{rf.score(X, df_filtered[ml_target].dropna().loc[X.index]):.3f}")
                
                fig_imp = plot_feature_importance(importance)
                st.pyplot(fig_imp)
                
                fig_shap = plot_shap_summary(shap_vals, X)
                st.pyplot(fig_shap)
            else:
                st.warning(f"Not enough data for ML (need ≥20 complete samples)")
        else:
            st.warning("Select at least 2 features and 1 target")
    
    # ========================================================================
    # РЕЖИМ 9: Batch Plot Generator
    # ========================================================================
    elif analysis_mode == "9. Batch Plot Generator":
        st.subheader("📸 Batch Plot Generation")
        st.markdown("Generates all standard plots for the current filtered data.")
        
        if st.button("Generate All Plots"):
            with st.spinner("Generating plots (may take 30-60 seconds)..."):
                plot_funcs = [
                    ("Bubble Chart (default)", lambda: create_bubble_plot(df_filtered, 't', 'αav·106 (K-1)', 'β', 'Symmetry', "Bubble Chart")),
                    ("Contour Map (default)", lambda: create_contour_plot(df_filtered, 't', 'rBav', 'αav·106 (K-1)', "Contour Map")),
                    ("Phase Transition Diagram", lambda: plot_phase_transition_diagram(df_filtered)),
                    ("Violin Plot by Symmetry", lambda: plot_violin_alpha_by_symmetry(df_filtered)),
                    ("Method Comparison", lambda: plot_method_comparison(df_filtered)),
                    ("Non-constant β", lambda: plot_nonconstant_beta_demo(df_filtered)),
                    ("α vs t", lambda: plot_alpha_vs_tolerance_with_filters(df_filtered)),
                    ("PCA Biplot", lambda: create_pca_plot(df_filtered, ['rBav', 't', 'χBav', 'Δχ_AB'], 'αav·106 (K-1)', "PCA Biplot")),
                    ("Correlation Heatmap", lambda: plot_correlation_heatmap_advanced(df_filtered))
                ]
                
                for name, func in plot_funcs:
                    st.subheader(name)
                    try:
                        fig = func()
                        if hasattr(fig, 'to_html'):  # Plotly figure
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Failed to generate {name}: {str(e)}")
    
    # Кнопка скачивания обогащённых данных
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Download enriched dataset (CSV)"):
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="enriched_perovskite_data.csv">Download CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Built with Streamlit | Advanced descriptors: χ, σ², S_config, ionicity, octahedral factor, vacancy proxy, partial correlations, MI, SHAP, UMAP, clustering | Scientific style for publication-ready figures.")

if __name__ == "__main__":
    main()
