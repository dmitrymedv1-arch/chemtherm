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
import re
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
# 2. УЛУЧШЕННАЯ ЗАГРУЗКА ДАННЫХ (С ПАРСИНГОМ ДИАПАЗОНОВ И МНОЖЕСТВЕННЫХ ЗНАЧЕНИЙ)
# ============================================================================

# Предзагруженные данные для первого листа (chem and therm expansion)
DEFAULT_THERM_DATA = """№	A	A'	B	B'	D1	D2	[A']	[B']	[D1]	[D2]	δ	rA	rA'	rAav	rB	rB'	rD1	rD2	rBav	t	rBav/rO	method	β	∆T, °C	α·106 (K-1)	T(bends), °C	αav·106 (K-1)	pH2O	Ref
1	Ba		Ce	Zr	Y	Yb		0.1	0.1	0.1	0.1	1.61		1.61	0.87	0.72	0.9	0.868	0.8578	0.942683768	0.612714286	dilatometry	0.0073	27-1000	10.6	400;600	10.6;4.73;10.1	0.0001	10.15826/chimtech.2024.11.4.22
2	Ba		Ce	Zr	Y	Yb		0.1	0.1	0.1	0.1	1.61		1.61	0.87	0.72	0.9	0.868	0.8578	0.942683768	0.612714286	HTXRD	0.0317	27-1000	10.6	300	10.7;8.7	0.02	10.15826/chimtech.2024.11.4.22
3	Ba		Ce	Zr	Y			0.1	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.858	0.942600271	0.612857143	HTND		20-900	11.2			0.00106	10.1021/acs.jpcc.1c08334
4	Ba		Ce	Zr	Y				0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.873	0.936379855	0.623571429	dilatometry	0.019	430-630		450		0.00106	10.1021/acs.jpcc.1c08334
5	Ba		Ce	Zr	Y			0.3	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.828	0.955292375	0.591428571	dilatometry	0.019	430-631				0.00106	10.1021/acs.jpcc.1c08334
6	Ba		Ce	Zr	Y			0.6	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.783	0.974984614	0.559285714	dilatometry	0.023	430-632				0.00106	10.1021/acs.jpcc.1c08334
7	Ba		Ce	Zr	Y			0.9	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.738	0.995505805	0.527142857	dilatometry	0.49	430-633				0.00106	10.1021/acs.jpcc.1c08334
8	Ba		Ce		Sm				0.2		0.1	1.61		1.61	0.87		0.958		0.8876	0.930403659	0.634	dilatometry	0.09718477	100-900	11.5	620	11.5;10.3		10.1016/j.jpowsour.2012.07.120
9	Ba		Ce		Nd				0.2		0.1	1.61		1.61	0.87		0.983		0.8926	0.928374514	0.637571429	dilatometry	0.0165	60-900	14	700	14;8.2	0.018	10.1016/j.jpowsour.2014.05.070
10	Ba		Ce	Zr	Y			0	0.2		0.1	1.61		1.61	0.87	0.72	0.9		0.876	0.935145611	0.625714286	dilatometry	0.00906816	100-900	11.6	620	11.6;8.3	0.018	10.1016/j.jpowsour.2014.12.024
11	Ba		Ce	Zr	Y			0.1	0.2		0.1	1.61		1.61	0.87	0.72	0.9		0.861	0.941349585	0.615	dilatometry		100-900	11.3	630	11.3;8.4	0.018	10.1016/j.jpowsour.2014.12.024
12	Ba		Ce	Zr	Y			0.2	0.2		0.1	1.61		1.61	0.87	0.72	0.9		0.846	0.947636425	0.604285714	dilatometry		100-900	11.32	620	11.32;8.4	0.018	10.1016/j.jpowsour.2014.12.024
13	Ba		Ce	Zr	Y			0.3	0.2		0.1	1.61		1.61	0.87	0.72	0.9		0.831	0.954007804	0.593571429	dilatometry		100-900	10.8	575	10.8;8.5	0.018	10.1016/j.jpowsour.2014.12.024"""

# Предзагруженные данные для второго листа (phase transition)
DEFAULT_PHASE_DATA = """№	A	A'	B	B'	D1	D2	[A']	[B']	[D1]	[D2]	δ	rA	rA'	rAav	rB	rB'	rD1	rD2	rBav	t	rBav/rO	pH2O	∆T, °C	Symmetry	Phase transitions (PT)	T (PT), °C	Ref
1	Ba		Ce	Zr	Y			0.36	0.1		0.05	1.61		1.61	0.87	0.72	0.9		0.819	0.959166927	0.585		30-1000				10.1063/1.5066970
2	Ba		Zr		Y				0		0	1.61		1.61	0.72		0.9		0.72	1.003958213	0.514285714		25	Cubic	Pm-3m		10.1088/1742-6596/1967/1/012015
3	Ba		Zr		Y				0.055		0.0275	1.61		1.61	0.72		0.9		0.7299	0.999291709	0.521357143		25	Cubic	Pm-3m		10.1088/1742-6596/1967/1/012015
4	Ba		Zr		Y				0.17		0.085	1.61		1.61	0.72		0.9		0.7506	0.989673306	0.536142857		25	Cubic	Pm-3m		10.1088/1742-6596/1967/1/012015
5	Ba		Sn		Y				0		0	1.61		1.61	0.69		0.9		0.69	1.018369096	0.492857143			Orthorombic;Rhombohedral;Cubic	Pm-3m, R-3c, Imma	352;476;711	10.1111/jace.12990
6	Ba		Sn		Y				0.05		0.025	1.61		1.61	0.69		0.9		0.7005	1.013278463	0.500357143			Cubic	Pm-3m		10.1111/jace.12990
7	Ba		Sn		Y				0.1		0.05	1.61		1.61	0.69		0.9		0.711	1.008238471	0.507857143			Cubic	Pm-3m		10.1111/jace.12990
8	Ba		Sn		Y				0.2		0.1	1.61		1.61	0.69		0.9		0.732	0.998307416	0.522857143			Cubic	Pm-3m		10.1111/jace.12990"""

def parse_temperature_range(temp_str):
    """Парсинг температурного диапазона вида '27-1000' в T_min и T_max"""
    if pd.isna(temp_str) or temp_str == '' or str(temp_str).strip() == '':
        return np.nan, np.nan
    
    temp_str = str(temp_str).strip()
    
    # Пробуем найти паттерн число-число
    match = re.match(r'(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)', temp_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Если одно число
    try:
        val = float(temp_str)
        return val, val
    except:
        return np.nan, np.nan

def parse_multiple_values(value_str):
    """Парсинг множественных значений вида '400;600' или '10.6;4.73;10.1' в список"""
    if pd.isna(value_str) or value_str == '' or str(value_str).strip() == '':
        return []
    
    value_str = str(value_str).strip()
    
    # Проверяем, есть ли разделитель
    if ';' in value_str:
        parts = value_str.split(';')
        result = []
        for part in parts:
            part = part.strip()
            if part:
                try:
                    # Замена запятой на точку для европейского формата
                    if ',' in part and '.' not in part:
                        part = part.replace(',', '.')
                    result.append(float(part))
                except:
                    result.append(np.nan)
        return result
    else:
        try:
            # Замена запятой на точку для европейского формата
            if ',' in value_str and '.' not in value_str:
                value_str = value_str.replace(',', '.')
            return [float(value_str)]
        except:
            return []

def expand_multiple_bends(df):
    """Разворачивает строки с множественными T(bends) в отдельные строки"""
    if 'T(bends), °C' not in df.columns:
        return df
    
    expanded_rows = []
    
    for idx, row in df.iterrows():
        bends_value = row['T(bends), °C']
        bends_list = parse_multiple_values(bends_value)
        
        if len(bends_list) <= 1:
            # Оставляем как есть
            expanded_rows.append(row)
        else:
            # Создаем отдельную строку для каждого значения
            for i, bend_val in enumerate(bends_list):
                if not pd.isna(bend_val):
                    new_row = row.copy()
                    new_row['T(bends), °C'] = bend_val
                    new_row['_bend_index'] = i
                    
                    # Если есть множественные α, берем соответствующее
                    if 'α·106 (K-1)' in df.columns:
                        alpha_values = parse_multiple_values(row['α·106 (K-1)'])
                        if i < len(alpha_values):
                            new_row['α·106 (K-1)'] = alpha_values[i]
                    
                    if 'αav·106 (K-1)' in df.columns:
                        alphaav_values = parse_multiple_values(row['αav·106 (K-1)'])
                        if i < len(alphaav_values):
                            new_row['αav·106 (K-1)'] = alphaav_values[i]
                    
                    expanded_rows.append(new_row)
    
    if len(expanded_rows) > len(df):
        df_expanded = pd.DataFrame(expanded_rows)
        return df_expanded
    else:
        return df

def parse_numeric_with_comma(value):
    """Преобразование числа с возможной запятой как десятичным разделителем"""
    if pd.isna(value) or value == '' or str(value).strip() == '':
        return np.nan
    try:
        value_str = str(value).strip()
        # Замена запятой на точку для европейского формата
        if ',' in value_str and '.' not in value_str:
            value_str = value_str.replace(',', '.')
        return float(value_str)
    except:
        return np.nan

@st.cache_data
def load_data_from_text(therm_text, phase_text, expand_bends=True):
    """Загрузка двух независимых таблиц с расширенным парсингом"""
    try:
        # Читаем как строки, сохраняя все колонки
        df_therm = pd.read_csv(io.StringIO(therm_text), sep='\t', dtype=str, keep_default_na=False)
        df_phase = pd.read_csv(io.StringIO(phase_text), sep='\t', dtype=str, keep_default_na=False)
        
        # Очистка колонок
        df_therm.columns = df_therm.columns.str.strip()
        df_phase.columns = df_phase.columns.str.strip()
        
        # Замена пустых строк и 'nan' на NaN
        df_therm = df_therm.replace(r'^\s*$', np.nan, regex=True)
        df_therm = df_therm.replace('nan', np.nan)
        df_therm = df_therm.replace('NaN', np.nan)
        
        df_phase = df_phase.replace(r'^\s*$', np.nan, regex=True)
        df_phase = df_phase.replace('nan', np.nan)
        df_phase = df_phase.replace('NaN', np.nan)
        
        # Создание колонки [A] из [A'] если её нет (A + A' = 1)
        if "[A']" in df_phase.columns and '[A]' not in df_phase.columns:
            conc_Ap = pd.to_numeric(df_phase["[A']"], errors='coerce')
            df_phase['[A]'] = 1 - conc_Ap
        
        # Парсинг температурных диапазонов для THERM
        if '∆T, °C' in df_therm.columns:
            temp_ranges = df_therm['∆T, °C'].apply(parse_temperature_range)
            df_therm['T_min'] = temp_ranges.apply(lambda x: x[0])
            df_therm['T_max'] = temp_ranges.apply(lambda x: x[1])
        
        # Парсинг температурных диапазонов для PHASE
        if '∆T, °C' in df_phase.columns:
            temp_ranges = df_phase['∆T, °C'].apply(parse_temperature_range)
            df_phase['T_min'] = temp_ranges.apply(lambda x: x[0])
            df_phase['T_max'] = temp_ranges.apply(lambda x: x[1])
        
        # Парсинг множественных T(PT) для PHASE
        if 'T (PT), °C' in df_phase.columns:
            # Оставляем оригинал, но создаем колонку с первым значением для простоты
            df_phase['T(PT)_first'] = df_phase['T (PT), °C'].apply(
                lambda x: parse_multiple_values(x)[0] if parse_multiple_values(x) else np.nan
            )
        
        # Список строковых колонок (которые НЕ преобразуем в числа)
        string_cols_therm = ['A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'method', 'Ref']
        string_cols_phase = ['A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Symmetry', 'Phase transitions (PT)', 'Ref']
        
        # Преобразование числовых колонок для THERM с поддержкой запятой
        for col in df_therm.columns:
            if col not in string_cols_therm and col != '№':
                df_therm[col] = df_therm[col].apply(parse_numeric_with_comma)
        
        # Преобразование числовых колонок для PHASE с поддержкой запятой
        for col in df_phase.columns:
            if col not in string_cols_phase and col != '№':
                if col == 'T (PT), °C':
                    # Для T(PT) сохраняем как строку для парсинга множественных значений
                    pass
                else:
                    df_phase[col] = df_phase[col].apply(parse_numeric_with_comma)
        
        # Разворачивание множественных T(bends) если нужно
        if expand_bends and 'T(bends), °C' in df_therm.columns:
            df_therm = expand_multiple_bends(df_therm)
        
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
# 4. ФУНКЦИИ СОПОСТАВЛЕНИЯ ДАННЫХ (CROSS-ANALYSIS) - УЛУЧШЕННЫЕ
# ============================================================================

def match_compositions_one_to_one(df_therm, df_phase):
    """
    Сопоставляет данные из THERM и PHASE по составу (A, B, [B'], D1, D2).
    Для каждого образца в THERM находит ПЕРВОЕ подходящее соответствие в PHASE.
    Используется для графиков, где нужно ОДНО значение Symmetry или T(PT).
    Улучшено: мягкое сравнение концентраций с округлением.
    """
    df_therm_matched = df_therm.copy()
    
    df_therm_matched['Symmetry'] = np.nan
    df_therm_matched['T(PT)_matched'] = np.nan
    df_therm_matched['Phase_transitions'] = np.nan
    
    for idx, row in df_therm_matched.iterrows():
        conc_Bp_therm = row.get('[B\']', 0)
        if pd.isna(conc_Bp_therm):
            conc_Bp_therm = 0
        
        # Округляем до 3 знаков для сравнения
        conc_Bp_therm_rounded = round(conc_Bp_therm, 3)
        
        mask = pd.Series([True] * len(df_phase))
        
        if 'A' in df_phase.columns and 'A' in row and not pd.isna(row['A']):
            mask = mask & (df_phase['A'] == row['A'])
        
        if 'B' in df_phase.columns and 'B' in row and not pd.isna(row['B']):
            mask = mask & (df_phase['B'] == row['B'])
        
        if '[B\']' in df_phase.columns:
            phase_conc = pd.to_numeric(df_phase['[B\']'], errors='coerce').fillna(0)
            phase_conc_rounded = np.round(phase_conc, 3)
            mask = mask & (np.abs(phase_conc_rounded - conc_Bp_therm_rounded) < 0.01)
        
        matches = df_phase[mask]
        
        if len(matches) > 0:
            first_match = matches.iloc[0]
            df_therm_matched.loc[idx, 'Symmetry'] = first_match.get('Symmetry', np.nan)
            
            # Используем первое значение T(PT) если их несколько
            tpt_value = first_match.get('T(PT)_first', first_match.get('T (PT), °C', np.nan))
            if pd.isna(tpt_value) and 'T (PT), °C' in first_match:
                tpt_list = parse_multiple_values(first_match['T (PT), °C'])
                tpt_value = tpt_list[0] if tpt_list else np.nan
            df_therm_matched.loc[idx, 'T(PT)_matched'] = tpt_value
            
            df_therm_matched.loc[idx, 'Phase_transitions'] = first_match.get('Phase transitions (PT)', np.nan)
    
    return df_therm_matched

def match_compositions_all_matches(df_therm, df_phase):
    """
    Сопоставляет данные из THERM и PHASE, сохраняя ВСЕ соответствия.
    Возвращает словарь {index_therm: list_of_matches}
    Используется для отображения нескольких вариантов в hover.
    Улучшено: мягкое сравнение концентраций с округлением.
    """
    matches_dict = {}
    
    for idx, row in df_therm.iterrows():
        conc_Bp_therm = row.get('[B\']', 0)
        if pd.isna(conc_Bp_therm):
            conc_Bp_therm = 0
        
        conc_Bp_therm_rounded = round(conc_Bp_therm, 3)
        
        mask = pd.Series([True] * len(df_phase))
        
        if 'A' in df_phase.columns and 'A' in row and not pd.isna(row['A']):
            mask = mask & (df_phase['A'] == row['A'])
        
        if 'B' in df_phase.columns and 'B' in row and not pd.isna(row['B']):
            mask = mask & (df_phase['B'] == row['B'])
        
        if '[B\']' in df_phase.columns:
            phase_conc = pd.to_numeric(df_phase['[B\']'], errors='coerce').fillna(0)
            phase_conc_rounded = np.round(phase_conc, 3)
            mask = mask & (np.abs(phase_conc_rounded - conc_Bp_therm_rounded) < 0.01)
        
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
# 5. УЛУЧШЕННАЯ ФИЛЬТРАЦИЯ ДАННЫХ (С МЯГКОЙ ЛОГИКОЙ И ОТЛАДКОЙ)
# ============================================================================

def create_filters(df_therm, df_phase):
    """Создание виджетов фильтрации (все фильтры по умолчанию НЕ активны)"""
    st.sidebar.markdown("## 🔍 Фильтрация данных")
    st.sidebar.markdown("---")
    
    # Инициализация состояния фильтров в session_state
    if 'filter_state' not in st.session_state:
        st.session_state.filter_state = {}
    
    available_a_therm = sorted(df_therm['A'].dropna().unique()) if 'A' in df_therm.columns and len(df_therm['A'].dropna()) > 0 else []
    available_a_phase = sorted(df_phase['A'].dropna().unique()) if df_phase is not None and 'A' in df_phase.columns and len(df_phase['A'].dropna()) > 0 else []
    available_a = sorted(list(set(available_a_therm + available_a_phase)))
    
    # Чекбокс для активации фильтра A-site
    enable_a = st.sidebar.checkbox("🔘 Filter by A-site", key="enable_a", value=False)
    
    if enable_a and available_a:
        selected_a = st.sidebar.multiselect(
            "A-site cations (A²⁺/A³⁺)",
            options=available_a,
            default=[],
            help="Выберите один или несколько катионов в A-позиции"
        )
    else:
        selected_a = []
    
    # B-site фильтр
    available_b_therm = sorted(df_therm['B'].dropna().unique()) if 'B' in df_therm.columns and len(df_therm['B'].dropna()) > 0 else []
    available_b_phase = sorted(df_phase['B'].dropna().unique()) if df_phase is not None and 'B' in df_phase.columns and len(df_phase['B'].dropna()) > 0 else []
    available_b = sorted(list(set(available_b_therm + available_b_phase)))
    
    enable_b = st.sidebar.checkbox("🔘 Filter by B-site", key="enable_b", value=False)
    
    if enable_b and available_b:
        selected_b = st.sidebar.multiselect(
            "B-site cations (B⁴⁺/B³⁺)",
            options=available_b,
            default=[],
            help="Выберите один или несколько катионов в B-позиции"
        )
    else:
        selected_b = []
    
    # D1 фильтр (мягкий - включает строки без допанта)
    if 'D1' in df_therm.columns:
        available_d1 = sorted(df_therm['D1'].dropna().unique())
    else:
        available_d1 = []
    
    enable_d1 = st.sidebar.checkbox("🔘 Filter by D1 dopant (soft: includes undoped)", key="enable_d1", value=False)
    
    if enable_d1 and available_d1:
        selected_d1 = st.sidebar.multiselect(
            "Dopant D1 (optional) - shows also samples without D1",
            options=available_d1,
            default=[],
            help="Легирующий элемент в позиции D1"
        )
    else:
        selected_d1 = []
    
    # D2 фильтр (мягкий - включает строки без допанта)
    if 'D2' in df_therm.columns:
        available_d2 = sorted(df_therm['D2'].dropna().unique())
    else:
        available_d2 = []
    
    enable_d2 = st.sidebar.checkbox("🔘 Filter by D2 dopant (soft: includes undoped)", key="enable_d2", value=False)
    
    if enable_d2 and available_d2:
        selected_d2 = st.sidebar.multiselect(
            "Dopant D2 (optional) - shows also samples without D2",
            options=available_d2,
            default=[],
            help="Легирующий элемент в позиции D2"
        )
    else:
        selected_d2 = []
    
    # Method фильтр
    if 'method' in df_therm.columns:
        available_methods = sorted(df_therm['method'].dropna().unique())
    else:
        available_methods = []
    
    enable_method = st.sidebar.checkbox("🔘 Filter by measurement method", key="enable_method", value=False)
    
    if enable_method and available_methods:
        selected_methods = st.sidebar.multiselect(
            "Measurement method",
            options=available_methods,
            default=[],
            help="Метод измерения термического расширения"
        )
    else:
        selected_methods = []
    
    # Symmetry фильтр
    if 'Symmetry' in df_phase.columns:
        available_sym = sorted(df_phase['Symmetry'].dropna().unique())
    else:
        available_sym = []
    
    enable_sym = st.sidebar.checkbox("🔘 Filter by crystal symmetry", key="enable_sym", value=False)
    
    if enable_sym and available_sym:
        selected_sym = st.sidebar.multiselect(
            "Crystal symmetry",
            options=available_sym,
            default=[],
            help="Кристаллографическая симметрия"
        )
    else:
        selected_sym = []
    
    # Температурный фильтр
    temp_range = (0, 1000)
    enable_temp = st.sidebar.checkbox("🔘 Filter by temperature range", key="enable_temp", value=False)
    
    if enable_temp and 'T_min' in df_therm.columns:
        temp_vals_min = df_therm['T_min'].dropna()
        temp_vals_max = df_therm['T_max'].dropna()
        if len(temp_vals_min) > 0 and len(temp_vals_max) > 0:
            all_temps = np.concatenate([temp_vals_min.values, temp_vals_max.values])
            try:
                temp_range = st.sidebar.slider(
                    "Temperature range (°C)",
                    min_value=float(np.nanmin(all_temps)),
                    max_value=float(np.nanmax(all_temps)),
                    value=(float(np.nanmin(all_temps)), float(np.nanmax(all_temps))),
                    step=50.0
                )
            except:
                pass
    
    # Кнопка сброса всех фильтров
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset all filters", use_container_width=True):
        st.session_state.filter_state = {}
        st.rerun()
    
    return {
        'A': selected_a,
        'B': selected_b,
        'D1': selected_d1,
        'D2': selected_d2,
        'method': selected_methods,
        'symmetry': selected_sym,
        'temp_range': temp_range,
        'enable_a': enable_a,
        'enable_b': enable_b,
        'enable_d1': enable_d1,
        'enable_d2': enable_d2,
        'enable_method': enable_method,
        'enable_sym': enable_sym,
        'enable_temp': enable_temp
    }

def apply_filters_to_therm(df_therm, filters):
    """Применение фильтров к THERM таблице с мягкой логикой (сохранение NaN)"""
    if df_therm is None or len(df_therm) == 0:
        return df_therm
    
    filtered_df = df_therm.copy()
    
    # A-site фильтр
    if filters['enable_a'] and filters['A']:
        filtered_df = filtered_df[filtered_df['A'].isin(filters['A']) | filtered_df['A'].isna()]
    
    # B-site фильтр
    if filters['enable_b'] and filters['B']:
        filtered_df = filtered_df[filtered_df['B'].isin(filters['B']) | filtered_df['B'].isna()]
    
    # D1 фильтр (мягкий: показываем если выбранный допант ИЛИ нет допанта)
    if filters['enable_d1'] and filters['D1']:
        filtered_df = filtered_df[filtered_df['D1'].isin(filters['D1']) | filtered_df['D1'].isna()]
    
    # D2 фильтр (мягкий)
    if filters['enable_d2'] and filters['D2']:
        filtered_df = filtered_df[filtered_df['D2'].isin(filters['D2']) | filtered_df['D2'].isna()]
    
    # Method фильтр
    if filters['enable_method'] and filters['method']:
        filtered_df = filtered_df[filtered_df['method'].isin(filters['method']) | filtered_df['method'].isna()]
    
    # Температурный фильтр (используем T_min и T_max)
    if filters['enable_temp'] and 'T_min' in filtered_df.columns and 'T_max' in filtered_df.columns:
        t_min, t_max = filters['temp_range']
        # Показываем строки, где интервал пересекается с выбранным диапазоном
        mask = (
            ((filtered_df['T_min'] <= t_max) | filtered_df['T_min'].isna()) &
            ((filtered_df['T_max'] >= t_min) | filtered_df['T_max'].isna())
        )
        filtered_df = filtered_df[mask]
    
    return filtered_df

def apply_filters_to_phase(df_phase, filters):
    """Применение фильтров к PHASE таблице с мягкой логикой (сохранение NaN)"""
    if df_phase is None or len(df_phase) == 0:
        return df_phase
    
    filtered_df = df_phase.copy()
    
    # A-site фильтр
    if filters['enable_a'] and filters['A']:
        filtered_df = filtered_df[filtered_df['A'].isin(filters['A']) | filtered_df['A'].isna()]
    
    # B-site фильтр
    if filters['enable_b'] and filters['B']:
        filtered_df = filtered_df[filtered_df['B'].isin(filters['B']) | filtered_df['B'].isna()]
    
    # D1 фильтр (мягкий)
    if filters['enable_d1'] and filters['D1'] and 'D1' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['D1'].isin(filters['D1']) | filtered_df['D1'].isna()]
    
    # D2 фильтр (мягкий)
    if filters['enable_d2'] and filters['D2'] and 'D2' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['D2'].isin(filters['D2']) | filtered_df['D2'].isna()]
    
    # Symmetry фильтр
    if filters['enable_sym'] and filters['symmetry']:
        filtered_df = filtered_df[filtered_df['Symmetry'].isin(filters['symmetry']) | filtered_df['Symmetry'].isna()]
    
    # Температурный фильтр для PHASE
    if filters['enable_temp'] and 'T_min' in filtered_df.columns and 'T_max' in filtered_df.columns:
        t_min, t_max = filters['temp_range']
        mask = (
            ((filtered_df['T_min'] <= t_max) | filtered_df['T_min'].isna()) &
            ((filtered_df['T_max'] >= t_min) | filtered_df['T_max'].isna())
        )
        filtered_df = filtered_df[mask]
    
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
        priority_numeric = ['t', 'rBav', 'rAav', 'χAav', 'χBav', 'Δχ_AB', 'σ²_rB', 'T(PT)_first']
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
# 7. ГРАФИКИ ТОЛЬКО ДЛЯ THERM (13 ГРАФИКОВ) - БЕЗ ИЗМЕНЕНИЙ
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
    
    if 'T_mid' not in df.columns:
        if 'T_min' in df.columns and 'T_max' in df.columns:
            df['T_mid'] = (df['T_min'] + df['T_max']) / 2
    
    if 'T_mid' not in df.columns or 'α·106 (K-1)' not in df.columns:
        ax.text(0.5, 0.5, 'No temperature or α data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['T_mid', 'α·106 (K-1)'])
    
    if len(df_plot) > 0:
        ax.scatter(df_plot['T_mid'], df_plot['α·106 (K-1)'], alpha=0.7, c='purple', edgecolors='k', s=50)
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
# 8. ГРАФИКИ ТОЛЬКО ДЛЯ PHASE (7 ГРАФИКОВ) - БЕЗ ИЗМЕНЕНИЙ
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
    
    y_col = 'T(PT)_first' if 'T(PT)_first' in df_cezr.columns else 'T (PT), °C'
    if y_col not in df_cezr.columns:
        df_cezr['T_dummy'] = 400 + 200 * df_cezr['[B\']'].fillna(0)
        y_col = 'T_dummy'
    
    fig = px.scatter(df_cezr, x='[B\']', y=y_col, color='Symmetry',
                     title='Phase transition diagram: Ce₁₋ₓZrₓO₃ system',
                     labels={'[B\']': 'Zr concentration (x)', y_col: 'Phase transition temperature (°C)'},
                     hover_data=['A', 'B', 'D1', 'D2', 'Phase transitions (PT)'])
    
    fig.update_layout(font_family="Times New Roman", width=800, height=600)
    return fig

def plot_phase_t_vs_tolerance(df):
    """График 15: T(PT) vs tolerance factor"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 't' not in df.columns:
        ax.text(0.5, 0.5, 'No t data', ha='center', va='center')
        return fig
    
    y_col = 'T(PT)_first' if 'T(PT)_first' in df.columns else 'T (PT), °C'
    if y_col not in df.columns:
        ax.text(0.5, 0.5, 'No T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['t', y_col])
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, 't'], df_plot.loc[mask, y_col], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['t'], df_plot[y_col], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Tolerance factor (t)', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperature vs tolerance factor', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_t_vs_disorder(df):
    """График 16: T(PT) vs σ²_rB (беспорядок)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'σ²_rB' not in df.columns:
        ax.text(0.5, 0.5, 'No disorder data', ha='center', va='center')
        return fig
    
    y_col = 'T(PT)_first' if 'T(PT)_first' in df.columns else 'T (PT), °C'
    if y_col not in df.columns:
        ax.text(0.5, 0.5, 'No T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['σ²_rB', y_col])
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, 'σ²_rB'], df_plot.loc[mask, y_col], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['σ²_rB'], df_plot[y_col], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
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
    
    if '[B\']' not in df.columns:
        ax.text(0.5, 0.5, 'No doping data', ha='center', va='center')
        return fig
    
    y_col = 'T(PT)_first' if 'T(PT)_first' in df.columns else 'T (PT), °C'
    if y_col not in df.columns:
        ax.text(0.5, 0.5, 'No T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['[B\']', y_col])
    
    if 'Symmetry' in df_plot.columns:
        symmetries = df_plot['Symmetry'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(symmetries)))
        for sym, color in zip(symmetries, colors):
            mask = df_plot['Symmetry'] == sym
            ax.scatter(df_plot.loc[mask, '[B\']'], df_plot.loc[mask, y_col], 
                      label=sym, color=color, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    else:
        ax.scatter(df_plot['[B\']'], df_plot[y_col], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Dopant concentration [B\']', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperature vs doping level', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_violin_by_symmetry(df):
    """График 19: Violin plot T(PT) по симметриям"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'Symmetry' not in df.columns:
        ax.text(0.5, 0.5, 'No Symmetry data', ha='center', va='center')
        return fig
    
    y_col = 'T(PT)_first' if 'T(PT)_first' in df.columns else 'T (PT), °C'
    if y_col not in df.columns:
        ax.text(0.5, 0.5, 'No T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['Symmetry', y_col])
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.violinplot(data=df_plot, x='Symmetry', y=y_col, ax=ax)
    ax.set_xlabel('Crystal symmetry', fontweight='bold')
    ax.set_ylabel('Phase transition temperature (°C)', fontweight='bold')
    ax.set_title('Distribution of PT temperature by symmetry', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

def plot_phase_t_distribution_by_type(df):
    """График 20: Распределение T(PT) по типам переходов"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'Phase transitions (PT)' not in df.columns:
        ax.text(0.5, 0.5, 'No PT type data', ha='center', va='center')
        return fig
    
    y_col = 'T(PT)_first' if 'T(PT)_first' in df.columns else 'T (PT), °C'
    if y_col not in df.columns:
        ax.text(0.5, 0.5, 'No T(PT) data', ha='center', va='center')
        return fig
    
    df_plot = df.dropna(subset=['Phase transitions (PT)', y_col])
    if len(df_plot) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    sns.boxplot(data=df_plot, x='Phase transitions (PT)', y=y_col, ax=ax)
    ax.set_xlabel('Phase transition type', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('Phase transition temperatures by transition type', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig

# ============================================================================
# 9. КРОСС-ГРАФИКИ (THERM + PHASE) — 10 ГРАФИКОВ - БЕЗ ИЗМЕНЕНИЙ
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
# 10. STREAMLIT UI (ГЛАВНОЕ ПРИЛОЖЕНИЕ) - С ДОБАВЛЕННОЙ ОТЛАДКОЙ
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
    
    # Опция для разворачивания множественных T(bends)
    expand_bends = st.sidebar.checkbox(
        "🔧 Expand multiple T(bends) into separate rows", 
        value=True,
        help="If enabled, samples with multiple bend temperatures (e.g., '400;600') will be split into multiple rows"
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
        df_therm_raw, df_phase_raw = load_data_from_text(therm_text, phase_text, expand_bends=expand_bends)
        
        if df_therm_raw is None or df_phase_raw is None:
            st.error("Failed to load data. Please check the format (tab-separated values).")
            return
        
        df_therm = add_all_descriptors(df_therm_raw, "therm")
        df_phase = add_all_descriptors(df_phase_raw, "phase")
        
        # Создаем T_mid для температурных графиков
        if 'T_min' in df_therm.columns and 'T_max' in df_therm.columns:
            df_therm['T_mid'] = (df_therm['T_min'] + df_therm['T_max']) / 2
        
        st.success(f"✅ THERM: {len(df_therm)} rows, {len(df_therm.columns)} columns | PHASE: {len(df_phase)} rows, {len(df_phase.columns)} columns")
        
        # Отладочная информация
        with st.expander("🔧 Debug Info - Check loaded data", expanded=False):
            st.write("### THERM columns:", list(df_therm.columns))
            st.write("### THERM first 3 rows:")
            st.dataframe(df_therm.head(3))
            st.write("### THERM dtypes:")
            st.write(df_therm.dtypes)
            
            st.write("### PHASE columns:", list(df_phase.columns))
            st.write("### PHASE first 3 rows:")
            st.dataframe(df_phase.head(3))
            st.write("### PHASE dtypes:")
            st.write(df_phase.dtypes)
            
            # Проверяем наличие ключевых колонок
            st.write("### Key columns check:")
            st.write(f"t in THERM: {'t' in df_therm.columns}")
            st.write(f"αav·106 (K-1) in THERM: {'αav·106 (K-1)' in df_therm.columns}")
            st.write(f"Symmetry in PHASE: {'Symmetry' in df_phase.columns}")
            st.write(f"T(bends) in THERM: {'T(bends), °C' in df_therm.columns}")
            st.write(f"T_min/T_max in THERM: {'T_min' in df_therm.columns} / {'T_max' in df_therm.columns}")
            
            # Статистика по NaN
            st.write("### NaN counts in THERM:")
            st.write(df_therm.isna().sum())
    
    # Фильтры
    filters = create_filters(df_therm, df_phase)
    df_therm_filtered = apply_filters_to_therm(df_therm, filters)
    df_phase_filtered = apply_filters_to_phase(df_phase, filters)
    
    # Отображение статистики фильтрации
    with st.sidebar.expander("📊 Filter statistics", expanded=True):
        st.write(f"THERM: **{len(df_therm_filtered)}** / {len(df_therm)} rows")
        st.write(f"PHASE: **{len(df_phase_filtered)}** / {len(df_phase)} rows")
        
        if len(df_therm_filtered) == 0 and len(df_therm) > 0:
            st.warning("⚠️ No data after filtering! Try:")
            st.write("- Disable some filters (uncheck the checkboxes)")
            st.write("- Click 'Reset all filters' button")
            st.write("- Check debug info above for data issues")
        
        # Показываем активные фильтры
        active_filters = []
        if filters['enable_a'] and filters['A']:
            active_filters.append(f"A={filters['A']}")
        if filters['enable_b'] and filters['B']:
            active_filters.append(f"B={filters['B']}")
        if filters['enable_method'] and filters['method']:
            active_filters.append(f"method={filters['method']}")
        if filters['enable_sym'] and filters['symmetry']:
            active_filters.append(f"symmetry={filters['symmetry']}")
        if filters['enable_temp']:
            active_filters.append(f"T={filters['temp_range']}°C")
        
        if active_filters:
            st.write("Active filters:", ", ".join(active_filters))
        else:
            st.write("No active filters - showing all data")
    
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
            st.warning("⚠️ No data after filtering. Please adjust filters or check the debug info above.")
        
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
            st.warning("⚠️ No data after filtering. Please adjust filters or check the debug info above.")
        
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
            st.warning("⚠️ Insufficient data in one or both tables after filtering. Please adjust filters.")
        
        st.subheader("📋 Matched Data Preview (THERM + Symmetry)")
        df_matched_preview = match_compositions_one_to_one(df_therm_filtered, df_phase_filtered)
        preview_cols = [c for c in ['A', 'B', '[B\']', 'αav·106 (K-1)', 'β', 'Symmetry', 'T(PT)_matched'] if c in df_matched_preview.columns]
        st.dataframe(df_matched_preview[preview_cols].head(50))
    
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
