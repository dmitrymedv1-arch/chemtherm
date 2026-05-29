# app.py
# Главное приложение для анализа термического и химического расширения перовскитов
# Интеграция с alpha_beta.py и phase_transitions.py

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
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
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
import warnings
warnings.filterwarnings('ignore')

# Импорт локальных модулей (предполагается, что они находятся в той же директории)
try:
    from alpha_beta import get_dataframe as get_alpha_beta_df
    from phase_transitions import get_dataframe as get_phase_transitions_df
    ALPHA_BETA_AVAILABLE = True
    PHASE_AVAILABLE = True
except ImportError:
    ALPHA_BETA_AVAILABLE = False
    PHASE_AVAILABLE = False
    st.warning("Модули alpha_beta.py или phase_transitions.py не найдены. Используются встроенные демо-данные.")

# ============================================================================
# 0. НАУЧНЫЙ СТИЛЬ ГРАФИКОВ (PUBLICATION-READY)
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
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'lines.linewidth': 0.6,
        'lines.markersize': 5,
        'errorbar.capsize': 3,
        'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
        'contour.negative_linestyle': 'dashed',
    })

apply_scientific_style()

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ ИЗ ЛОКАЛЬНЫХ МОДУЛЕЙ
# ============================================================================

@st.cache_data
def load_and_merge_data():
    """Загрузка данных из alpha_beta.py и phase_transitions.py, объединение"""
    
    if ALPHA_BETA_AVAILABLE and PHASE_AVAILABLE:
        df_alpha_beta = get_alpha_beta_df()
        df_phase = get_phase_transitions_df()
        
        # Очистка колонок
        df_alpha_beta.columns = df_alpha_beta.columns.str.strip()
        df_phase.columns = df_phase.columns.str.strip()
        
        # Замена пустых строк на NaN
        df_alpha_beta.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df_phase.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Ключи для объединения
        merge_cols = ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', '[A\']', '[B\']', '[D1]', '[D2]']
        for col in merge_cols:
            if col not in df_phase.columns:
                df_phase[col] = np.nan
            if col not in df_alpha_beta.columns:
                df_alpha_beta[col] = np.nan
        
        # Объединение
        df = pd.merge(df_alpha_beta, df_phase, on=merge_cols, how='left', suffixes=('', '_phase'))
        
        # Преобразование числовых колонок
        numeric_cols = ['[A\']', '[B\']', '[D1]', '[D2]', 'δ', 'rA', 'rA\'', 'rB', 'rB\'', 'rD1', 'rD2', 
                        't', 'β', 'pH2O', 'α·106 (K-1)', 'αav·106 (K-1)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    else:
        # Демо-данные если модули не найдены
        st.info("Используются демо-данные. Для полной функциональности создайте alpha_beta.py и phase_transitions.py")
        return create_demo_data()

def create_demo_data():
    """Создание демонстрационных данных при отсутствии модулей"""
    np.random.seed(42)
    n_samples = 50
    demo_data = []
    
    a_elements = ['Ba', 'Sr', 'Ca', 'La']
    b_elements = ['Ce', 'Zr', 'Sn', 'Ti']
    dopants = ['Y', 'Yb', 'Sc', 'In', 'Fe']
    methods = ['dilatometry', 'HTXRD', 'HTND']
    symmetries = ['Cubic', 'Orthorhombic', 'Rhombohedral', 'Tetragonal']
    
    for i in range(n_samples):
        a = np.random.choice(a_elements)
        b = np.random.choice(b_elements)
        dopant = np.random.choice(dopants)
        method = np.random.choice(methods)
        symmetry = np.random.choice(symmetries)
        
        rA = {'Ba': 1.61, 'Sr': 1.44, 'Ca': 1.34, 'La': 1.36}.get(a, 1.5)
        rB = {'Ce': 0.87, 'Zr': 0.72, 'Sn': 0.69, 'Ti': 0.605}.get(b, 0.8)
        rD = {'Y': 0.9, 'Yb': 0.868, 'Sc': 0.745, 'In': 0.8, 'Fe': 0.645}.get(dopant, 0.8)
        
        conc_dopant = np.random.uniform(0.05, 0.3)
        rBav = rB * (1 - conc_dopant) + rD * conc_dopant
        t = (rA + 1.4) / (np.sqrt(2) * (rBav + 1.4))
        
        alpha = 5 + 15 * np.random.random() + 5 * (1 - t)
        beta = 0.001 + 0.05 * np.random.random()
        ph2o = np.random.choice([0.0001, 0.018, 0.0312, 0.02])
        
        demo_data.append({
            '№': i+1, 'A': a, 'A\'': '', 'B': b, 'B\'': '', 'D1': dopant, 'D2': '',
            '[A\']': np.nan, '[B\']': conc_dopant, '[D1]': conc_dopant, '[D2]': np.nan,
            'δ': np.random.uniform(0, 0.1), 'rA': rA, 'rA\'': np.nan, 'rAav': rA,
            'rB': rB, 'rB\'': np.nan, 'rD1': rD, 'rD2': np.nan, 'rBav': rBav,
            't': t, 'rBav/rO': rBav/1.4, 'method': method, 'β': beta,
            '∆T, °C': f"25-{np.random.randint(600, 1000)}", 'α·106 (K-1)': alpha,
            'T(bends), °C': np.random.randint(300, 700), 'αav·106 (K-1)': alpha,
            'pH2O': ph2o, 'Ref': 'demo', 'Symmetry': symmetry,
            'Phase transitions (PT)': f"{symmetry} phase", 'T (PT), °C': np.random.randint(400, 900)
        })
    
    return pd.DataFrame(demo_data)

# ============================================================================
# 2. ДОБАВЛЕНИЕ НОВЫХ ДЕСКРИПТОРОВ (Feature Engineering)
# ============================================================================

# Таблица электроотрицательностей (Pauling scale)
CHI_TABLE = {
    'Ba': 0.89, 'Sr': 0.95, 'Ca': 1.00, 'La': 1.10, 'Ce': 1.12,
    'Zr': 1.33, 'Sn': 1.96, 'Ti': 1.54, 'Sc': 1.36, 'Y': 1.22,
    'Yb': 1.10, 'In': 1.78, 'Fe': 1.83, 'Gd': 1.20, 'Sm': 1.17,
    'Nd': 1.14, 'Eu': 1.20, 'Dy': 1.22, 'Zn': 1.65, 'Pr': 1.13,
    'Ho': 1.23, 'Tm': 1.25, 'Tb': 1.20, 'Hf': 1.30, 'Pb': 2.33,
    'Bi': 2.02, 'Al': 1.61, 'Ga': 1.81, 'Ge': 2.01, 'Si': 1.90
}

# Таблица валентностей
VALENCE_TABLE = {
    'Ba': 2, 'Sr': 2, 'Ca': 2, 'La': 3, 'Ce': 4, 'Zr': 4, 'Y': 3, 'Yb': 3,
    'Sc': 3, 'In': 3, 'Fe': 3, 'Zn': 2, 'Sn': 4, 'Ti': 4, 'Gd': 3, 'Sm': 3,
    'Nd': 3, 'Eu': 3, 'Dy': 3, 'Pr': 3, 'Ho': 3, 'Tm': 3, 'Tb': 3, 'Hf': 4
}

def get_chi(el):
    """Получение электроотрицательности элемента"""
    if pd.isna(el) or el == '':
        return np.nan
    return CHI_TABLE.get(str(el), np.nan)

def get_valence(el):
    """Получение валентности элемента"""
    if pd.isna(el) or el == '':
        return 0
    return VALENCE_TABLE.get(str(el), 0)

def add_electronegativity_descriptors(df):
    """Добавление электроотрицательности и производных"""
    if df is None or len(df) == 0:
        return df
    
    # Добавление χ для каждого катиона
    for pos in ['A', 'A\'', 'B', 'B\'', 'D1', 'D2']:
        if pos in df.columns:
            df[f'χ{pos}'] = df[pos].apply(get_chi)
    
    # Средние значения χ на A-позиции
    if 'χA' in df.columns and 'χA\'' in df.columns and '[A\']' in df.columns:
        df['χAav'] = df['χA'] * (1 - df['[A\']'].fillna(0)) + df['χA\''] * df['[A\']'].fillna(0)
    elif 'χA' in df.columns:
        df['χAav'] = df['χA']
    
    # Средние значения χ на B-позиции
    if all(col in df.columns for col in ['χB', 'χB\'', 'χD1', 'χD2']):
        conc_Bp = df['[B\']'].fillna(0) if '[B\']' in df.columns else 0
        conc_D1 = df['[D1]'].fillna(0) if '[D1]' in df.columns else 0
        conc_D2 = df['[D2]'].fillna(0) if '[D2]' in df.columns else 0
        total_conc = conc_Bp + conc_D1 + conc_D2
        
        df['χBav'] = (df['χB'] * (1 - total_conc) +
                      df['χB\''] * conc_Bp +
                      df['χD1'] * conc_D1 +
                      df['χD2'] * conc_D2)
    elif 'χB' in df.columns:
        df['χBav'] = df['χB']
    
    # Производные
    if 'χAav' in df.columns and 'χBav' in df.columns:
        df['Δχ_AB'] = np.abs(df['χAav'] - df['χBav'])
        df['χ_ratio_AB'] = df['χAav'] / df['χBav']
        df['χ_total'] = (df['χAav'] + df['χBav']) / 2
        
        # Ионность связи по Поллингу (χO = 3.44)
        chi_O = 3.44
        df['ionicity_AO'] = 1 - np.exp(-0.25 * (df['χAav'] - chi_O)**2)
        df['ionicity_BO'] = 1 - np.exp(-0.25 * (df['χBav'] - chi_O)**2)
    
    return df

def add_geometric_descriptors(df):
    """Геометрические дескрипторы"""
    if df is None or len(df) == 0:
        return df
    
    r_O = 1.4
    
    # Октаэдрический фактор
    if 'rBav' in df.columns:
        df['octahedral_factor'] = df['rBav'] / r_O
    
    # Глобальная нестабильность
    if 't' in df.columns:
        df['D_t'] = np.abs(1 - df['t'])
    
    # Разность радиусов A и B
    if 'rAav' in df.columns and 'rBav' in df.columns:
        df['Δr_AB'] = np.abs(df['rAav'] - df['rBav'])
        df['Δr_AB_norm'] = df['Δr_AB'] / r_O
    
    # Дисперсия ионных радиусов (беспорядок подрешётки B)
    if all(col in df.columns for col in ['rB', 'rB\'', 'rD1', 'rD2']):
        conc_Bp = df['[B\']'].fillna(0) if '[B\']' in df.columns else 0
        conc_D1 = df['[D1]'].fillna(0) if '[D1]' in df.columns else 0
        conc_D2 = df['[D2]'].fillna(0) if '[D2]' in df.columns else 0
        total_conc = conc_Bp + conc_D1 + conc_D2
        
        rad_B = df['rB'].fillna(0)
        rad_Bp = df['rB\''].fillna(0)
        rad_D1 = df['rD1'].fillna(0)
        rad_D2 = df['rD2'].fillna(0)
        
        rBav = df['rBav'].values if 'rBav' in df.columns else rad_B
        
        sum_sq = (rad_B**2 * (1 - total_conc) +
                  rad_Bp**2 * conc_Bp +
                  rad_D1**2 * conc_D1 +
                  rad_D2**2 * conc_D2)
        df['σ²_rB'] = sum_sq - rBav**2
    
    # Альтернативный фактор толерантности
    if 'rAav' in df.columns and 'rBav' in df.columns:
        df['t_alt'] = (df['rAav'] + r_O) / (np.sqrt(2) * (df['rBav'] + r_O))
    
    return df

def add_thermodynamic_descriptors(df):
    """Энтропия, валентность, концентрация вакансий"""
    if df is None or len(df) == 0:
        return df
    
    R_gas = 8.314
    
    # Конфигурационная энтропия для A-позиции
    if '[A\']' in df.columns:
        x_A = 1 - df['[A\']'].fillna(0)
        x_Ap = df['[A\']'].fillna(0)
        entropy_A = np.zeros(len(df))
        mask = (x_A > 0) & (x_Ap > 0)
        entropy_A[mask] = -R_gas * (x_A[mask] * np.log(x_A[mask]) + x_Ap[mask] * np.log(x_Ap[mask]))
        df['S_config_A'] = entropy_A
    
    # Конфигурационная энтропия для B-позиции
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
    
    # Добавление валентностей
    for pos in ['B', 'B\'', 'D1', 'D2']:
        if pos in df.columns:
            df[f'V{pos}'] = df[pos].apply(get_valence)
    
    # Средняя валентность B-подрешётки
    if all(col in df.columns for col in ['VB', 'VB\'', 'VD1', 'VD2']):
        conc_Bp = df['[B\']'].fillna(0) if '[B\']' in df.columns else 0
        conc_D1 = df['[D1]'].fillna(0) if '[D1]' in df.columns else 0
        conc_D2 = df['[D2]'].fillna(0) if '[D2]' in df.columns else 0
        total_conc = conc_Bp + conc_D1 + conc_D2
        
        df['V_Bav'] = (df['VB'] * (1 - total_conc) +
                       df['VB\''] * conc_Bp +
                       df['VD1'] * conc_D1 +
                       df['VD2'] * conc_D2)
        
        # Концентрация кислородных вакансий (прокси)
        df['Vo_proxy'] = (4 - df['V_Bav']) / 2
    
    return df

def add_physics_inspired_descriptors(df):
    """Физически мотивированные комбинированные дескрипторы"""
    if df is None or len(df) == 0:
        return df
    
    if 'Δχ_AB' in df.columns and 't' in df.columns:
        df['Δχ_div_t'] = df['Δχ_AB'] / df['t']
        df['Δχ_mul_t'] = df['Δχ_AB'] * df['t']
    
    if 'σ²_rB' in df.columns and 'D_t' in df.columns:
        df['disorder_over_distortion'] = df['σ²_rB'] / (df['D_t'] + 1e-6)
    
    if 'ionicity_BO' in df.columns and 'octahedral_factor' in df.columns:
        df['ionic_x_octa'] = df['ionicity_BO'] * df['octahedral_factor']
    
    if 'χ_ratio_AB' in df.columns and 't' in df.columns:
        df['chi_ratio_t'] = df['χ_ratio_AB'] * df['t']
    
    # Суммарная концентрация допантов
    if '[B\']' in df.columns:
        df['total_dopant_B'] = df['[B\']'].fillna(0)
    if '[D1]' in df.columns and '[D2]' in df.columns:
        df['total_dopant_D'] = df['[D1]'].fillna(0) + df['[D2]'].fillna(0)
    
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
# 3. ПОЛУЧЕНИЕ СПИСКОВ ДЕСКРИПТОРОВ ДЛЯ UI
# ============================================================================

def get_target_variables():
    """Список целевых переменных (Y)"""
    return ['αav·106 (K-1)', 'α·106 (K-1)', 'β', 't', 'T(bends), °C', 'T (PT), °C']

def get_descriptor_list():
    """Полный список дескрипторов для выбора в UI"""
    return [
        # Концентрационные
        '[B\']', '[D1]', '[D2]', 'total_dopant_B', 'total_dopant_D',
        # Электроотрицательности
        'χAav', 'χBav', 'Δχ_AB', 'χ_ratio_AB', 'χ_total',
        # Геометрические
        'rAav', 'rBav', 't', 't_alt', 'D_t', 'Δr_AB', 'Δr_AB_norm',
        'octahedral_factor', 'σ²_rB',
        # Термодинамические
        'S_config_A', 'S_config_B', 'V_Bav', 'Vo_proxy',
        # Химические
        'ionicity_AO', 'ionicity_BO', 'pH2O', 'δ',
        # Комбинированные
        'Δχ_div_t', 'Δχ_mul_t', 'disorder_over_distortion', 'ionic_x_octa', 'chi_ratio_t'
    ]

def get_all_plot_parameters():
    """Все параметры для графиков (X, Y, Size, Color)"""
    return get_descriptor_list() + get_target_variables() + ['method', 'Symmetry', 'A', 'B', 'D1', 'D2']

# ============================================================================
# 4. ВИЗУАЛИЗАЦИИ (ГРАФИКИ) - С МНОГОПАРАМЕТРИЧЕСКИМИ ВОЗМОЖНОСТЯМИ
# ============================================================================

def plot_scatter_with_filters(df, x_var, y_var, color_var=None, size_var=None, 
                              title="Scatter Plot", xlabel=None, ylabel=None):
    """Универсальный scatter plot с возможностью выбора color и size"""
    
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        return fig
    
    # Удаление NaN
    plot_df = df[[x_var, y_var]].dropna()
    if color_var and color_var in df.columns:
        plot_df = plot_df.join(df[color_var])
    if size_var and size_var in df.columns:
        plot_df = plot_df.join(df[size_var])
    
    if len(plot_df) < 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Недостаточно данных (n={len(plot_df)})', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if color_var and size_var:
        scat = ax.scatter(plot_df[x_var], plot_df[y_var], 
                         c=plot_df[color_var], s=plot_df[size_var]*100, 
                         alpha=0.7, edgecolors='k', linewidth=0.5, cmap='plasma')
        plt.colorbar(scat, ax=ax, label=color_var)
    elif color_var:
        scat = ax.scatter(plot_df[x_var], plot_df[y_var], 
                         c=plot_df[color_var], s=50, 
                         alpha=0.7, edgecolors='k', linewidth=0.5, cmap='viridis')
        plt.colorbar(scat, ax=ax, label=color_var)
    elif size_var:
        scat = ax.scatter(plot_df[x_var], plot_df[y_var], 
                         s=plot_df[size_var]*100, 
                         alpha=0.7, edgecolors='k', linewidth=0.5)
    else:
        ax.scatter(plot_df[x_var], plot_df[y_var], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel(xlabel if xlabel else x_var, fontweight='bold')
    ax.set_ylabel(ylabel if ylabel else y_var, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    # Добавление линии тренда если есть линейная зависимость
    if len(plot_df) > 5:
        try:
            z = np.polyfit(plot_df[x_var], plot_df[y_var], 1)
            p = np.poly1d(z)
            x_range = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=1.5, label='Trend line')
            ax.legend()
        except:
            pass
    
    plt.tight_layout()
    return fig

def plot_contour_map(df, x_var, y_var, z_var, title="Contour Map"):
    """Контурная карта с многопараметрическим выбором"""
    
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        return fig
    
    plot_df = df[[x_var, y_var, z_var]].dropna()
    if len(plot_df) < 10:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Недостаточно данных для контурной карты (n={len(plot_df)}<10)', 
                ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Создание сетки для контуров
    xi = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 50)
    yi = np.linspace(plot_df[y_var].min(), plot_df[y_var].max(), 50)
    xi, yi = np.meshgrid(xi, yi)
    
    # Интерполяция (простой метод)
    from scipy.interpolate import griddata
    zi = griddata((plot_df[x_var], plot_df[y_var]), plot_df[z_var], (xi, yi), method='cubic')
    
    # Контурный график
    contour = ax.contourf(xi, yi, zi, levels=15, cmap='plasma', alpha=0.8)
    ax.contour(xi, yi, zi, levels=15, colors='black', linewidths=0.5, alpha=0.3)
    
    # Точки данных
    ax.scatter(plot_df[x_var], plot_df[y_var], c=plot_df[z_var], s=30, 
               edgecolors='k', linewidth=0.5, cmap='plasma', zorder=5)
    
    cbar = plt.colorbar(contour, ax=ax, label=z_var)
    ax.set_xlabel(x_var, fontweight='bold')
    ax.set_ylabel(y_var, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_bubble_chart_plotly(df, x_var, y_var, size_var, color_var, title="Bubble Chart"):
    """Интерактивная пузырьковая диаграмма с Plotly"""
    
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df = df[[x_var, y_var]].dropna()
    if size_var and size_var in df.columns:
        plot_df = plot_df.join(df[size_var])
    if color_var and color_var in df.columns:
        plot_df = plot_df.join(df[color_var])
    
    if len(plot_df) < 3:
        fig = go.Figure()
        fig.add_annotation(text=f"Недостаточно данных (n={len(plot_df)})", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Определение размера маркеров
    if size_var and size_var in plot_df.columns:
        size_vals = plot_df[size_var].fillna(plot_df[size_var].median())
        size_norm = 20 + 80 * (size_vals - size_vals.min()) / (size_vals.max() - size_vals.min() + 1e-6)
    else:
        size_norm = 30
    
    # Определение цвета
    if color_var and color_var in plot_df.columns:
        color_vals = plot_df[color_var]
        color_norm = color_vals
    else:
        color_norm = 'blue'
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_df[x_var], y=plot_df[y_var],
        mode='markers',
        marker=dict(
            size=size_norm if size_var else 15,
            color=color_norm if color_var else 'royalblue',
            colorscale='Viridis' if color_var else None,
            showscale=True if color_var else False,
            colorbar=dict(title=color_var) if color_var else None,
            line=dict(width=0.5, color='black')
        ),
        text=[f"{x_var}={row[x_var]:.3f}<br>{y_var}={row[y_var]:.3f}<br>A={row.get('A', '')}<br>B={row.get('B', '')}" 
              for _, row in plot_df.iterrows()],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title=dict(text=x_var, font=dict(size=12, weight='bold')),
        yaxis_title=dict(text=y_var, font=dict(size=12, weight='bold')),
        width=800, height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    )
    
    return fig

def plot_violin_by_category(df, y_var, category_var, title="Distribution by Category"):
    """Violin plot для категориальных переменных"""
    
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        return fig
    
    plot_df = df[[y_var, category_var]].dropna()
    if len(plot_df) < 5:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Недостаточно данных (n={len(plot_df)})', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ограничение количества категорий для читаемости
    unique_cats = plot_df[category_var].nunique()
    if unique_cats > 10:
        top_cats = plot_df[category_var].value_counts().head(10).index
        plot_df = plot_df[plot_df[category_var].isin(top_cats)]
    
    sns.violinplot(data=plot_df, x=category_var, y=y_var, ax=ax, palette='Set2')
    sns.stripplot(data=plot_df, x=category_var, y=y_var, ax=ax, 
                  color='black', alpha=0.3, size=3)
    
    ax.set_xlabel(category_var, fontweight='bold')
    ax.set_ylabel(y_var, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_pca_biplot_with_filters(df, feature_cols, color_var, title="PCA Biplot"):
    """PCA Biplot с выбором цветовой переменной"""
    
    if df is None or len(df) == 0 or len(feature_cols) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно дескрипторов для PCA", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df = df[feature_cols].dropna()
    if len(plot_df) < 5:
        fig = go.Figure()
        fig.add_annotation(text=f"Мало данных после удаления NaN (n={len(plot_df)})", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    X = plot_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Получение цветовой переменной
    if color_var and color_var in df.columns:
        color_vals = df.loc[plot_df.index, color_var].values
    else:
        color_vals = np.ones(len(plot_df))
    
    fig = go.Figure()
    
    # Точки образцов
    fig.add_trace(go.Scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=color_vals,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_var) if color_var else None,
            line=dict(width=0.5, color='black')
        ),
        text=[f"Sample {i}" for i in range(len(plot_df))],
        hoverinfo='text'
    ))
    
    # Векторы признаков
    for i, feature in enumerate(feature_cols[:10]):  # Ограничиваем до 10 для читаемости
        fig.add_annotation(
            x=pca.components_[0, i] * 3, y=pca.components_[1, i] * 3,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text=feature,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='red'
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title=dict(text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', font=dict(size=12)),
        yaxis_title=dict(text=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', font=dict(size=12)),
        width=900, height=700,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_umap_projection_with_filters(df, feature_cols, color_var, title="UMAP Projection"):
    """UMAP проекция с выбором цветовой переменной"""
    
    if df is None or len(df) == 0 or len(feature_cols) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно дескрипторов для UMAP", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df = df[feature_cols].dropna()
    if len(plot_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text=f"Мало данных для UMAP (n={len(plot_df)}<10)", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    X = plot_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(plot_df)-1))
        X_umap = reducer.fit_transform(X_scaled)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"UMAP ошибка: {str(e)[:50]}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    if color_var and color_var in df.columns:
        color_vals = df.loc[plot_df.index, color_var].values
    else:
        color_vals = np.ones(len(plot_df))
    
    fig = px.scatter(
        x=X_umap[:, 0], y=X_umap[:, 1], 
        color=color_vals,
        title=title,
        labels={'x': 'UMAP1', 'y': 'UMAP2', 'color': color_var},
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        width=800, height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_correlation_heatmap(df, corr_method='spearman', title="Correlation Matrix"):
    """Тепловая карта корреляций с кластеризацией"""
    
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        return fig
    
    # Выбор числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Недостаточно числовых колонок', ha='center', va='center')
        return fig
    
    # Ограничиваем количество колонок для читаемости
    if len(numeric_cols) > 30:
        # Выбираем колонки с наименьшим количеством NaN
        na_counts = df[numeric_cols].isna().sum()
        numeric_cols = na_counts.nsmallest(25).index.tolist()
    
    df_corr = df[numeric_cols].dropna()
    if len(df_corr) < 5:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, f'Мало данных после удаления NaN (n={len(df_corr)})', 
                ha='center', va='center')
        return fig
    
    if corr_method == 'spearman':
        corr_matrix = df_corr.corr(method='spearman')
    else:
        corr_matrix = df_corr.corr(method='pearson')
    
    # Кластеризованная тепловая карта
    g = sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       figsize=(14, 12), dendrogram_ratio=0.15, 
                       cbar_pos=(0.02, 0.8, 0.03, 0.18))
    plt.title(title, fontweight='bold')
    return g.figure

# ============================================================================
# 5. ФИЛЬТРАЦИЯ ДАННЫХ
# ============================================================================

def apply_filters(df, filters):
    """Применение фильтров к DataFrame"""
    if df is None or len(df) == 0:
        return df
    
    filtered_df = df.copy()
    
    # Фильтр по типу перовскита (A-позиция)
    if filters.get('a_site_type'):
        a_types = filters['a_site_type']
        if 'Ba,Sr,Ca (A²⁺)' in a_types:
            mask_a = filtered_df['A'].isin(['Ba', 'Sr', 'Ca'])
        else:
            mask_a = pd.Series([False] * len(filtered_df))
        if 'La (A³⁺)' in a_types:
            mask_a = mask_a | (filtered_df['A'] == 'La')
        filtered_df = filtered_df[mask_a]
    
    # Фильтр по B-элементу
    if filters.get('b_site_elements') and len(filters['b_site_elements']) > 0:
        filtered_df = filtered_df[filtered_df['B'].isin(filters['b_site_elements'])]
    
    # Фильтр по допантам
    if filters.get('dopants') and len(filters['dopants']) > 0:
        mask_d = (filtered_df['D1'].isin(filters['dopants'])) | (filtered_df['D2'].isin(filters['dopants']))
        filtered_df = filtered_df[mask_d]
    
    # Фильтр по методу
    if filters.get('methods') and len(filters['methods']) > 0:
        filtered_df = filtered_df[filtered_df['method'].isin(filters['methods'])]
    
    # Фильтр по симметрии
    if filters.get('symmetries') and len(filters['symmetries']) > 0 and 'Symmetry' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Symmetry'].isin(filters['symmetries'])]
    
    # Числовые диапазоны
    if filters.get('t_min') is not None:
        filtered_df = filtered_df[filtered_df['t'] >= filters['t_min']]
    if filters.get('t_max') is not None:
        filtered_df = filtered_df[filtered_df['t'] <= filters['t_max']]
    
    if filters.get('alpha_min') is not None and 'αav·106 (K-1)' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['αav·106 (K-1)'] >= filters['alpha_min']]
    if filters.get('alpha_max') is not None and 'αav·106 (K-1)' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['αav·106 (K-1)'] <= filters['alpha_max']]
    
    if filters.get('beta_min') is not None and 'β' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['β'] >= filters['beta_min']]
    if filters.get('beta_max') is not None and 'β' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['β'] <= filters['beta_max']]
    
    return filtered_df

# ============================================================================
# 6. ML МОДЕЛЬ И SHAP
# ============================================================================

def train_rf_and_shap(df, target, feature_cols):
    """Обучение RandomForest и расчёт SHAP значений"""
    
    if df is None or len(df) < 20:
        return None, None, None, None
    
    df_ml = df[feature_cols + [target]].dropna()
    if len(df_ml) < 20:
        return None, None, None, None
    
    X = df_ml[feature_cols]
    y = df_ml[target]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    
    # SHAP
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        shap_values = None
    
    return rf, importance, shap_values, X

# ============================================================================
# 7. СТРАНИЦА ФИЛЬТРОВ
# ============================================================================

def render_filters_sidebar(df):
    """Рендеринг панели фильтров в боковой панели"""
    
    st.sidebar.markdown("## 🔍 Фильтры данных")
    
    filters = {}
    
    # Фильтр по типу перовскита
    st.sidebar.markdown("### Перовскит тип")
    a_site_type = st.sidebar.multiselect(
        "A-позиция", 
        ['Ba,Sr,Ca (A²⁺)', 'La (A³⁺)'],
        default=['Ba,Sr,Ca (A²⁺)', 'La (A³⁺)']
    )
    filters['a_site_type'] = a_site_type
    
    # Фильтр по B-элементу
    if 'B' in df.columns:
        b_options = sorted(df['B'].dropna().unique().tolist())
        b_site_elements = st.sidebar.multiselect("B-элемент", b_options, default=[])
        filters['b_site_elements'] = b_site_elements
    
    # Фильтр по допантам
    dopant_options = []
    if 'D1' in df.columns:
        dopant_options.extend(df['D1'].dropna().unique().tolist())
    if 'D2' in df.columns:
        dopant_options.extend(df['D2'].dropna().unique().tolist())
    dopant_options = sorted(set([d for d in dopant_options if d and d != '']))
    
    if dopant_options:
        dopants = st.sidebar.multiselect("Допанты (D1/D2)", dopant_options, default=[])
        filters['dopants'] = dopants
    
    # Фильтр по методу
    if 'method' in df.columns:
        method_options = sorted(df['method'].dropna().unique().tolist())
        methods = st.sidebar.multiselect("Метод измерения", method_options, default=[])
        filters['methods'] = methods
    
    # Фильтр по симметрии
    if 'Symmetry' in df.columns:
        sym_options = sorted(df['Symmetry'].dropna().unique().tolist())
        symmetries = st.sidebar.multiselect("Симметрия", sym_options, default=[])
        filters['symmetries'] = symmetries
    
    st.sidebar.markdown("### Числовые диапазоны")
    
    # Диапазон t
    if 't' in df.columns:
        t_min = st.sidebar.number_input("t min", value=float(df['t'].min()), format="%.3f")
        t_max = st.sidebar.number_input("t max", value=float(df['t'].max()), format="%.3f")
        filters['t_min'] = t_min
        filters['t_max'] = t_max
    
    # Диапазон α
    if 'αav·106 (K-1)' in df.columns:
        alpha_min = st.sidebar.number_input("α min (×10⁶)", value=float(df['αav·106 (K-1)'].min()), format="%.1f")
        alpha_max = st.sidebar.number_input("α max (×10⁶)", value=float(df['αav·106 (K-1)'].max()), format="%.1f")
        filters['alpha_min'] = alpha_min
        filters['alpha_max'] = alpha_max
    
    # Диапазон β
    if 'β' in df.columns:
        beta_min = st.sidebar.number_input("β min", value=float(df['β'].min()), format="%.4f")
        beta_max = st.sidebar.number_input("β max", value=float(df['β'].max()), format="%.4f")
        filters['beta_min'] = beta_min
        filters['beta_max'] = beta_max
    
    return filters

# ============================================================================
# 8. ГЛАВНОЕ ПРИЛОЖЕНИЕ STREAMLIT
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Perovskite Expansion Analyzer", page_icon="🧪")
    st.title("📊 Perovskite Thermal & Chemical Expansion Analyzer")
    st.markdown("### Advanced Materials Informatics for SOFC Materials")
    
    # Загрузка данных
    with st.spinner("Загрузка данных и расчёт дескрипторов..."):
        df_raw = load_and_merge_data()
        df = add_all_descriptors(df_raw)
    
    if df is None or len(df) == 0:
        st.error("Нет данных. Убедитесь, что alpha_beta.py и phase_transitions.py находятся в той же директории.")
        return
    
    st.success(f"✅ Загружено {len(df)} записей, {len(df.columns)} колонок (включая новые дескрипторы)")
    
    # Панель фильтров
    filters = render_filters_sidebar(df)
    df_filtered = apply_filters(df, filters)
    
    st.info(f"📊 После фильтрации: {len(df_filtered)} записей")
    
    # Основная навигация
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Scatter Plot", "🗺️ Contour Map", "🫧 Bubble Chart", 
        "📊 Violin/Box", "🔬 PCA/UMAP", "📉 Correlations", 
        "🤖 ML & SHAP", "📋 Data"
    ])
    
    # Получение списков для выбора
    all_params = get_all_plot_parameters()
    descriptor_list = get_descriptor_list()
    target_list = get_target_variables()
    
    # ==================== TAB 1: SCATTER PLOT ====================
    with tab1:
        st.subheader("Scatter Plot (с линией тренда)")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-axis", all_params, index=0, key='scatter_x')
            y_var = st.selectbox("Y-axis", target_list + descriptor_list, index=0, key='scatter_y')
        with col2:
            color_var = st.selectbox("Color (optional)", ['None'] + all_params, index=0, key='scatter_color')
            size_var = st.selectbox("Size (optional)", ['None'] + all_params, index=0, key='scatter_size')
        
        color_var = None if color_var == 'None' else color_var
        size_var = None if size_var == 'None' else size_var
        
        if st.button("Generate Scatter Plot", key='scatter_btn'):
            fig = plot_scatter_with_filters(
                df_filtered, x_var, y_var, color_var, size_var,
                title=f"{y_var} vs {x_var}"
            )
            st.pyplot(fig)
            
            # Кнопка экспорта
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.download_button("Download as SVG", data=b64, file_name="scatter_plot.svg", mime="image/svg+xml")
    
    # ==================== TAB 2: CONTOUR MAP ====================
    with tab2:
        st.subheader("Contour Map")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_contour = st.selectbox("X-axis", descriptor_list, index=0, key='contour_x')
        with col2:
            y_contour = st.selectbox("Y-axis", descriptor_list, index=1 if len(descriptor_list)>1 else 0, key='contour_y')
        with col3:
            z_contour = st.selectbox("Z-axis (color)", target_list + descriptor_list, index=0, key='contour_z')
        
        if st.button("Generate Contour Map", key='contour_btn'):
            fig = plot_contour_map(df_filtered, x_contour, y_contour, z_contour, 
                                   title=f"Contour: {z_contour} vs {x_contour} vs {y_contour}")
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.download_button("Download as SVG", data=b64, file_name="contour_map.svg", mime="image/svg+xml")
    
    # ==================== TAB 3: BUBBLE CHART ====================
    with tab3:
        st.subheader("Interactive Bubble Chart (Plotly)")
        
        col1, col2 = st.columns(2)
        with col1:
            x_bubble = st.selectbox("X-axis", all_params, index=0, key='bubble_x')
            y_bubble = st.selectbox("Y-axis", target_list, index=0, key='bubble_y')
        with col2:
            size_bubble = st.selectbox("Bubble size", ['None'] + all_params, index=0, key='bubble_size')
            color_bubble = st.selectbox("Bubble color", ['None'] + all_params, index=0, key='bubble_color')
        
        size_bubble = None if size_bubble == 'None' else size_bubble
        color_bubble = None if color_bubble == 'None' else color_bubble
        
        if st.button("Generate Bubble Chart", key='bubble_btn'):
            fig = plot_bubble_chart_plotly(
                df_filtered, x_bubble, y_bubble, size_bubble, color_bubble,
                title=f"{y_bubble} vs {x_bubble}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 4: VIOLIN/BOX PLOT ====================
    with tab4:
        st.subheader("Distribution Analysis by Category")
        
        col1, col2 = st.columns(2)
        with col1:
            y_violin = st.selectbox("Y-axis (property)", target_list, index=0, key='violin_y')
        with col2:
            cat_violin = st.selectbox("Category", ['Symmetry', 'method', 'A', 'B', 'D1'], key='violin_cat')
        
        if st.button("Generate Violin Plot", key='violin_btn'):
            fig = plot_violin_by_category(df_filtered, y_violin, cat_violin,
                                          title=f"Distribution of {y_violin} by {cat_violin}")
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.download_button("Download as SVG", data=b64, file_name="violin_plot.svg", mime="image/svg+xml")
    
    # ==================== TAB 5: PCA/UMAP ====================
    with tab5:
        st.subheader("Dimensionality Reduction")
        
        # Выбор дескрипторов для PCA
        default_pca_features = [d for d in descriptor_list if d in df_filtered.columns][:6]
        pca_features = st.multiselect("Select descriptors for PCA", descriptor_list, default=default_pca_features)
        
        if len(pca_features) >= 3:
            color_pca = st.selectbox("Color by", ['αav·106 (K-1)', 'β', 't', 'Symmetry', 'method'], index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run PCA", key='pca_btn'):
                    fig = plot_pca_biplot_with_filters(df_filtered, pca_features, color_pca,
                                                       title=f"PCA Biplot (color={color_pca})")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if st.button("Run UMAP", key='umap_btn'):
                    fig = plot_umap_projection_with_filters(df_filtered, pca_features, color_pca,
                                                            title=f"UMAP Projection (color={color_pca})")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least 3 descriptors for PCA/UMAP")
    
    # ==================== TAB 6: CORRELATIONS ====================
    with tab6:
        st.subheader("Correlation Analysis")
        
        corr_method = st.radio("Correlation method", ['spearman', 'pearson'], horizontal=True)
        
        if st.button("Generate Correlation Heatmap", key='corr_btn'):
            fig = plot_correlation_heatmap(df_filtered, corr_method=corr_method,
                                          title=f"{corr_method.capitalize()} Correlation Matrix with Clustering")
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.download_button("Download as SVG", data=b64, file_name="correlation_heatmap.svg", mime="image/svg+xml")
        
        # Mutual Information
        st.subheader("Mutual Information (non-linear dependencies)")
        if st.button("Compute Mutual Information", key='mi_btn'):
            mi_features = [d for d in descriptor_list if d in df_filtered.columns][:10]
            mi_targets = [t for t in target_list if t in df_filtered.columns]
            
            if len(mi_features) > 0 and len(mi_targets) > 0:
                mi_results = {}
                for target in mi_targets:
                    df_mi = df_filtered[mi_features + [target]].dropna()
                    if len(df_mi) > 10:
                        X_mi = df_mi[mi_features]
                        y_mi = df_mi[target]
                        mi = mutual_info_regression(X_mi, y_mi, random_state=42)
                        mi_results[target] = dict(zip(mi_features, mi))
                
                if mi_results:
                    mi_df = pd.DataFrame(mi_results).T
                    fig, ax = plt.subplots(figsize=(12, max(4, len(mi_results)*1.5)))
                    sns.heatmap(mi_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
                    ax.set_title('Mutual Information between descriptors and targets')
                    st.pyplot(fig)
                else:
                    st.warning("Not enough data for MI computation")
            else:
                st.warning("Select at least 3 descriptors and 1 target")
    
    # ==================== TAB 7: ML & SHAP ====================
    with tab7:
        st.subheader("Random Forest + SHAP Interpretability")
        
        target_ml = st.selectbox("Target property", target_list, index=0, key='ml_target')
        ml_features = st.multiselect("Select features", descriptor_list, 
                                     default=[d for d in descriptor_list if d in df_filtered.columns][:8])
        
        if len(ml_features) >= 3 and st.button("Train Model", key='ml_btn'):
            rf, importance, shap_values, X = train_rf_and_shap(df_filtered, target_ml, ml_features)
            
            if rf is not None:
                st.metric("Model R² (train)", f"{rf.score(X, df_filtered[target_ml].dropna().loc[X.index]):.3f}")
                
                # Feature importance
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance.head(10), x='importance', y='feature', ax=ax1)
                ax1.set_title('Random Forest Feature Importance (top-10)')
                st.pyplot(fig1)
                
                # SHAP summary
                if shap_values is not None:
                    st.subheader("SHAP Summary Plot")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X, show=False)
                    st.pyplot(fig2)
                    
                    buf = io.BytesIO()
                    fig2.savefig(buf, format='svg', dpi=300, bbox_inches='tight')
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    st.download_button("Download SHAP as SVG", data=b64, file_name="shap_summary.svg", mime="image/svg+xml")
            else:
                st.warning(f"Not enough data to train model for {target_ml}. Need at least 20 complete samples.")
    
    # ==================== TAB 8: DATA ====================
    with tab8:
        st.subheader("Filtered Dataset")
        st.dataframe(df_filtered)
        
        st.subheader("Column Description")
        col_info = pd.DataFrame({
            'Column': df_filtered.columns,
            'Type': df_filtered.dtypes.values,
            'Non-null count': df_filtered.count().values,
            'Null count': df_filtered.isna().sum().values
        })
        st.dataframe(col_info)
        
        # Export
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        st.download_button("Download filtered data as CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")

if __name__ == "__main__":
    main()