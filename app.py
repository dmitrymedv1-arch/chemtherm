# ============================================================================
# 1. ИМПОРТЫ И НАСТРОЙКИ
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Статистика и ML
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

# Корреляции
import pingouin as pg
from scipy.stats import spearmanr, pearsonr

# Специализированные
import networkx as nx
from adjustText import adjust_text
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Proton-Conducting Perovskites Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 2. КОНСТАНТЫ И БАЗЫ ДАННЫХ
# ============================================================================

# 2.1. Ионные радиусы Шеннона (12-коорд для A, 6-коорд для B)
IONIC_RADII = {
    # A-позиция (12-координация)
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
    'O': 1.40,  # O2- для 6-координации
}

# 2.2. Электроотрицательности по Поллингу
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

# 2.3. Валентности
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

# 2.4. Молярные массы
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

# 2.5. Цветовые палитры для научного стиля
COLOR_PALETTES = {
    'method': {'dilatometry': '#2C3E50', 'HT XRD': '#E67E22', 'HT ND': '#8E44AD'},
    'B_cation': {'Ce': '#E74C3C', 'Zr': '#3498DB', 'Sn': '#2ECC71', 'Ti': '#F39C12'},
    'A_cation': {'Ba': '#1A5276', 'Sr': '#2471A3', 'Ca': '#5DADE2', 'La': '#F39C12'},
    'continuous': 'viridis',
    'diverging': 'coolwarm',
    'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}

# 2.6. Список целевых переменных
TARGET_VARIABLES = ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'T(bends), °C']

# 2.7. Список всех возможных катионов
ALL_CATIONS = list(IONIC_RADII.keys())
ALL_CATIONS.remove('O')

# 2.8. Обязательные колонки для данных
REQUIRED_COLUMNS = ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', '[A\']', '[B\']', '[D1]', '[D2]', 
                   'δ', 'method', 'β', '∆T, °C', 'α·106 (K-1)', 'T(bends), °C', 
                   'αav·106 (K-1)', 'pH2O', 'Ref']

# ============================================================================
# 3. НАСТРОЙКИ НАУЧНОГО СТИЛЯ
# ============================================================================

def apply_scientific_style():
    """Улучшенный научный стиль для matplotlib для материаловедческих публикаций"""
    plt.style.use('seaborn-v0_8-whitegrid')
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
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        
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
        'xtick.minor.size': 3,
        'xtick.minor.width': 1.0,
        'ytick.minor.size': 3,
        'ytick.minor.width': 1.0,
        
        # Легенда
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#000000',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.8,
        
        # Фигура
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'figure.figsize': (8, 6),
        
        # Линии
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'lines.markeredgewidth': 1.0,
        'errorbar.capsize': 3,
        
        # PDF для публикаций
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

def get_plotly_layout(title, xlabel, ylabel, width=None, height=None, showlegend=True):
    """Стандартный layout для Plotly графиков в научном стиле"""
    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(size=14, family='Times New Roman', color='#000000'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(size=12, family='Times New Roman', color='#000000')),
            tickfont=dict(size=11, family='Times New Roman', color='#000000'),
            showline=True,
            linewidth=1.5,
            linecolor='#000000',
            mirror=True,
            ticks='inside',
            tickwidth=1.5,
            ticklen=6,
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(size=12, family='Times New Roman', color='#000000')),
            tickfont=dict(size=11, family='Times New Roman', color='#000000'),
            showline=True,
            linewidth=1.5,
            linecolor='#000000',
            mirror=True,
            ticks='inside',
            tickwidth=1.5,
            ticklen=6,
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Times New Roman', color='#000000'),
        legend=dict(
            font=dict(size=10, family='Times New Roman'),
            borderwidth=1.5,
            bordercolor='#000000',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ) if showlegend else dict(showlegend=False),
        width=width,
        height=height,
        margin=dict(l=80, r=40, t=60, b=60),
        hovermode='closest'
    )
    return layout

# ============================================================================
# 4. КЛАССЫ ДЛЯ ОБРАБОТКИ ДАННЫХ
# ============================================================================

class DataProcessor:
    """Обработка загруженных данных"""
    
    def __init__(self):
        self.df = None
        self.raw_data = None
    
    @staticmethod
    def parse_text_data(text):
        """Парсинг текста из textarea в DataFrame"""
        if not text or text.strip() == '':
            return None
        
        lines = text.strip().split('\n')
        
        # Определение разделителя
        header = lines[0]
        if '\t' in header:
            delimiter = '\t'
        elif ';' in header:
            delimiter = ';'
        else:
            delimiter = r'\s+'
        
        # Парсинг данных
        data = []
        for line in lines:
            if delimiter == r'\s+':
                row = re.split(r'\s+', line.strip())
            else:
                row = line.split(delimiter)
            data.append(row)
        
        # Создание DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Замена '-' на NaN
        df = df.replace('-', np.nan)
        
        # Преобразование типов для числовых колонок
        numeric_cols = ['№', '[A\']', '[B\']', '[D1]', '[D2]', 'δ', 'β', 'pH2O']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Преобразование температурных диапазонов
        if '∆T, °C' in df.columns:
            df['T_min'] = df['∆T, °C'].apply(
                lambda x: float(str(x).split('-')[0]) if pd.notna(x) and '-' in str(x) else np.nan
            )
            df['T_max'] = df['∆T, °C'].apply(
                lambda x: float(str(x).split('-')[1]) if pd.notna(x) and '-' in str(x) else np.nan
            )
        
        # Преобразование α
        if 'α·106 (K-1)' in df.columns:
            df['α·106 (K-1)'] = pd.to_numeric(df['α·106 (K-1)'], errors='coerce')
        
        # Обработка T(bends) - извлечение первого значения
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
        
        # Обработка αav - извлечение значений
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
        
        return df
    
    def clean_data(self, df):
        """Очистка данных от выбросов и аномалий"""
        if df is None:
            return None
        
        df_clean = df.copy()
        
        # Удаление строк с полностью отсутствующими ключевыми данными
        required_for_analysis = ['A', 'B', 'method']
        df_clean = df_clean.dropna(subset=required_for_analysis)
        
        # Фильтрация по разумным значениям α (обычно 5-20 × 10⁻⁶ K⁻¹)
        if 'α·106 (K-1)' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['α·106 (K-1)'].isna()) | 
                ((df_clean['α·106 (K-1)'] >= 5) & (df_clean['α·106 (K-1)'] <= 25))
            ]
        
        # Фильтрация по разумным значениям β (обычно 0-1)
        if 'β' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['β'].isna()) | 
                ((df_clean['β'] >= 0) & (df_clean['β'] <= 1))
            ]
        
        return df_clean


class DescriptorEngine:
    """Расчёт всех дескрипторов на основе химического состава"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.ionic_radii = IONIC_RADII
        self.electronegativity = ELECTRONEGATIVITY
        self.valences = VALENCES
        self.molar_mass = MOLAR_MASS
        self.descriptor_names = []
        
    def get_radius(self, element, site='B'):
        """Получение ионного радиуса элемента для заданной позиции"""
        if element == '-' or pd.isna(element):
            return np.nan
        
        # Для A-позиции используем 12-координацию
        if site == 'A':
            key = element
        else:
            key = element
        
        if key in self.ionic_radii:
            return self.ionic_radii[key]
        else:
            return np.nan
    
    def get_electronegativity(self, element):
        """Получение электроотрицательности элемента"""
        if element == '-' or pd.isna(element):
            return np.nan
        if element in self.electronegativity:
            return self.electronegativity[element]
        else:
            return np.nan
    
    def get_valence(self, element):
        """Получение валентности элемента"""
        if element == '-' or pd.isna(element):
            return np.nan
        if element in self.valences:
            return self.valences[element]
        else:
            return np.nan
    
    def get_molar_mass(self, element):
        """Получение молярной массы элемента"""
        if element == '-' or pd.isna(element):
            return np.nan
        if element in self.molar_mass:
            return self.molar_mass[element]
        else:
            return np.nan
    
    def calculate_all(self):
        """Расчёт всех 63+ дескрипторов"""
        
        # --- Группа 1: Геометрические дескрипторы ---
        self.calculate_geometric()
        
        # --- Группа 2: Электроотрицательные дескрипторы ---
        self.calculate_electronegativity()
        
        # --- Группа 3: Термодинамические дескрипторы ---
        self.calculate_thermodynamic()
        
        # --- Группа 4: Массовые дескрипторы ---
        self.calculate_mass()
        
        # --- Группа 5: Дефектные дескрипторы ---
        self.calculate_defect()
        
        # --- Группа 6: Специфические дескрипторы ---
        self.calculate_specific()
        
        # --- Группа 7: Составные дескрипторы ---
        self.calculate_compositional()
        
        return self.df
    
    def calculate_geometric(self):
        """Расчёт геометрических дескрипторов"""
        
        # Добавляем оценку rO для расчётов (должно быть до использования)
        self.df['rO_est'] = self.ionic_radii['O']
        
        # 1. Средний радиус A-site
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
            else:
                return rA * (1 - A_prime_conc) + rA_prime * A_prime_conc
        
        self.df['rAav'] = self.df.apply(calc_rAav, axis=1)
        
        # 2. Средний радиус B-site
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
        
        # 3. Толерант-фактор Гольдшмидта
        rO = self.ionic_radii['O']
        self.df['t'] = (self.df['rAav'] + rO) / (np.sqrt(2) * (self.df['rBav'] + rO))
        
        # 4. Отклонение толерант-фактора
        self.df['D_t'] = np.abs(1 - self.df['t'])
        
        # 5. Октаэдрический фактор
        self.df['octahedral_factor'] = self.df['rBav'] / rO
        
        # 6. Разница радиусов A и B
        self.df['Δr_AB'] = np.abs(self.df['rAav'] - self.df['rBav'])
        
        # 7. Нормированная разница радиусов
        self.df['Δr_AB_norm'] = self.df['Δr_AB'] / rO
        
        # 8. Дисперсия радиусов A-site
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
            
            result = (1 - A_prime_conc) * (rA - rAav)**2 + A_prime_conc * (rA_prime - rAav)**2
            return result
        
        self.df['σ²_rA'] = self.df.apply(calc_sigma2_rA, axis=1)
        
        # 9. Дисперсия радиусов B-site
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
        
        # 10. Расчётный объём элементарной ячейки (псевдокубический)
        # Используем rO_est, который создан в начале метода
        self.df['V_cell'] = (2 * (self.df['rBav'] + self.df['rO_est']))**3
        
        # 11. Отношение радиусов rA/rB
        self.df['r_ratio_AB'] = self.df['rAav'] / self.df['rBav']
    
    def calculate_electronegativity(self):
        """Расчёт электроотрицательных дескрипторов"""
        
        # Используем константу напрямую, без создания колонки rO_est
        rO = self.ionic_radii['O']
        
        # 1. Средняя электроотрицательность A-site
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
            else:
                return chiA * (1 - A_prime_conc) + chiA_prime * A_prime_conc
        
        self.df['χAav'] = self.df.apply(calc_chiAav, axis=1)
        
        # 2. Средняя электроотрицательность B-site
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
        
        # 3. Разница электроотрицательностей
        self.df['Δχ_AB'] = np.abs(self.df['χAav'] - self.df['χBav'])
        
        # 4. Отношение электроотрицательностей
        self.df['χ_ratio_AB'] = self.df['χAav'] / self.df['χBav']
        
        # 5. Ионность связи A-O
        chiO = self.electronegativity['O']
        self.df['ionicity_AO'] = 1 - np.exp(-0.25 * (self.df['χAav'] - chiO)**2)
        
        # 6. Ионность связи B-O
        self.df['ionicity_BO'] = 1 - np.exp(-0.25 * (self.df['χBav'] - chiO)**2)
        
        # 7. Кислотность A-O (обратная электроотрицательность)
        self.df['acidity_AO'] = 1 / self.df['χAav']
        
        # 8. Кислотность B-O
        self.df['acidity_BO'] = 1 / self.df['χBav']
        
        # 9. Разница кислотностей
        self.df['Δacidity'] = self.df['acidity_BO'] - self.df['acidity_AO']
        
        # 10. Средняя электроотрицательность
        self.df['χ_total'] = (self.df['χAav'] + self.df['χBav']) / 2
        
        # 11. Произведение χ_ratio на толерант-фактор
        self.df['χ_ratio_t'] = self.df['χ_ratio_AB'] * self.df['t']
    
    def calculate_thermodynamic(self):
        """Расчёт термодинамических дескрипторов"""
        
        R = 8.314  # Универсальная газовая постоянная
        
        # 1. Энтропия конфигурации A-site
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
        
        # 2. Энтропия конфигурации B-site
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
        
        # 3. Средняя валентность B-site
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
        
        # 4. Прокси кислородных вакансий (для Ce⁴⁺/Zr⁴⁺)
        self.df['Vo_proxy'] = (4 - self.df['V_Bav']) / 2
        
        # 5. Расчётная энтальпия гидратации
        chiO = self.electronegativity['O']
        self.df['ΔH_hydr'] = 1 / (self.df['rBav'] + 0.1) * (self.df['χBav'] - chiO)**2
        
        # 6. Энергия связи B-O (кулоновская)
        self.df['E_BO'] = (self.df['V_Bav'] * 2) / (self.df['rBav'] + self.df['rO_est'])
        
        # 7. Массовая плотность (относительная)
        self.df['ρ'] = self.df['M_Bav'] / self.df['V_cell']
    
    def calculate_mass(self):
        """Расчёт массовых дескрипторов"""
        
        # 1. Средняя молярная масса A-site
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
            else:
                return mA * (1 - A_prime_conc) + mA_prime * A_prime_conc
        
        self.df['M_Aav'] = self.df.apply(calc_MAav, axis=1)
        
        # 2. Средняя молярная масса B-site
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
        
        # 3. Общая молярная масса (ABO3)
        self.df['M_total'] = self.df['M_Aav'] + self.df['M_Bav'] + 3 * self.molar_mass['O']
        
        # 4. Отношение масс A/B
        self.df['M_ratio_AB'] = self.df['M_Aav'] / self.df['M_Bav']
        
        # 5. Произведение массы и радиуса A
        self.df['M_rA'] = self.df['M_Aav'] * self.df['rAav']
        
        # 6. Произведение массы и электроотрицательности A
        self.df['M_χA'] = self.df['M_Aav'] * self.df['χAav']
    
    def calculate_defect(self):
        """Расчёт дефектных дескрипторов"""
        
        # 1. Концентрация вакансий (пересчёт для проверки)
        def calc_delta_actual(row):
            D1_conc = row['[D1]'] if pd.notna(row['[D1]']) else 0
            D2_conc = row['[D2]'] if pd.notna(row['[D2]']) else 0
            return D1_conc / 2 + D2_conc / 2
        
        self.df['δ_actual'] = self.df.apply(calc_delta_actual, axis=1)
        
        # 2. Эффективный заряд B-site
        self.df['Z_eff_B'] = 4 - 2 * self.df['δ_actual']
        
        # 3. Сродство к протону
        self.df['proton_affinity'] = 1 / ((self.df['rBav'] + 0.1) * self.df['χBav'])
        
        # 4. Энергия образования вакансии
        chiO = self.electronegativity['O']
        self.df['E_vac'] = 1 / ((self.df['rBav'] + 0.1)**2) * (self.df['χBav'] - chiO)
    
    def calculate_specific(self):
        """Расчёт специфических дескрипторов для T(bends)"""
        
        # 1. Отношение α/β (если есть оба)
        self.df['alpha_beta_ratio'] = self.df['α·106 (K-1)'] / self.df['β']
        
        # 2. Температурная стабильность протона
        R = 8.314
        self.df['T_stab'] = -self.df['ΔH_hydr'] / R
        
        # 3. Произведение δ и χBav
        self.df['δ_χB'] = self.df['δ_actual'] * self.df['χBav']
        
        # 4. Произведение δ и rBav
        self.df['δ_rB'] = self.df['δ_actual'] * self.df['rBav']
    
    def calculate_compositional(self):
        """Расчёт составных дескрипторов"""
        
        # 1. Концентрация изовалентного допанта B'
        self.df['B\'_conc'] = self.df['[B\']']
        
        # 2. Суммарная концентрация акцепторов D1 + D2
        self.df['D_total'] = self.df['[D1]'] + self.df['[D2]']
        
        # 3. Отношение D_total к B'
        self.df['D_B_ratio'] = self.df['D_total'] / (self.df['B\'_conc'] + 0.001)
        
        # 4. Суммарная концентрация всех допантов на B-site
        self.df['total_dopant'] = self.df['[B\']'] + self.df['[D1]'] + self.df['[D2]']


class CorrelationAnalyzer:
    """Расширенный корреляционный анализ"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.targets = ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'T(bends), °C']
        self.all_descriptors = []
        self.top_features = {}
    
    def get_numeric_features(self):
        """Получение всех числовых признаков (кроме целевых и идентификаторов)"""
        exclude_cols = ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Ref', 'method',
                       '∆T, °C', 'T(bends), °C', 'αav·106 (K-1)']
        exclude_cols.extend(self.targets)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Убираем колонки с более чем 50% пропусков
        for col in features[:]:
            if self.df[col].isna().sum() / len(self.df) > 0.5:
                features.remove(col)
        
        self.all_descriptors = features
        return features
    
    def pearson_correlation(self, features=None, target=None):
        """Корреляция Пирсона с p-value"""
        if features is None:
            features = self.get_numeric_features()
        
        if target is None:
            target = self.targets[0]
        
        if target not in self.df.columns:
            return None
        
        results = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                data = self.df[[feature, target]].dropna()
                if len(data) > 2:
                    corr, p_value = pearsonr(data[feature], data[target])
                    if not np.isnan(corr):
                        results.append({
                            'feature': feature,
                            'correlation': corr,
                            'p_value': p_value,
                            'n': len(data)
                        })
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['significant'] = results_df['p_value'] < 0.05
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)
        
        return results_df
    
    def spearman_correlation(self, features=None, target=None):
        """Корреляция Спирмена (для нелинейных зависимостей)"""
        if features is None:
            features = self.get_numeric_features()
        
        if target is None:
            target = self.targets[0]
        
        if target not in self.df.columns:
            return None
        
        results = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                data = self.df[[feature, target]].dropna()
                if len(data) > 2:
                    corr, p_value = spearmanr(data[feature], data[target])
                    if not np.isnan(corr):
                        results.append({
                            'feature': feature,
                            'correlation': corr,
                            'p_value': p_value,
                            'n': len(data)
                        })
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['significant'] = results_df['p_value'] < 0.05
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)
        
        return results_df
    
    def partial_correlation(self, target, control_vars=['pH2O'], features=None):
        """Частичная корреляция с контролем третьих переменных"""
        if features is None:
            features = self.get_numeric_features()
        
        if target not in self.df.columns:
            return None
        
        # Подготовка данных
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
                except:
                    continue
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['significant'] = results_df['p_value'] < 0.05
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)
        
        return results_df
    
    def get_top_features(self, target=None, n=20):
        """Топ-N дескрипторов для целевой переменной"""
        if target is None:
            target = self.targets[0]
        
        if target not in self.df.columns:
            return []
        
        # Получаем корреляции Пирсона и Спирмена
        pearson_results = self.pearson_correlation(target=target)
        spearman_results = self.spearman_correlation(target=target)
        
        if pearson_results is None or len(pearson_results) == 0:
            return []
        
        # Объединяем результаты
        merged = pearson_results.merge(
            spearman_results, 
            on='feature', 
            suffixes=('_pearson', '_spearman')
        )
        
        # Вычисляем комбинированный рейтинг
        merged['abs_pearson'] = np.abs(merged['correlation_pearson'])
        merged['abs_spearman'] = np.abs(merged['correlation_spearman'])
        merged['combined_score'] = (merged['abs_pearson'] + merged['abs_spearman']) / 2
        
        # Сортируем по комбинированному рейтингу
        merged = merged.sort_values('combined_score', ascending=False)
        
        # Выбираем топ-N
        top_features = merged['feature'].head(n).tolist()
        
        # Проверяем на мультиколлинеарность
        if len(top_features) > 1:
            try:
                data = self.df[top_features].dropna()
                if len(data) > len(top_features):
                    X = add_constant(data)
                    vif = pd.DataFrame()
                    vif['Variable'] = X.columns
                    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    vif = vif[vif['Variable'] != 'const']
                    vif = vif[vif['VIF'] < 10]  # Убираем сильно коллинеарные
                    top_features = vif['Variable'].tolist()
            except:
                pass
        
        self.top_features[target] = top_features
        return top_features[:n]
    
    def correlation_matrix_with_filter(self, features=None, threshold=0.3):
        """Матрица корреляций с фильтром значимых связей"""
        if features is None:
            features = self.get_numeric_features()
        
        # Оставляем только признаки с достаточным количеством данных
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
        
        # Вычисляем корреляционную матрицу
        corr_matrix = data.corr()
        
        # Фильтруем по порогу
        mask = np.abs(corr_matrix) < threshold
        corr_filtered = corr_matrix.copy()
        corr_filtered[mask] = 0
        
        # Заменяем диагональ на 1
        np.fill_diagonal(corr_filtered.values, 1)
        
        return corr_filtered
    
    def correlation_network(self, features=None, threshold=0.5):
        """Сетевой граф корреляций"""
        if features is None:
            features = self.get_numeric_features()
        
        # Оставляем только признаки с достаточным количеством данных
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
        
        # Вычисляем корреляционную матрицу
        corr_matrix = data.corr()
        
        # Создаем граф
        G = nx.Graph()
        
        # Добавляем узлы
        for feature in valid_features:
            G.add_node(feature)
        
        # Добавляем рёбра для значимых корреляций
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                if i < j:
                    corr = corr_matrix.loc[feat1, feat2]
                    if abs(corr) >= threshold and not np.isnan(corr):
                        G.add_edge(feat1, feat2, weight=abs(corr), sign=np.sign(corr))
        
        return G


class VisualizationEngine:
    """Генерация всех графиков"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.targets = ['α·106 (K-1)', 'β', 'αav·106 (K-1)', 'T(bends), °C']
        self.available_targets = [t for t in self.targets if t in self.df.columns]
        self.compositional = ['[B\']', 'D_total', 'δ']
        self.all_features = []
    
    def set_features(self, features):
        """Установка списка дескрипторов"""
        self.all_features = features
    
    # --- КАТЕГОРИЯ 1: ОБЗОРНЫЕ ГРАФИКИ ---
    
    def plot_distributions(self, features=None):
        """1. Гистограммы всех числовых колонок"""
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
                    axes[i].set_xlabel(feature, fontsize=10)
                    axes[i].set_ylabel('Frequency', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
        
        # Скрываем пустые подграфики
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_missing_matrix(self):
        """2. Матрица пропусков"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Сортируем колонки по количеству пропусков
        missing = self.df.isna().sum() / len(self.df) * 100
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            ax.text(0.5, 0.5, 'No missing data', transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Создаём матрицу пропусков
        cols = missing.index.tolist()
        data = self.df[cols]
        
        # Визуализируем
        im = ax.imshow(data.isna().T, aspect='auto', cmap='Reds', interpolation='none')
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Feature', fontsize=10)
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Missing', fontsize=10)
        
        # Добавляем процент пропусков
        for i, col in enumerate(cols):
            pct = missing[col]
            ax.text(len(data) + 1, i, f'{pct:.1f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_box_by_method(self, target=None):
        """3. Box plot по методам измерения"""
        if target is None:
            target = 'α·106 (K-1)'
        
        if target not in self.df.columns or 'method' not in self.df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Группируем данные
        data = self.df[['method', target]].dropna()
        
        if len(data) == 0:
            return None
        
        # Создаём box plot
        methods = data['method'].unique()
        box_data = [data[data['method'] == m][target].values for m in methods]
        
        bp = ax.boxplot(box_data, labels=methods, patch_artist=True)
        
        # Оформление
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(target, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_violin_by_B(self, target=None):
        """4. Violin plot по B-катиону"""
        if target is None:
            target = 'α·106 (K-1)'
        
        if target not in self.df.columns or 'B' not in self.df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Группируем данные
        data = self.df[['B', target]].dropna()
        
        if len(data) == 0:
            return None
        
        # Создаём violin plot
        b_cations = data['B'].unique()
        parts = ax.violinplot(
            [data[data['B'] == b][target].values for b in b_cations],
            positions=range(len(b_cations)),
            showmeans=True,
            showmedians=True,
            showextrema=True
        )
        
        # Оформление
        for pc in parts['bodies']:
            pc.set_facecolor('#3498DB')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')
        
        ax.set_xticks(range(len(b_cations)))
        ax.set_xticklabels(b_cations, fontsize=10)
        ax.set_xlabel('B-cation', fontsize=12, fontweight='bold')
        ax.set_ylabel(target, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # --- КАТЕГОРИЯ 2: КОРРЕЛЯЦИОННЫЕ ГРАФИКИ ---
    
    def plot_correlation_matrix(self, features=None):
        """5. Полная матрица корреляций (Пирсон)"""
        if features is None:
            features = self.all_features[:15] if len(self.all_features) > 0 else []
        
        if len(features) < 2:
            return None
        
        # Оставляем только признаки с достаточным количеством данных
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
        
        # Вычисляем корреляционную матрицу
        corr_matrix = data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Маска для верхней половины (показываем только нижнюю)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Тепловая карта
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
            annot_kws={'size': 8},
            ax=ax
        )
        
        ax.set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_partial_correlation_matrix(self, target=None, control_vars=['pH2O']):
        """6. Частичная корреляция"""
        if target is None:
            target = 'α·106 (K-1)'
        
        if target not in self.df.columns:
            return None
        
        features = self.all_features[:10] if len(self.all_features) > 0 else []
        if len(features) < 2:
            return None
        
        # Подготовка данных
        cols = features + [target] + control_vars
        cols = [c for c in cols if c in self.df.columns]
        data = self.df[cols].dropna()
        
        if len(data) < 10:
            return None
        
        # Вычисляем частичные корреляции
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
                except:
                    continue
        
        if len(results) == 0:
            return None
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('correlation', key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Bar plot
        colors = ['red' if c < 0 else 'blue' for c in results_df['correlation'].values]
        ax.barh(results_df['feature'], results_df['correlation'], color=colors, alpha=0.7)
        
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax.set_xlabel(f'Partial Correlation with {target}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Partial Correlation (controlling for {", ".join(control_vars)})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_network(self, features=None, threshold=0.5):
        """7. Сетевой граф корреляций"""
        if features is None:
            features = self.all_features[:15] if len(self.all_features) > 0 else []
        
        if len(features) < 2:
            return None
        
        # Оставляем только признаки с достаточным количеством данных
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
        
        # Вычисляем корреляционную матрицу
        corr_matrix = data.corr()
        
        # Создаём граф
        G = nx.Graph()
        
        # Добавляем узлы
        for feature in valid_features:
            G.add_node(feature)
        
        # Добавляем рёбра для значимых корреляций
        edge_list = []
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features):
                if i < j:
                    corr = corr_matrix.loc[feat1, feat2]
                    if abs(corr) >= threshold and not np.isnan(corr):
                        edge_list.append((feat1, feat2, abs(corr), np.sign(corr)))
        
        # Сортируем рёбра по весу
        edge_list = sorted(edge_list, key=lambda x: x[2], reverse=True)
        
        # Добавляем только топ-30 рёбер
        for edge in edge_list[:30]:
            G.add_edge(edge[0], edge[1], weight=edge[2], sign=edge[3])
        
        if len(G.edges()) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Позиционирование
        pos = nx.spring_layout(G, k=1, seed=42)
        
        # Положительные и отрицательные рёбра
        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('sign', 1) > 0]
        negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('sign', 1) < 0]
        
        # Рисуем рёбра
        if len(positive_edges) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=positive_edges, 
                                   edge_color='blue', alpha=0.6, width=2, ax=ax)
        if len(negative_edges) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=negative_edges, 
                                   edge_color='red', alpha=0.6, width=2, ax=ax)
        
        # Рисуем узлы
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                               node_size=1000, alpha=0.8, ax=ax)
        
        # Рисуем метки
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='Positive correlation'),
            Patch(facecolor='red', alpha=0.6, label='Negative correlation')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f'Correlation Network (threshold = {threshold})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_pairplot_top5(self, features=None, target=None):
        """8. Pairplot топ-5 дескрипторов с целевой переменной"""
        if features is None:
            features = self.all_features[:4] if len(self.all_features) > 0 else []
        
        if target is None:
            target = 'α·106 (K-1)'
        
        if len(features) < 2 or target not in self.df.columns:
            return None
        
        # Оставляем только признаки с достаточным количеством данных
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)
        
        if len(valid_features) < 2:
            return None
        
        # Подготовка данных
        plot_cols = valid_features[:4] + [target]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()
        
        if len(data) < 5:
            return None
        
        # Создаём pairplot
        g = sns.pairplot(
            data,
            vars=valid_features[:4],
            hue=target,
            palette='viridis',
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30},
            diag_kws={'alpha': 0.6}
        )
        
        g.fig.suptitle('Pairplot of Top Features', fontsize=14, fontweight='bold', y=1.02)
        g.fig.set_size_inches(12, 10)
        
        return g.fig
    
    def plot_best_linear_relationships(self, features=None, target=None, n=5):
        """9. Scatter + регрессия (топ-5 зависимостей)"""
        if features is None:
            features = self.all_features[:10] if len(self.all_features) > 0 else []
        
        if target is None:
            target = 'α·106 (K-1)'
        
        if target not in self.df.columns:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                data = self.df[[feature, target]].dropna()
                if len(data) > 10:
                    valid_features.append(feature)
        
        if len(valid_features) == 0:
            return None
        
        # Выбираем топ-5 по корреляции
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
        
        # Создаём подграфики
        n_cols = 2
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                ax = axes[i]
                data = self.df[[feature, target]].dropna()
                
                if len(data) > 2:
                    # Scatter plot
                    ax.scatter(data[feature], data[target], alpha=0.6, color='#3498DB', s=30)
                    
                    # Регрессия
                    try:
                        X = data[[feature]].values
                        y = data[target].values
                        
                        reg = LinearRegression().fit(X, y)
                        y_pred = reg.predict(X)
                        
                        # Линия регрессии
                        x_range = np.linspace(X.min(), X.max(), 100)
                        y_range = reg.predict(x_range.reshape(-1, 1))
                        ax.plot(x_range, y_range, color='red', linewidth=2, label='Regression')
                        
                        # R²
                        r2 = reg.score(X, y)
                        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                    
                    ax.set_xlabel(feature, fontsize=10)
                    ax.set_ylabel(target, fontsize=10)
                    ax.grid(True, alpha=0.3)
        
        # Скрываем пустые подграфики
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, feature=None, target=None):
        """10. График остатков"""
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
            
            # Остатки vs предсказанные значения
            axes[0].scatter(y_pred, residuals, alpha=0.6, color='#3498DB', s=30)
            axes[0].axhline(0, color='red', linewidth=2, linestyle='--')
            axes[0].set_xlabel('Predicted Values', fontsize=12)
            axes[0].set_ylabel('Residuals', fontsize=12)
            axes[0].set_title('Residuals vs Fitted', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # QQ-plot для остатков
            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title('Normal Q-Q Plot', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except:
            return None
    
    def plot_heatmap_top_features(self, features=None):
        """11. Тепловая карта топ-20 дескрипторов"""
        if features is None:
            features = self.all_features[:20] if len(self.all_features) > 0 else []
        
        if len(features) < 2:
            return None
        
        # Оставляем только признаки с достаточным количеством данных
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
        
        # Вычисляем корреляционную матрицу
        corr_matrix = data.corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Тепловая карта
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
            annot_kws={'size': 7},
            ax=ax
        )
        
        ax.set_title('Correlation Heatmap of Top Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # --- КАТЕГОРИЯ 3: PCA И КЛАСТЕРИЗАЦИЯ ---
    
    def plot_elbow(self, features=None, max_components=15):
        """12. Elbow-plot для PCA"""
        if features is None:
            features = self.all_features
        
        if len(features) < 2:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10:
                    valid_features.append(feature)
        
        if len(valid_features) < 2:
            return None
        
        data = self.df[valid_features].dropna()
        if len(data) < 5:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA
        pca = PCA()
        pca.fit(data_scaled)
        
        # Кумулятивная дисперсия
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, 'o-', color='#3498DB', linewidth=2)
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Elbow Plot', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Кумулятивная дисперсия
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'o-', color='#E74C3C', linewidth=2)
        axes[1].axhline(0.95, color='green', linewidth=2, linestyle='--', label='95% variance')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Variance', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_pca_biplot(self, features=None, n_components=2):
        """13. Biplot PCA"""
        if features is None:
            features = self.all_features
        
        if len(features) < 2:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10:
                    valid_features.append(feature)
        
        if len(valid_features) < 2:
            return None
        
        data = self.df[valid_features].dropna()
        if len(data) < 5:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)
        
        # Создаём biplot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Точки
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                           alpha=0.6, c='#3498DB', s=30)
        
        # Стрелки для признаков
        for i, feature in enumerate(valid_features):
            ax.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3,
                    head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
            ax.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2,
                   feature, fontsize=8, ha='center', va='center', color='red')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA Biplot', fontsize=14, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_pca_3d(self, features=None):
        """14. 3D PCA (интерактивный)"""
        if features is None:
            features = self.all_features
        
        if len(features) < 3:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10:
                    valid_features.append(feature)
        
        if len(valid_features) < 3:
            return None
        
        data = self.df[valid_features].dropna()
        if len(data) < 5:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_scaled)
        
        # Создаём 3D scatter plot с помощью Plotly
        fig = go.Figure()
        
        # Добавляем точки
        fig.add_trace(go.Scatter3d(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            z=pca_result[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=pca_result[:, 0],
                colorscale='Viridis',
                colorbar=dict(title='PC1'),
                showscale=True,
                opacity=0.8
            ),
            text=data.index,
            hoverinfo='text'
        ))
        
        # Настройки layout
        fig.update_layout(
            title=dict(
                text='3D PCA Projection',
                font=dict(size=14, family='Times New Roman', color='#000000'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                yaxis=dict(
                    title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                zaxis=dict(
                    title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                )
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def plot_tsne_umap(self, features=None, perplexity=30, n_neighbors=15, min_dist=0.1):
        """15. t-SNE и UMAP"""
        if features is None:
            features = self.all_features
        
        if len(features) < 2:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10:
                    valid_features.append(feature)
        
        if len(valid_features) < 2:
            return None
        
        data = self.df[valid_features].dropna()
        if len(data) < 10:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=min(perplexity, len(data) - 1), random_state=42)
            tsne_result = tsne.fit_transform(data_scaled)
        except:
            tsne_result = None
        
        # UMAP
        try:
            umap_result = umap.UMAP(
                n_neighbors=min(n_neighbors, len(data) - 1),
                min_dist=min_dist,
                random_state=42
            ).fit_transform(data_scaled)
        except:
            umap_result = None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # t-SNE
        if tsne_result is not None:
            axes[0].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          alpha=0.6, color='#3498DB', s=30)
            axes[0].set_xlabel('t-SNE 1', fontsize=12)
            axes[0].set_ylabel('t-SNE 2', fontsize=12)
            axes[0].set_title(f't-SNE (perplexity={perplexity})', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 't-SNE failed', transform=axes[0].transAxes, 
                        ha='center', va='center')
        
        # UMAP
        if umap_result is not None:
            axes[1].scatter(umap_result[:, 0], umap_result[:, 1], 
                          alpha=0.6, color='#E74C3C', s=30)
            axes[1].set_xlabel('UMAP 1', fontsize=12)
            axes[1].set_ylabel('UMAP 2', fontsize=12)
            axes[1].set_title(f'UMAP (n_neighbors={n_neighbors})', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'UMAP failed', transform=axes[1].transAxes, 
                        ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette(self, features=None, max_clusters=10):
        """16. Silhouette plot"""
        if features is None:
            features = self.all_features
        
        if len(features) < 2:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 10:
                    valid_features.append(feature)
        
        if len(valid_features) < 2:
            return None
        
        data = self.df[valid_features].dropna()
        if len(data) < 10:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Вычисляем силуэт для разных K
        silhouette_scores = []
        for k in range(2, min(max_clusters, len(data) - 1) + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data_scaled)
                score = silhouette_score(data_scaled, labels)
                silhouette_scores.append((k, score))
            except:
                continue
        
        if len(silhouette_scores) == 0:
            return None
        
        # Оптимальное K
        silhouette_scores = np.array(silhouette_scores)
        optimal_k = silhouette_scores[np.argmax(silhouette_scores[:, 1]), 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График силуэта
        axes[0].plot(silhouette_scores[:, 0], silhouette_scores[:, 1], 
                    'o-', color='#3498DB', linewidth=2, markersize=8)
        axes[0].axvline(optimal_k, color='red', linewidth=2, linestyle='--', 
                       label=f'Optimal K = {int(optimal_k)}')
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
        axes[0].set_ylabel('Silhouette Score', fontsize=12)
        axes[0].set_title('Silhouette Analysis', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette для оптимального K
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
            axes[1].set_xlabel('Silhouette Coefficient', fontsize=12)
            axes[1].set_ylabel('Cluster', fontsize=12)
            axes[1].set_title(f'Silhouette Plot (K={int(optimal_k)})', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        except:
            axes[1].text(0.5, 0.5, 'Could not compute silhouette for optimal K', 
                        transform=axes[1].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    # --- КАТЕГОРИЯ 4: КОНЦЕНТРАЦИОННЫЕ КАРТЫ ---
    
    def plot_heatmap_2d(self, x_feature, y_feature, target, grid_resolution=50):
        """17. 2D Heatmap"""
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None
        
        if target not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_feature, target]].dropna()
        
        if len(data) < 10:
            return None
        
        # Создаём сетку для интерполяции
        x = data[x_feature].values
        y = data[y_feature].values
        z = data[target].values
        
        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Интерполяция
        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except:
                return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Тепловая карта
        im = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
        
        # Точки данных
        ax.scatter(x, y, c='red', s=20, alpha=0.5, edgecolors='black', linewidth=0.5)
        
        # Оформление
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(target, fontsize=12, fontweight='bold')
        
        ax.set_xlabel(x_feature, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_feature, fontsize=12, fontweight='bold')
        ax.set_title(f'{target} vs {x_feature} and {y_feature}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    def plot_contour(self, x_feature, y_feature, target, grid_resolution=50):
        """18. Contour plot с изолиниями"""
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None
        
        if target not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_feature, target]].dropna()
        
        if len(data) < 10:
            return None
        
        # Создаём сетку для интерполяции
        x = data[x_feature].values
        y = data[y_feature].values
        z = data[target].values
        
        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Интерполяция
        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except:
                return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Контурный график
        contour = ax.contour(xi, yi, zi, levels=15, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=10, fmt='%1.2f')
        
        # Точки данных
        ax.scatter(x, y, c='red', s=20, alpha=0.5, edgecolors='black', linewidth=0.5)
        
        # Оформление
        ax.set_xlabel(x_feature, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_feature, fontsize=12, fontweight='bold')
        ax.set_title(f'{target} Contour Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap_with_points(self, x_feature, y_feature, target, grid_resolution=50):
        """19. Heatmap + overlay точек"""
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None
        
        if target not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_feature, target]].dropna()
        
        if len(data) < 10:
            return None
        
        # Создаём сетку для интерполяции
        x = data[x_feature].values
        y = data[y_feature].values
        z = data[target].values
        
        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Интерполяция
        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        except:
            try:
                zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            except:
                return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Тепловая карта
        im = ax.imshow(zi.T, origin='lower', extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                      aspect='auto', cmap='viridis', alpha=0.8)
        
        # Точки данных с размером по значению
        scatter = ax.scatter(x, y, c=z, s=50, cmap='viridis', 
                           edgecolors='black', linewidth=1, vmin=zi.min(), vmax=zi.max())
        
        # Оформление
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(target, fontsize=12, fontweight='bold')
        
        ax.set_xlabel(x_feature, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_feature, fontsize=12, fontweight='bold')
        ax.set_title(f'{target} Heatmap with Data Points', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    def plot_ternary(self, a_feature='[B\']', b_feature='[D1]', c_feature='[D2]', target=None):
        """20. Ternary plot ([B'], [D1], [D2])"""
        if target is None:
            target = 'α·106 (K-1)'
        
        if target not in self.df.columns:
            return None
        
        # Проверяем наличие всех трёх компонентов
        if a_feature not in self.df.columns or b_feature not in self.df.columns or c_feature not in self.df.columns:
            return None
        
        data = self.df[[a_feature, b_feature, c_feature, target]].dropna()
        
        if len(data) < 5:
            return None
        
        # Нормализация для ternary plot
        total = data[a_feature] + data[b_feature] + data[c_feature]
        data_a = data[a_feature] / total
        data_b = data[b_feature] / total
        data_c = data[c_feature] / total
        
        # Преобразование в декартовы координаты для ternary plot
        x_ternary = 0.5 * (2 * data_b + data_c) / (data_a + data_b + data_c)
        y_ternary = (np.sqrt(3) / 2) * data_c / (data_a + data_b + data_c)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Создаём ternary plot с помощью scatter
        scatter = ax.scatter(x_ternary, y_ternary, c=data[target], 
                           cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Рисуем границы треугольника
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        
        # Подписи вершин
        ax.text(-0.05, -0.05, a_feature, fontsize=12, fontweight='bold')
        ax.text(1.05, -0.05, b_feature, fontsize=12, fontweight='bold')
        ax.text(0.5, np.sqrt(3)/2 + 0.05, c_feature, fontsize=12, fontweight='bold')
        
        # Оформление
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(target, fontsize=12, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Ternary Plot: {a_feature} - {b_feature} - {c_feature}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_temperature_slice(self, feature=None, target=None):
        """21. α vs T для разных составов"""
        if target is None:
            target = 'α·106 (K-1)'
        
        if target not in self.df.columns:
            return None
        
        if 'T_min' not in self.df.columns or 'T_max' not in self.df.columns:
            return None
        
        # Используем feature для группировки
        if feature is None:
            feature = 'B'
        
        if feature not in self.df.columns:
            return None
        
        data = self.df[[feature, 'T_min', 'T_max', target]].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по feature
        groups = data.groupby(feature)
        
        for name, group in groups:
            if len(group) > 1:
                # Средние значения по температуре
                temp_avg = (group['T_min'] + group['T_max']) / 2
                ax.scatter(temp_avg, group[target], label=name, s=50, alpha=0.7)
        
        ax.set_xlabel('Average Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel(target, fontsize=12, fontweight='bold')
        ax.set_title(f'{target} vs Temperature (colored by {feature})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # --- КАТЕГОРИЯ 5: ПУЗЫРЬКОВЫЕ ДИАГРАММЫ ---
    
    def plot_bubble_4d(self, x_feature, y_target, color_feature, size_feature, shape_by='method'):
        """22. 4D Bubble с формами маркеров"""
        if x_feature not in self.df.columns or color_feature not in self.df.columns:
            return None
        
        if size_feature not in self.df.columns:
            return None
        
        if y_target not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_target, color_feature, size_feature]].dropna()
        
        if len(data) < 5:
            return None
        
        # Создаём интерактивный график с Plotly
        fig = go.Figure()
        
        # Преобразование размера
        size_norm = (data[size_feature] - data[size_feature].min()) / (data[size_feature].max() - data[size_feature].min() + 1e-10)
        size_scaled = 10 + size_norm * 40
        
        # Добавляем точки
        fig.add_trace(go.Scatter(
            x=data[x_feature],
            y=data[y_target],
            mode='markers',
            marker=dict(
                size=size_scaled,
                color=data[color_feature],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_feature),
                line=dict(width=1, color='black'),
                opacity=0.7
            ),
            text=[f'{x_feature}: {x:.3f}<br>{y_target}: {y:.3f}<br>{color_feature}: {c:.3f}<br>{size_feature}: {s:.3f}' 
                  for x, y, c, s in zip(data[x_feature], data[y_target], data[color_feature], data[size_feature])],
            hoverinfo='text'
        ))
        
        # Настройки layout
        layout = get_plotly_layout(
            title=f'{y_target} vs {x_feature}',
            xlabel=x_feature,
            ylabel=y_target,
            width=900,
            height=600
        )
        fig.update_layout(layout)
        
        return fig
    
    def plot_compositional_bubble(self, target='α·106 (K-1)'):
        """23. Compositional Bubble ([B'] vs α)"""
        if target not in self.df.columns:
            return None
        
        if '[B\']' not in self.df.columns:
            return None
        
        data = self.df[['[B\']', target, 'B', 'δ']].dropna()
        
        if len(data) < 5:
            return None
        
        fig = go.Figure()
        
        # Группировка по B-катиону
        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            
            fig.add_trace(go.Scatter(
                x=group['[B\']'],
                y=group[target],
                mode='markers',
                name=b_cation,
                marker=dict(
                    size=10 + group['δ'] * 50,
                    sizemin=5,
                    line=dict(width=1, color='black')
                ),
                text=[f'B: {b_cation}<br>[B\']: {b:.3f}<br>{target}: {t:.3f}<br>δ: {d:.3f}' 
                      for b, t, d in zip(group['[B\']'], group[target], group['δ'])],
                hoverinfo='text'
            ))
        
        # Настройки layout
        layout = get_plotly_layout(
            title=f'{target} vs [B\'] (Size = δ)',
            xlabel='[B\'] Concentration',
            ylabel=target,
            width=900,
            height=600
        )
        fig.update_layout(layout)
        
        return fig
    
    def plot_scatter_3d(self, x_feature, y_feature, z_feature, color_by='method'):
        """24. 3D Scatter (интерактивный)"""
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None
        
        if z_feature not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_feature, z_feature]].dropna()
        
        if len(data) < 5:
            return None
        
        fig = go.Figure()
        
        # Используем цвет по категории
        if color_by in self.df.columns:
            color_values = self.df.loc[data.index, color_by]
            
            if color_by == 'method':
                color_map = COLOR_PALETTES['method']
                colors = [color_map.get(c, '#2C3E50') for c in color_values]
                
                fig.add_trace(go.Scatter3d(
                    x=data[x_feature],
                    y=data[y_feature],
                    z=data[z_feature],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=0.5, color='black')
                    ),
                    text=[f'{x_feature}: {x:.3f}<br>{y_feature}: {y:.3f}<br>{z_feature}: {z:.3f}' 
                          for x, y, z in zip(data[x_feature], data[y_feature], data[z_feature])],
                    hoverinfo='text'
                ))
            else:
                # Непрерывный цвет
                fig.add_trace(go.Scatter3d(
                    x=data[x_feature],
                    y=data[y_feature],
                    z=data[z_feature],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=data[color_by] if color_by in data.columns else data[z_feature],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_by),
                        opacity=0.8,
                        line=dict(width=0.5, color='black')
                    ),
                    text=[f'{x_feature}: {x:.3f}<br>{y_feature}: {y:.3f}<br>{z_feature}: {z:.3f}' 
                          for x, y, z in zip(data[x_feature], data[y_feature], data[z_feature])],
                    hoverinfo='text'
                ))
        else:
            # Без цвета
            fig.add_trace(go.Scatter3d(
                x=data[x_feature],
                y=data[y_feature],
                z=data[z_feature],
                mode='markers',
                marker=dict(
                    size=5,
                    color='#3498DB',
                    opacity=0.8,
                    line=dict(width=0.5, color='black')
                ),
                text=[f'{x_feature}: {x:.3f}<br>{y_feature}: {y:.3f}<br>{z_feature}: {z:.3f}' 
                      for x, y, z in zip(data[x_feature], data[y_feature], data[z_feature])],
                hoverinfo='text'
            ))
        
        # Настройки layout
        fig.update_layout(
            title=dict(
                text=f'3D Scatter: {x_feature} vs {y_feature} vs {z_feature}',
                font=dict(size=14, family='Times New Roman', color='#000000'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(title=x_feature, gridcolor='lightgray'),
                yaxis=dict(title=y_feature, gridcolor='lightgray'),
                zaxis=dict(title=z_feature, gridcolor='lightgray')
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def plot_bubble_with_trend(self, x_feature, y_target, color_feature=None):
        """25. Bubble + trend line"""
        if x_feature not in self.df.columns or y_target not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_target]].dropna()
        
        if len(data) < 5:
            return None
        
        # Линейная регрессия
        try:
            X = data[[x_feature]].values
            y = data[y_target].values
            
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = reg.score(X, y)
            
            x_range = np.linspace(X.min(), X.max(), 100)
            y_range = reg.predict(x_range.reshape(-1, 1))
        except:
            y_pred = None
            r2 = None
            x_range = None
            y_range = None
        
        fig = go.Figure()
        
        # Точки
        if color_feature and color_feature in self.df.columns:
            color_data = self.df.loc[data.index, color_feature]
            fig.add_trace(go.Scatter(
                x=data[x_feature],
                y=data[y_target],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_feature),
                    line=dict(width=1, color='black'),
                    opacity=0.7
                ),
                text=[f'{x_feature}: {x:.3f}<br>{y_target}: {y:.3f}' for x, y in zip(data[x_feature], data[y_target])],
                hoverinfo='text',
                name='Data points'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data[x_feature],
                y=data[y_target],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#3498DB',
                    line=dict(width=1, color='black'),
                    opacity=0.7
                ),
                text=[f'{x_feature}: {x:.3f}<br>{y_target}: {y:.3f}' for x, y in zip(data[x_feature], data[y_target])],
                hoverinfo='text',
                name='Data points'
            ))
        
        # Линия регрессии
        if x_range is not None and y_range is not None:
            fig.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_range.flatten(),
                mode='lines',
                name=f'Regression (R² = {r2:.3f})',
                line=dict(color='red', width=2)
            ))
            
            # Добавляем R² в аннотацию
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'R² = {r2:.3f}',
                showarrow=False,
                font=dict(size=12, color='red'),
                bordercolor='black',
                borderwidth=1,
                borderpad=4,
                bgcolor='white',
                opacity=0.8
            )
        
        # Настройки layout
        layout = get_plotly_layout(
            title=f'{y_target} vs {x_feature}',
            xlabel=x_feature,
            ylabel=y_target,
            width=900,
            height=600
        )
        fig.update_layout(layout)
        
        return fig
    
    def plot_bubble_filtered_by_ph2o(self, x_feature, y_target, color_feature=None):
        """26. Bubble с фильтрацией по pH₂O"""
        if x_feature not in self.df.columns or y_target not in self.df.columns:
            return None
        
        if 'pH2O' not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_target, 'pH2O']].dropna()
        
        if len(data) < 5:
            return None
        
        # Создаём интерактивный график с цветом по pH2O
        fig = go.Figure()
        
        # Группировка по диапазонам pH2O
        ph2o_groups = pd.cut(data['pH2O'], bins=4, labels=['Very Low', 'Low', 'Medium', 'High'])
        
        # Цветовая карта для групп
        colors = ['#2C3E50', '#3498DB', '#F39C12', '#E74C3C']
        
        for i, group in enumerate(ph2o_groups.cat.categories):
            group_data = data[ph2o_groups == group]
            if len(group_data) > 0:
                fig.add_trace(go.Scatter(
                    x=group_data[x_feature],
                    y=group_data[y_target],
                    mode='markers',
                    name=group,
                    marker=dict(
                        size=12,
                        color=colors[i % len(colors)],
                        line=dict(width=1, color='black'),
                        opacity=0.7
                    ),
                    text=[f'{x_feature}: {x:.3f}<br>{y_target}: {y:.3f}<br>pH2O: {p:.5f}' 
                          for x, y, p in zip(group_data[x_feature], group_data[y_target], group_data['pH2O'])],
                    hoverinfo='text'
                ))
        
        # Настройки layout
        layout = get_plotly_layout(
            title=f'{y_target} vs {x_feature} (colored by pH₂O)',
            xlabel=x_feature,
            ylabel=y_target,
            width=900,
            height=600
        )
        fig.update_layout(layout)
        
        return fig
    
    def plot_bubble_size_delta(self, x_feature, y_target, color_feature=None):
        """27. Bubble с размером = δ"""
        if x_feature not in self.df.columns or y_target not in self.df.columns:
            return None
        
        if 'δ' not in self.df.columns:
            return None
        
        data = self.df[[x_feature, y_target, 'δ']].dropna()
        
        if len(data) < 5:
            return None
        
        # Создаём интерактивный график с размером по δ
        fig = go.Figure()
        
        # Размер пропорционален δ
        size_norm = (data['δ'] - data['δ'].min()) / (data['δ'].max() - data['δ'].min() + 1e-10)
        size_scaled = 10 + size_norm * 50
        
        # Цвет по целевому параметру или по feature
        if color_feature and color_feature in self.df.columns:
            color_data = self.df.loc[data.index, color_feature]
        else:
            color_data = data[y_target]
        
        fig.add_trace(go.Scatter(
            x=data[x_feature],
            y=data[y_target],
            mode='markers',
            marker=dict(
                size=size_scaled,
                color=color_data,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_feature if color_feature else y_target),
                line=dict(width=1, color='black'),
                opacity=0.7,
                sizemin=5
            ),
            text=[f'{x_feature}: {x:.3f}<br>{y_target}: {y:.3f}<br>δ: {d:.3f}' 
                  for x, y, d in zip(data[x_feature], data[y_target], data['δ'])],
            hoverinfo='text'
        ))
        
        # Настройки layout
        layout = get_plotly_layout(
            title=f'{y_target} vs {x_feature} (Size = δ)',
            xlabel=x_feature,
            ylabel=y_target,
            width=900,
            height=600
        )
        fig.update_layout(layout)
        
        return fig
    
    # --- КАТЕГОРИЯ 6: СПЕЦИАЛИЗИРОВАННЫЕ ГРАФИКИ ---
    
    def plot_t_bends_vs_delta(self):
        """28. T(bends) vs δ"""
        if 'δ' not in self.df.columns or 'T_bends_1' not in self.df.columns:
            return None
        
        data = self.df[['δ', 'T_bends_1', 'B', 'method']].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по B-катиону
        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['δ'], group['T_bends_1'], label=b_cation, s=50, alpha=0.7)
        
        ax.set_xlabel('δ (Oxygen Vacancy Concentration)', fontsize=12, fontweight='bold')
        ax.set_ylabel('T(bends) (°C)', fontsize=12, fontweight='bold')
        ax.set_title('T(bends) vs δ (colored by B-cation)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_alpha_vs_beta(self):
        """29. α vs β (компромиссная диаграмма)"""
        if 'α·106 (K-1)' not in self.df.columns or 'β' not in self.df.columns:
            return None
        
        data = self.df[['α·106 (K-1)', 'β', 'B', 'method']].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по B-катиону
        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['α·106 (K-1)'], group['β'], label=b_cation, s=50, alpha=0.7)
        
        ax.set_xlabel('α (10⁻⁶ K⁻¹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('β (Chemical Expansion Coefficient)', fontsize=12, fontweight='bold')
        ax.set_title('α vs β: Compromise Diagram (colored by B-cation)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_beta_vs_ph2o(self):
        """30. β vs pH₂O"""
        if 'β' not in self.df.columns or 'pH2O' not in self.df.columns:
            return None
        
        data = self.df[['β', 'pH2O', 'B']].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по B-катиону
        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['pH2O'], group['β'], label=b_cation, s=50, alpha=0.7)
        
        ax.set_xlabel('pH₂O (log scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('β (Chemical Expansion Coefficient)', fontsize=12, fontweight='bold')
        ax.set_title('β vs pH₂O (colored by B-cation)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_alpha_vs_rAav(self):
        """31. α vs rAav"""
        if 'α·106 (K-1)' not in self.df.columns or 'rAav' not in self.df.columns:
            return None
        
        data = self.df[['α·106 (K-1)', 'rAav', 'A', 'method']].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по A-катиону
        for a_cation in data['A'].unique():
            group = data[data['A'] == a_cation]
            ax.scatter(group['rAav'], group['α·106 (K-1)'], label=a_cation, s=50, alpha=0.7)
        
        ax.set_xlabel('rAav (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('α (10⁻⁶ K⁻¹)', fontsize=12, fontweight='bold')
        ax.set_title('α vs rAav (colored by A-cation)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_beta_vs_chiBav(self):
        """32. β vs χBav"""
        if 'β' not in self.df.columns or 'χBav' not in self.df.columns:
            return None
        
        data = self.df[['β', 'χBav', 'B']].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по B-катиону
        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['χBav'], group['β'], label=b_cation, s=50, alpha=0.7)
        
        ax.set_xlabel('χBav (Average Electronegativity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('β (Chemical Expansion Coefficient)', fontsize=12, fontweight='bold')
        ax.set_title('β vs χBav (colored by B-cation)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_t_bends_vs_t_stab(self):
        """33. T(bends) vs T_stab"""
        if 'T_bends_1' not in self.df.columns or 'T_stab' not in self.df.columns:
            return None
        
        data = self.df[['T_bends_1', 'T_stab', 'B']].dropna()
        
        if len(data) < 5:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Группировка по B-катиону
        for b_cation in data['B'].unique():
            group = data[data['B'] == b_cation]
            ax.scatter(group['T_stab'], group['T_bends_1'], label=b_cation, s=50, alpha=0.7)
        
        ax.set_xlabel('T_stab (Temperature Stability of Protons, K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('T(bends) (°C)', fontsize=12, fontweight='bold')
        ax.set_title('T(bends) vs T_stab (colored by B-cation)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # --- ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ---
    
    def plot_pairplot_colored(self, features, hue='method'):
        """34. Pairplot с многоцветным исполнением"""
        if len(features) < 2 or len(features) > 5:
            return None
        
        if hue not in self.df.columns:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)
        
        if len(valid_features) < 2:
            return None
        
        # Подготовка данных
        plot_cols = valid_features + [hue]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()
        
        if len(data) < 5:
            return None
        
        # Цветовая палитра для hue
        if hue == 'method':
            palette = COLOR_PALETTES['method']
        elif hue == 'B':
            palette = COLOR_PALETTES['B_cation']
        elif hue == 'A':
            palette = COLOR_PALETTES['A_cation']
        else:
            palette = 'viridis'
        
        # Создаём pairplot
        g = sns.pairplot(
            data,
            vars=valid_features,
            hue=hue,
            palette=palette,
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30},
            diag_kws={'alpha': 0.6}
        )
        
        g.fig.suptitle(f'Pairplot of Selected Features (colored by {hue})', 
                      fontsize=14, fontweight='bold', y=1.02)
        g.fig.set_size_inches(12, 10)
        
        return g.fig
    
    def plot_radar_chart(self, features, group_by='method'):
        """35. Радарная диаграмма для сравнения групп"""
        if len(features) < 3:
            return None
        
        if group_by not in self.df.columns:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)
        
        if len(valid_features) < 3:
            return None
        
        # Группируем данные
        groups = self.df.groupby(group_by)
        
        # Вычисляем средние значения для каждой группы
        means = {}
        for name, group in groups:
            if len(group) > 3:
                means[name] = group[valid_features].mean().values
        
        if len(means) == 0:
            return None
        
        # Нормализация для радарной диаграммы
        all_values = np.array(list(means.values()))
        all_values_norm = (all_values - all_values.min(axis=0)) / (all_values.max(axis=0) - all_values.min(axis=0) + 1e-10)
        
        # Создаём радарную диаграмму
        angles = np.linspace(0, 2 * np.pi, len(valid_features), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
        
        for i, (name, values_norm) in enumerate(zip(means.keys(), all_values_norm)):
            values = values_norm.tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(valid_features, fontsize=10)
        ax.set_title(f'Radar Chart: Comparison by {group_by}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_parallel_coordinates(self, features, target='α·106 (K-1)'):
        """36. Параллельные координаты (опционально)"""
        if len(features) < 3:
            return None
        
        if target not in self.df.columns:
            return None
        
        # Фильтруем признаки
        valid_features = []
        for feature in features:
            if feature in self.df.columns:
                non_na = self.df[feature].dropna()
                if len(non_na) > 5:
                    valid_features.append(feature)
        
        if len(valid_features) < 3:
            return None
        
        # Подготовка данных
        plot_cols = valid_features + [target]
        plot_cols = [c for c in plot_cols if c in self.df.columns]
        data = self.df[plot_cols].dropna()
        
        if len(data) < 5:
            return None
        
        # Стандартизация
        scaler = StandardScaler()
        data_scaled = data.copy()
        data_scaled[valid_features] = scaler.fit_transform(data[valid_features])
        
        # Создаём параллельные координаты с Plotly
        fig = go.Figure()
        
        # Сортируем по целевому параметру для лучшей визуализации
        data_sorted = data_scaled.sort_values(target)
        
        for i, row in data_sorted.iterrows():
            fig.add_trace(go.Scatter(
                x=valid_features + [target],
                y=row[valid_features + [target]].values,
                mode='lines',
                line=dict(color=plt.cm.viridis(row[target] / data_sorted[target].max())),
                showlegend=False,
                hoverinfo='text',
                text=[f'{col}: {val:.3f}' for col, val in zip(valid_features + [target], row[valid_features + [target]].values)]
            ))
        
        # Настройки layout
        fig.update_layout(
            title=dict(
                text=f'Parallel Coordinates (colored by {target})',
                font=dict(size=14, family='Times New Roman', color='#000000'),
                x=0.5
            ),
            xaxis=dict(
                title='Features',
                tickfont=dict(size=10, family='Times New Roman'),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Standardized Value',
                gridcolor='lightgray'
            ),
            width=1000,
            height=600,
            margin=dict(l=80, r=40, t=60, b=60)
        )
        
        return fig


# ============================================================================
# 5. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ
# ============================================================================

def main():
    """Главная функция приложения"""
    
    # Применяем научный стиль
    apply_scientific_style()
    
    # --- SIDEBAR: Загрузка данных и фильтры ---
    with st.sidebar:
        st.title("⚡ Perovskite Analyzer")
        st.markdown("---")
        
        # Виджет загрузки текста
        st.subheader("📂 Загрузка данных")
        data_text = st.text_area(
            "Вставьте данные в формате TSV/CSV:",
            height=200,
            placeholder="№\tA\tA'\tB\tB'\tD1\tD2\t[A']\t[B']\t[D1]\t[D2]\tδ\tmethod\tβ\t∆T, °C\tα·106 (K-1)\tT(bends), °C\tαav·106 (K-1)\tpH2O\tRef\n1\tBa\t-\tCe\tZr\tY\tYb\t0\t0.1\t0.1\t0.1\t0.1\tdilatometry\t0.0073\t27-1000\t10.6\t400;600\t10.6;4.73;10.1\t0.0001\t10.15826/chimtech.2024.11.4.22"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            load_button = st.button("🚀 Загрузить и обработать", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Очистить", use_container_width=True)
        
        if clear_button:
            # Очищаем все ключи session_state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if load_button and data_text:
            with st.spinner("Обработка данных..."):
                # Загрузка данных
                processor = DataProcessor()
                df_raw = processor.parse_text_data(data_text)
                
                if df_raw is not None and len(df_raw) > 0:
                    # Очистка данных
                    df_clean = processor.clean_data(df_raw)
                    
                    # Сохранение в сессию
                    st.session_state['df_raw'] = df_raw
                    st.session_state['df_clean'] = df_clean
                    st.session_state['data_loaded'] = True
                    
                    # Расчёт дескрипторов
                    engine = DescriptorEngine(df_clean)
                    df_descriptors = engine.calculate_all()
                    st.session_state['df_descriptors'] = df_descriptors
                    
                    # Сохраняем отфильтрованные данные как копию дескрипторов
                    st.session_state['filtered_df'] = df_descriptors.copy()
                    
                    st.success(f"✅ Загружено {len(df_clean)} записей")
                    st.info(f"📊 Рассчитано {len(df_descriptors.columns) - len(df_clean.columns)} дескрипторов")
                else:
                    st.error("❌ Ошибка загрузки данных. Проверьте формат.")
        
        st.markdown("---")
        
        # Фильтры (появляются после загрузки данных)
        # Проверяем, что данные загружены и ключ существует
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            if 'df_descriptors' in st.session_state:
                st.subheader("🔍 Фильтры")
                
                df = st.session_state['df_descriptors']
                
                # Уровень 1: Базовые фильтры
                st.subheader("Метод измерения")
                method_options = df['method'].unique().tolist() if 'method' in df.columns else []
                method_filter = st.multiselect(
                    "Выберите методы:",
                    options=method_options,
                    default=method_options
                )
                
                st.subheader("A-катион")
                a_options = df['A'].unique().tolist() if 'A' in df.columns else []
                a_filter = st.multiselect(
                    "Выберите A-катионы:",
                    options=a_options,
                    default=a_options
                )
                
                st.subheader("B-катион")
                b_options = df['B'].unique().tolist() if 'B' in df.columns else []
                b_filter = st.multiselect(
                    "Выберите B-катионы:",
                    options=b_options,
                    default=b_options
                )
                
                # Уровень 2: Расширенные фильтры
                st.subheader("Диапазон δ")
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
                
                # Уровень 3: Дескрипторные фильтры
                with st.expander("Дескрипторные фильтры"):
                    if 't' in df.columns:
                        t_min = float(df['t'].min())
                        t_max = float(df['t'].max())
                        t_range = st.slider(
                            "Толерант-фактор (t):",
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
                
                # Применение фильтров
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
                st.caption(f"📊 Данных после фильтрации: {len(filtered_df)} строк")
            else:
                st.warning("⚠️ Данные загружены, но дескрипторы не рассчитаны. Нажмите 'Загрузить и обработать'.")
    
    # --- MAIN AREA: Tabs ---
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        if 'df_descriptors' in st.session_state and 'filtered_df' in st.session_state:
            
            # Получаем данные
            df = st.session_state['filtered_df'] if 'filtered_df' in st.session_state else st.session_state['df_descriptors']
            
            # Инициализация визуализационного движка
            viz = VisualizationEngine(df)
            
            # Получаем список дескрипторов
            all_descriptors = [col for col in df.columns if col not in ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Ref', 'method', '∆T, °C']]
            all_descriptors = [col for col in all_descriptors if df[col].dtype in ['float64', 'int64']]
            all_descriptors = [col for col in all_descriptors if df[col].isna().sum() / len(df) < 0.5]
            
            viz.set_features(all_descriptors)
            
            # Создание вкладок
            tabs = st.tabs([
                "📊 Обзор данных",
                "🧮 Дескрипторы",
                "📈 Корреляции",
                "🗺️ Карты и кластеры",
                "🎯 Визуализация",
                "📤 Экспорт"
            ])
            
            # --- TAB 1: Обзор данных ---
            with tabs[0]:
                st.header("📊 Обзор данных")
                
                # Статистика
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Всего записей", len(df))
                with col2:
                    unique_compositions = df.groupby(['A', 'B']).ngroups if 'A' in df.columns and 'B' in df.columns else 0
                    st.metric("Уникальных составов", unique_compositions)
                with col3:
                    methods = df['method'].nunique() if 'method' in df.columns else 0
                    st.metric("Методов измерения", methods)
                with col4:
                    refs = df['Ref'].nunique() if 'Ref' in df.columns else 0
                    st.metric("Источников (Ref)", refs)
                
                # Предпросмотр данных
                st.subheader("📄 Данные")
                st.dataframe(df, use_container_width=True, height=400)
                
                # Статистика по колонкам
                st.subheader("📊 Статистика числовых колонок")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                # Распределение целевых переменных
                st.subheader("📈 Распределение целевых переменных")
                
                target_cols = [t for t in TARGET_VARIABLES if t in df.columns]
                if target_cols:
                    cols = st.columns(min(len(target_cols), 3))
                    for i, target in enumerate(target_cols[:3]):
                        with cols[i]:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            data = df[target].dropna()
                            if len(data) > 0:
                                ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='#3498DB')
                                ax.set_xlabel(target, fontsize=11, fontweight='bold')
                                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            else:
                                st.write("Нет данных")
            
            # --- TAB 2: Дескрипторы ---
            with tabs[1]:
                st.header("🧮 Расчёт дескрипторов")
                
                # Информация о дескрипторах
                descriptor_cols = [col for col in df.columns if col not in ['№', 'A', 'A\'', 'B', 'B\'', 'D1', 'D2', 'Ref', 'method', '∆T, °C']]
                descriptor_cols = [col for col in descriptor_cols if df[col].dtype in ['float64', 'int64']]
                
                st.info(f"📊 Всего рассчитано: {len(descriptor_cols)} дескрипторов")
                
                # Категории дескрипторов
                with st.expander("📋 Список всех дескрипторов по категориям"):
                    categories = {
                        "Геометрические": ['rAav', 'rBav', 't', 'D_t', 'octahedral_factor', 'Δr_AB', 'Δr_AB_norm', 'σ²_rA', 'σ²_rB', 'V_cell', 'r_ratio_AB'],
                        "Электроотрицательные": ['χAav', 'χBav', 'Δχ_AB', 'χ_ratio_AB', 'ionicity_AO', 'ionicity_BO', 'acidity_AO', 'acidity_BO', 'Δacidity', 'χ_total', 'χ_ratio_t'],
                        "Термодинамические": ['S_config_A', 'S_config_B', 'V_Bav', 'Vo_proxy', 'ΔH_hydr', 'E_BO', 'ρ'],
                        "Массовые": ['M_Aav', 'M_Bav', 'M_total', 'M_ratio_AB', 'M_rA', 'M_χA'],
                        "Дефектные": ['δ_actual', 'Z_eff_B', 'proton_affinity', 'E_vac'],
                        "Специфические": ['alpha_beta_ratio', 'T_stab', 'δ_χB', 'δ_rB'],
                        "Составные": ['B\'_conc', 'D_total', 'D_B_ratio', 'total_dopant']
                    }
                    
                    for category, descs in categories.items():
                        present = [d for d in descs if d in df.columns]
                        st.write(f"**{category}**: {len(present)}/{len(descs)} дескрипторов")
                        st.write(", ".join(present))
                        st.write("")
                
                # Таблица с дескрипторами
                st.subheader("📊 Таблица дескрипторов")
                if descriptor_cols:
                    display_cols = ['№' if '№' in df.columns else None, 'A' if 'A' in df.columns else None, 'B' if 'B' in df.columns else None] + descriptor_cols[:10]
                    display_cols = [c for c in display_cols if c is not None]
                    st.dataframe(df[display_cols], use_container_width=True, height=300)
                
                # Распределение дескрипторов
                st.subheader("📈 Распределение дескрипторов")
                
                if descriptor_cols:
                    selected_desc = st.selectbox("Выберите дескриптор для отображения:", descriptor_cols)
                    if selected_desc:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        data = df[selected_desc].dropna()
                        if len(data) > 0:
                            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='#2ECC71')
                            ax.set_xlabel(selected_desc, fontsize=12, fontweight='bold')
                            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        else:
                            st.warning("Нет данных для выбранного дескриптора")
            
            # --- TAB 3: Корреляции ---
            with tabs[2]:
                st.header("📈 Расширенный корреляционный анализ")
                
                if len(all_descriptors) > 1:
                    # Выбор целевой переменной
                    target_options = [t for t in TARGET_VARIABLES if t in df.columns]
                    if target_options:
                        target = st.selectbox(
                            "Выберите целевую переменную:",
                            options=target_options
                        )
                        
                        # Выбор топ-N
                        n_features = st.slider("Количество топ-дескрипторов:", 5, 30, 20)
                        
                        # Анализ корреляций
                        if st.button("🔍 Анализировать корреляции", use_container_width=True):
                            with st.spinner("Анализ корреляций..."):
                                analyzer = CorrelationAnalyzer(df)
                                
                                # Топ-дескрипторы
                                top_features = analyzer.get_top_features(target, n=n_features)
                                
                                if top_features:
                                    st.success(f"✅ Найдено {len(top_features)} значимых дескрипторов")
                                    
                                    # Таблица топ-дескрипторов
                                    st.subheader(f"🏆 Топ-{len(top_features)} дескрипторов для {target}")
                                    
                                    pearson_results = analyzer.pearson_correlation(target=target)
                                    if pearson_results is not None:
                                        top_df = pearson_results[pearson_results['feature'].isin(top_features)]
                                        st.dataframe(top_df[['feature', 'correlation', 'p_value', 'significant']], use_container_width=True)
                                    
                                    # Графики в колонках
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("📊 Топ-20 корреляций")
                                        if pearson_results is not None:
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            top_plot = pearson_results.head(20)
                                            colors = ['red' if c < 0 else 'blue' for c in top_plot['correlation'].values]
                                            ax.barh(top_plot['feature'], top_plot['correlation'], color=colors, alpha=0.7)
                                            ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
                                            ax.set_xlabel(f'Correlation with {target}', fontsize=12, fontweight='bold')
                                            ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
                                            ax.set_title(f'Top 20 Correlations with {target}', fontsize=14, fontweight='bold')
                                            ax.grid(True, alpha=0.3)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                    
                                    with col2:
                                        st.subheader("🔗 Сетевой граф")
                                        G = analyzer.correlation_network(features=top_features[:15], threshold=0.4)
                                        if G is not None and len(G.edges()) > 0:
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            pos = nx.spring_layout(G, k=1, seed=42)
                                            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                                                  node_size=500, alpha=0.8, ax=ax)
                                            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
                                            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)
                                            ax.set_title('Correlation Network', fontsize=14, fontweight='bold')
                                            ax.axis('off')
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                        else:
                                            st.warning("Недостаточно связей для построения сети")
                                    
                                    # Матрица корреляций
                                    st.subheader("📊 Матрица корреляций топ-дескрипторов")
                                    corr_fig = viz.plot_correlation_matrix(features=top_features[:10])
                                    if corr_fig is not None:
                                        st.pyplot(corr_fig)
                                    
                                    # Pairplot
                                    st.subheader("🎨 Pairplot топ-5 дескрипторов")
                                    pair_fig = viz.plot_pairplot_top5(features=top_features[:4], target=target)
                                    if pair_fig is not None:
                                        st.pyplot(pair_fig)
                                else:
                                    st.warning("Не удалось найти значимые корреляции")
                    else:
                        st.warning("Нет целевых переменных в данных")
                else:
                    st.warning("Недостаточно дескрипторов для корреляционного анализа")
            
            # --- TAB 4: Карты и кластеры ---
            with tabs[3]:
                st.header("🗺️ Концентрационные карты и кластеризация")
                
                # Подвкладки
                sub_tabs = st.tabs(["Концентрационные карты", "PCA", "Кластеризация"])
                
                with sub_tabs[0]:
                    st.subheader("🗺️ Концентрационные карты")
                    
                    # Выбор параметров
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_options = ['δ', '[B\']', 'D_total', 'rAav', 'rBav', 't', 'χBav', 'r_ratio_AB', 'δ_actual', 'B\'_conc']
                        x_options = [x for x in x_options if x in df.columns]
                        x_axis = st.selectbox("X-axis:", options=x_options, index=0 if x_options else None)
                    with col2:
                        y_options = ['[B\']', 'D_total', 'δ', 'rAav', 'rBav', 't', 'χBav', 'r_ratio_AB', 'B\'_conc']
                        y_options = [y for y in y_options if y in df.columns]
                        y_axis = st.selectbox("Y-axis:", options=y_options, index=min(1, len(y_options)-1) if y_options else None)
                    with col3:
                        color_options = [t for t in TARGET_VARIABLES if t in df.columns]
                        color_by = st.selectbox("Color (целевая переменная):", options=color_options, index=0 if color_options else None)
                    
                    # Тип карты
                    map_type = st.radio(
                        "Тип карты:",
                        options=['Heatmap', 'Contour', 'Heatmap + точки'],
                        horizontal=True
                    )
                    
                    if x_axis and y_axis and color_by:
                        if st.button("🗺️ Построить карту", use_container_width=True):
                            with st.spinner("Построение карты..."):
                                if map_type == 'Heatmap':
                                    fig = viz.plot_heatmap_2d(x_axis, y_axis, color_by)
                                elif map_type == 'Contour':
                                    fig = viz.plot_contour(x_axis, y_axis, color_by)
                                else:
                                    fig = viz.plot_heatmap_with_points(x_axis, y_axis, color_by)
                                
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Недостаточно данных для построения карты")
                    
                    # Ternary plot
                    st.subheader("📐 Ternary Plot")
                    if '[B\']' in df.columns and '[D1]' in df.columns and '[D2]' in df.columns:
                        ternary_target = st.selectbox(
                            "Целевая переменная для Ternary Plot:",
                            options=color_options,
                            key="ternary_target"
                        )
                        if st.button("🔺 Построить Ternary Plot", use_container_width=True):
                            fig = viz.plot_ternary('[B\']', '[D1]', '[D2]', ternary_target)
                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.warning("Недостаточно данных для Ternary Plot")
                
                with sub_tabs[1]:
                    st.subheader("📊 PCA анализ")
                    
                    # Выбор компонент
                    pc_x = st.selectbox("PC X:", [1, 2, 3], index=0)
                    pc_y = st.selectbox("PC Y:", [1, 2, 3], index=1)
                    
                    # Biplot
                    if st.button("📊 Построить Biplot", use_container_width=True):
                        with st.spinner("Построение Biplot..."):
                            fig = viz.plot_pca_biplot()
                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.warning("Недостаточно данных для PCA")
                    
                    # 3D PCA
                    if st.button("🎯 Построить 3D PCA", use_container_width=True):
                        with st.spinner("Построение 3D PCA..."):
                            fig = viz.plot_pca_3d()
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Недостаточно данных для 3D PCA")
                    
                    # Elbow plot
                    st.subheader("📉 Elbow Plot")
                    max_components = st.slider("Максимальное число компонент:", 5, 20, 10)
                    if st.button("📉 Построить Elbow Plot", use_container_width=True):
                        fig = viz.plot_elbow(max_components=max_components)
                        if fig is not None:
                            st.pyplot(fig)
                        else:
                            st.warning("Недостаточно данных для Elbow Plot")
                
                with sub_tabs[2]:
                    st.subheader("🔍 Кластеризация")
                    
                    # Выбор алгоритма
                    algo = st.selectbox(
                        "Алгоритм:",
                        options=['K-means', 'DBSCAN', 'Иерархическая']
                    )
                    
                    if algo == 'K-means':
                        n_clusters = st.slider("K (количество кластеров):", 2, 10, 3)
                        if st.button("🔍 Кластеризовать (K-means)", use_container_width=True):
                            with st.spinner("Выполнение K-means кластеризации..."):
                                # Выполняем кластеризацию
                                features = all_descriptors[:20]  # Используем топ-20
                                valid_features = []
                                for feat in features:
                                    if feat in df.columns and df[feat].isna().sum() / len(df) < 0.3:
                                        valid_features.append(feat)
                                
                                if len(valid_features) > 1:
                                    data = df[valid_features].dropna()
                                    if len(data) > 10:
                                        scaler = StandardScaler()
                                        data_scaled = scaler.fit_transform(data)
                                        
                                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                        labels = kmeans.fit_predict(data_scaled)
                                        
                                        # Добавляем метки в DataFrame
                                        df_clustered = data.copy()
                                        df_clustered['Cluster'] = labels
                                        
                                        # Silhouette score
                                        score = silhouette_score(data_scaled, labels)
                                        st.success(f"✅ Silhouette Score: {score:.3f}")
                                        
                                        # Визуализация с помощью PCA
                                        pca = PCA(n_components=2)
                                        pca_result = pca.fit_transform(data_scaled)
                                        
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                                           c=labels, cmap='viridis', s=50, alpha=0.7)
                                        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
                                        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
                                        ax.set_title(f'K-means Clustering (K={n_clusters})', fontsize=14, fontweight='bold')
                                        plt.colorbar(scatter, ax=ax, label='Cluster')
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Недостаточно данных для кластеризации")
                                else:
                                    st.warning("Недостаточно признаков для кластеризации")
                    
                    elif algo == 'DBSCAN':
                        eps = st.slider("eps (максимальное расстояние):", 0.1, 5.0, 0.5, 0.1)
                        min_samples = st.slider("min_samples:", 2, 10, 5)
                        if st.button("🔍 Кластеризовать (DBSCAN)", use_container_width=True):
                            with st.spinner("Выполнение DBSCAN кластеризации..."):
                                features = all_descriptors[:20]
                                valid_features = []
                                for feat in features:
                                    if feat in df.columns and df[feat].isna().sum() / len(df) < 0.3:
                                        valid_features.append(feat)
                                
                                if len(valid_features) > 1:
                                    data = df[valid_features].dropna()
                                    if len(data) > 10:
                                        scaler = StandardScaler()
                                        data_scaled = scaler.fit_transform(data)
                                        
                                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                                        labels = dbscan.fit_predict(data_scaled)
                                        
                                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                        n_noise = list(labels).count(-1)
                                        
                                        st.info(f"📊 Найдено кластеров: {n_clusters}, Шумовых точек: {n_noise}")
                                        
                                        # Визуализация с помощью PCA
                                        pca = PCA(n_components=2)
                                        pca_result = pca.fit_transform(data_scaled)
                                        
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                                           c=labels, cmap='viridis', s=50, alpha=0.7)
                                        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
                                        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
                                        ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})', 
                                                    fontsize=14, fontweight='bold')
                                        plt.colorbar(scatter, ax=ax, label='Cluster')
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Недостаточно данных для кластеризации")
                                else:
                                    st.warning("Недостаточно признаков для кластеризации")
                    
                    else:  # Иерархическая
                        if st.button("🔍 Кластеризовать (Иерархическая)", use_container_width=True):
                            with st.spinner("Выполнение иерархической кластеризации..."):
                                # Используем silhouette для оптимального K
                                fig = viz.plot_silhouette()
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Недостаточно данных для кластеризации")
            
            # --- TAB 5: Визуализация ---
            with tabs[4]:
                st.header("🎯 Интерактивная визуализация")
                
                # Категории графиков
                plot_category = st.selectbox(
                    "Выберите категорию графиков:",
                    options=[
                        "Пузырьковые диаграммы",
                        "Специализированные графики",
                        "Pairplot с цветом",
                        "Радарные диаграммы"
                    ]
                )
                
                if plot_category == "Пузырьковые диаграммы":
                    st.subheader("💎 Пузырьковая диаграмма")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        x_feature = st.selectbox("X:", options=all_descriptors, index=0 if all_descriptors else None)
                    with col2:
                        y_target = st.selectbox("Y (целевая):", options=[t for t in TARGET_VARIABLES if t in df.columns], index=0)
                    with col3:
                        color_options = [d for d in all_descriptors if d != x_feature and d != y_target]
                        color_feature = st.selectbox("Color:", options=color_options, index=0 if color_options else None)
                    with col4:
                        size_options = [d for d in all_descriptors if d != x_feature and d != y_target and d != color_feature]
                        size_feature = st.selectbox("Size:", options=size_options, index=0 if size_options else None)
                    
                    if x_feature and y_target and color_feature and size_feature:
                        if st.button("💎 Построить пузырьковую диаграмму", use_container_width=True):
                            with st.spinner("Построение..."):
                                fig = viz.plot_bubble_4d(x_feature, y_target, color_feature, size_feature)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Недостаточно данных для построения")
                    
                    st.subheader("📊 Compositional Bubble")
                    if st.button("📊 Построить Compositional Bubble", use_container_width=True):
                        with st.spinner("Построение..."):
                            fig = viz.plot_compositional_bubble()
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Недостаточно данных")
                    
                    st.subheader("🎯 3D Scatter")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_3d = st.selectbox("X (3D):", options=all_descriptors, key="x_3d", index=0)
                    with col2:
                        y_3d = st.selectbox("Y (3D):", options=all_descriptors, key="y_3d", index=min(1, len(all_descriptors)-1))
                    with col3:
                        z_3d = st.selectbox("Z (3D):", options=all_descriptors, key="z_3d", index=min(2, len(all_descriptors)-1))
                    
                    if x_3d and y_3d and z_3d:
                        if st.button("🎯 Построить 3D Scatter", use_container_width=True):
                            with st.spinner("Построение..."):
                                fig = viz.plot_scatter_3d(x_3d, y_3d, z_3d)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Недостаточно данных")
                    
                    st.subheader("📈 Bubble с размером = δ")
                    if st.button("📈 Построить Bubble с размером = δ", use_container_width=True):
                        with st.spinner("Построение..."):
                            x_feat = st.selectbox("X:", options=all_descriptors, key="bubble_delta_x")
                            y_targ = st.selectbox("Y:", options=[t for t in TARGET_VARIABLES if t in df.columns], key="bubble_delta_y")
                            if x_feat and y_targ:
                                fig = viz.plot_bubble_size_delta(x_feat, y_targ)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Недостаточно данных")
                
                elif plot_category == "Специализированные графики":
                    st.subheader("🎯 Специализированные графики")
                    
                    spec_plots = [
                        "T(bends) vs δ",
                        "α vs β (компромисс)",
                        "β vs pH₂O",
                        "α vs rAav",
                        "β vs χBav",
                        "T(bends) vs T_stab"
                    ]
                    
                    selected_plot = st.selectbox("Выберите график:", spec_plots)
                    
                    if st.button("📊 Построить", use_container_width=True):
                        with st.spinner("Построение..."):
                            fig = None
                            if selected_plot == "T(bends) vs δ":
                                fig = viz.plot_t_bends_vs_delta()
                            elif selected_plot == "α vs β (компромисс)":
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
                                st.warning("Недостаточно данных для построения графика")
                
                elif plot_category == "Pairplot с цветом":
                    st.subheader("🎨 Pairplot с многоцветным исполнением")
                    
                    # Выбор признаков
                    features = st.multiselect(
                        "Выберите признаки (2-5):",
                        options=all_descriptors,
                        default=all_descriptors[:3] if len(all_descriptors) >= 3 else all_descriptors
                    )
                    
                    hue_by = st.selectbox(
                        "Цвет по:",
                        options=['method', 'B', 'A'] + [t for t in TARGET_VARIABLES if t in df.columns],
                        index=0
                    )
                    
                    if len(features) >= 2:
                        if st.button("🎨 Построить Pairplot", use_container_width=True):
                            with st.spinner("Построение..."):
                                fig = viz.plot_pairplot_colored(features, hue_by)
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Недостаточно данных")
                    else:
                        st.warning("Выберите как минимум 2 признака")
                
                elif plot_category == "Радарные диаграммы":
                    st.subheader("📊 Радарная диаграмма для сравнения групп")
                    
                    radar_features = st.multiselect(
                        "Выберите признаки (минимум 3):",
                        options=all_descriptors,
                        default=all_descriptors[:3] if len(all_descriptors) >= 3 else all_descriptors
                    )
                    
                    group_by = st.selectbox(
                        "Группировка по:",
                        options=['method', 'B', 'A'],
                        index=0
                    )
                    
                    if len(radar_features) >= 3:
                        if st.button("📊 Построить радарную диаграмму", use_container_width=True):
                            with st.spinner("Построение..."):
                                fig = viz.plot_radar_chart(radar_features, group_by)
                                if fig is not None:
                                    st.pyplot(fig)
                                else:
                                    st.warning("Недостаточно данных")
                    else:
                        st.warning("Выберите как минимум 3 признака")
            
            # --- TAB 6: Экспорт ---
            with tabs[5]:
                st.header("📤 Экспорт результатов")
                
                st.subheader("📊 Экспорт данных")
                
                # Экспорт дескрипторов
                if st.button("📥 Скачать данные с дескрипторами (CSV)", use_container_width=True):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать CSV",
                        data=csv,
                        file_name="perovskite_descriptors.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.subheader("🖼️ Экспорт графиков")
                
                # Выбор графиков для экспорта
                export_plots = st.multiselect(
                    "Выберите графики для экспорта:",
                    options=[
                        "Корреляционные графики",
                        "PCA и кластеризация",
                        "Концентрационные карты",
                        "Пузырьковые диаграммы",
                        "Специализированные графики",
                        "Pairplot и радарные диаграммы"
                    ]
                )
                
                if export_plots:
                    st.info(f"ℹ️ Выбрано {len(export_plots)} категорий графиков")
                    
                    if st.button("📦 Сгенерировать графики для экспорта", use_container_width=True):
                        with st.spinner("Генерация графиков..."):
                            st.warning("⚠️ Функция экспорта графиков в ZIP находится в разработке")
                
                st.subheader("📄 Экспорт отчёта")
                
                if st.button("📄 Сгенерировать HTML-отчёт", use_container_width=True):
                    with st.spinner("Генерация отчёта..."):
                        st.warning("⚠️ Функция генерации HTML-отчёта находится в разработке")
        
        else:
            st.warning("⚠️ Данные загружены, но дескрипторы не рассчитаны. Пожалуйста, нажмите кнопку 'Загрузить и обработать' в боковой панели.")
    
    else:
        # Отображение приветствия при отсутствии данных
        st.info("👈 Загрузите данные через боковую панель (Sidebar) для начала анализа")
        
        st.markdown("""
        ### 📋 Формат данных
        
        Данные должны быть в формате TSV (табуляция) или CSV с колонками:
        
        - `№` - номер записи
        - `A` - A-катион
        - `A'` - допант в A-позиции
        - `B` - B-катион
        - `B'` - изовалентный допант
        - `D1`, `D2` - акцепторные допанты
        - `[A']`, `[B']`, `[D1]`, `[D2]` - концентрации
        - `δ` - концентрация вакансий
        - `method` - метод измерения
        - `β` - коэффициент химического расширения
        - `∆T, °C` - температурный диапазон
        - `α·106 (K-1)` - коэффициент термического расширения
        - `T(bends), °C` - температуры изломов
        - `αav·106 (K-1)` - средний КТР на участках
        - `pH2O` - парциальное давление воды
        - `Ref` - ссылка на источник
        
        ### 🚀 Возможности приложения
        
        - 📊 Расчёт 63+ дескрипторов
        - 📈 Расширенный корреляционный анализ
        - 🗺️ Концентрационные карты и кластеризация
        - 🎯 36 типов интерактивных графиков
        - 🔍 Фильтрация по составу и условиям эксперимента
        """)

# ============================================================================
# 6. ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == "__main__":
    main()
