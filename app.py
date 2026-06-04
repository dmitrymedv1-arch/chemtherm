"""
Streamlit Application for Analysis of Thermal and Chemical Expansion
of Proton-Conducting Perovskite Oxides

Version: 3.0 (with NaN handling, extended descriptors, enhanced visualizations)
Author: Materials Informatics Research
Description: Comprehensive analysis tool for understanding composition-structure-property
             relationships in proton-conducting perovskites with focus on thermal
             expansion (α), chemical expansion (β), and phase transitions.
             
Features:
- Robust handling of '-' and missing values (now using NaN instead of 0)
- Upload and process two independent datasets via text/CSV/TSV input
- Calculate 35+ structural, electronegativity, and thermodynamic descriptors
- Interactive visualizations with scientific styling
- Machine learning models for property prediction
- SHAP analysis for interpretability
- Phase transition impact analysis with advanced plots
- Clustering and dimensionality reduction
- Enhanced pairplot with user-selectable descriptors
- Bubble charts with density contours (heatmap-style)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import re
from scipy import stats
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
from io import BytesIO
import csv
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# 0. УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ БЕЗОПАСНОГО ПРЕОБРАЗОВАНИЯ В FLOAT (теперь возвращает NaN)
# ============================================================================

def safe_float_conversion(value: Any, default: float = np.nan) -> float:
    """
    Safely convert any value to float, handling:
    - '-', '—', '', None, NaN
    - Strings with commas (European format: '0,0971847700154417')
    - Lists, tuples, and other sequences
    
    Returns NaN for missing values instead of 0 to preserve physical meaning.
    """
    # Handle None, NaN, and empty values
    if value is None:
        return default
    if pd.isna(value):
        return default
    
    # Handle string values
    if isinstance(value, str):
        value_str = value.strip()
        # Check for dash/empty placeholders
        if value_str == '' or value_str == '-' or value_str == '—' or value_str == '–':
            return default
        # Replace comma with dot (European decimal format)
        if ',' in value_str and '.' not in value_str:
            value_str = value_str.replace(',', '.')
        # Handle semicolon-separated values (take first)
        if ';' in value_str:
            value_str = value_str.split(';')[0].strip()
        # Handle slash-separated values (take first)
        if '/' in value_str:
            value_str = value_str.split('/')[0].strip()
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return default
    
    # Handle list/tuple (take first element if numeric)
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            return safe_float_conversion(value[0], default)
        return default
    
    # Handle numeric types directly
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    
    # Fallback
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_parse_temperature_range(temp_range_str: Any) -> Tuple[float, float, float, float]:
    """
    Safely parse temperature range string like '27-1000' or '430-630'
    Returns (T_min, T_max, T_span, T_mid)
    """
    T_min = np.nan
    T_max = np.nan
    T_span = np.nan
    T_mid = np.nan
    
    if pd.isna(temp_range_str) or temp_range_str == '-' or temp_range_str == '':
        return T_min, T_max, T_span, T_mid
    
    temp_str = str(temp_range_str).strip()
    if '-' in temp_str:
        parts = temp_str.split('-')
        if len(parts) == 2:
            try:
                T_min = safe_float_conversion(parts[0].strip(), np.nan)
                T_max = safe_float_conversion(parts[1].strip(), np.nan)
                if not np.isnan(T_min) and not np.isnan(T_max) and T_max > T_min:
                    T_span = T_max - T_min
                    T_mid = (T_min + T_max) / 2
            except (ValueError, TypeError):
                pass
    
    return T_min, T_max, T_span, T_mid


def safe_parse_semicolon_values(value: Any) -> List[float]:
    """
    Safely parse semicolon-separated values like '400;600' or '10.6;4.73;10.1'
    Returns list of floats (empty list for missing)
    """
    if pd.isna(value) or value == '-' or value == '':
        return []
    
    value_str = str(value).strip()
    if ';' in value_str:
        parts = value_str.split(';')
        result = []
        for part in parts:
            part = part.strip()
            if part and part != '-':
                val = safe_float_conversion(part, np.nan)
                if not np.isnan(val):
                    result.append(val)
        return result
    else:
        val = safe_float_conversion(value_str, np.nan)
        return [val] if not np.isnan(val) else []

# ============================================================================
# 1. НАСТРОЙКИ СТРАНИЦЫ И СТИЛЯ
# ============================================================================

st.set_page_config(
    page_title="Perovskite Expansion Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_scientific_css():
    """Enhanced scientific CSS styling for publication-ready appearance"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #ffffff;
    }
    
    /* Scientific card styling */
    .card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
        border: none;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2A9D8F;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #cccccc;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-family: 'Times New Roman', serif;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2A9D8F;
        color: white;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.85rem;
        font-family: 'Times New Roman', serif;
        border-collapse: collapse;
        width: 100%;
    }
    
    .dataframe th {
        background-color: #2c3e50;
        color: white;
        padding: 8px;
        text-align: center;
    }
    
    .dataframe td {
        padding: 6px;
        border-bottom: 1px solid #ddd;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', serif;
        font-weight: bold;
        color: #1a1a2e;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2A9D8F 0%, #1D3557 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 2px 8px rgba(42,157,143,0.3);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f0f0;
        border-radius: 6px;
        font-weight: bold;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2A9D8F;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #e67e22;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Plot container */
    .plot-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. БАЗЫ ДАННЫХ ДЛЯ РАСЧЁТА ДЕСКРИПТОРОВ
# ============================================================================

@dataclass
class IonicRadii:
    """Ionic radii database (Shannon) for different coordination numbers"""
    # CN = 12 (A-site, cuboctahedral)
    cn12: Dict[str, float] = field(default_factory=lambda: {
        "Ba": 1.61, "Sr": 1.44, "Ca": 1.34, "La": 1.36, "Nd": 1.27, "Pr": 1.29,
        "Sm": 1.24, "Eu": 1.20, "Gd": 1.19, "Tb": 1.18, "Dy": 1.17, "Ho": 1.16,
        "Er": 1.15, "Tm": 1.14, "Yb": 1.13, "Lu": 1.12, "Y": 1.19, "Sc": 0.95,
        "Pb": 1.49, "Bi": 1.30, "K": 1.64, "Na": 1.39, "Li": 0.92, "Ag": 1.54,
        "Cd": 1.31, "Hg": 1.40, "Sn": 1.28, "Pb": 1.49
    })
    
    # CN = 6 (B-site, octahedral)
    cn6: Dict[str, float] = field(default_factory=lambda: {
        "Zr": 0.72, "Ce": 0.87, "Sn": 0.69, "Ti": 0.605, "Hf": 0.71, "Si": 0.40,
        "Ge": 0.53, "Al": 0.535, "Ga": 0.62, "In": 0.80, "Sc": 0.745, "Y": 0.90,
        "La": 1.032, "Nd": 0.983, "Sm": 0.958, "Eu": 0.947, "Gd": 0.938,
        "Tb": 0.923, "Dy": 0.912, "Ho": 0.901, "Er": 0.89, "Tm": 0.88, "Yb": 0.868,
        "Lu": 0.861, "Fe": 0.645, "Mn": 0.645, "Co": 0.545, "Ni": 0.69,
        "Cu": 0.73, "Zn": 0.74, "Cr": 0.615, "V": 0.64, "Mo": 0.59, "W": 0.60,
        "Ru": 0.68, "Rh": 0.665, "Ir": 0.63, "Pt": 0.625, "Pd": 0.86, "Ag": 0.94,
        "Cd": 0.95, "Hg": 1.02, "Sn": 0.83, "Pb": 0.775, "Bi": 0.76, "Sb": 0.76,
        "Te": 0.70, "Se": 0.50, "S": 0.37
    })
    
    # CN = 4 (tetrahedral, for some dopants)
    cn4: Dict[str, float] = field(default_factory=lambda: {
        "Zn": 0.60, "Fe": 0.49, "Co": 0.58, "Ni": 0.55, "Cu": 0.57, "Mn": 0.39,
        "Cr": 0.44, "V": 0.42, "Ti": 0.42, "Sn": 0.55, "Ge": 0.39, "Si": 0.26,
        "Al": 0.39, "Ga": 0.47, "In": 0.62, "B": 0.12
    })
    
    # CN = 8 (for some A-site substitutions)
    cn8: Dict[str, float] = field(default_factory=lambda: {
        "Ba": 1.42, "Sr": 1.26, "Ca": 1.12, "La": 1.16, "Nd": 1.09, "Pr": 1.10,
        "Sm": 1.07, "Eu": 1.06, "Gd": 1.053, "Tb": 1.04, "Dy": 1.027, "Ho": 1.015,
        "Y": 1.019, "Yb": 0.985, "Lu": 0.977, "Sc": 0.87, "Pb": 1.29, "Bi": 1.17
    })

@dataclass
class ElementProperties:
    """Comprehensive element properties for descriptor calculation"""
    electronegativity: Dict[str, float] = field(default_factory=lambda: {
        "Ba": 0.89, "Sr": 0.95, "Ca": 1.00, "Mg": 1.31, "La": 1.10, "Ce": 1.12,
        "Pr": 1.13, "Nd": 1.14, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.20,
        "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27,
        "Y": 1.22, "Sc": 1.36, "Zr": 1.33, "Hf": 1.30, "Ti": 1.54, "Sn": 1.96,
        "Si": 1.90, "Ge": 2.01, "Al": 1.61, "Ga": 1.81, "In": 1.78, "Fe": 1.83,
        "Mn": 1.55, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Cr": 1.66,
        "V": 1.63, "Mo": 2.16, "W": 2.36, "Ru": 2.20, "Rh": 2.28, "Ir": 2.20,
        "Pt": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "Hg": 2.00, "Pb": 2.33,
        "Bi": 2.02, "Sb": 2.05, "Te": 2.10, "Se": 2.55, "S": 2.58, "O": 3.44
    })
    
    polarizability: Dict[str, float] = field(default_factory=lambda: {
        "Ba": 6.8, "Sr": 4.2, "Ca": 3.2, "Mg": 1.3, "La": 4.8, "Ce": 4.7, "Pr": 4.6,
        "Nd": 4.5, "Sm": 4.3, "Eu": 4.2, "Gd": 4.1, "Tb": 4.0, "Dy": 3.9, "Ho": 3.8,
        "Y": 3.9, "Yb": 3.7, "Sc": 2.1, "Zr": 3.2, "Ti": 2.9, "Sn": 3.9, "In": 3.1,
        "Fe": 2.0, "Zn": 2.0, "Al": 1.8, "Ga": 2.4, "Ge": 2.2, "Si": 1.4, "O": 3.9
    })
    
    ionization_potential: Dict[str, float] = field(default_factory=lambda: {
        "Ba": 5.21, "Sr": 5.69, "Ca": 6.11, "Mg": 7.65, "La": 5.58, "Ce": 5.54,
        "Pr": 5.46, "Nd": 5.53, "Sm": 5.64, "Eu": 5.67, "Gd": 6.15, "Y": 6.38,
        "Yb": 6.25, "Sc": 6.56, "Zr": 6.84, "Ti": 6.83, "Sn": 7.34, "In": 5.79,
        "Fe": 7.90, "Zn": 9.39, "Al": 5.99, "Ga": 6.00, "Ge": 7.90, "Si": 8.15,
        "O": 13.62
    })
    
    valency: Dict[str, int] = field(default_factory=lambda: {
        "Ba": 2, "Sr": 2, "Ca": 2, "Mg": 2, "La": 3, "Ce": 4, "Pr": 4, "Nd": 3,
        "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 4, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3,
        "Yb": 3, "Lu": 3, "Y": 3, "Sc": 3, "Zr": 4, "Hf": 4, "Ti": 4, "Sn": 4,
        "Si": 4, "Ge": 4, "Al": 3, "Ga": 3, "In": 3, "Fe": 3, "Mn": 4, "Co": 2,
        "Ni": 2, "Cu": 2, "Zn": 2, "Cr": 3, "V": 5, "Mo": 6, "W": 6, "Ru": 4,
        "Rh": 3, "Ir": 4, "Pt": 4, "Pd": 2, "Ag": 1, "Cd": 2, "Hg": 2, "Pb": 2,
        "Bi": 3, "Sb": 3, "Te": 4, "Se": -2, "S": -2, "O": -2
    })
    
    atomic_weight: Dict[str, float] = field(default_factory=lambda: {
        "Ba": 137.33, "Sr": 87.62, "Ca": 40.08, "Mg": 24.31, "La": 138.91,
        "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36, "Eu": 151.96,
        "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
        "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Y": 88.91, "Sc": 44.96,
        "Zr": 91.22, "Hf": 178.49, "Ti": 47.87, "Sn": 118.71, "Si": 28.09,
        "Ge": 72.63, "Al": 26.98, "Ga": 69.72, "In": 114.82, "Fe": 55.85,
        "Mn": 54.94, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
        "Cr": 52.00, "V": 50.94, "Mo": 95.94, "W": 183.84, "Pb": 207.20,
        "Bi": 208.98, "O": 16.00
    })

# Initialize databases
ionic_radii = IonicRadii()
element_props = ElementProperties()

# Gas constant (J/(mol·K))
R_GAS = 8.314

# ============================================================================
# 3. КЛАСС ДЛЯ РАСЧЁТА ДЕСКРИПТОРОВ
# ============================================================================

class PerovskiteDescriptorCalculator:
    """Calculate 35+ structural, electronic, and thermodynamic descriptors"""
    
    def __init__(self):
        self.r_o = 1.4  # Ionic radius of O2- (CN=4,6)
        self.chi_o = 3.44  # Electronegativity of oxygen
        
    def get_ionic_radius(self, element: str, site: str) -> float:
        """Get ionic radius for element based on site (A or B)"""
        if pd.isna(element) or element == '-' or element == '':
            return np.nan
        
        # Determine coordination number based on site
        if site.upper() == 'A':
            # A-site typically CN=12, fallback to CN=8
            radius = ionic_radii.cn12.get(element, None)
            if radius is None:
                radius = ionic_radii.cn8.get(element, np.nan)
        else:  # B-site
            # B-site typically CN=6, fallback to CN=4
            radius = ionic_radii.cn6.get(element, None)
            if radius is None:
                radius = ionic_radii.cn4.get(element, np.nan)
        return radius if radius is not None else np.nan
    
    def get_electronegativity(self, element: str) -> float:
        """Get Pauling electronegativity"""
        if pd.isna(element) or element == '-' or element == '':
            return np.nan
        return element_props.electronegativity.get(element, np.nan)
    
    def get_polarizability(self, element: str) -> float:
        """Get ionic polarizability"""
        if pd.isna(element) or element == '-' or element == '':
            return np.nan
        return element_props.polarizability.get(element, np.nan)
    
    def get_ionization_potential(self, element: str) -> float:
        """Get first ionization potential (eV)"""
        if pd.isna(element) or element == '-' or element == '':
            return np.nan
        return element_props.ionization_potential.get(element, np.nan)
    
    def get_valency(self, element: str) -> int:
        """Get common oxidation state"""
        if pd.isna(element) or element == '-' or element == '':
            return np.nan
        return element_props.valency.get(element, np.nan)
    
    def calculate_average_radius(self, elements: List[str], concentrations: List[float], site: str) -> float:
        """Calculate weighted average ionic radius (returns NaN if no valid data)"""
        total = 0.0
        weight_sum = 0.0
        for elem, conc in zip(elements, concentrations):
            if not pd.isna(conc) and conc > 0 and elem not in [None, '-', ''] and not pd.isna(elem):
                rad = self.get_ionic_radius(elem, site)
                if not np.isnan(rad):
                    total += conc * rad
                    weight_sum += conc
        return total / weight_sum if weight_sum > 0 else np.nan
    
    def calculate_average_electronegativity(self, elements: List[str], concentrations: List[float]) -> float:
        """Calculate weighted average electronegativity (returns NaN if no valid data)"""
        total = 0.0
        weight_sum = 0.0
        for elem, conc in zip(elements, concentrations):
            if not pd.isna(conc) and conc > 0 and elem not in [None, '-', ''] and not pd.isna(elem):
                chi = self.get_electronegativity(elem)
                if not np.isnan(chi):
                    total += conc * chi
                    weight_sum += conc
        return total / weight_sum if weight_sum > 0 else np.nan
    
    def calculate_average_polarizability(self, elements: List[str], concentrations: List[float]) -> float:
        """Calculate weighted average polarizability (returns NaN if no valid data)"""
        total = 0.0
        weight_sum = 0.0
        for elem, conc in zip(elements, concentrations):
            if not pd.isna(conc) and conc > 0 and elem not in [None, '-', ''] and not pd.isna(elem):
                pol = self.get_polarizability(elem)
                if not np.isnan(pol):
                    total += conc * pol
                    weight_sum += conc
        return total / weight_sum if weight_sum > 0 else np.nan
    
    def calculate_average_valency(self, elements: List[str], concentrations: List[float]) -> float:
        """Calculate weighted average valency (returns NaN if no valid data)"""
        total = 0.0
        weight_sum = 0.0
        for elem, conc in zip(elements, concentrations):
            if not pd.isna(conc) and conc > 0 and elem not in [None, '-', ''] and not pd.isna(elem):
                val = self.get_valency(elem)
                if not np.isnan(val):
                    total += conc * val
                    weight_sum += conc
        return total / weight_sum if weight_sum > 0 else np.nan
    
    def calculate_configurational_entropy(self, concentrations: List[float]) -> float:
        """Calculate configurational entropy: S = -R * Σ(x_i * ln(x_i))"""
        total = 0.0
        valid_concs = [c for c in concentrations if not pd.isna(c) and c > 0]
        if not valid_concs:
            return np.nan
        # Normalize to sum=1 if not already
        sum_conc = sum(valid_concs)
        if sum_conc > 0:
            valid_concs = [c/sum_conc for c in valid_concs]
        for x in valid_concs:
            if x > 0:
                total -= x * np.log(x)
        return R_GAS * total if total > 0 else 0.0
    
    def calculate_all_descriptors(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate all 35+ descriptors for a given composition row
        
        Returns dictionary with descriptor names and values (NaN for missing)
        """
        descriptors = {}
        
        # Extract composition data (these are strings, no conversion needed)
        A = row.get('A', None) if not pd.isna(row.get('A', None)) else None
        A_prime = row.get("A'", None) if not pd.isna(row.get("A'", None)) else None
        B = row.get('B', None) if not pd.isna(row.get('B', None)) else None
        B_prime = row.get("B'", None) if not pd.isna(row.get("B'", None)) else None
        D1 = row.get('D1', None) if not pd.isna(row.get('D1', None)) else None
        D2 = row.get('D2', None) if not pd.isna(row.get('D2', None)) else None
        
        # Concentrations - using safe_float_conversion (returns NaN for missing)
        conc_A_prime = safe_float_conversion(row.get("[A']", np.nan), np.nan)
        conc_B_prime = safe_float_conversion(row.get("[B']", np.nan), np.nan)
        conc_D1 = safe_float_conversion(row.get("[D1]", np.nan), np.nan)
        conc_D2 = safe_float_conversion(row.get("[D2]", np.nan), np.nan)
        
        # Ensure concentrations are within [0, 1] if not NaN
        if not np.isnan(conc_A_prime):
            conc_A_prime = max(0.0, min(1.0, conc_A_prime))
        if not np.isnan(conc_B_prime):
            conc_B_prime = max(0.0, min(1.0, conc_B_prime))
        if not np.isnan(conc_D1):
            conc_D1 = max(0.0, min(1.0, conc_D1))
        if not np.isnan(conc_D2):
            conc_D2 = max(0.0, min(1.0, conc_D2))
        
        # Calculate remaining concentrations (handling NaNs)
        conc_A = 1.0 - (conc_A_prime if not np.isnan(conc_A_prime) else 0)
        
        # total_dopant_B is ONLY D1+D2 (acceptor dopants creating vacancies)
        total_dopant_B_val = 0.0
        if not np.isnan(conc_D1):
            total_dopant_B_val += conc_D1
        if not np.isnan(conc_D2):
            total_dopant_B_val += conc_D2
        
        # B concentration = 1 - B_prime - D1 - D2
        conc_B = 1.0
        if not np.isnan(conc_B_prime):
            conc_B -= conc_B_prime
        if not np.isnan(conc_D1):
            conc_B -= conc_D1
        if not np.isnan(conc_D2):
            conc_B -= conc_D2
        conc_B = max(0.0, conc_B)  # Ensure non-negative
        
        # Store individual concentration descriptors
        descriptors['conc_A_prime'] = conc_A_prime if not np.isnan(conc_A_prime) else 0.0
        descriptors['conc_B_prime'] = conc_B_prime if not np.isnan(conc_B_prime) else 0.0
        descriptors['conc_D1'] = conc_D1 if not np.isnan(conc_D1) else 0.0
        descriptors['conc_D2'] = conc_D2 if not np.isnan(conc_D2) else 0.0
        descriptors['conc_A'] = conc_A
        descriptors['conc_B'] = conc_B
        
        # ====================================================================
        # Category 1: Geometric descriptors
        # ====================================================================
        
        # A-site elements and concentrations
        A_elements = []
        A_concentrations = []
        if A and A not in [None, '-', '']:
            A_elements.append(A)
            A_concentrations.append(conc_A)
        if A_prime and A_prime not in [None, '-', ''] and conc_A_prime > 0:
            A_elements.append(A_prime)
            A_concentrations.append(conc_A_prime)
        
        # B-site elements and concentrations
        B_elements = []
        B_concentrations = []
        if B and B not in [None, '-', ''] and conc_B > 0:
            B_elements.append(B)
            B_concentrations.append(conc_B)
        if B_prime and B_prime not in [None, '-', ''] and conc_B_prime > 0:
            B_elements.append(B_prime)
            B_concentrations.append(conc_B_prime)
        if D1 and D1 not in [None, '-', ''] and conc_D1 > 0:
            B_elements.append(D1)
            B_concentrations.append(conc_D1)
        if D2 and D2 not in [None, '-', ''] and conc_D2 > 0:
            B_elements.append(D2)
            B_concentrations.append(conc_D2)
        
        # Calculate average radii
        rA_avg = self.calculate_average_radius(A_elements, A_concentrations, 'A')
        rB_avg = self.calculate_average_radius(B_elements, B_concentrations, 'B')
        
        descriptors['rA_avg'] = rA_avg if not np.isnan(rA_avg) else 0.0
        descriptors['rB_avg'] = rB_avg if not np.isnan(rB_avg) else 0.0
        descriptors['rA_rB_ratio'] = (rA_avg / rB_avg) if (not np.isnan(rA_avg) and not np.isnan(rB_avg) and rB_avg > 0) else 0.0
        
        # Tolerance factor (Goldschmidt)
        sqrt2 = np.sqrt(2)
        denominator = sqrt2 * (rB_avg + self.r_o) if not np.isnan(rB_avg) else 0
        t = (rA_avg + self.r_o) / denominator if (not np.isnan(rA_avg) and denominator > 0) else 0
        descriptors['tolerance_factor'] = t
        descriptors['tolerance_deviation'] = abs(1 - t)
        
        # Octahedral factor
        octahedral = rB_avg / self.r_o if (not np.isnan(rB_avg) and self.r_o > 0) else 0
        descriptors['octahedral_factor'] = octahedral
        
        # Radius difference
        if not np.isnan(rA_avg) and not np.isnan(rB_avg):
            delta_r_AB = abs(rA_avg - rB_avg)
        else:
            delta_r_AB = 0.0
        descriptors['delta_r_AB'] = delta_r_AB
        descriptors['delta_r_AB_norm'] = delta_r_AB / self.r_o if self.r_o > 0 else 0
        
        # Variance of B-site radii
        if len(B_elements) > 1:
            valid_radii = []
            valid_concs = []
            for e, c in zip(B_elements, B_concentrations):
                rad = self.get_ionic_radius(e, 'B')
                if not np.isnan(rad) and c > 0:
                    valid_radii.append(rad)
                    valid_concs.append(c)
            if len(valid_radii) > 1:
                total_conc = sum(valid_concs)
                if total_conc > 0:
                    rB_squared_avg = sum(c * (r**2) for r, c in zip(valid_radii, valid_concs)) / total_conc
                    rB_avg_sq = (descriptors['rB_avg'])**2
                    descriptors['variance_rB'] = max(0, rB_squared_avg - rB_avg_sq)
                else:
                    descriptors['variance_rB'] = 0.0
            else:
                descriptors['variance_rB'] = 0.0
        else:
            descriptors['variance_rB'] = 0.0
        
        # Variance of A-site radii
        if len(A_elements) > 1:
            valid_radii = []
            valid_concs = []
            for e, c in zip(A_elements, A_concentrations):
                rad = self.get_ionic_radius(e, 'A')
                if not np.isnan(rad) and c > 0:
                    valid_radii.append(rad)
                    valid_concs.append(c)
            if len(valid_radii) > 1:
                total_conc = sum(valid_concs)
                if total_conc > 0:
                    rA_squared_avg = sum(c * (r**2) for r, c in zip(valid_radii, valid_concs)) / total_conc
                    rA_avg_sq = (descriptors['rA_avg'])**2
                    descriptors['variance_rA'] = max(0, rA_squared_avg - rA_avg_sq)
                else:
                    descriptors['variance_rA'] = 0.0
            else:
                descriptors['variance_rA'] = 0.0
        else:
            descriptors['variance_rA'] = 0.0
        
        # ====================================================================
        # Category 2: Electronegativity descriptors
        # ====================================================================
        
        # Individual element electronegativities
        descriptors['chi_A'] = self.get_electronegativity(A) if A and A not in [None, '-', ''] else np.nan
        descriptors['chi_A_prime'] = self.get_electronegativity(A_prime) if A_prime and A_prime not in [None, '-', ''] else np.nan
        descriptors['chi_B'] = self.get_electronegativity(B) if B and B not in [None, '-', ''] else np.nan
        descriptors['chi_B_prime'] = self.get_electronegativity(B_prime) if B_prime and B_prime not in [None, '-', ''] else np.nan
        descriptors['chi_D1'] = self.get_electronegativity(D1) if D1 and D1 not in [None, '-', ''] else np.nan
        descriptors['chi_D2'] = self.get_electronegativity(D2) if D2 and D2 not in [None, '-', ''] else np.nan
        
        # Replace NaN with 0 for compatibility
        for key in ['chi_A', 'chi_A_prime', 'chi_B', 'chi_B_prime', 'chi_D1', 'chi_D2']:
            if np.isnan(descriptors[key]):
                descriptors[key] = 0.0
        
        # Average electronegativities
        chiA_avg = self.calculate_average_electronegativity(A_elements, A_concentrations)
        chiB_avg = self.calculate_average_electronegativity(B_elements, B_concentrations)
        
        descriptors['chiA_avg'] = chiA_avg if not np.isnan(chiA_avg) else 0.0
        descriptors['chiB_avg'] = chiB_avg if not np.isnan(chiB_avg) else 0.0
        
        # Difference and ratio
        delta_chi_AB = abs(descriptors['chiA_avg'] - descriptors['chiB_avg'])
        descriptors['delta_chi_AB'] = delta_chi_AB
        descriptors['chi_ratio_AB'] = descriptors['chiA_avg'] / descriptors['chiB_avg'] if descriptors['chiB_avg'] > 0 else 0
        
        # Total average electronegativity
        descriptors['chi_total_avg'] = (descriptors['chiA_avg'] + descriptors['chiB_avg']) / 2
        
        # Ionicity (Pauling formula)
        ionicity_AO = 1 - np.exp(-0.25 * (descriptors['chiA_avg'] - self.chi_o)**2)
        ionicity_BO = 1 - np.exp(-0.25 * (descriptors['chiB_avg'] - self.chi_o)**2)
        descriptors['ionicity_AO'] = ionicity_AO
        descriptors['ionicity_BO'] = ionicity_BO
        
        # ====================================================================
        # Category 3: Thermodynamic descriptors
        # ====================================================================
        
        # Configurational entropy
        descriptors['S_config_A'] = self.calculate_configurational_entropy(A_concentrations)
        descriptors['S_config_B'] = self.calculate_configurational_entropy(B_concentrations)
        descriptors['S_config_total'] = descriptors['S_config_A'] + descriptors['S_config_B']
        
        # Valency
        descriptors['valency_A'] = self.get_valency(A) if A and A not in [None, '-', ''] else 0
        descriptors['valency_A_prime'] = self.get_valency(A_prime) if A_prime and A_prime not in [None, '-', ''] else 0
        descriptors['valency_B'] = self.get_valency(B) if B and B not in [None, '-', ''] else 0
        descriptors['valency_B_prime'] = self.get_valency(B_prime) if B_prime and B_prime not in [None, '-', ''] else 0
        descriptors['valency_D1'] = self.get_valency(D1) if D1 and D1 not in [None, '-', ''] else 0
        descriptors['valency_D2'] = self.get_valency(D2) if D2 and D2 not in [None, '-', ''] else 0
        
        # Average valency on B-site
        VB_avg = self.calculate_average_valency(B_elements, B_concentrations)
        descriptors['VB_avg'] = VB_avg if not np.isnan(VB_avg) else 0.0
        
        # Oxygen vacancy proxy (for Ce4+/Zr4+ based perovskites)
        # Vacancy concentration = [D1]/2 + [D2]/2 (only from acceptor dopants, NOT from B')
        Vo_proxy_val = 0.0
        if not np.isnan(conc_D1):
            Vo_proxy_val += conc_D1 / 2
        if not np.isnan(conc_D2):
            Vo_proxy_val += conc_D2 / 2
        descriptors['Vo_proxy'] = Vo_proxy_val
        
        # ====================================================================
        # Category 4: Combined (physics-inspired) descriptors
        # ====================================================================
        
        # Delta chi divided by tolerance factor
        descriptors['delta_chi_div_t'] = delta_chi_AB / t if t > 0 else 0
        
        # Delta chi multiplied by tolerance factor
        descriptors['delta_chi_mul_t'] = delta_chi_AB * t
        
        # Disorder over distortion ratio
        variance_rB = descriptors['variance_rB']
        t_dev = descriptors['tolerance_deviation']
        descriptors['disorder_over_distortion'] = variance_rB / (t_dev + 1e-6)
        
        # Ionicity × octahedral factor
        descriptors['ionic_x_octa'] = ionicity_BO * octahedral
        
        # Chi ratio × tolerance factor
        descriptors['chi_ratio_t'] = descriptors['chi_ratio_AB'] * t
        
        # rB × chiB
        descriptors['rB_x_chiB'] = descriptors['rB_avg'] * descriptors['chiB_avg']
        
        # ====================================================================
        # Category 5: Additional useful descriptors
        # ====================================================================
        
        # Total dopant concentrations (distinguishing B' from D1+D2)
        descriptors['total_dopant_A'] = conc_A_prime if not np.isnan(conc_A_prime) else 0.0
        descriptors['total_dopant_B'] = total_dopant_B_val  # ONLY D1+D2, NOT including B'
        descriptors['total_dopant_B_with_Bprime'] = total_dopant_B_val + (conc_B_prime if not np.isnan(conc_B_prime) else 0.0)
        descriptors['total_dopant'] = descriptors['total_dopant_A'] + descriptors['total_dopant_B']
        
        # Oxygen stoichiometry parameter delta
        delta = safe_float_conversion(row.get('δ', np.nan), np.nan)
        descriptors['delta'] = delta if not np.isnan(delta) else 0.0
        
        # Chemical expansion beta
        beta_val = safe_float_conversion(row.get('β', np.nan), np.nan)
        descriptors['beta'] = beta_val if not np.isnan(beta_val) else 0.0
        
        # True thermal expansion coefficient alpha
        alpha_val = safe_float_conversion(row.get('α·106 (K-1)', np.nan), np.nan)
        descriptors['alpha_true'] = alpha_val if not np.isnan(alpha_val) else 0.0
        
        # Apparent thermal expansion coefficient alpha_av
        alpha_av_raw = row.get('αav·106 (K-1)', np.nan)
        if pd.isna(alpha_av_raw) or alpha_av_raw == '-' or alpha_av_raw == '':
            descriptors['alpha_apparent'] = np.nan
        else:
            alpha_av_list = safe_parse_semicolon_values(alpha_av_raw)
            descriptors['alpha_apparent'] = alpha_av_list[0] if alpha_av_list else np.nan
        if np.isnan(descriptors['alpha_apparent']):
            descriptors['alpha_apparent'] = 0.0
        
        # Water partial pressure
        pH2O = safe_float_conversion(row.get('pH2O', np.nan), np.nan)
        descriptors['log_pH2O'] = np.log10(pH2O) if (not np.isnan(pH2O) and pH2O > 0) else -10
        
        # Temperature range span
        temp_range = row.get('∆T, °C', '')
        T_min, T_max, T_span, T_mid = safe_parse_temperature_range(temp_range)
        descriptors['T_min'] = T_min if not np.isnan(T_min) else 0.0
        descriptors['T_max'] = T_max if not np.isnan(T_max) else 0.0
        descriptors['T_span'] = T_span if not np.isnan(T_span) else 0.0
        descriptors['T_mid'] = T_mid if not np.isnan(T_mid) else 0.0
        
        # Bends temperatures
        bends_raw = row.get('T(bends), °C', '')
        bends_list = safe_parse_semicolon_values(bends_raw)
        descriptors['T_bends_first'] = bends_list[0] if bends_list else np.nan
        descriptors['T_bends_count'] = len(bends_list)
        if np.isnan(descriptors['T_bends_first']):
            descriptors['T_bends_first'] = 0.0
        
        # Method encoding (for ML)
        method = row.get('method', '')
        descriptors['method_HT_XRD'] = 1 if 'HT XRD' in str(method) else 0
        descriptors['method_dilatometry'] = 1 if 'dilatometry' in str(method) else 0
        descriptors['method_HT_ND'] = 1 if 'HT ND' in str(method) else 0
        
        return descriptors

# ============================================================================
# 4. ПАРСЕРЫ ДЛЯ СПЕЦИАЛЬНЫХ ПОЛЕЙ
# ============================================================================

class DataParser:
    """Parse specialized fields like T(bends), αav, and phase transitions"""
    
    @staticmethod
    def parse_bends_temperatures(value: Any) -> List[float]:
        """Parse T(bends) field - can be semicolon or comma separated"""
        return safe_parse_semicolon_values(value)
    
    @staticmethod
    def parse_alpha_av(value: Any) -> List[float]:
        """Parse αav field - semicolon or comma separated values"""
        return safe_parse_semicolon_values(value)
    
    @staticmethod
    def parse_phase_transition(phase_str: str, temp_str: Any) -> List[Dict]:
        """
        Parse phase transition data from string like:
        "Orthorombic;Monoclinic" and temperatures "400"
        or "Pm-3m, R-3c, Imma" and temperatures "352;476;711"
        
        Returns list of transitions: [{'symmetry': 'Orthorombic', 'space_group': '...', 'T': 400}, ...]
        """
        transitions = []
        
        if pd.isna(phase_str) or phase_str == '-':
            return transitions
        
        # Parse symmetries and space groups
        phase_parts = str(phase_str).split(';')
        
        # Parse temperatures
        temps = DataParser.parse_bends_temperatures(temp_str)
        
        # Determine if phase_str contains space groups (with parentheses or common groups)
        # Simple heuristic: if phase_str contains common space group symbols
        has_space_groups = any(sg in phase_str for sg in ['Pm-3m', 'Imma', 'I2/m', 'R-3c', 'Pbnm', 'Cmcm', 'I4/mcm', 'Fm-3m'])
        
        if has_space_groups and len(temps) == len(phase_parts) - 1:
            # Format: "Symmetry1;Symmetry2" or "SG1;SG2" with temperatures between
            pass
        elif len(temps) == len(phase_parts) - 1:
            # Symmetry names with transition temperatures between them
            for i in range(len(phase_parts) - 1):
                transitions.append({
                    'symmetry_from': phase_parts[i].strip(),
                    'symmetry_to': phase_parts[i+1].strip(),
                    'temperature': temps[i] if i < len(temps) else None
                })
        elif len(temps) == len(phase_parts):
            # Each phase has its own temperature (stability range boundaries)
            for i, phase in enumerate(phase_parts):
                transitions.append({
                    'symmetry': phase.strip(),
                    'temperature': temps[i] if i < len(temps) else None
                })
        else:
            # Just store as is
            transitions.append({
                'phases': phase_parts,
                'temperatures': temps,
                'raw': phase_str
            })
        
        return transitions

# ============================================================================
# 5. ПРОГРЕСС БАР С ETA
# ============================================================================

class ModernProgressBar:
    """Modern progress bar with ETA estimation"""
    
    def __init__(self, total: int, desc: str = "Processing", unit: str = "it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()
        self.current = 0
        
    def update(self, n: int = 1, custom_msg: str = None):
        """Update progress by n steps"""
        self.current += n
        progress = self.current / self.total
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{int(eta//60)}m {int(eta%60)}s" if eta < 3600 else f"{eta/3600:.1f}h"
        else:
            eta_str = "calculating..."
        
        # Update UI
        self.progress_bar.progress(progress)
        
        if custom_msg:
            msg = custom_msg
        else:
            msg = f"{self.desc}: {self.current}/{self.total} {self.unit} | Progress: {progress*100:.1f}% | ETA: {eta_str}"
        
        self.status_text.info(msg)
        
    def finish(self):
        """Mark as complete"""
        self.progress_bar.empty()
        self.status_text.success(f"✅ {self.desc} completed! Time: {time.time() - self.start_time:.1f}s")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИИ (РАСШИРЕННЫЕ)
# ============================================================================

class ScientificVisualizer:
    """Create publication-ready scientific visualizations"""
    
    # Color palette for scientific publications
    COLORS = {
        'primary': '#2A9D8F',
        'secondary': '#E63946',
        'tertiary': '#1D3557',
        'quaternary': '#F4A261',
        'quinary': '#6A4E9B',
        'success': '#4CAF50',
        'warning': '#FFC107',
        'info': '#2196F3',
        'dark': '#2B2D42',
        'light': '#8D99AE',
        'white': '#FFFFFF',
        'black': '#000000'
    }
    
    CMAPS = {
        'thermal': 'viridis',
        'chemical': 'plasma',
        'diverging': 'RdBu_r',
        'sequential': 'YlOrRd'
    }
    
    @staticmethod
    def apply_style():
        """Apply matplotlib style for scientific plots"""
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
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.framealpha': 0.95,
            'legend.edgecolor': '#2c3e50',
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'figure.facecolor': 'white',
            'figure.constrained_layout.use': True,
            'lines.linewidth': 1.2,
            'lines.markersize': 5,
            'lines.markeredgewidth': 0.5,
            'errorbar.capsize': 3,
        })
    
    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, title: str = None, bins: int = 30):
        """Plot histogram with KDE and statistics (ignores NaN)"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Filter out NaNs for distribution
        data = df[column].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No valid data available", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Histogram
        n, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7, 
                                   color=ScientificVisualizer.COLORS['primary'], 
                                   edgecolor='black', linewidth=0.8)
        
        # KDE
        if len(data) > 1:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range) * len(data) * (bins_edges[1]-bins_edges[0]), 
                   'r-', linewidth=2, label='KDE')
        
        # Statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(column, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(title or f'Distribution of {column}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'n = {len(data)}\nStd = {std_val:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        return fig
    
    @staticmethod
    def plot_boxplot_comparison(df: pd.DataFrame, x_col: str, y_col: str, 
                                 title: str = None, palette: str = 'Set2'):
        """Create boxplot for categorical comparison (ignores NaN)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter valid data
        plot_df = df[[x_col, y_col]].dropna()
        
        if len(plot_df) > 0:
            groups = plot_df.groupby(x_col)[y_col].apply(list).to_dict()
            positions = range(len(groups))
            
            bp = ax.boxplot(list(groups.values()), positions=positions, widths=0.7,
                           patch_artist=True, showmeans=True,
                           meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 6})
            
            # Set colors
            colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(list(groups.keys()), rotation=45, ha='right')
            ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
            ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
            ax.set_title(title or f'{y_col} by {x_col}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No valid data for comparison", transform=ax.transAxes, ha='center', va='center')
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, features: List[str], 
                                 target: str = None, top_k: int = 15):
        """
        Create enhanced correlation matrix highlighting top correlations with target
        """
        # Filter features that exist and have valid data (at least 5 non-NaN)
        valid_features = [f for f in features if f in df.columns and df[f].notna().sum() > 5]
        
        if len(valid_features) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough valid features for correlation analysis", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Calculate correlations
        corr_matrix = df[valid_features].corr()
        
        if target and target in corr_matrix.columns:
            # Sort features by correlation with target
            target_corr = corr_matrix[target].abs().sort_values(ascending=False)
            top_features = target_corr.head(top_k).index.tolist()
            if target in top_features:
                top_features.remove(target)
            top_features = [target] + top_features[:top_k-1]
            # Only keep features that exist
            top_features = [f for f in top_features if f in corr_matrix.columns]
            corr_matrix = corr_matrix.loc[top_features, top_features] if top_features else corr_matrix
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Custom diverging colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap=cmap, center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot_kws={'size': 8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        return fig
    
    @staticmethod
    def plot_pairplot_interactive(df: pd.DataFrame, selected_features: List[str], 
                                   color_by: str = None, title: str = "Pairplot"):
        """
        Create enhanced pairplot with user-selected descriptors.
        Features: colored points by target variable, KDE on diagonal, customizable.
        """
        if len(selected_features) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Select at least 2 features", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Filter data: keep rows where all selected features are non-NaN
        plot_df = df[selected_features].dropna()
        
        if len(plot_df) < 3:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Not enough valid data (n={len(plot_df)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Add color column if specified and available
        if color_by and color_by in df.columns:
            color_data = df.loc[plot_df.index, color_by]
            plot_df = plot_df.copy()
            plot_df['_color'] = color_data
        
        # Create pairplot
        if color_by and color_by in df.columns and '_color' in plot_df.columns:
            # Filter out NaN in color column
            plot_df_color = plot_df.dropna(subset=['_color'])
            if len(plot_df_color) > 0:
                g = sns.pairplot(plot_df_color, vars=selected_features, hue='_color',
                                 diag_kind='kde', palette=ScientificVisualizer.CMAPS['thermal'],
                                 plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
                                 diag_kws={'fill': True, 'alpha': 0.7})
                g.fig.suptitle(f'{title}\nColor by: {color_by}', y=1.02, fontsize=14, fontweight='bold')
            else:
                g = sns.pairplot(plot_df, vars=selected_features, diag_kind='kde',
                                 plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5})
                g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        else:
            g = sns.pairplot(plot_df, vars=selected_features, diag_kind='kde',
                             plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5})
            g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        
        return g.fig
    
    @staticmethod
    def plot_regression_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = 'Model'):
        """Plot actual vs predicted with residuals (ignores NaN)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_true_valid) == 0:
            ax1.text(0.5, 0.5, "No valid predictions", transform=ax1.transAxes, ha='center', va='center')
            ax2.text(0.5, 0.5, "No valid predictions", transform=ax2.transAxes, ha='center', va='center')
            return fig
        
        # Actual vs Predicted
        ax1.scatter(y_true_valid, y_pred_valid, alpha=0.6, c=ScientificVisualizer.COLORS['primary'], 
                   edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect prediction')
        
        # Regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_valid, y_pred_valid)
        ax1.plot([min_val, max_val], [slope*min_val+intercept, slope*max_val+intercept], 
                'b-', linewidth=1, alpha=0.7, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
        
        ax1.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
        ax1.set_title(f'{model_name}: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add R² and RMSE
        r2 = r2_score(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Residuals plot
        residuals = y_true_valid - y_pred_valid
        ax2.scatter(y_pred_valid, residuals, alpha=0.6, c=ScientificVisualizer.COLORS['secondary'],
                   edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        
        ax2.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax2.set_title('Residuals Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add residual statistics
        resid_mean = np.mean(residuals)
        resid_std = np.std(residuals)
        textstr2 = f'Mean residual = {resid_mean:.3f}\nStd residual = {resid_std:.3f}'
        ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                                 title: str = 'Feature Importance', top_k: int = 15):
        """Plot feature importance bar chart"""
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_k]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices if i < len(feature_names)]
        
        if len(sorted_features) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No feature importance data available", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importances)))
        bars = ax.barh(range(len(sorted_importances)), sorted_importances[:len(sorted_features)], 
                       color=colors[:len(sorted_features)], edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_importances[:len(sorted_features)])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None,
                         size_col: str = None, title: str = None):
        """Create 2D scatter plot with optional color and size mapping (ignores NaN)"""
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Prepare data - drop NaN in x and y
        plot_df = df[[x_col, y_col]].dropna()
        
        if color_col and color_col in df.columns:
            plot_df = plot_df.join(df[color_col])
        if size_col and size_col in df.columns:
            plot_df = plot_df.join(df[size_col])
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, "No valid data for scatter plot", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        if color_col and color_col in plot_df.columns:
            scatter = ax.scatter(plot_df[x_col], plot_df[y_col], 
                                c=plot_df[color_col],
                                s=plot_df[size_col]*50 if size_col and size_col in plot_df.columns else 50,
                                cmap=ScientificVisualizer.CMAPS['thermal'],
                                alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_col, fontsize=10)
        else:
            ax.scatter(plot_df[x_col], plot_df[y_col], 
                      s=plot_df[size_col]*50 if size_col and size_col in plot_df.columns else 50,
                      c=ScientificVisualizer.COLORS['primary'],
                      alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax.set_title(title or f'{y_col} vs {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_bubble_with_density(df: pd.DataFrame, x_col: str, y_col: str, 
                                  size_col: str = None, color_col: str = None,
                                  title: str = None):
        """
        Create bubble chart with density contour overlay (heatmap-style).
        Features: colorbar on right, size legend, density contours.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data - drop NaN in x and y
        plot_df = df[[x_col, y_col]].dropna()
        
        # Add size column if specified - using direct assignment to avoid column overlap
        if size_col and size_col in df.columns:
            plot_df[size_col] = df.loc[plot_df.index, size_col]
        
        # Add color column if specified - using direct assignment to avoid column overlap
        if color_col and color_col in df.columns:
            plot_df[color_col] = df.loc[plot_df.index, color_col]
        
        if len(plot_df) < 3:
            ax.text(0.5, 0.5, f"Not enough valid data (n={len(plot_df)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Bubble sizes (scale to reasonable range)
        if size_col and size_col in plot_df.columns:
            sizes = plot_df[size_col].values
            # Remove NaN for scaling
            sizes = sizes[~np.isnan(sizes)]
            if len(sizes) > 1 and sizes.max() > sizes.min():
                sizes_scaled = 20 + 480 * (plot_df[size_col].values - sizes.min()) / (sizes.max() - sizes.min())
                # Handle any remaining NaN after scaling
                sizes_scaled = np.nan_to_num(sizes_scaled, nan=100)
            else:
                sizes_scaled = np.full(len(plot_df), 100)
        else:
            sizes_scaled = np.full(len(plot_df), 100)
        
        # Create scatter with bubbles
        if color_col and color_col in plot_df.columns:
            # Filter out NaN in color column for scatter
            color_valid_mask = ~np.isnan(plot_df[color_col].values)
            if color_valid_mask.any():
                scatter = ax.scatter(plot_df[x_col].values[color_valid_mask], 
                                    plot_df[y_col].values[color_valid_mask], 
                                    s=sizes_scaled[color_valid_mask], 
                                    c=plot_df[color_col].values[color_valid_mask],
                                    cmap=ScientificVisualizer.CMAPS['thermal'],
                                    alpha=0.6, edgecolors='black', linewidth=0.8)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_col, fontsize=10, fontweight='bold')
            else:
                # Fallback: no valid color data
                ax.scatter(plot_df[x_col], plot_df[y_col], 
                          s=sizes_scaled, c=ScientificVisualizer.COLORS['primary'],
                          alpha=0.6, edgecolors='black', linewidth=0.8)
        else:
            ax.scatter(plot_df[x_col], plot_df[y_col], 
                      s=sizes_scaled, c=ScientificVisualizer.COLORS['primary'],
                      alpha=0.6, edgecolors='black', linewidth=0.8)
        
        # Add density contours (heatmap-style)
        try:
            from scipy.stats import gaussian_kde
            x_vals = plot_df[x_col].values
            y_vals = plot_df[y_col].values
            
            # Remove NaN for KDE
            kde_valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_vals_clean = x_vals[kde_valid_mask]
            y_vals_clean = y_vals[kde_valid_mask]
            
            if len(x_vals_clean) >= 4:
                # Create grid for contour
                x_grid = np.linspace(x_vals_clean.min(), x_vals_clean.max(), 50)
                y_grid = np.linspace(y_vals_clean.min(), y_vals_clean.max(), 50)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                
                # Calculate KDE
                xy = np.vstack([x_vals_clean, y_vals_clean])
                kde = gaussian_kde(xy)
                Z = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
                
                # Plot contours
                contour = ax.contour(X_grid, Y_grid, Z, levels=5, 
                                    colors='white', linewidths=0.8, alpha=0.6)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        except Exception:
            pass  # Skip density contours if fails
        
        # Add size legend
        if size_col and size_col in plot_df.columns:
            size_vals = plot_df[size_col].dropna().unique()
            if len(size_vals) > 3:
                size_vals = np.linspace(size_vals.min(), size_vals.max(), 4)
            for size_val in size_vals:
                if not np.isnan(size_val):
                    # Calculate scaled size for legend
                    sizes_clean = plot_df[size_col].dropna().values
                    if len(sizes_clean) > 1 and sizes_clean.max() > sizes_clean.min():
                        scaled_size = 20 + 480 * (size_val - sizes_clean.min()) / (sizes_clean.max() - sizes_clean.min())
                    else:
                        scaled_size = 100
                    ax.scatter([], [], s=scaled_size, c='gray', alpha=0.6, 
                              edgecolors='black', linewidth=0.8,
                              label=f'{size_col} = {size_val:.3f}')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax.set_title(title or f'Bubble Chart: {y_col} vs {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_pca_2d(X_pca: np.ndarray, y: np.ndarray, 
                    labels: List[str] = None, title: str = 'PCA Projection'):
        """Create 2D PCA scatter plot with color mapping (ignores NaN)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter out invalid points
        valid_mask = ~np.isnan(y)
        X_valid = X_pca[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) == 0:
            ax.text(0.5, 0.5, "No valid PCA data", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        scatter = ax.scatter(X_valid[:, 0], X_valid[:, 1], c=y_valid, cmap=ScientificVisualizer.CMAPS['thermal'],
                            alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Property', fontsize=10)
        
        ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Annotate points if labels provided and not too many
        if labels and len(labels) < 50:
            labels_valid = [labels[i] for i in range(len(labels)) if valid_mask[i]]
            for i, label in enumerate(labels_valid[:min(50, len(labels_valid))]):
                ax.annotate(label, (X_valid[i, 0], X_valid[i, 1]), fontsize=7, alpha=0.7)
        
        return fig
    
    @staticmethod
    def plot_concentration_heatmap(df: pd.DataFrame, x_col: str, y_col: str, 
                                    z_col: str, bins: int = 20):
        """Create 2D heatmap for concentration dependence (ignores NaN)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create grid
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x = x[mask]
        y = y[mask]
        z = z[mask]
        
        if len(x) < 4:
            ax.text(0.5, 0.5, f"Not enough data points for heatmap (n={len(x)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Create grid
        xi = np.linspace(x.min(), x.max(), bins)
        yi = np.linspace(y.min(), y.max(), bins)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate
        try:
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
            
            # Plot
            im = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap=ScientificVisualizer.CMAPS['thermal'])
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(z_col, fontsize=10)
            
            # Scatter original points
            ax.scatter(x, y, c='black', s=20, alpha=0.5, edgecolors='white', linewidth=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Interpolation error: {str(e)[:50]}", 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax.set_title(f'{z_col} concentration map', fontsize=12, fontweight='bold')
        
        return fig
    
    @staticmethod
    def plot_phase_transition_vs_dopant(df_phase: pd.DataFrame, df_chem: pd.DataFrame,
                                          dopant_col: str, temp_col: str = 'T_transition',
                                          title: str = None):
        """
        Plot phase transition temperature vs dopant concentration.
        Merges phase and chemical data by composition.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Attempt to merge by common composition columns
        merge_cols = ['A', 'B', '[D1]', '[D2]', 'δ']
        available_merge = [c for c in merge_cols if c in df_phase.columns and c in df_chem.columns]
        
        if not available_merge:
            ax.text(0.5, 0.5, "Cannot merge phase and chemical data: no common columns", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            merged = pd.merge(df_phase, df_chem, on=available_merge, how='inner')
        except:
            ax.text(0.5, 0.5, "Merge failed", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        if len(merged) == 0:
            ax.text(0.5, 0.5, "No matching samples between phase and chemical datasets", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Get temperature data
        if temp_col not in merged.columns:
            ax.text(0.5, 0.5, f"Temperature column '{temp_col}' not found", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        plot_df = merged[[dopant_col, temp_col]].dropna()
        
        if len(plot_df) < 3:
            ax.text(0.5, 0.5, f"Not enough valid data (n={len(plot_df)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Scatter plot with trend line
        ax.scatter(plot_df[dopant_col], plot_df[temp_col], 
                  c=ScientificVisualizer.COLORS['primary'], s=60,
                  alpha=0.7, edgecolors='black', linewidth=0.8)
        
        # Add trend line
        if len(plot_df) > 3:
            z = np.polyfit(plot_df[dopant_col], plot_df[temp_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(plot_df[dopant_col].min(), plot_df[dopant_col].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.7, 
                   label=f'Trend: {z[0]:.1f}·x + {z[1]:.1f}')
        
        ax.set_xlabel(dopant_col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Phase Transition Temperature (°C)', fontsize=11, fontweight='bold')
        ax.set_title(title or f'Phase Transition Temperature vs {dopant_col}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_transition_count_vs_dopant(df_phase: pd.DataFrame, df_chem: pd.DataFrame,
                                          dopant_col: str, title: str = None):
        """
        Plot number of phase transitions vs dopant concentration.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Merge data
        merge_cols = ['A', 'B', '[D1]', '[D2]', 'δ']
        available_merge = [c for c in merge_cols if c in df_phase.columns and c in df_chem.columns]
        
        if not available_merge:
            ax.text(0.5, 0.5, "Cannot merge phase and chemical data", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            merged = pd.merge(df_phase, df_chem, on=available_merge, how='inner')
        except:
            ax.text(0.5, 0.5, "Merge failed", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        if len(merged) == 0:
            ax.text(0.5, 0.5, "No matching samples", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Count transitions from Phase transitions (PT) column or parse
        transition_counts = []
        valid_dopants = []
        
        for idx, row in merged.iterrows():
            pt_str = row.get('Phase transitions (PT)', '')
            if pd.isna(pt_str) or pt_str == '-':
                continue
            
            # Count semicolons to estimate number of phases
            n_transitions = pt_str.count(';')
            if n_transitions > 0:
                transition_counts.append(n_transitions)
            else:
                transition_counts.append(1)  # At least one phase
            
            dopant_val = row.get(dopant_col, np.nan)
            if not np.isnan(dopant_val):
                valid_dopants.append(dopant_val)
        
        if len(transition_counts) < 3:
            ax.text(0.5, 0.5, f"Not enough data (n={len(transition_counts)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Create boxplot by dopant bins
        dopant_array = np.array(valid_dopants)
        counts_array = np.array(transition_counts)
        
        # Bin dopant concentrations
        bins = np.linspace(dopant_array.min(), dopant_array.max(), 5)
        bin_indices = np.digitize(dopant_array, bins)
        
        box_data = []
        bin_labels = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.any():
                box_data.append(counts_array[mask])
                bin_labels.append(f'{bins[i-1]:.2f}-{bins[i]:.2f}')
        
        if box_data:
            bp = ax.boxplot(box_data, labels=bin_labels, patch_artist=True,
                           showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'red'})
            
            for patch in bp['boxes']:
                patch.set_facecolor(ScientificVisualizer.COLORS['primary'])
                patch.set_alpha(0.7)
        
        ax.set_xlabel(dopant_col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Phase Transitions', fontsize=11, fontweight='bold')
        ax.set_title(title or f'Phase Transition Count vs {dopant_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig

# ============================================================================
# 7. ML МОДЕЛИ
# ============================================================================

class MLModelManager:
    """Manage machine learning models for property prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame, feature_cols: List[str], 
                          target_col: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare feature matrix and optional target vector (ignores NaN)"""
        # Filter valid rows - exclude rows with NaN in any feature or target
        if target_col:
            valid_df = df[feature_cols + [target_col]].dropna()
            if len(valid_df) == 0:
                return np.array([]), np.array([])
            X = valid_df[feature_cols].values
            y = valid_df[target_col].values
        else:
            X = df[feature_cols].values
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y if target_col else X_scaled
    
    def train_regression_ensemble(self, X: np.ndarray, y: np.ndarray, 
                                   test_size: float = 0.2) -> Dict:
        """Train ensemble of regression models (RF, GBM, XGB)"""
        if len(X) < 10:
            return {'error': 'Insufficient data for training (need at least 10 samples)'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if len(X_train) < 5 or len(X_test) < 2:
            return {'error': 'Train/test split resulted in too few samples'}
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        }
        
        results = {}
        predictions = {}
        
        # Train each model
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[name] = {
                    'model': model,
                    'r2': r2_score(y_test, y_pred) if len(y_test) > 1 else 0,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
                predictions[name] = y_pred
            except Exception as e:
                results[name] = {'error': str(e)}
                predictions[name] = np.zeros_like(y_test)
            
        # Ensemble prediction (weighted average) - only use models that succeeded
        weights = {'Random Forest': 0.4, 'Gradient Boosting': 0.3, 'XGBoost': 0.3}
        ensemble_pred = np.zeros_like(y_test)
        weight_sum = 0
        for name, weight in weights.items():
            if name in predictions and len(predictions[name]) == len(y_test):
                ensemble_pred += weight * predictions[name]
                weight_sum += weight
        
        if weight_sum > 0:
            ensemble_pred = ensemble_pred / weight_sum
            ensemble_r2 = r2_score(y_test, ensemble_pred) if len(y_test) > 1 else 0
        else:
            ensemble_r2 = 0
        
        return {
            'models': results,
            'ensemble_r2': ensemble_r2,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'ensemble_pred': ensemble_pred,
            'predictions': predictions
        }
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray,
                         test_size: float = 0.2) -> Dict:
        """Train classifier for method prediction"""
        if len(X) < 10:
            return {'error': 'Insufficient data for training (need at least 10 samples)'}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if len(X_train) < 5 or len(X_test) < 2:
            return {'error': 'Train/test split resulted in too few samples'}
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            return {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            importances = np.zeros(len(feature_names))
        
        # Ensure lengths match
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

# ============================================================================
# 8. ОСНОВНОЕ ПРИЛОЖЕНИЕ STREAMLIT
# ============================================================================

def parse_text_data(text_data: str) -> Optional[pd.DataFrame]:
    """Parse text data (CSV or TSV) into DataFrame"""
    if not text_data.strip():
        return None
    
    try:
        # Detect separator
        first_line = text_data.strip().split('\n')[0]
        if '\t' in first_line:
            sep = '\t'
        elif ',' in first_line:
            sep = ','
        else:
            # Try space-separated
            sep = None
        
        if sep:
            df = pd.read_csv(StringIO(text_data), sep=sep)
        else:
            df = pd.read_csv(StringIO(text_data), sep=r'\s+', engine='python')
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None

def main():
    """Main application entry point"""
    
    # Apply styling
    apply_scientific_css()
    ScientificVisualizer.apply_style()
    
    # Header
    st.title("🔥 Perovskite Expansion Analyzer")
    st.markdown("""
    <div class="info-box">
    <b>📊 Comprehensive Analysis of Thermal and Chemical Expansion</b><br>
    Proton-conducting perovskite oxides: BaZrO₃, BaCeO₃, BaSnO₃ and derivatives.<br>
    Analyze composition → structure → property relationships using 35+ descriptors.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chem_df' not in st.session_state:
        st.session_state.chem_df = None
    if 'phase_df' not in st.session_state:
        st.session_state.phase_df = None
    if 'chem_with_descriptors' not in st.session_state:
        st.session_state.chem_with_descriptors = None
    if 'descriptor_names' not in st.session_state:
        st.session_state.descriptor_names = []
    if 'chem_data_text' not in st.session_state:
        st.session_state.chem_data_text = ""
    if 'phase_data_text' not in st.session_state:
        st.session_state.phase_data_text = ""
    
    # Sidebar for data input
    with st.sidebar:
        st.header("📁 Data Input")
        
        # Dataset 1: Chemical & Thermal Expansion
        with st.expander("📊 Dataset 1: Chemical & Thermal Expansion", expanded=True):
            st.markdown("""
            <div class="info-box" style="font-size: 0.8rem; padding: 0.5rem;">
            <b>Format:</b> CSV (comma) or TSV (tab). First row must be headers.<br>
            <b>Note:</b> Missing values marked as '-' are automatically handled.
            </div>
            """, unsafe_allow_html=True)
            
            chem_text_input = st.text_area(
                "Paste Chemical Data (CSV/TSV):",
                height=250,
                key="chem_text_area",
                placeholder="""№,A,A',B,B',D1,D2,[A'],[B'],[D1],[D2],δ,method,β,∆T,°C,α·106 (K-1),T(bends),°C,αav·106 (K-1),pH2O,Ref
1,Ba,-,Ce,Zr,Y,Yb,0,0.1,0.1,0.1,0.1,dilatometry,0.0073,27-1000,10.6,400;600,10.6;4.73;10.1,0.0001,10.15826/chimtech.2024.11.4.22
2,Ba,-,Ce,Zr,Y,Yb,0,0.1,0.1,0.1,0.1,HT XRD,0.0317,27-1000,10.6,300,10.7;8.7,0.02,10.15826/chimtech.2024.11.4.22"""
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Load Chemical Data", key="load_chem_btn", use_container_width=True):
                    if chem_text_input.strip():
                        chem_df = parse_text_data(chem_text_input)
                        if chem_df is not None:
                            st.session_state.chem_df = chem_df
                            st.session_state.chem_data_text = chem_text_input
                            st.success(f"✅ Loaded {len(chem_df)} samples with {len(chem_df.columns)} columns")
                        else:
                            st.error("Failed to parse chemical data")
                    else:
                        st.warning("Please paste chemical data first")
            
            with col2:
                if st.button("📋 Load Example Chem Data", key="load_example_chem_btn", use_container_width=True):
                    example_chem = """№,A,A',B,B',D1,D2,[A'],[B'],[D1],[D2],δ,method,β,∆T,°C,α·106 (K-1),T(bends),°C,αav·106 (K-1),pH2O,Ref
1,Ba,-,Ce,Zr,Y,Yb,0,0.1,0.1,0.1,0.1,dilatometry,0.0073,27-1000,10.6,400;600,10.6;4.73;10.1,0.0001,10.15826/chimtech.2024.11.4.22
2,Ba,-,Ce,Zr,Y,Yb,0,0.1,0.1,0.1,0.1,HT XRD,0.0317,27-1000,10.6,300,10.7;8.7,0.02,10.15826/chimtech.2024.11.4.22
3,Ba,-,Ce,Zr,Y,-,0,0.1,0.1,0,0.05,HT ND,-,20-900,11.2,-,-,0.00106,10.1021/acs.jpcc.1c08334
4,Ba,-,Ce,Zr,Y,-,0,0,0.1,0,0.05,dilatometry,0.019,430-630,-,450,-,0.00106,10.1021/acs.jpcc.1c08334"""
                    st.session_state.chem_text_area = example_chem
                    st.rerun()
        
        # Dataset 2: Phase Transitions
        with st.expander("🔬 Dataset 2: Phase Transitions", expanded=False):
            st.markdown("""
            <div class="info-box" style="font-size: 0.8rem; padding: 0.5rem;">
            <b>Format:</b> CSV (comma) or TSV (tab). First row must be headers.<br>
            <b>Note:</b> Missing values marked as '-' are automatically handled.
            </div>
            """, unsafe_allow_html=True)
            
            phase_text_input = st.text_area(
                "Paste Phase Transition Data (CSV/TSV):",
                height=200,
                key="phase_text_area",
                placeholder="""№,A,A',B,B',D1,D2,[A],[B'],[D1],[D2],δ,pH2O,∆T,°C,Symmetry,Phase transitions (PT),T (PT),°C,Ref
1,Ba,-,Ce,Zr,Y,-,0,0.36,0.1,0,0.05,-,30-1000,-,-,-,10.1063/1.5066970
2,Ba,-,Zr,-,Y,-,0,0,0,0,0,-,25,Cubic,Pm-3m,-,10.1088/1742-6596/1967/1/012015"""
            )
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("📊 Load Phase Data", key="load_phase_btn", use_container_width=True):
                    if phase_text_input.strip():
                        phase_df = parse_text_data(phase_text_input)
                        if phase_df is not None:
                            st.session_state.phase_df = phase_df
                            st.session_state.phase_data_text = phase_text_input
                            st.success(f"✅ Loaded {len(phase_df)} phase transition records")
                        else:
                            st.error("Failed to parse phase transition data")
                    else:
                        st.warning("Please paste phase transition data first")
            
            with col4:
                if st.button("📋 Load Example Phase Data", key="load_example_phase_btn", use_container_width=True):
                    example_phase = """№,A,A',B,B',D1,D2,[A],[B'],[D1],[D2],δ,pH2O,∆T,°C,Symmetry,Phase transitions (PT),T (PT),°C,Ref
1,Ba,-,Zr,-,Y,-,0,0,0,0,0,-,25,Cubic,Pm-3m,-,10.1088/1742-6596/1967/1/012015
2,Ba,-,Zr,-,Y,-,0,0,0.055,0,0.0275,-,25,Cubic,Pm-3m,-,10.1088/1742-6596/1967/1/012015
3,Ba,-,Sn,-,Y,-,0,0,0,0,0,-,-,Orthorombic;Rhombohedral;Cubic,Pm-3m, R-3c, Imma,352;476;711,10.1111/jace.12990"""
                    st.session_state.phase_text_area = example_phase
                    st.rerun()
        
        st.markdown("---")
        
        # Filters section (will be populated after data load)
        st.header("🔍 Filters")
        
        if st.session_state.chem_with_descriptors is not None:
            df = st.session_state.chem_with_descriptors
            
            # A-site filter
            if 'A' in df.columns:
                a_sites = df['A'].dropna().unique().tolist()
                a_sites = [a for a in a_sites if a not in [None, '-', '']]
                if a_sites:
                    selected_a = st.multiselect("A-site elements", a_sites, default=a_sites[:3] if len(a_sites) > 3 else a_sites)
                else:
                    selected_a = []
            else:
                selected_a = []
            
            # B-site filter
            if 'B' in df.columns:
                b_sites = df['B'].dropna().unique().tolist()
                b_sites = [b for b in b_sites if b not in [None, '-', '']]
                if b_sites:
                    selected_b = st.multiselect("B-site elements", b_sites, default=b_sites[:3] if len(b_sites) > 3 else b_sites)
                else:
                    selected_b = []
            else:
                selected_b = []
            
            # Method filter
            if 'method' in df.columns:
                methods = df['method'].dropna().unique().tolist()
                methods = [m for m in methods if m not in [None, '-', '']]
                if methods:
                    selected_methods = st.multiselect("Measurement method", methods, default=methods)
                else:
                    selected_methods = []
            else:
                selected_methods = []
            
            # pH2O range
            if 'log_pH2O' in df.columns:
                pH2O_vals = df['log_pH2O'].dropna()
                if len(pH2O_vals) > 0:
                    pH2O_min = float(pH2O_vals.min())
                    pH2O_max = float(pH2O_vals.max())
                    if pH2O_min < pH2O_max:
                        pH2O_range = st.slider("log(pH₂O)", pH2O_min, pH2O_max, (pH2O_min, pH2O_max))
                    else:
                        pH2O_range = (pH2O_min, pH2O_max)
                else:
                    pH2O_range = (-10, 0)
            else:
                pH2O_range = (-10, 0)
            
            # Alpha range
            alpha_col = 'alpha_true'
            if alpha_col in df.columns:
                alpha_vals = df[alpha_col].dropna()
                if len(alpha_vals) > 0:
                    alpha_min = float(alpha_vals.min())
                    alpha_max = float(alpha_vals.max())
                    if alpha_min < alpha_max:
                        alpha_range = st.slider("α (×10⁶ K⁻¹)", alpha_min, alpha_max, (alpha_min, alpha_max))
                    else:
                        alpha_range = (alpha_min, alpha_max)
                else:
                    alpha_range = (0, 20)
            else:
                alpha_range = (0, 20)
            
            # Apply filters button
            if st.button("Apply Filters", use_container_width=True):
                filtered_df = df.copy()
                if selected_a:
                    filtered_df = filtered_df[filtered_df['A'].isin(selected_a)]
                if selected_b:
                    filtered_df = filtered_df[filtered_df['B'].isin(selected_b)]
                if selected_methods:
                    filtered_df = filtered_df[filtered_df['method'].isin(selected_methods)]
                if 'log_pH2O' in df.columns:
                    filtered_df = filtered_df[
                        (filtered_df['log_pH2O'] >= pH2O_range[0]) & 
                        (filtered_df['log_pH2O'] <= pH2O_range[1])
                    ]
                if alpha_col in df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[alpha_col] >= alpha_range[0]) & 
                        (filtered_df[alpha_col] <= alpha_range[1])
                    ]
                st.session_state.filtered_df = filtered_df
                st.success(f"Filtered to {len(filtered_df)} samples")
    
    # Process chemical data if loaded
    if st.session_state.chem_df is not None:
        chem_df = st.session_state.chem_df
        
        with st.spinner("Processing chemical data and calculating descriptors..."):
            pb = ModernProgressBar(len(chem_df), "Calculating descriptors")
            
            # Initialize descriptor calculator
            calculator = PerovskiteDescriptorCalculator()
            
            # Calculate descriptors for each row
            descriptors_list = []
            for idx, row in chem_df.iterrows():
                try:
                    desc = calculator.calculate_all_descriptors(row)
                    descriptors_list.append(desc)
                except Exception as e:
                    st.warning(f"Error processing row {idx}: {e}")
                    # Append empty descriptors
                    empty_desc = {k: np.nan for k in PerovskiteDescriptorCalculator().calculate_all_descriptors(chem_df.iloc[0]).keys()}
                    descriptors_list.append(empty_desc)
                
                if idx % 5 == 0:
                    pb.update(1, f"Processing row {idx+1}/{len(chem_df)}")
            
            pb.update(len(chem_df) - pb.current, "Finalizing...")
            
            # Combine with original data
            desc_df = pd.DataFrame(descriptors_list) if descriptors_list else pd.DataFrame()
            if len(desc_df) > 0:
                chem_df_full = pd.concat([chem_df.reset_index(drop=True), desc_df], axis=1)
            else:
                chem_df_full = chem_df.copy()
            
            # Store in session state
            st.session_state.chem_with_descriptors = chem_df_full
            st.session_state.descriptor_names = list(desc_df.columns) if len(desc_df) > 0 else []
            st.session_state.filtered_df = chem_df_full
            
            pb.finish()
            
            st.success(f"✅ Processed {len(chem_df)} samples with {len(desc_df.columns)} descriptors")
    
    # Main content with tabs
    if st.session_state.chem_with_descriptors is not None:
        df = st.session_state.filtered_df if 'filtered_df' in st.session_state else st.session_state.chem_with_descriptors
        
        # Create tabs (removed 3D Scatter tab)
        tabs = st.tabs([
            "📊 Data Overview",
            "📈 EDA & Distributions",
            "🔥 Correlation Analysis",
            "🗺️ Concentration Maps",
            "💨 Bubble Charts",
            "🧠 Machine Learning",
            "🔬 Phase Transitions",
            "🎯 Clustering & PCA",
            "📉 Advanced ML (SHAP)",
            "📤 Export"
        ])
        
        # Tab 1: Data Overview
        with tabs[0]:
            st.header("📊 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Descriptors", len(st.session_state.descriptor_names))
            with col3:
                alpha_col = 'alpha_true'
                if alpha_col in df.columns:
                    alpha_vals = df[alpha_col].dropna()
                    mean_alpha = alpha_vals.mean() if len(alpha_vals) > 0 else 0
                    st.metric("Mean α (×10⁶ K⁻¹)", f"{mean_alpha:.2f}")
            with col4:
                beta_col = 'beta'
                if beta_col in df.columns:
                    beta_vals = df[beta_col].dropna()
                    mean_beta = beta_vals.mean() if len(beta_vals) > 0 else 0
                    st.metric("Mean β", f"{mean_beta:.4f}")
            
            st.markdown("---")
            
            st.subheader("Sample Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Descriptor Summary")
            if len(st.session_state.descriptor_names) > 0:
                desc_stats = df[st.session_state.descriptor_names].describe()
                st.dataframe(desc_stats, use_container_width=True)
            else:
                st.info("No descriptors calculated yet")
        
        # Tab 2: EDA & Distributions
        with tabs[1]:
            st.header("📈 Exploratory Data Analysis")
            
            # Select property for distribution
            target_options = ['alpha_true', 'alpha_apparent', 'beta', 'delta', 'tolerance_factor', 'chiB_avg']
            available_targets = [t for t in target_options if t in df.columns]
            
            if available_targets:
                selected_target = st.selectbox("Select property to analyze", available_targets)
                
                # Display nice name for the selected target
                target_display = {
                    'alpha_true': 'α·10⁶ (K⁻¹) - True Thermal Expansion',
                    'alpha_apparent': 'αav·10⁶ (K⁻¹) - Apparent Thermal Expansion',
                    'beta': 'β - Chemical Expansion',
                    'delta': 'δ - Oxygen Non-stoichiometry',
                    'tolerance_factor': 't - Tolerance Factor',
                    'chiB_avg': 'χB_avg - Average B-site Electronegativity'
                }.get(selected_target, selected_target)
                
                # Distribution plot
                fig = ScientificVisualizer.plot_distribution(df, selected_target, 
                                                              title=f'Distribution of {target_display}')
                st.pyplot(fig)
                
                # Boxplot by method
                if 'method' in df.columns and selected_target in df.columns:
                    fig2 = ScientificVisualizer.plot_boxplot_comparison(df, 'method', selected_target,
                                                                         title=f'{target_display} by measurement method')
                    st.pyplot(fig2)
                
                # Violin plot by A-site
                if 'A' in df.columns and len(df['A'].unique()) > 1 and selected_target in df.columns:
                    st.subheader("Distribution by A-site Element")
                    fig3, ax = plt.subplots(figsize=(10, 6))
                    valid_a_sites = [a for a in df['A'].unique() if a not in [None, '-', '']]
                    data_to_plot = []
                    positions = []
                    for i, a in enumerate(valid_a_sites):
                        vals = df[df['A'] == a][selected_target].dropna().values
                        if len(vals) > 0:
                            data_to_plot.append(vals)
                            positions.append(i)
                    
                    if data_to_plot:
                        violin_parts = ax.violinplot(data_to_plot, positions=positions, 
                                                    showmeans=True, showmedians=True)
                        ax.set_xticks(positions)
                        ax.set_xticklabels(valid_a_sites[:len(positions)])
                        ax.set_ylabel(target_display)
                        ax.set_title(f'{target_display} by A-site')
                        st.pyplot(fig3)
            else:
                st.warning("No target properties available for analysis")
        
        # Tab 3: Correlation Analysis
        with tabs[2]:
            st.header("🔥 Correlation Analysis")
            
            # Select features for correlation
            default_features = ['rB_avg', 'tolerance_factor', 'chiB_avg', 'delta_chi_AB',
                               'variance_rB', 'S_config_B', 'VB_avg', 'Vo_proxy',
                               'conc_B_prime', 'conc_D1', 'conc_D2', 'total_dopant_B',
                               'alpha_true', 'beta']
            available_features = [f for f in default_features if f in df.columns]
            
            if len(available_features) > 2:
                fig = ScientificVisualizer.plot_correlation_matrix(df, available_features, 
                                                                    target='alpha_true' if 'alpha_true' in df.columns else None)
                st.pyplot(fig)
            else:
                st.warning("Not enough features for correlation analysis")
            
            # Pairplot with user-selectable descriptors (NEW ENHANCED FEATURE)
            st.subheader("Interactive Pairplot of Selected Descriptors")
            
            # Extended list of 20+ descriptors for pairplot
            pairplot_descriptor_options = [
                'tolerance_factor', 'rA_avg', 'rB_avg', 'rA_rB_ratio',
                'chiA_avg', 'chiB_avg', 'delta_chi_AB', 'chi_ratio_AB',
                'conc_B_prime', 'conc_D1', 'conc_D2', 'total_dopant_B',
                'Vo_proxy', 'S_config_B', 'variance_rB', 'VB_avg',
                'ionicity_BO', 'delta_chi_div_t', 'octahedral_factor',
                'alpha_true', 'beta', 'delta'
            ]
            available_pairplot = [d for d in pairplot_descriptor_options if d in df.columns]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_pairplot_features = st.multiselect(
                    "Select descriptors for pairplot (2-6 recommended)",
                    available_pairplot,
                    default=available_pairplot[:4] if len(available_pairplot) >= 4 else available_pairplot,
                    key="pairplot_features"
                )
            with col2:
                color_options = [None] + [c for c in ['alpha_true', 'beta', 'method', 'A', 'B'] if c in df.columns]
                pairplot_color = st.selectbox("Color points by", color_options, index=0, key="pairplot_color")
            
            if len(selected_pairplot_features) >= 2:
                fig = ScientificVisualizer.plot_pairplot_interactive(
                    df, selected_pairplot_features, 
                    color_by=pairplot_color if pairplot_color else None,
                    title="Composition-Property Relationships"
                )
                st.pyplot(fig)
            else:
                st.info("Select at least 2 descriptors for pairplot")
        
        # Tab 4: Concentration Maps (EXPANDED: 18 descriptors including conc_B_prime)
        with tabs[3]:
            st.header("🗺️ Concentration Maps")
            
            # EXPANDED list of 18+ descriptors for 2D maps
            map_x_options = [
                'conc_D1', 'conc_D2', 'total_dopant_B', 'conc_B_prime',
                'tolerance_factor', 'chiB_avg', 'rB_avg', 'delta_chi_AB',
                'chi_ratio_AB', 'Vo_proxy', 'S_config_B', 'variance_rB',
                'tolerance_deviation', 'octahedral_factor', 'VB_avg'
            ]
            map_x_available = [c for c in map_x_options if c in df.columns]
            
            map_y_options = [
                'conc_D2', 'conc_D1', 'total_dopant_B', 'conc_B_prime',
                'variance_rB', 'S_config_B', 'VB_avg', 'Vo_proxy',
                'delta_chi_AB', 'chi_ratio_AB', 'tolerance_deviation'
            ]
            map_y_available = [c for c in map_y_options if c in df.columns]
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis (concentration/descriptor)", 
                                      map_x_available, key="conc_x")
            with col2:
                y_axis = st.selectbox("Y-axis (concentration/descriptor)",
                                      map_y_available, key="conc_y")
            
            z_options = ['alpha_true', 'alpha_apparent', 'beta', 'tolerance_factor', 'delta']
            z_axis = st.selectbox("Color by (property)", 
                                  [c for c in z_options if c in df.columns])
            
            # 2D Scatter
            st.subheader("2D Concentration Map")
            fig = ScientificVisualizer.plot_scatter_2d(df, x_axis, y_axis, color_col=z_axis)
            st.pyplot(fig)
            
            # Heatmap
            st.subheader("2D Heatmap")
            fig2 = ScientificVisualizer.plot_concentration_heatmap(df, x_axis, y_axis, z_axis)
            st.pyplot(fig2)
        
        # Tab 5: Bubble Charts (with density contours, heatmap-style)
        with tabs[4]:
            st.header("💨 Bubble Charts with Density Contours")
            
            st.markdown("""
            <div class="info-box">
            <b>Interactive bubble charts with heatmap-style density contours:</b><br>
            - Bubble size represents a third variable<br>
            - Color shows a fourth variable (with colorbar on right)<br>
            - Density contours overlay the plot (like a 2D heatmap)
            </div>
            """, unsafe_allow_html=True)
            
            # Extended list of descriptors for bubble chart
            bubble_x_options = [
                'tolerance_factor', 'chiB_avg', 'rB_avg', 'delta_chi_AB',
                'conc_D1', 'conc_D2', 'total_dopant_B', 'conc_B_prime',
                'Vo_proxy', 'S_config_B', 'variance_rB'
            ]
            bubble_x_available = [c for c in bubble_x_options if c in df.columns]
            
            bubble_y_options = ['alpha_true', 'alpha_apparent', 'beta']
            bubble_y_available = [c for c in bubble_y_options if c in df.columns]
            
            bubble_size_options = ['delta', 'total_dopant_B', 'S_config_B', 'Vo_proxy', 'beta']
            bubble_size_available = [c for c in bubble_size_options if c in df.columns]
            
            bubble_color_options = ['alpha_true', 'beta', 'tolerance_factor', 'delta', 'chiB_avg']
            bubble_color_available = [c for c in bubble_color_options if c in df.columns]
            
            col1, col2 = st.columns(2)
            with col1:
                x_bubble = st.selectbox("X-axis", bubble_x_available, key="bubble_x")
                size_bubble = st.selectbox("Bubble size", bubble_size_available, key="bubble_size")
            with col2:
                y_bubble = st.selectbox("Y-axis (target property)", bubble_y_available, key="bubble_y")
                color_bubble = st.selectbox("Color (colorbar on right)", bubble_color_available, key="bubble_color")
            
            if len(df) > 0:
                # Filter out NaN in y
                plot_df = df.dropna(subset=[y_bubble])
                if len(plot_df) > 0:
                    fig = ScientificVisualizer.plot_bubble_with_density(
                        plot_df, x_bubble, y_bubble, 
                        size_col=size_bubble, color_col=color_bubble,
                        title=f'{y_bubble} vs {x_bubble} (bubble size: {size_bubble})'
                    )
                    st.pyplot(fig)
                else:
                    st.info(f"No valid data for {y_bubble}")
        
        # Tab 6: Machine Learning
        with tabs[5]:
            st.header("🧠 Machine Learning Models")
            
            # Select target for prediction
            ml_target_options = ['alpha_true', 'alpha_apparent', 'beta']
            ml_target = st.selectbox("Predict target property",
                                     [c for c in ml_target_options if c in df.columns],
                                     key="ml_target")
            
            # Select features (expanded list)
            default_ml_features = ['tolerance_factor', 'chiB_avg', 'delta_chi_AB', 'rB_avg',
                                   'variance_rB', 'S_config_B', 'VB_avg', 'Vo_proxy',
                                   'conc_B_prime', 'conc_D1', 'conc_D2', 'total_dopant_B',
                                   'log_pH2O', 'T_span']
            available_ml_features = [f for f in default_ml_features if f in df.columns]
            
            selected_features = st.multiselect("Select features for ML", 
                                               available_ml_features,
                                               default=available_ml_features[:5] if len(available_ml_features) > 5 else available_ml_features)
            
            if len(selected_features) > 0 and len(df) > 10:
                if st.button("Train Model", type="primary", use_container_width=True):
                    with st.spinner("Training ensemble model..."):
                        # Prepare data
                        ml_manager = MLModelManager()
                        X_scaled, y = ml_manager.prepare_features(df, selected_features, ml_target)
                        
                        if len(X_scaled) > 0 and len(np.unique(y)) > 1 and len(X_scaled) >= 10:
                            # Train model
                            results = ml_manager.train_regression_ensemble(X_scaled, y, test_size=0.2)
                            
                            if 'error' in results:
                                st.error(results['error'])
                            else:
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Ensemble R²", f"{results['ensemble_r2']:.3f}")
                                with col2:
                                    if 'Random Forest' in results['models'] and 'r2' in results['models']['Random Forest']:
                                        st.metric("Random Forest R²", f"{results['models']['Random Forest']['r2']:.3f}")
                                with col3:
                                    if 'XGBoost' in results['models'] and 'r2' in results['models']['XGBoost']:
                                        st.metric("XGBoost R²", f"{results['models']['XGBoost']['r2']:.3f}")
                                
                                # Actual vs Predicted plot
                                fig = ScientificVisualizer.plot_regression_analysis(
                                    results['y_test'], results['ensemble_pred'], 'Ensemble Model'
                                )
                                st.pyplot(fig)
                                
                                # Feature importance
                                if 'Random Forest' in results['models'] and 'model' in results['models']['Random Forest']:
                                    rf_model = results['models']['Random Forest']['model']
                                    importance_df = ml_manager.get_feature_importance(rf_model, selected_features)
                                    
                                    st.subheader("Feature Importance")
                                    fig2 = ScientificVisualizer.plot_feature_importance(
                                        importance_df['importance'].values,
                                        importance_df['feature'].tolist(),
                                        title='Random Forest Feature Importance'
                                    )
                                    st.pyplot(fig2)
                                    
                                    # Store model results
                                    st.session_state.ml_results = results
                                    st.session_state.ml_features = selected_features
                        else:
                            if len(X_scaled) < 10:
                                st.warning("Not enough valid data for training (need at least 10 samples)")
                            else:
                                st.warning("Insufficient variation in target variable")
            else:
                st.info("Select at least one feature and ensure sufficient data")
        
        # Tab 7: Phase Transitions (ENHANCED with 3 new plots)
        with tabs[6]:
            st.header("🔬 Phase Transition Analysis")
            
            if st.session_state.phase_df is not None:
                phase_df = st.session_state.phase_df
                st.subheader("Phase Transition Data")
                st.dataframe(phase_df, use_container_width=True)
                
                # Analyze phase transitions
                st.subheader("Transition Temperatures Distribution")
                phase_temps = []
                for _, row in phase_df.iterrows():
                    temps = DataParser.parse_bends_temperatures(row.get('T (PT), °C', ''))
                    phase_temps.extend(temps)
                
                if phase_temps:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(phase_temps, bins=20, color=ScientificVisualizer.COLORS['primary'],
                           edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Transition Temperature (°C)', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    ax.set_title('Distribution of Phase Transition Temperatures', fontsize=12, fontweight='bold')
                    st.pyplot(fig)
                else:
                    st.info("No valid transition temperatures found")
                
                # Symmetry distribution
                st.subheader("Symmetry Types")
                symmetries = []
                for _, row in phase_df.iterrows():
                    sym = row.get('Symmetry', '')
                    if sym and sym != '-':
                        symmetries.extend(sym.split(';'))
                
                if symmetries:
                    sym_counts = pd.Series(symmetries).value_counts()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sym_counts.plot(kind='bar', ax=ax, color=ScientificVisualizer.COLORS['secondary'])
                    ax.set_xlabel('Symmetry', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                    ax.set_title('Phase Symmetry Distribution', fontsize=12, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.info("No symmetry information found")
                
                # NEW PLOT 1: Phase transition temperature vs dopant concentration
                if st.session_state.chem_with_descriptors is not None:
                    st.subheader("Phase Transition Temperature vs Dopant Concentration")
                    
                    chem_df_full = st.session_state.chem_with_descriptors
                    
                    dopant_options = ['total_dopant_B', 'conc_B_prime', 'conc_D1', 'conc_D2']
                    available_dopants = [d for d in dopant_options if d in chem_df_full.columns]
                    
                    if available_dopants:
                        selected_dopant = st.selectbox("Select dopant variable", available_dopants, key="pt_dopant")
                        
                        # Parse transition temperatures from phase_df
                        phase_data_list = []
                        for idx, row in phase_df.iterrows():
                            temps = DataParser.parse_bends_temperatures(row.get('T (PT), °C', ''))
                            if temps:
                                # Use first transition temperature
                                phase_data_list.append({
                                    'T_transition': temps[0],
                                    **{col: row.get(col, np.nan) for col in ['A', 'B', '[D1]', '[D2]', 'δ'] if col in phase_df.columns}
                                })
                        
                        if phase_data_list:
                            phase_temp_df = pd.DataFrame(phase_data_list)
                            
                            fig = ScientificVisualizer.plot_phase_transition_vs_dopant(
                                phase_temp_df, chem_df_full, selected_dopant, 'T_transition',
                                title=f'Phase Transition Temperature vs {selected_dopant}'
                            )
                            st.pyplot(fig)
                        else:
                            st.info("No transition temperatures found in phase data")
                    else:
                        st.info("Dopant concentration variables not available")
                    
                    # NEW PLOT 2: Number of phase transitions vs dopant concentration
                    st.subheader("Number of Phase Transitions vs Dopant Concentration")
                    
                    if available_dopants:
                        selected_dopant2 = st.selectbox("Select dopant variable for count plot", available_dopants, key="pt_count_dopant")
                        
                        fig2 = ScientificVisualizer.plot_transition_count_vs_dopant(
                            phase_df, chem_df_full, selected_dopant2,
                            title=f'Phase Transition Count vs {selected_dopant2}'
                        )
                        st.pyplot(fig2)
                    
                    # NEW PLOT 3: Symmetry type vs composition (scatter with colors)
                    st.subheader("Crystal Symmetry vs Composition")
                    
                    # Merge phase and chemical data
                    merge_cols = ['A', 'B', '[D1]', '[D2]', 'δ']
                    available_merge = [c for c in merge_cols if c in phase_df.columns and c in chem_df_full.columns]
                    
                    if available_merge:
                        try:
                            merged = pd.merge(phase_df, chem_df_full, on=available_merge, how='inner')
                            if len(merged) > 0:
                                # Get symmetry column
                                sym_col = 'Symmetry'
                                if sym_col in merged.columns:
                                    plot_df = merged[[sym_col, 'total_dopant_B']].dropna()
                                    plot_df = plot_df[plot_df[sym_col] != '-']
                                    
                                    if len(plot_df) > 0:
                                        fig3, ax = plt.subplots(figsize=(12, 6))
                                        
                                        # Create grouped boxplot or scatter
                                        unique_sym = plot_df[sym_col].unique()
                                        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sym)))
                                        
                                        for i, sym in enumerate(unique_sym):
                                            subset = plot_df[plot_df[sym_col] == sym]['total_dopant_B'].dropna()
                                            if len(subset) > 0:
                                                # Jittered scatter
                                                x_jitter = np.random.normal(i, 0.05, len(subset))
                                                ax.scatter(x_jitter, subset, alpha=0.6, s=40,
                                                          c=[colors[i]], edgecolors='black', linewidth=0.5,
                                                          label=sym)
                                        
                                        ax.set_xticks(range(len(unique_sym)))
                                        ax.set_xticklabels(unique_sym, rotation=45, ha='right')
                                        ax.set_ylabel('Total Dopant Concentration (D1+D2)', fontsize=11, fontweight='bold')
                                        ax.set_title('Symmetry Distribution vs Dopant Concentration', fontsize=12, fontweight='bold')
                                        ax.legend(loc='upper right', fontsize=8)
                                        ax.grid(True, alpha=0.3, axis='y')
                                        st.pyplot(fig3)
                                    else:
                                        st.info("No valid symmetry data for plotting")
                                else:
                                    st.info("Symmetry column not found")
                            else:
                                st.info("No matching samples between phase and chemical data")
                        except Exception as e:
                            st.warning(f"Could not create symmetry plot: {str(e)[:100]}")
            else:
                st.info("Upload phase transition data to enable this analysis")
        
        # Tab 8: Clustering & PCA
        with tabs[7]:
            st.header("🎯 Clustering & Dimensionality Reduction")
            
            # Select features for clustering (expanded)
            cluster_options = ['tolerance_factor', 'chiB_avg', 'delta_chi_AB', 'rB_avg', 
                               'variance_rB', 'S_config_B', 'conc_B_prime', 'total_dopant_B',
                               'Vo_proxy', 'VB_avg']
            cluster_features = st.multiselect("Select features for clustering",
                                             [c for c in cluster_options if c in df.columns],
                                             default=[c for c in ['tolerance_factor', 'chiB_avg', 'delta_chi_AB'] if c in df.columns][:3])
            
            if len(cluster_features) >= 2 and len(df) > 0:
                # Prepare data - drop rows with NaN in any feature
                cluster_df = df[cluster_features].dropna()
                
                if len(cluster_df) >= 5:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(cluster_df)
                    
                    # PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    st.subheader("PCA Projection")
                    target_for_color = 'alpha_true' if 'alpha_true' in df.columns else None
                    if target_for_color and target_for_color in df.columns:
                        y_colors = df.loc[cluster_df.index, target_for_color].values
                        # Filter out NaN
                        valid_mask = ~np.isnan(y_colors)
                        X_pca_valid = X_pca[valid_mask]
                        y_colors_valid = y_colors[valid_mask]
                        if len(X_pca_valid) > 0:
                            fig = ScientificVisualizer.plot_pca_2d(X_pca_valid, y_colors_valid,
                                                                   title='PCA of Composition Space')
                            st.pyplot(fig)
                    else:
                        fig = ScientificVisualizer.plot_pca_2d(X_pca, np.zeros(len(X_pca)),
                                                               title='PCA of Composition Space')
                        st.pyplot(fig)
                    
                    st.write(f"Explained variance: PC1 = {pca.explained_variance_ratio_[0]:.2%}, PC2 = {pca.explained_variance_ratio_[1]:.2%}")
                    
                    # K-Means clustering
                    st.subheader("K-Means Clustering")
                    n_clusters = st.slider("Number of clusters", 2, 10, 3)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Visualize clusters
                    fig2, ax = plt.subplots(figsize=(10, 7))
                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis',
                                        alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                              c='red', marker='X', s=200, edgecolors='black', linewidth=1.5,
                              label='Centroids')
                    ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
                    ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
                    ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=12, fontweight='bold')
                    ax.legend()
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Cluster', fontsize=10)
                    st.pyplot(fig2)
                    
                    # Cluster characteristics
                    st.subheader("Cluster Characteristics")
                    cluster_df_copy = cluster_df.copy()
                    cluster_df_copy['Cluster'] = clusters
                    cluster_means = cluster_df_copy.groupby('Cluster')[cluster_features].mean()
                    st.dataframe(cluster_means, use_container_width=True)
                    
                    # Dendrogram
                    if len(cluster_df) < 200 and len(cluster_df) >= 10:
                        st.subheader("Hierarchical Clustering Dendrogram")
                        fig3, ax = plt.subplots(figsize=(12, 6))
                        linkage_matrix = linkage(X_scaled[:min(100, len(X_scaled))], method='ward')
                        dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=30)
                        ax.set_title('Dendrogram (top 30 samples)', fontsize=12, fontweight='bold')
                        ax.set_xlabel('Sample index', fontsize=11)
                        ax.set_ylabel('Distance', fontsize=11)
                        st.pyplot(fig3)
                else:
                    st.warning(f"Not enough valid data for clustering (need at least 5 samples, have {len(cluster_df)})")
            else:
                st.info("Select at least 2 features for clustering")
        
        # Tab 9: Advanced ML (SHAP)
        with tabs[8]:
            st.header("📉 Advanced ML Analysis with SHAP")
            
            if 'ml_results' in st.session_state and st.session_state.ml_results and 'error' not in st.session_state.ml_results:
                st.subheader("SHAP Analysis for Model Interpretability")
                
                if st.button("Run SHAP Analysis", use_container_width=True):
                    with st.spinner("Computing SHAP values..."):
                        try:
                            # Get the best model
                            results = st.session_state.ml_results
                            features = st.session_state.ml_features
                            
                            # Use Random Forest model for SHAP
                            if 'Random Forest' in results['models'] and 'model' in results['models']['Random Forest']:
                                rf_model = results['models']['Random Forest']['model']
                                X_train = results['X_train']
                                
                                if len(X_train) > 0 and len(features) > 0:
                                    # Create SHAP explainer
                                    explainer = shap.TreeExplainer(rf_model)
                                    shap_values = explainer.shap_values(X_train[:min(100, len(X_train))])  # Limit for performance
                                    
                                    # SHAP summary plot
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    shap.summary_plot(shap_values, X_train[:min(100, len(X_train))], feature_names=features, show=False)
                                    st.pyplot(fig)
                                    
                                    # SHAP bar plot
                                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                                    shap.summary_plot(shap_values, X_train[:min(100, len(X_train))], feature_names=features, 
                                                     plot_type="bar", show=False)
                                    st.pyplot(fig2)
                                    
                                    st.success("SHAP analysis completed")
                                else:
                                    st.warning("Insufficient data for SHAP analysis")
                            else:
                                st.warning("Random Forest model not available for SHAP analysis")
                        except Exception as e:
                            st.error(f"SHAP analysis failed: {str(e)[:100]}")
            else:
                st.info("Train a model in the Machine Learning tab first")
        
        # Tab 10: Export
        with tabs[9]:
            st.header("📤 Export Results")
            
            st.subheader("Export Data with Descriptors")
            
            # CSV export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Data as CSV",
                data=csv,
                file_name=f"perovskite_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Export report
            st.subheader("Generate Analysis Report")
            
            if st.button("📊 Generate Summary Report", use_container_width=True):
                report = []
                report.append("# Perovskite Expansion Analysis Report")
                report.append(f"\n## Summary Statistics")
                report.append(f"\n- Total samples: {len(df)}")
                report.append(f"- Descriptors calculated: {len(st.session_state.descriptor_names)}")
                
                alpha_col = 'alpha_true'
                if alpha_col in df.columns:
                    alpha_vals = df[alpha_col].dropna()
                    if len(alpha_vals) > 0:
                        report.append(f"\n### Thermal Expansion (α)")
                        report.append(f"- Mean: {alpha_vals.mean():.3f} ×10⁻⁶ K⁻¹")
                        report.append(f"- Std: {alpha_vals.std():.3f}")
                        report.append(f"- Min: {alpha_vals.min():.3f}")
                        report.append(f"- Max: {alpha_vals.max():.3f}")
                        report.append(f"- Valid samples: {len(alpha_vals)}")
                    else:
                        report.append(f"\n### Thermal Expansion (α)")
                        report.append(f"- No valid data available")
                
                beta_col = 'beta'
                if beta_col in df.columns:
                    beta_vals = df[beta_col].dropna()
                    if len(beta_vals) > 0:
                        report.append(f"\n### Chemical Expansion (β)")
                        report.append(f"- Mean: {beta_vals.mean():.4f}")
                        report.append(f"- Std: {beta_vals.std():.4f}")
                        report.append(f"- Valid samples: {len(beta_vals)}")
                    else:
                        report.append(f"\n### Chemical Expansion (β)")
                        report.append(f"- No valid data available")
                
                # Top correlations
                if alpha_col in df.columns and len(st.session_state.descriptor_names) > 0:
                    numeric_cols = st.session_state.descriptor_names + [alpha_col]
                    numeric_cols = [c for c in numeric_cols if c in df.columns]
                    if len(numeric_cols) > 1:
                        # Filter out rows with NaN
                        temp_df = df[numeric_cols].dropna()
                        if len(temp_df) > 5:
                            corrs = temp_df.corr()[alpha_col].abs().sort_values(ascending=False)
                            top_corrs = corrs.head(10)
                            
                            report.append(f"\n### Top 10 Correlations with {alpha_col}")
                            for feat, corr in top_corrs.items():
                                if feat != alpha_col and not pd.isna(corr):
                                    report.append(f"- {feat}: {corr:.3f}")
                
                # Save report
                report_text = "\n".join(report)
                st.download_button(
                    label="📄 Download Report (Markdown)",
                    data=report_text,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
                st.success("Report generated!")
    
    else:
        # No data loaded yet
        st.info("👈 Please paste chemical data in the sidebar to begin analysis")
        
        # Show expected format
        with st.expander("📖 Expected Data Format"):
            st.markdown("""
            ### Dataset 1: Chemical & Thermal Expansion
            
            Required columns:
            - `№` - Unique identifier
            - `A`, `A'` - A-site elements
            - `B`, `B'` - B-site elements  
            - `D1`, `D2` - Dopants
            - `[A']`, `[B']`, `[D1]`, `[D2]` - Concentrations
            - `δ` - Oxygen non-stoichiometry
            - `method` - Measurement method
            - `β` - Chemical expansion coefficient
            - `∆T, °C` - Temperature range
            - `α·106 (K-1)` - True thermal expansion coefficient
            - `αav·106 (K-1)` - Apparent thermal expansion
            - `pH2O` - Water partial pressure
            - `Ref` - Reference DOI
            
            **Note:** Missing values can be marked as `-` (dash) and will be handled automatically.
            
            ### Dataset 2: Phase Transitions
            
            Required columns:
            - `№` - Identifier
            - `A`, `A'`, `B`, `B'`, `D1`, `D2` - Composition
            - `[A]`, `[B']`, `[D1]`, `[D2]` - Concentrations
            - `Symmetry` - Crystal symmetry
            - `Phase transitions (PT)` - Phase sequence
            - `T (PT), °C` - Transition temperature(s)
            """)

# ============================================================================
# 9. ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == "__main__":
    main()
