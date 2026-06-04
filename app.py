"""
Streamlit Application for Analysis of Thermal and Chemical Expansion
of Proton-Conducting Perovskite Oxides

Version: 2.3 (with enhanced visualizations: expanded pairplot, concentration maps,
             bubble charts, phase transition analysis, regplot, PDP, elbow/silhouette)
Author: Materials Informatics Research
Description: Comprehensive analysis tool for understanding composition-structure-property
             relationships in proton-conducting perovskites with focus on thermal
             expansion (α), chemical expansion (β), and phase transitions.
             
Features:
- Robust handling of '-' and missing values in all data fields
- Upload and process two independent datasets via text/CSV/TSV input
- Calculate 35+ structural, electronegativity, and thermodynamic descriptors
- Interactive visualizations with scientific styling
- Machine learning models for property prediction
- SHAP analysis for interpretability
- Phase transition impact analysis with multiple plots
- Clustering and dimensionality reduction with elbow/silhouette
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
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import xgboost as xgb
import shap
from io import BytesIO
import csv
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# 0. УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ БЕЗОПАСНОГО ПРЕОБРАЗОВАНИЯ В FLOAT
# ============================================================================

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert any value to float, handling:
    - '-', '—', '', None, NaN
    - Strings with commas (European format: '0,0971847700154417')
    - Lists, tuples, and other sequences
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
    T_min = 0.0
    T_max = 0.0
    T_span = 0.0
    T_mid = 0.0
    
    if pd.isna(temp_range_str) or temp_range_str == '-' or temp_range_str == '':
        return T_min, T_max, T_span, T_mid
    
    temp_str = str(temp_range_str).strip()
    if '-' in temp_str:
        parts = temp_str.split('-')
        if len(parts) == 2:
            try:
                T_min = safe_float_conversion(parts[0].strip(), 0.0)
                T_max = safe_float_conversion(parts[1].strip(), 0.0)
                if T_min > 0 and T_max > 0 and T_max > T_min:
                    T_span = T_max - T_min
                    T_mid = (T_min + T_max) / 2
            except (ValueError, TypeError):
                pass
    
    return T_min, T_max, T_span, T_mid


def safe_parse_semicolon_values(value: Any) -> List[float]:
    """
    Safely parse semicolon-separated values like '400;600' or '10.6;4.73;10.1'
    Returns list of floats
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
                val = safe_float_conversion(part, None)
                if val is not None:
                    result.append(val)
        return result
    else:
        val = safe_float_conversion(value_str, None)
        return [val] if val is not None else []

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
            return 0.0
        
        # Determine coordination number based on site
        if site.upper() == 'A':
            # A-site typically CN=12, fallback to CN=8
            radius = ionic_radii.cn12.get(element, None)
            if radius is None:
                radius = ionic_radii.cn8.get(element, 0.0)
        else:  # B-site
            # B-site typically CN=6, fallback to CN=4
            radius = ionic_radii.cn6.get(element, None)
            if radius is None:
                radius = ionic_radii.cn4.get(element, 0.0)
        return radius if radius is not None else 0.0
    
    def get_electronegativity(self, element: str) -> float:
        """Get Pauling electronegativity"""
        if pd.isna(element) or element == '-' or element == '':
            return 0.0
        return element_props.electronegativity.get(element, 0.0)
    
    def get_polarizability(self, element: str) -> float:
        """Get ionic polarizability"""
        if pd.isna(element) or element == '-' or element == '':
            return 0.0
        return element_props.polarizability.get(element, 0.0)
    
    def get_ionization_potential(self, element: str) -> float:
        """Get first ionization potential (eV)"""
        if pd.isna(element) or element == '-' or element == '':
            return 0.0
        return element_props.ionization_potential.get(element, 0.0)
    
    def get_valency(self, element: str) -> int:
        """Get common oxidation state"""
        if pd.isna(element) or element == '-' or element == '':
            return 0
        return element_props.valency.get(element, 0)
    
    def calculate_average_radius(self, elements: List[str], concentrations: List[float], site: str) -> float:
        """Calculate weighted average ionic radius"""
        total = 0.0
        for elem, conc in zip(elements, concentrations):
            if conc > 0 and elem not in [None, '-', '']:
                total += conc * self.get_ionic_radius(elem, site)
        return total if total > 0 else 0.0
    
    def calculate_average_electronegativity(self, elements: List[str], concentrations: List[float]) -> float:
        """Calculate weighted average electronegativity"""
        total = 0.0
        for elem, conc in zip(elements, concentrations):
            if conc > 0 and elem not in [None, '-', '']:
                total += conc * self.get_electronegativity(elem)
        return total if total > 0 else 0.0
    
    def calculate_average_polarizability(self, elements: List[str], concentrations: List[float]) -> float:
        """Calculate weighted average polarizability"""
        total = 0.0
        for elem, conc in zip(elements, concentrations):
            if conc > 0 and elem not in [None, '-', '']:
                total += conc * self.get_polarizability(elem)
        return total if total > 0 else 0.0
    
    def calculate_average_valency(self, elements: List[str], concentrations: List[float]) -> float:
        """Calculate weighted average valency"""
        total = 0.0
        for elem, conc in zip(elements, concentrations):
            if conc > 0 and elem not in [None, '-', '']:
                total += conc * self.get_valency(elem)
        return total if total > 0 else 0.0
    
    def calculate_configurational_entropy(self, concentrations: List[float]) -> float:
        """Calculate configurational entropy: S = -R * Σ(x_i * ln(x_i))"""
        total = 0.0
        valid_concs = [c for c in concentrations if c > 0]
        if not valid_concs:
            return 0.0
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
        
        Returns dictionary with descriptor names and values
        """
        descriptors = {}
        
        # Extract composition data (these are strings, no conversion needed)
        A = row.get('A', None) if not pd.isna(row.get('A', None)) else None
        A_prime = row.get("A'", None) if not pd.isna(row.get("A'", None)) else None
        B = row.get('B', None) if not pd.isna(row.get('B', None)) else None
        B_prime = row.get("B'", None) if not pd.isna(row.get("B'", None)) else None
        D1 = row.get('D1', None) if not pd.isna(row.get('D1', None)) else None
        D2 = row.get('D2', None) if not pd.isna(row.get('D2', None)) else None
        
        # Concentrations - using safe_float_conversion for all
        conc_A_prime = safe_float_conversion(row.get("[A']", 0), 0.0)
        conc_B_prime = safe_float_conversion(row.get("[B']", 0), 0.0)
        conc_D1 = safe_float_conversion(row.get("[D1]", 0), 0.0)
        conc_D2 = safe_float_conversion(row.get("[D2]", 0), 0.0)
        
        # Ensure concentrations are within [0, 1]
        conc_A_prime = max(0.0, min(1.0, conc_A_prime))
        conc_B_prime = max(0.0, min(1.0, conc_B_prime))
        conc_D1 = max(0.0, min(1.0, conc_D1))
        conc_D2 = max(0.0, min(1.0, conc_D2))
        
        # Calculate remaining concentrations
        conc_A = 1.0 - conc_A_prime
        conc_B = 1.0 - conc_B_prime - conc_D1 - conc_D2
        conc_B = max(0.0, conc_B)  # Ensure non-negative
        
        # ====================================================================
        # Category 1: Geometric descriptors
        # ====================================================================
        
        # A-site elements and concentrations
        A_elements = [A, A_prime] if A_prime not in [None, '-', ''] else [A]
        A_concentrations = [conc_A, conc_A_prime] if A_prime not in [None, '-', ''] else [1.0]
        
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
        
        descriptors['rA_avg'] = rA_avg
        descriptors['rB_avg'] = rB_avg
        descriptors['rA_rB_ratio'] = rA_avg / rB_avg if rB_avg > 0 else 0
        
        # Tolerance factor (Goldschmidt)
        sqrt2 = np.sqrt(2)
        denominator = sqrt2 * (rB_avg + self.r_o)
        t = (rA_avg + self.r_o) / denominator if denominator > 0 else 0
        descriptors['tolerance_factor'] = t
        descriptors['tolerance_deviation'] = abs(1 - t)
        
        # Octahedral factor
        octahedral = rB_avg / self.r_o if self.r_o > 0 else 0
        descriptors['octahedral_factor'] = octahedral
        
        # Radius difference
        delta_r_AB = abs(rA_avg - rB_avg)
        descriptors['delta_r_AB'] = delta_r_AB
        descriptors['delta_r_AB_norm'] = delta_r_AB / self.r_o if self.r_o > 0 else 0
        
        # Variance of B-site radii
        if len(B_elements) > 1:
            rB_squared_avg = sum(c * (self.get_ionic_radius(e, 'B')**2) 
                                 for e, c in zip(B_elements, B_concentrations) if e not in [None, '-', ''])
            rB_avg_sq = descriptors['rB_avg']**2
            descriptors['variance_rB'] = max(0, rB_squared_avg - rB_avg_sq)
        else:
            descriptors['variance_rB'] = 0.0
        
        # Variance of A-site radii
        if len(A_elements) > 1:
            rA_squared_avg = sum(c * (self.get_ionic_radius(e, 'A')**2)
                                 for e, c in zip(A_elements, A_concentrations) if e not in [None, '-', ''])
            rA_avg_sq = descriptors['rA_avg']**2
            descriptors['variance_rA'] = max(0, rA_squared_avg - rA_avg_sq)
        else:
            descriptors['variance_rA'] = 0.0
        
        # ====================================================================
        # Category 2: Electronegativity descriptors
        # ====================================================================
        
        # Individual element electronegativities
        descriptors['chi_A'] = self.get_electronegativity(A) if A else 0
        descriptors['chi_A_prime'] = self.get_electronegativity(A_prime) if A_prime else 0
        descriptors['chi_B'] = self.get_electronegativity(B) if B else 0
        descriptors['chi_B_prime'] = self.get_electronegativity(B_prime) if B_prime else 0
        descriptors['chi_D1'] = self.get_electronegativity(D1) if D1 else 0
        descriptors['chi_D2'] = self.get_electronegativity(D2) if D2 else 0
        
        # Average electronegativities
        chiA_avg = self.calculate_average_electronegativity(A_elements, A_concentrations)
        chiB_avg = self.calculate_average_electronegativity(B_elements, B_concentrations)
        descriptors['chiA_avg'] = chiA_avg
        descriptors['chiB_avg'] = chiB_avg
        
        # Difference and ratio
        delta_chi_AB = abs(chiA_avg - chiB_avg)
        descriptors['delta_chi_AB'] = delta_chi_AB
        descriptors['chi_ratio_AB'] = chiA_avg / chiB_avg if chiB_avg > 0 else 0
        
        # Total average electronegativity
        descriptors['chi_total_avg'] = (chiA_avg + chiB_avg) / 2
        
        # Ionicity (Pauling formula)
        ionicity_AO = 1 - np.exp(-0.25 * (chiA_avg - self.chi_o)**2)
        ionicity_BO = 1 - np.exp(-0.25 * (chiB_avg - self.chi_o)**2)
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
        descriptors['valency_A'] = self.get_valency(A) if A else 0
        descriptors['valency_A_prime'] = self.get_valency(A_prime) if A_prime else 0
        descriptors['valency_B'] = self.get_valency(B) if B else 0
        descriptors['valency_B_prime'] = self.get_valency(B_prime) if B_prime else 0
        descriptors['valency_D1'] = self.get_valency(D1) if D1 else 0
        descriptors['valency_D2'] = self.get_valency(D2) if D2 else 0
        
        # Average valency on B-site
        VB_avg = self.calculate_average_valency(B_elements, B_concentrations)
        descriptors['VB_avg'] = VB_avg
        
        # Oxygen vacancy proxy (for Ce4+/Zr4+ based perovskites)
        # Assuming perovskite with A2+B4+O3, dopants with M3+ create oxygen vacancies
        # Vacancy concentration = [D1]/2 + [D2]/2
        descriptors['Vo_proxy'] = (conc_D1 + conc_D2) / 2
        
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
        descriptors['rB_x_chiB'] = rB_avg * chiB_avg
        
        # ====================================================================
        # Category 5: Additional useful descriptors
        # ====================================================================
        
        # Total dopant concentrations
        descriptors['total_dopant_A'] = conc_A_prime
        descriptors['total_dopant_B'] = conc_B_prime + conc_D1 + conc_D2
        descriptors['total_dopant'] = conc_A_prime + conc_B_prime + conc_D1 + conc_D2
        
        # Oxygen stoichiometry parameter delta - using safe_float_conversion
        delta = safe_float_conversion(row.get('δ', 0), 0.0)
        descriptors['delta'] = delta
        
        # Chemical expansion beta - using safe_float_conversion
        beta_val = safe_float_conversion(row.get('β', 0), 0.0)
        descriptors['beta'] = beta_val
        
        # True thermal expansion coefficient alpha - using safe_float_conversion
        alpha_val = safe_float_conversion(row.get('α·106 (K-1)', 0), 0.0)
        descriptors['alpha_true'] = alpha_val
        
        # Apparent thermal expansion coefficient alpha_av - using safe_float_conversion
        alpha_av_raw = row.get('αav·106 (K-1)', 0)
        if pd.isna(alpha_av_raw) or alpha_av_raw == '-' or alpha_av_raw == '':
            descriptors['alpha_apparent'] = 0.0
        else:
            alpha_av_list = safe_parse_semicolon_values(alpha_av_raw)
            descriptors['alpha_apparent'] = alpha_av_list[0] if alpha_av_list else 0.0
        
        # Water partial pressure - using safe_float_conversion
        pH2O = safe_float_conversion(row.get('pH2O', 0), 0.0)
        descriptors['log_pH2O'] = np.log10(pH2O) if pH2O > 0 else -10
        
        # Temperature range span - using safe_parse_temperature_range
        temp_range = row.get('∆T, °C', '')
        T_min, T_max, T_span, T_mid = safe_parse_temperature_range(temp_range)
        descriptors['T_min'] = T_min
        descriptors['T_max'] = T_max
        descriptors['T_span'] = T_span
        descriptors['T_mid'] = T_mid
        
        # Bends temperatures - using safe_parse_semicolon_values
        bends_raw = row.get('T(bends), °C', '')
        bends_list = safe_parse_semicolon_values(bends_raw)
        descriptors['T_bends_first'] = bends_list[0] if bends_list else 0.0
        descriptors['T_bends_count'] = len(bends_list)
        
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
# 6. ВИЗУАЛИЗАЦИИ
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
    
    # Extended list of available descriptors for axis selection
    AVAILABLE_DESCRIPTORS = [
        # Concentrations
        '[D1]', '[D2]', "[B']", 'total_dopant_A', 'total_dopant_B', 'total_dopant', 'sum_B_site_dopants',
        # Geometric
        'tolerance_factor', 'tolerance_deviation', 'octahedral_factor', 'delta_r_AB', 'delta_r_AB_norm',
        'rA_rB_ratio', 'rA_avg', 'rB_avg', 'variance_rA', 'variance_rB',
        # Electronegativity
        'chiA_avg', 'chiB_avg', 'delta_chi_AB', 'chi_ratio_AB', 'chi_total_avg',
        'ionicity_AO', 'ionicity_BO',
        # Thermodynamic
        'S_config_A', 'S_config_B', 'S_config_total', 'VB_avg', 'Vo_proxy',
        # Combined
        'delta_chi_div_t', 'delta_chi_mul_t', 'disorder_over_distortion',
        'ionic_x_octa', 'chi_ratio_t', 'rB_x_chiB',
        # Experimental conditions
        'log_pH2O', 'T_min', 'T_max', 'T_span', 'T_mid',
        # Targets
        'alpha_true', 'alpha_apparent', 'beta', 'delta'
    ]
    
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
        """Plot histogram with KDE and statistics"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Filter out zeros and NaNs for distribution (zeros may be from missing data)
        data = df[column].dropna()
        data = data[data != 0]  # Exclude zeros that came from missing data
        
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
        """Create boxplot for categorical comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter valid data (exclude zeros that may come from missing data)
        plot_df = df[[x_col, y_col]].dropna()
        plot_df = plot_df.reset_index(drop=True)
        plot_df = plot_df[plot_df[y_col] != 0]  
        
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
        # Filter features that exist and have valid data
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
    def plot_regplot(df: pd.DataFrame, x_col: str, y_col: str, 
                     title: str = None, ci: float = 0.95):
        """Create regression plot with confidence interval (regplot)"""
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Filter valid data (exclude zeros)
        plot_df = df[[x_col, y_col]].dropna()
        plot_df = plot_df[(plot_df[x_col] != 0) & (plot_df[y_col] != 0)]
        
        if len(plot_df) < 3:
            ax.text(0.5, 0.5, f"Not enough data points for regression (n={len(plot_df)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Create regplot
        sns.regplot(data=plot_df, x=x_col, y=y_col, ax=ax, 
                   scatter_kws={'alpha': 0.6, 's': 50, 'color': ScientificVisualizer.COLORS['primary'],
                               'edgecolor': 'black', 'linewidth': 0.5},
                   line_kws={'color': ScientificVisualizer.COLORS['secondary'], 'linewidth': 2},
                   ci=ci)
        
        # Calculate R²
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(plot_df[x_col], plot_df[y_col])
        r2 = r_value ** 2
        
        # Add R² text
        textstr = f'R² = {r2:.3f}\nSlope = {slope:.3f}\np-value = {p_value:.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax.set_title(title or f'Regression: {y_col} vs {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_pairplot_custom(df: pd.DataFrame, features: List[str], 
                              diag_kind: str = 'kde', title: str = None):
        """Create custom pairplot with user-selected features"""
        if len(features) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Select at least 2 features for pairplot", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Filter data
        plot_df = df[features].dropna()
        for col in features:
            plot_df = plot_df[plot_df[col] != 0]
        
        if len(plot_df) < 3:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Not enough data (n={len(plot_df)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Create pairplot
        fig = sns.pairplot(plot_df, diag_kind=diag_kind,
                          plot_kws={'alpha': 0.6, 's': 40, 
                                   'color': ScientificVisualizer.COLORS['primary'],
                                   'edgecolor': 'black', 'linewidth': 0.5},
                          diag_kws={'color': ScientificVisualizer.COLORS['secondary']})
        
        if title:
            fig.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        return fig
    
    @staticmethod
    def plot_regression_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = 'Model'):
        """Plot actual vs predicted with residuals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Filter out zero values that may indicate missing data
        valid_mask = (y_true != 0) & (y_pred != 0)
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
        """Create 2D scatter plot with optional color and size mapping"""
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Prepare data - start with all three columns if color_col provided
        if color_col and color_col in df.columns:
            base_cols = [x_col, y_col, color_col]
        else:
            base_cols = [x_col, y_col]
        
        if size_col and size_col in df.columns:
            base_cols.append(size_col)
        
        # Create a copy and reset index to avoid duplicate index issues
        plot_df = df[base_cols].copy().reset_index(drop=True)
        plot_df = plot_df.dropna()
        
        # Filter out zeros in y_col (target property)
        if y_col in plot_df.columns:
            plot_df = plot_df[plot_df[y_col] != 0]
        
        # Also filter out zeros in x_col if needed
        if x_col in plot_df.columns and len(plot_df) > 0:
            plot_df = plot_df[plot_df[x_col] != 0]
        
        # Filter out zeros in color_col if it's being used for coloring
        if color_col and color_col in plot_df.columns and len(plot_df) > 0:
            plot_df = plot_df[plot_df[color_col] != 0]
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, "No valid data for scatter plot", transform=ax.transAxes, ha='center', va='center')
            return fig
        
        if color_col:
            scatter = ax.scatter(plot_df[x_col].values, plot_df[y_col].values, 
                                c=plot_df[color_col].values,
                                s=plot_df[size_col].values * 50 if size_col else 50,
                                cmap=ScientificVisualizer.CMAPS['thermal'],
                                alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_col, fontsize=10)
        else:
            ax.scatter(plot_df[x_col].values, plot_df[y_col].values, 
                      s=plot_df[size_col].values * 50 if size_col else 50,
                      c=ScientificVisualizer.COLORS['primary'],
                      alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax.set_title(title or f'{y_col} vs {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_hexbin_heatmap(df: pd.DataFrame, x_col: str, y_col: str, 
                             z_col: str = None, gridsize: int = 30,
                             title: str = None):
        """Create hexbin heatmap (2D density plot with optional color mapping)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter data
        plot_df = df[[x_col, y_col]].dropna()
        if z_col and z_col in df.columns:
            plot_df = plot_df.join(df[z_col])
            plot_df = plot_df[(plot_df[x_col] != 0) & (plot_df[y_col] != 0) & (plot_df[z_col] != 0)]
        else:
            plot_df = plot_df[(plot_df[x_col] != 0) & (plot_df[y_col] != 0)]
        
        if len(plot_df) < 5:
            ax.text(0.5, 0.5, f"Not enough data for hexbin (n={len(plot_df)})", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        if z_col and z_col in plot_df.columns:
            # Colored hexbin
            hb = ax.hexbin(plot_df[x_col], plot_df[y_col], C=plot_df[z_col],
                          gridsize=gridsize, cmap=ScientificVisualizer.CMAPS['thermal'],
                          edgecolors='none', alpha=0.8)
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label(z_col, fontsize=10)
        else:
            # Density hexbin
            hb = ax.hexbin(plot_df[x_col], plot_df[y_col], gridsize=gridsize,
                          cmap=ScientificVisualizer.CMAPS['thermal'],
                          edgecolors='none', alpha=0.8)
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Density (points per bin)', fontsize=10)
        
        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        ax.set_title(title or f'2D Hexbin Heatmap: {y_col} vs {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_concentration_heatmap(df: pd.DataFrame, x_col: str, y_col: str, 
                                    z_col: str, bins: int = 20):
        """Create 2D heatmap for concentration dependence"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create grid
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        
        # Remove NaN and zero values (zero may indicate missing data for z)
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x = x[mask]
        y = y[mask]
        z = z[mask]
        
        # Also filter out zero z values (likely missing data)
        nonzero_mask = z != 0
        x = x[nonzero_mask]
        y = y[nonzero_mask]
        z = z[nonzero_mask]
        
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
    def plot_pca_2d(X_pca: np.ndarray, y: np.ndarray, 
                    labels: List[str] = None, title: str = 'PCA Projection'):
        """Create 2D PCA scatter plot with color mapping"""
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
    def plot_elbow_curve(X_scaled: np.ndarray, max_k: int = 15):
        """Plot elbow curve for KMeans clustering (inertia vs k)"""
        inertias = []
        K = range(1, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K, inertias, 'o-', color=ScientificVisualizer.COLORS['primary'], 
               linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        ax.set_xlabel('Number of clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Inertia', fontsize=11, fontweight='bold')
        ax.set_title('Elbow Method for Optimal k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_silhouette_analysis(X_scaled: np.ndarray, k: int):
        """Plot silhouette analysis for KMeans clustering"""
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Compute silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        y_lower = 10
        
        for i in range(k):
            # Aggregate silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(i / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax.set_xlabel('Silhouette coefficient', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cluster label', fontsize=11, fontweight='bold')
        ax.set_title(f'Silhouette plot for k={k} (avg silhouette: {silhouette_avg:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=1.5)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
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
        """Prepare feature matrix and optional target vector"""
        # Filter valid rows - exclude rows where target is zero (likely missing)
        if target_col:
            valid_df = df[feature_cols + [target_col]].dropna()
            valid_df = valid_df[valid_df[target_col] != 0]  # Exclude zeros from missing data
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
    
    def compute_partial_dependence(self, model, X: np.ndarray, feature_idx: int,
                                    feature_names: List[str], grid_resolution: int = 50):
        """Compute partial dependence for a single feature"""
        try:
            results = partial_dependence(
                model, X, [feature_idx], grid_resolution=grid_resolution,
                kind='average'
            )
            return results['values'][0], results['average'][0]
        except Exception as e:
            return None, None

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
                pH2O_min = float(df['log_pH2O'].min())
                pH2O_max = float(df['log_pH2O'].max())
                if pH2O_min < pH2O_max:
                    pH2O_range = st.slider("log(pH₂O)", pH2O_min, pH2O_max, (pH2O_min, pH2O_max))
                else:
                    pH2O_range = (pH2O_min, pH2O_max)
            else:
                pH2O_range = (-10, 0)
            
            # Alpha range
            alpha_col = 'alpha_true'
            if alpha_col in df.columns:
                alpha_values = df[alpha_col][df[alpha_col] != 0]  # Exclude zeros from missing data
                if len(alpha_values) > 0:
                    alpha_min = float(alpha_values.min())
                    alpha_max = float(alpha_values.max())
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
                    empty_desc = {k: 0.0 for k in PerovskiteDescriptorCalculator().calculate_all_descriptors(chem_df.iloc[0]).keys()}
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
        
        # Create tabs
        tabs = st.tabs([
            "📊 Data Overview",
            "📈 EDA & Distributions",
            "🔥 Correlation Analysis",
            "🗺️ Concentration Maps",
            "💨 Bubble Charts",
            "🔬 Phase Transition Analysis",
            "🧠 Machine Learning",
            "🎯 Clustering & PCA",
            "📉 Advanced ML (SHAP & PDP)",
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
                    alpha_vals = df[alpha_col][df[alpha_col] != 0]
                    mean_alpha = alpha_vals.mean() if len(alpha_vals) > 0 else 0
                    st.metric("Mean α (×10⁶ K⁻¹)", f"{mean_alpha:.2f}")
            with col4:
                beta_col = 'beta'
                if beta_col in df.columns:
                    beta_vals = df[beta_col][df[beta_col] != 0]
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
                        vals = vals[vals != 0]  # Exclude zeros from missing data
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
            
            # Sub-tabs for different correlation views
            corr_tab1, corr_tab2, corr_tab3 = st.tabs(["📊 Correlation Matrix", "📈 Pairplot", "📉 Regression Plots"])
            
            with corr_tab1:
                # Select features for correlation
                default_features = ['rB_avg', 'tolerance_factor', 'chiB_avg', 'delta_chi_AB',
                                   'variance_rB', 'S_config_B', 'VB_avg', 'Vo_proxy',
                                   'alpha_true', 'beta']
                available_features = [f for f in default_features if f in df.columns]
                
                if len(available_features) > 2:
                    fig = ScientificVisualizer.plot_correlation_matrix(df, available_features, 
                                                                        target='alpha_true' if 'alpha_true' in df.columns else None)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough features for correlation analysis")
            
            with corr_tab2:
                st.subheader("Custom Pairplot")
                
                # Select features for pairplot
                all_numeric_features = [f for f in ScientificVisualizer.AVAILABLE_DESCRIPTORS if f in df.columns]
                
                if len(all_numeric_features) >= 2:
                    selected_pair_features = st.multiselect(
                        "Select features for pairplot (2-6 features):",
                        all_numeric_features,
                        default=all_numeric_features[:4] if len(all_numeric_features) >= 4 else all_numeric_features
                    )
                    
                    if len(selected_pair_features) >= 2:
                        diag_type = st.radio("Diagonal plot type:", ["kde", "hist"], horizontal=True)
                        fig = ScientificVisualizer.plot_pairplot_custom(
                            df, selected_pair_features, 
                            diag_kind=diag_type,
                            title=f'Pairplot: {", ".join(selected_pair_features)}'
                        )
                        st.pyplot(fig)
                    else:
                        st.info("Select at least 2 features for pairplot")
                else:
                    st.warning(f"Not enough numeric features (found {len(all_numeric_features)})")
            
            with corr_tab3:
                st.subheader("Regression Plots with Confidence Intervals")
                
                # Select X and Y for regression
                regression_features = [f for f in ['alpha_true', 'beta', 'tolerance_factor', 'chiB_avg', 
                                                   'delta_chi_AB', 'rB_avg', 'Vo_proxy'] if f in df.columns]
                
                if len(regression_features) >= 2:
                    col1_reg, col2_reg = st.columns(2)
                    with col1_reg:
                        x_reg = st.selectbox("X-axis for regression:", regression_features, key="reg_x")
                    with col2_reg:
                        y_options = [f for f in ['alpha_true', 'beta', 'alpha_apparent'] if f in df.columns]
                        y_reg = st.selectbox("Y-axis for regression:", y_options if y_options else regression_features, key="reg_y")
                    
                    if x_reg != y_reg:
                        ci_level = st.slider("Confidence interval:", 0.8, 0.99, 0.95, 0.01)
                        fig = ScientificVisualizer.plot_regplot(df, x_reg, y_reg, ci=ci_level)
                        st.pyplot(fig)
                else:
                    st.warning("Not enough features for regression analysis")
        
        # Tab 4: Concentration Maps
        with tabs[3]:
            st.header("🗺️ Concentration Maps")
            
            # Get available descriptors for axis selection (excluding targets for X/Y)
            axis_options = [d for d in ScientificVisualizer.AVAILABLE_DESCRIPTORS 
                           if d in df.columns and d not in ['alpha_true', 'alpha_apparent', 'beta']]
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis (descriptor):", axis_options, key="conc_x")
            with col2:
                y_axis = st.selectbox("Y-axis (descriptor):", axis_options, key="conc_y")
            
            # Color options (target properties)
            color_options = [d for d in ['alpha_true', 'alpha_apparent', 'beta', 'tolerance_factor', 'delta_chi_AB'] 
                            if d in df.columns]
            z_axis = st.selectbox("Color by (property):", color_options, key="conc_z")
            
            # 2D Scatter
            st.subheader("2D Concentration Map (Scatter)")
            fig = ScientificVisualizer.plot_scatter_2d(df, x_axis, y_axis, color_col=z_axis)
            st.pyplot(fig)
            
            # 2D Heatmap
            st.subheader("2D Concentration Map (Heatmap)")
            fig2 = ScientificVisualizer.plot_concentration_heatmap(df, x_axis, y_axis, z_axis)
            st.pyplot(fig2)
        
        # Tab 5: Bubble Charts
        with tabs[4]:
            st.header("💨 Bubble Charts")
            
            st.markdown("""
            <div class="info-box">
            <b>Interactive bubble charts:</b> Y-axis shows target property (α, αav, or β),<br>
            while X-axis, bubble size, and color can be customized from 35+ descriptors.
            </div>
            """, unsafe_allow_html=True)
            
            # Bubble chart sub-tabs
            bubble_tab1, bubble_tab2 = st.tabs(["🫧 Bubble Scatter", "🔥 2D Bubble Heatmap (Hexbin)"])
            
            with bubble_tab1:
                # Select Y-axis (target property)
                bubble_y_options = ['alpha_true', 'alpha_apparent', 'beta']
                y_bubble = st.selectbox("Y-axis (target property):",
                                        [c for c in bubble_y_options if c in df.columns],
                                        key="bubble_y")
                
                # X-axis options (all descriptors)
                x_options = [d for d in ScientificVisualizer.AVAILABLE_DESCRIPTORS if d in df.columns]
                x_bubble = st.selectbox("X-axis:", x_options, key="bubble_x")
                
                col1, col2 = st.columns(2)
                with col1:
                    size_options = [d for d in ['delta', 'total_dopant_B', 'S_config_B', 'beta', 'Vo_proxy'] 
                                   if d in df.columns]
                    size_bubble = st.selectbox("Bubble size:", size_options, key="bubble_size")
                with col2:
                    color_options_bubble = [d for d in ['method', 'A', 'B', 'alpha_true', 'beta', 'tolerance_factor'] 
                                           if d in df.columns]
                    color_bubble = st.selectbox("Color:", color_options_bubble, key="bubble_color")
                
                # Create plotly bubble chart
                if len(df) > 0:
                    # Filter out zeros for y-axis
                    plot_df = df[df[y_bubble] != 0].copy()
                    if len(plot_df) > 0:
                        fig = px.scatter(plot_df, x=x_bubble, y=y_bubble,
                                         size=size_bubble if size_bubble else None,
                                         color=color_bubble,
                                         hover_data=['A', 'B', '[D1]', '[D2]', 'method', 'alpha_true', 'beta'],
                                         title=f'Bubble Chart: {y_bubble} vs {x_bubble}',
                                         size_max=30)
                        fig.update_layout(
                            template='plotly_white',
                            font=dict(family='Times New Roman', size=12),
                            xaxis_title=x_bubble,
                            yaxis_title=y_bubble
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid data for bubble chart")
            
            with bubble_tab2:
                st.subheader("2D Bubble Heatmap (Hexbin)")
                st.markdown("Hexbin plot showing point density with optional color mapping.")
                
                # X and Y selection
                x_hex = st.selectbox("X-axis:", [d for d in ScientificVisualizer.AVAILABLE_DESCRIPTORS if d in df.columns], key="hex_x")
                y_hex = st.selectbox("Y-axis:", [d for d in ['alpha_true', 'alpha_apparent', 'beta'] if d in df.columns], key="hex_y")
                
                # Optional color mapping
                z_hex_options = [d for d in ['beta', 'delta', 'total_dopant_B', 'Vo_proxy', 'tolerance_factor'] if d in df.columns]
                z_hex = st.selectbox("Color mapping (optional):", [None] + z_hex_options, key="hex_z")
                
                gridsize = st.slider("Grid size (resolution):", 20, 60, 30, 5)
                
                if y_hex in df.columns:
                    fig = ScientificVisualizer.plot_hexbin_heatmap(
                        df, x_hex, y_hex, 
                        z_col=z_hex if z_hex != "None" else None,
                        gridsize=gridsize,
                        title=f'Hexbin Heatmap: {y_hex} vs {x_hex}'
                    )
                    st.pyplot(fig)
        
        # Tab 6: Phase Transition Analysis
        with tabs[5]:
            st.header("🔬 Phase Transition Analysis")
            
            if st.session_state.phase_df is not None:
                phase_df = st.session_state.phase_df
                
                # Create sub-tabs for phase transition plots
                pt_tab1, pt_tab2, pt_tab3, pt_tab4 = st.tabs([
                    "📋 Data Overview", 
                    "📊 Temperature Analysis", 
                    "🔗 Correlation Plots",
                    "📈 Composition Dependence"
                ])
                
                with pt_tab1:
                    st.subheader("Phase Transition Data")
                    st.dataframe(phase_df, use_container_width=True)
                    
                    # Basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(phase_df))
                    with col2:
                        # Count unique symmetries
                        all_sym = []
                        for _, row in phase_df.iterrows():
                            sym = row.get('Symmetry', '')
                            if sym and sym != '-':
                                all_sym.extend(sym.split(';'))
                        st.metric("Unique Symmetries", len(set(all_sym)))
                    with col3:
                        # Count transitions with temperatures
                        temp_count = 0
                        for _, row in phase_df.iterrows():
                            temps = DataParser.parse_bends_temperatures(row.get('T (PT), °C', ''))
                            if temps:
                                temp_count += 1
                        st.metric("Records with T(PT)", temp_count)
                
                with pt_tab2:
                    st.subheader("Transition Temperatures Distribution")
                    
                    # Extract all transition temperatures
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
                    
                    st.subheader("Symmetry Types Distribution")
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
                    
                    st.subheader("Transition Type Distribution")
                    transition_types = []
                    for _, row in phase_df.iterrows():
                        pt_str = row.get('Phase transitions (PT)', '')
                        if pt_str and pt_str != '-':
                            # Count number of transitions (number of arrows between phases)
                            if ';' in pt_str:
                                parts = pt_str.split(';')
                                transition_types.append(f"{len(parts)}-phase sequence")
                            else:
                                transition_types.append("Single phase")
                    
                    if transition_types:
                        type_counts = pd.Series(transition_types).value_counts()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        type_counts.plot(kind='bar', ax=ax, color=ScientificVisualizer.COLORS['primary'])
                        ax.set_xlabel('Transition Type', fontsize=11, fontweight='bold')
                        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                        ax.set_title('Phase Transition Types', fontsize=12, fontweight='bold')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                
                with pt_tab3:
                    st.subheader("Correlation: Phase Transitions vs Composition")
                    
                    # Create combined dataset from phase_df and chemical data
                    # For each phase record, try to match with chemical data by composition
                    if st.session_state.chem_with_descriptors is not None:
                        chem_df_pt = st.session_state.chem_with_descriptors
                        
                        # Try to match by A, B, D1, [D1] approximately
                        matched_data = []
                        for _, pt_row in phase_df.iterrows():
                            pt_a = pt_row.get('A', '')
                            pt_b = pt_row.get('B', '')
                            pt_d1 = pt_row.get('D1', '')
                            pt_d1_conc = safe_float_conversion(pt_row.get('[D1]', 0), 0)
                            
                            # Find matching chem rows
                            matches = chem_df_pt[
                                (chem_df_pt['A'] == pt_a) & 
                                (chem_df_pt['B'] == pt_b) &
                                (abs(chem_df_pt.get('[D1]', 0) - pt_d1_conc) < 0.02)
                            ]
                            
                            if len(matches) > 0:
                                for _, match in matches.iterrows():
                                    temps = DataParser.parse_bends_temperatures(pt_row.get('T (PT), °C', ''))
                                    for t_pt in temps:
                                        matched_data.append({
                                            'T_PT': t_pt,
                                            'alpha_true': match.get('alpha_true', 0),
                                            'beta': match.get('beta', 0),
                                            'total_dopant_B': match.get('total_dopant_B', 0),
                                            'tolerance_factor': match.get('tolerance_factor', 0),
                                            'delta_chi_AB': match.get('delta_chi_AB', 0),
                                            'symmetry': pt_row.get('Symmetry', 'Unknown')
                                        })
                        
                        if matched_data:
                            match_df = pd.DataFrame(matched_data)
                            match_df = match_df[match_df['T_PT'] > 0]
                            
                            if len(match_df) > 0:
                                # α vs T(PT) plot
                                st.subheader("Thermal Expansion vs Transition Temperature")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                scatter = ax.scatter(match_df['T_PT'], match_df['alpha_true'], 
                                                   c=match_df['beta'] if 'beta' in match_df.columns else 'blue',
                                                   cmap='plasma', alpha=0.7, s=80,
                                                   edgecolors='black', linewidth=0.5)
                                ax.set_xlabel('Phase Transition Temperature (°C)', fontsize=11, fontweight='bold')
                                ax.set_ylabel('α (×10⁶ K⁻¹)', fontsize=11, fontweight='bold')
                                ax.set_title('α vs T(PT)', fontsize=12, fontweight='bold')
                                cbar = plt.colorbar(scatter, ax=ax)
                                cbar.set_label('β (Chemical Expansion)', fontsize=10)
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                
                                # β vs T(PT)
                                if 'beta' in match_df.columns:
                                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                                    scatter2 = ax2.scatter(match_df['T_PT'], match_df['beta'],
                                                          c=match_df['alpha_true'], cmap='viridis',
                                                          alpha=0.7, s=80,
                                                          edgecolors='black', linewidth=0.5)
                                    ax2.set_xlabel('Phase Transition Temperature (°C)', fontsize=11, fontweight='bold')
                                    ax2.set_ylabel('β (Chemical Expansion)', fontsize=11, fontweight='bold')
                                    ax2.set_title('β vs T(PT)', fontsize=12, fontweight='bold')
                                    cbar2 = plt.colorbar(scatter2, ax=ax2)
                                    cbar2.set_label('α (×10⁶ K⁻¹)', fontsize=10)
                                    ax2.grid(True, alpha=0.3)
                                    st.pyplot(fig2)
                            else:
                                st.info("No matched data with valid temperatures")
                        else:
                            st.info("No matching compositions found between phase transition and chemical data")
                    else:
                        st.info("Load chemical data first to enable correlation plots")
                
                with pt_tab4:
                    st.subheader("Composition Dependence of Phase Transitions")
                    
                    # Extract composition data from phase_df
                    comp_data = []
                    for _, row in phase_df.iterrows():
                        # Get dopant concentrations
                        d1_conc = safe_float_conversion(row.get('[D1]', 0), 0)
                        d2_conc = safe_float_conversion(row.get('[D2]', 0), 0)
                        b_prime_conc = safe_float_conversion(row.get("[B']", 0), 0)
                        total_dopant = d1_conc + d2_conc + b_prime_conc
                        
                        # Get temperatures
                        temps = DataParser.parse_bends_temperatures(row.get('T (PT), °C', ''))
                        num_transitions = len(temps)
                        
                        # Get tolerance factor if available (approximate from A/B)
                        a_elem = row.get('A', '')
                        b_elem = row.get('B', '')
                        
                        for t_pt in temps:
                            comp_data.append({
                                'T_PT': t_pt,
                                'num_transitions': num_transitions,
                                'total_dopant_B': total_dopant,
                                '[D1]': d1_conc,
                                '[D2]': d2_conc,
                                "[B']": b_prime_conc,
                                'A': a_elem,
                                'B': b_elem
                            })
                    
                    if comp_data:
                        comp_df = pd.DataFrame(comp_data)
                        comp_df = comp_df[comp_df['T_PT'] > 0]
                        
                        if len(comp_df) > 0:
                            # T(PT) vs total_dopant_B
                            st.subheader("Transition Temperature vs Total B-site Doping")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(comp_df['total_dopant_B'], comp_df['T_PT'], 
                                      alpha=0.7, s=60, c=ScientificVisualizer.COLORS['primary'],
                                      edgecolors='black', linewidth=0.5)
                            ax.set_xlabel('Total B-site Doping Concentration', fontsize=11, fontweight='bold')
                            ax.set_ylabel('Phase Transition Temperature (°C)', fontsize=11, fontweight='bold')
                            ax.set_title('T(PT) vs Total B-site Doping', fontsize=12, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            # T(PT) vs [B']
                            st.subheader("Transition Temperature vs B' Concentration")
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            ax2.scatter(comp_df["[B']"], comp_df['T_PT'], 
                                       alpha=0.7, s=60, c=ScientificVisualizer.COLORS['secondary'],
                                       edgecolors='black', linewidth=0.5)
                            ax2.set_xlabel("[B'] Concentration", fontsize=11, fontweight='bold')
                            ax2.set_ylabel('Phase Transition Temperature (°C)', fontsize=11, fontweight='bold')
                            ax2.set_title('T(PT) vs [B\']', fontsize=12, fontweight='bold')
                            ax2.grid(True, alpha=0.3)
                            st.pyplot(fig2)
                            
                            # Number of transitions vs total_dopant
                            st.subheader("Number of Phase Transitions vs Total Doping")
                            unique_comp = comp_df.groupby('total_dopant_B')['num_transitions'].first().reset_index()
                            fig3, ax3 = plt.subplots(figsize=(10, 6))
                            ax3.scatter(unique_comp['total_dopant_B'], unique_comp['num_transitions'],
                                       alpha=0.7, s=100, c=ScientificVisualizer.COLORS['quaternary'],
                                       edgecolors='black', linewidth=0.5)
                            ax3.set_xlabel('Total B-site Doping Concentration', fontsize=11, fontweight='bold')
                            ax3.set_ylabel('Number of Phase Transitions', fontsize=11, fontweight='bold')
                            ax3.set_title('Number of PT vs Total Doping', fontsize=12, fontweight='bold')
                            ax3.grid(True, alpha=0.3)
                            st.pyplot(fig3)
                            
                            # T(PT) boxplot by A-site
                            if 'A' in comp_df.columns:
                                st.subheader("Transition Temperature by A-site Element")
                                fig4, ax4 = plt.subplots(figsize=(10, 6))
                                valid_a = [a for a in comp_df['A'].unique() if a not in [None, '-', '']]
                                data_by_a = [comp_df[comp_df['A'] == a]['T_PT'].dropna().values for a in valid_a]
                                data_by_a = [d for d in data_by_a if len(d) > 0]
                                if data_by_a:
                                    bp = ax4.boxplot(data_by_a, labels=valid_a[:len(data_by_a)],
                                                    patch_artist=True, showmeans=True)
                                    for patch in bp['boxes']:
                                        patch.set_facecolor(ScientificVisualizer.COLORS['primary'])
                                        patch.set_alpha(0.7)
                                    ax4.set_xlabel('A-site Element', fontsize=11, fontweight='bold')
                                    ax4.set_ylabel('Phase Transition Temperature (°C)', fontsize=11, fontweight='bold')
                                    ax4.set_title('T(PT) by A-site', fontsize=12, fontweight='bold')
                                    ax4.grid(True, alpha=0.3)
                                    st.pyplot(fig4)
                    else:
                        st.info("No valid composition data extracted from phase transitions")
            else:
                st.info("Upload phase transition data to enable this analysis")
        
        # Tab 7: Machine Learning
        with tabs[6]:
            st.header("🧠 Machine Learning Models")
            
            # Select target for prediction
            ml_target_options = ['alpha_true', 'alpha_apparent', 'beta']
            ml_target = st.selectbox("Predict target property",
                                     [c for c in ml_target_options if c in df.columns],
                                     key="ml_target")
            
            # Select features
            all_numeric_features = [f for f in ScientificVisualizer.AVAILABLE_DESCRIPTORS if f in df.columns]
            selected_features = st.multiselect("Select features for ML", 
                                               all_numeric_features,
                                               default=all_numeric_features[:5] if len(all_numeric_features) > 5 else all_numeric_features)
            
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
                                    st.session_state.ml_model = rf_model
                                    st.session_state.ml_X_train = results['X_train']
                        else:
                            if len(X_scaled) < 10:
                                st.warning("Not enough valid data for training (need at least 10 samples)")
                            else:
                                st.warning("Insufficient variation in target variable")
            else:
                st.info("Select at least one feature and ensure sufficient data")
        
        # Tab 8: Clustering & PCA
        with tabs[7]:
            st.header("🎯 Clustering & Dimensionality Reduction")
            
            # Sub-tabs for clustering analysis
            cluster_tab1, cluster_tab2, cluster_tab3 = st.tabs(["📊 PCA & Projections", "🔬 K-Means Clustering", "📈 Cluster Quality"])
            
            with cluster_tab1:
                # Select features for PCA
                pca_features = st.multiselect("Select features for PCA",
                                             [f for f in ScientificVisualizer.AVAILABLE_DESCRIPTORS if f in df.columns],
                                             default=[f for f in ['tolerance_factor', 'chiB_avg', 'delta_chi_AB', 'rB_avg'] if f in df.columns])
                
                if len(pca_features) >= 2 and len(df) > 0:
                    # Prepare data
                    pca_df = df[pca_features].dropna()
                    for col in pca_features:
                        pca_df = pca_df[pca_df[col] != 0]
                    
                    if len(pca_df) >= 5:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(pca_df)
                        
                        # PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        st.subheader("PCA Projection")
                        target_for_color = 'alpha_true' if 'alpha_true' in df.columns else None
                        if target_for_color and target_for_color in df.columns:
                            y_colors = df.loc[pca_df.index, target_for_color].values
                            valid_mask = y_colors != 0
                            if np.any(valid_mask):
                                X_pca_valid = X_pca[valid_mask]
                                y_colors_valid = y_colors[valid_mask]
                                fig = ScientificVisualizer.plot_pca_2d(X_pca_valid, y_colors_valid,
                                                                       title='PCA of Composition Space')
                                st.pyplot(fig)
                        else:
                            fig = ScientificVisualizer.plot_pca_2d(X_pca, np.zeros(len(X_pca)),
                                                                   title='PCA of Composition Space')
                            st.pyplot(fig)
                        
                        st.write(f"Explained variance: PC1 = {pca.explained_variance_ratio_[0]:.2%}, PC2 = {pca.explained_variance_ratio_[1]:.2%}")
                        
                        # PCA loadings
                        st.subheader("PCA Loadings")
                        loadings_df = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_features)
                        st.dataframe(loadings_df, use_container_width=True)
                    else:
                        st.warning(f"Not enough valid data for PCA (need at least 5 samples, have {len(pca_df)})")
            
            with cluster_tab2:
                # Select features for clustering
                cluster_features = st.multiselect("Select features for clustering",
                                                 [f for f in ScientificVisualizer.AVAILABLE_DESCRIPTORS if f in df.columns],
                                                 default=[f for f in ['tolerance_factor', 'chiB_avg', 'delta_chi_AB'] if f in df.columns])
                
                if len(cluster_features) >= 2:
                    # Prepare data
                    cluster_df = df[cluster_features].dropna()
                    for col in cluster_features:
                        cluster_df = cluster_df[cluster_df[col] != 0]
                    
                    if len(cluster_df) >= 5:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(cluster_df)
                        
                        # PCA for visualization
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # K-Means clustering
                        n_clusters = st.slider("Number of clusters", 2, 10, 3, key="cluster_k")
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Visualize clusters
                        fig, ax = plt.subplots(figsize=(10, 7))
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
                        st.pyplot(fig)
                        
                        # Cluster characteristics
                        st.subheader("Cluster Characteristics")
                        cluster_df_copy = cluster_df.copy()
                        cluster_df_copy['Cluster'] = clusters
                        cluster_means = cluster_df_copy.groupby('Cluster')[cluster_features].mean()
                        st.dataframe(cluster_means, use_container_width=True)
                        
                        # Store for quality plots
                        st.session_state.cluster_X_scaled = X_scaled
                        st.session_state.cluster_labels = clusters
                        st.session_state.cluster_k = n_clusters
                    else:
                        st.warning(f"Not enough valid data for clustering (need at least 5 samples, have {len(cluster_df)})")
            
            with cluster_tab3:
                st.subheader("Cluster Quality Assessment")
                
                if 'cluster_X_scaled' in st.session_state and 'cluster_labels' in st.session_state:
                    X_scaled = st.session_state.cluster_X_scaled
                    labels = st.session_state.cluster_labels
                    k = st.session_state.cluster_k
                    
                    # Elbow plot
                    st.subheader("Elbow Curve for Optimal k")
                    fig_elbow = ScientificVisualizer.plot_elbow_curve(X_scaled, max_k=15)
                    st.pyplot(fig_elbow)
                    
                    # Silhouette plot for current k
                    st.subheader(f"Silhouette Analysis (k={k})")
                    fig_sil = ScientificVisualizer.plot_silhouette_analysis(X_scaled, k)
                    st.pyplot(fig_sil)
                    
                    # Dendrogram
                    if len(X_scaled) < 200 and len(X_scaled) >= 10:
                        st.subheader("Hierarchical Clustering Dendrogram")
                        fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6))
                        linkage_matrix = linkage(X_scaled[:min(100, len(X_scaled))], method='ward')
                        dendrogram(linkage_matrix, ax=ax_dendro, truncate_mode='lastp', p=30)
                        ax_dendro.set_title('Dendrogram (top 30 samples)', fontsize=12, fontweight='bold')
                        ax_dendro.set_xlabel('Sample index', fontsize=11)
                        ax_dendro.set_ylabel('Distance', fontsize=11)
                        st.pyplot(fig_dendro)
                else:
                    st.info("Run K-Means clustering first in the Clustering tab")
        
        # Tab 9: Advanced ML (SHAP & PDP)
        with tabs[8]:
            st.header("📉 Advanced ML Analysis")
            
            adv_tab1, adv_tab2 = st.tabs(["📊 SHAP Analysis", "📈 Partial Dependence Plots (PDP)"])
            
            with adv_tab1:
                if 'ml_results' in st.session_state and st.session_state.ml_results and 'error' not in st.session_state.ml_results:
                    st.subheader("SHAP Analysis for Model Interpretability")
                    
                    if st.button("Run SHAP Analysis", use_container_width=True):
                        with st.spinner("Computing SHAP values..."):
                            try:
                                results = st.session_state.ml_results
                                features = st.session_state.ml_features
                                
                                if 'Random Forest' in results['models'] and 'model' in results['models']['Random Forest']:
                                    rf_model = results['models']['Random Forest']['model']
                                    X_train = results['X_train']
                                    
                                    if len(X_train) > 0 and len(features) > 0:
                                        explainer = shap.TreeExplainer(rf_model)
                                        shap_values = explainer.shap_values(X_train[:min(100, len(X_train))])
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        shap.summary_plot(shap_values, X_train[:min(100, len(X_train))], 
                                                        feature_names=features, show=False)
                                        st.pyplot(fig)
                                        
                                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                                        shap.summary_plot(shap_values, X_train[:min(100, len(X_train))], 
                                                         feature_names=features, plot_type="bar", show=False)
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
            
            with adv_tab2:
                if 'ml_model' in st.session_state and 'ml_X_train' in st.session_state and 'ml_features' in st.session_state:
                    st.subheader("Partial Dependence Plots (PDP)")
                    st.markdown("PDP shows how the predicted target changes with a single feature, averaging out other features.")
                    
                    model = st.session_state.ml_model
                    X_train = st.session_state.ml_X_train
                    features = st.session_state.ml_features
                    
                    if len(features) > 0:
                        selected_pdp_feature = st.selectbox("Select feature for PDP:", features)
                        
                        if st.button(f"Generate PDP for {selected_pdp_feature}", use_container_width=True):
                            with st.spinner("Computing partial dependence..."):
                                try:
                                    from sklearn.inspection import PartialDependenceDisplay
                                    
                                    fig, ax = plt.subplots(figsize=(9, 6))
                                    PartialDependenceDisplay.from_estimator(
                                        model, X_train, [features.index(selected_pdp_feature)],
                                        feature_names=features, ax=ax, grid_resolution=50
                                    )
                                    ax.set_title(f'Partial Dependence: {selected_pdp_feature}', fontsize=12, fontweight='bold')
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"PDP computation failed: {str(e)[:100]}")
                    else:
                        st.warning("No features available for PDP")
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
                    alpha_vals = df[alpha_col][df[alpha_col] != 0]
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
                    beta_vals = df[beta_col][df[beta_col] != 0]
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
                        temp_df = df[numeric_cols].copy()
                        for col in numeric_cols:
                            temp_df = temp_df[temp_df[col] != 0]
                        if len(temp_df) > 5:
                            corrs = temp_df.corr()[alpha_col].abs().sort_values(ascending=False)
                            top_corrs = corrs.head(10)
                            
                            report.append(f"\n### Top 10 Correlations with {alpha_col}")
                            for feat, corr in top_corrs.items():
                                if feat != alpha_col and not pd.isna(corr):
                                    report.append(f"- {feat}: {corr:.3f}")
                
                report.append(f"\n## Phase Transition Summary")
                if st.session_state.phase_df is not None:
                    phase_df_local = st.session_state.phase_df
                    report.append(f"- Total phase transition records: {len(phase_df_local)}")
                    
                    # Count transitions with temperatures
                    temp_count_local = 0
                    for _, row in phase_df_local.iterrows():
                        temps = DataParser.parse_bends_temperatures(row.get('T (PT), °C', ''))
                        if temps:
                            temp_count_local += 1
                    report.append(f"- Records with temperature data: {temp_count_local}")
                    
                    # Unique symmetries
                    all_sym_local = []
                    for _, row in phase_df_local.iterrows():
                        sym = row.get('Symmetry', '')
                        if sym and sym != '-':
                            all_sym_local.extend(sym.split(';'))
                    report.append(f"- Unique symmetries: {len(set(all_sym_local))}")
                else:
                    report.append(f"- No phase transition data loaded")
                
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
