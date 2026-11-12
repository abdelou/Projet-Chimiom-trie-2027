"""
Preprocessing Module - Chimiométrie 2027 Challenge
Outils de prétraitements chimiométriques pour les spectres NIR et les données physiques (morphologie et couleur).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class ChemometricPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformateur de prétraitement pour spectres NIR chimiométriques.
    Supporte :
    - SNV (Standard Normal Variate)
    - Savitzky-Golay (lissage, dérivée 1ère, dérivée 2nde)
    - Centrage / Réduction
    """
    def __init__(self, apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1):
        self.apply_snv = apply_snv
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.sg_deriv = sg_deriv
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = np.array(X, dtype=np.float64, copy=True)
        
        # 1. SNV (Standard Normal Variate) par spectre (axe des lignes)
        if self.apply_snv:
            mean = np.mean(X_out, axis=1, keepdims=True)
            std = np.std(X_out, axis=1, keepdims=True)
            std[std == 0] = 1.0  # éviter division par zéro
            X_out = (X_out - mean) / std
            
        # 2. Filtre de Savitzky-Golay
        if self.sg_window is not None and self.sg_window > self.sg_poly:
            X_out = savgol_filter(
                X_out,
                window_length=self.sg_window,
                polyorder=self.sg_poly,
                deriv=self.sg_deriv,
                axis=1
            )
            
        return X_out


class BlockScaler(BaseEstimator, TransformerMixin):
    """
    Normalisation par bloc pour la fusion précoce (Early Fusion).
    Chaque bloc de variables est divisé par la racine carrée de sa variance totale
    (norme de Frobenius), garantissant qu'aucun bloc (ex: 512 longueurs d'onde NIR)
    n'écrase artificiellement un autre bloc (ex: 4 variables morphologiques).
    """
    def __init__(self):
        self.scalers_ = {}
        self.block_weights_ = {}
        
    def fit(self, blocks_dict, y=None):
        """
        blocks_dict : dict de la forme {'nom_bloc': np.ndarray (n_samples, n_features)}
        """
        self.scalers_ = {}
        self.block_weights_ = {}
        for name, X_b in blocks_dict.items():
            sc = StandardScaler()
            X_scaled = sc.fit_transform(X_b)
            # Poids du bloc = racine carrée de la variance totale
            frob_norm = np.linalg.norm(X_scaled) / np.sqrt(X_scaled.shape[0])
            self.scalers_[name] = sc
            self.block_weights_[name] = frob_norm if frob_norm > 0 else 1.0
        return self
        
    def transform(self, blocks_dict):
        scaled_blocks = []
        for name, X_b in blocks_dict.items():
            X_scaled = self.scalers_[name].transform(X_b)
            # Division par le poids du bloc
            scaled_blocks.append(X_scaled / self.block_weights_[name])
        return np.hstack(scaled_blocks)


def clean_physical_outliers(df, is_train=True):
    """
    Corrige les anomalies physiques détectées lors de l'EDA.
    (Notamment la ligne Atoll avec largeur < 0 et volume <= 0).
    """
    df_clean = df.copy()
    morpho_cols = ['volume [mm^3]', 'length [mm]', 'width [mm]', 'thickness [mm]']
    
    # Remplacer les valeurs physiques <= 0 par NaN puis imputer par la médiane de la variété (train) ou médiane globale (test)
    for col in morpho_cols:
        invalid = df_clean[col] <= 0
        if invalid.any():
            print(f"Correction de {invalid.sum()} valeur(s) aberrante(s) (<= 0) dans '{col}'.")
            df_clean.loc[invalid, col] = np.nan
            if is_train and 'Variety' in df_clean.columns:
                df_clean[col] = df_clean.groupby('Variety')[col].transform(lambda s: s.fillna(s.median()))
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
    return df_clean


def extract_modalities(df):
    """
    Extrait et sépare les blocs :
    - y : cibles (Varieties, si présentes)
    - meta : Batch, ImageName, ID
    - X_nir : spectres NIR (512 variables)
    - X_morpho : mesures morphologiques (4 variables)
    - X_color : composantes couleur CIELAB (3 variables)
    - wavelengths : longueurs d'onde en float
    """
    meta_cols = [c for c in ['ID', 'Batch', 'Variety', 'ImageName'] if c in df.columns]
    morpho_cols = ['volume [mm^3]', 'length [mm]', 'width [mm]', 'thickness [mm]']
    color_cols = ['L', 'A', 'B']
    
    non_nir = set(meta_cols + morpho_cols + color_cols)
    nir_cols = [c for c in df.columns if c not in non_nir]
    wavelengths = np.array([float(c) for c in nir_cols])
    
    y = np.array(df['Variety'].astype(str)) if 'Variety' in df.columns else None
    meta = df[meta_cols] if len(meta_cols) > 0 else None
    X_nir = df[nir_cols].values.astype(np.float64)
    X_morpho = df[morpho_cols].values.astype(np.float64)
    X_color = df[color_cols].values.astype(np.float64)
    
    return {
        'y': y,
        'meta': meta,
        'X_nir': X_nir,
        'X_morpho': X_morpho,
        'X_color': X_color,
        'nir_cols': nir_cols,
        'morpho_cols': morpho_cols,
        'color_cols': color_cols,
        'wavelengths': wavelengths
    }
