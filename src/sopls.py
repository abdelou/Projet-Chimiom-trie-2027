"""
Sequential and Orthogonalized PLS (SO-PLS) Classifier
Implémentation chimiométrique multivoie de SO-PLS pour la classification variétale.
Références : Næs et al. (2013), Rinde et al. (2022).
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from preprocessing import ChemometricPreprocessor, clean_physical_outliers, extract_modalities

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class SOPLSClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificateur SO-PLS (Sequential and Orthogonalized Partial Least Squares).
    
    Architecture à deux blocs :
    - Bloc 1 (X1) : Spectres NIR chimiométriques (SNV + Savitzky-Golay)
    - Bloc 2 (X2) : Données physiques (Morphologie + Couleur CIELAB)
    
    Procédure :
    1. Régression PLS de X1 sur la matrice indicatrice Y -> Calcul des scores T1 et prédictions Y1.
    2. Orthogonalisation de X2 par rapport aux scores T1 :
       X2_orth = (I - T1 (T1' T1)^-1 T1') X2
    3. Régression PLS de X2_orth sur les résidus Y_res = Y - Y1 -> Prédictions Y2.
    4. Prédiction combinée : Y_pred = Y1 + Y2
    """
    def __init__(self, n_comp_x1=15, n_comp_x2=4):
        self.n_comp_x1 = n_comp_x1
        self.n_comp_x2 = n_comp_x2
        self.pls1 = None
        self.pls2 = None
        self.classes_ = None
        
    def _orthogonalize(self, X, T):
        """Orthogonalise les colonnes de X par rapport aux colonnes de T (projecteur orthogonal)."""
        # P_T = T (T^T T)^(-1) T^T
        # X_orth = X - T (T^T T)^(-1) T^T X
        # Calcul stable via pseudo-inverse ou moindres carrés
        beta, _, _, _ = np.linalg.lstsq(T, X, rcond=None)
        X_proj = T @ beta
        return X - X_proj, beta
        
    def fit(self, X_blocks, y):
        """
        X_blocks : tuple (X1, X2)
        y : array 1D des classes
        """
        X1, X2 = X_blocks
        self.classes_ = np.unique(y)
        Y_onehot = (y[:, None] == self.classes_).astype(float)
        
        # 1. Ajustement PLS sur le Bloc 1 (NIR)
        self.pls1 = PLSRegression(n_components=self.n_comp_x1)
        self.pls1.fit(X1, Y_onehot)
        
        # Extraction des scores T1 et des premières prédictions
        T1 = self.pls1.x_scores_
        Y1_pred = self.pls1.predict(X1)
        Y_res = Y_onehot - Y1_pred
        
        # 2. Orthogonalisation du Bloc 2 par rapport à T1
        X2_orth, self.orth_beta_ = self._orthogonalize(X2, T1)
        
        # 3. Ajustement PLS sur le Bloc 2 orthogonalisé pour modéliser les résidus
        n_comp_2 = min(self.n_comp_x2, X2_orth.shape[1])
        self.pls2 = PLSRegression(n_components=n_comp_2)
        self.pls2.fit(X2_orth, Y_res)
        
        return self
        
    def predict_continuous(self, X_blocks):
        X1, X2 = X_blocks
        # Étape 1 : Prédiction Bloc 1
        Y1_pred = self.pls1.predict(X1)
        
        # Projection de X1 pour estimer T1 sur de nouvelles données
        # T1_new = (X1 - X1_mean) @ W_star
        T1_new = (X1 - self.pls1._x_mean) @ self.pls1.x_rotations_
        
        # Étape 2 : Orthogonalisation de X2_new avec la matrice beta apprise
        X2_orth_new = X2 - (T1_new @ self.orth_beta_)
        
        # Étape 3 : Prédiction résiduelle Bloc 2
        Y2_pred = self.pls2.predict(X2_orth_new)
        
        return Y1_pred + Y2_pred
        
    def predict(self, X_blocks):
        Y_total = self.predict_continuous(X_blocks)
        return self.classes_[np.argmax(Y_total, axis=1)]
        
    def predict_proba(self, X_blocks):
        Y_total = self.predict_continuous(X_blocks)
        exp_Y = np.exp(Y_total - np.max(Y_total, axis=1, keepdims=True))
        return exp_Y / np.sum(exp_Y, axis=1, keepdims=True)


def benchmark_sopls():
    """Évalue SO-PLS sur différentes combinaisons de composantes latentes."""
    print("="*65)
    print("BENCHMARK MULTIVOIE : SEQUENTIAL AND ORTHOGONALIZED PLS (SO-PLS)")
    print("="*65)
    
    df_raw = pd.read_excel(os.path.join(DATA_DIR, "Calibration_set_challenge.xlsx"))
    df = clean_physical_outliers(df_raw, is_train=True)
    modalities = extract_modalities(df)
    
    y = modalities['y']
    X_nir = modalities['X_nir']
    X_phys = np.hstack([modalities['X_morpho'], modalities['X_color']])
    
    # Prétraitement chimiométrique
    prep = ChemometricPreprocessor(apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1)
    X_nir_prep = prep.fit_transform(X_nir)
    
    # Standardisation physique
    X_phys_sc = (X_phys - np.mean(X_phys, axis=0)) / np.std(X_phys, axis=0)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_configs = [
        (10, 3),
        (15, 3),
        (15, 5),
        (20, 3),
        (20, 5),
        (22, 5)
    ]
    
    for n1, n2 in grid_configs:
        scores = []
        for train_idx, val_idx in skf.split(X_nir_prep, y):
            X1_tr, X1_va = X_nir_prep[train_idx], X_nir_prep[val_idx]
            X2_tr, X2_va = X_phys_sc[train_idx], X_phys_sc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            sopls = SOPLSClassifier(n_comp_x1=n1, n_comp_x2=n2)
            sopls.fit((X1_tr, X2_tr), y_tr)
            preds = sopls.predict((X1_va, X2_va))
            scores.append(accuracy_score(y_va, preds))
            
        print(f"SO-PLS (NIR: {n1:2d} LV | Phys: {n2:1d} LV) -> Accuracy CV : {np.mean(scores)*100:6.2f} % (± {np.std(scores)*100:4.2f} %)")


if __name__ == "__main__":
    benchmark_sopls()
