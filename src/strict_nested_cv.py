"""
Strict Zero-Leakage & Nested Cross-Validation Verification
Répond aux questions méthodologiques de Marcos :
1. Imputation sans label (label-free median imputation sur le train fold uniquement)
2. Zero data leakage : BlockScaler et scalers ajustés à 100% à l'intérieur de chaque pli train
3. Nested Cross-Validation (Outer 5-fold, Inner 3-fold GridSearchCV)
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import ChemometricPreprocessor, BlockScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def run_strict_nested_cv():
    print("="*70)
    print("STRICT ZERO-LEAKAGE & NESTED CROSS-VALIDATION EXPERIMENT")
    print("="*70)
    
    # 1. Charger les données brutes sans AUCUN prétraitement préalable
    df_cal = pd.read_excel(os.path.join(DATA_DIR, "Calibration_set_challenge.xlsx"))
    
    morpho_cols = ['volume [mm^3]', 'length [mm]', 'width [mm]', 'thickness [mm]']
    color_cols = ['L', 'A', 'B']
    meta_cols = ['ID', 'Batch', 'Variety', 'ImageName']
    nir_cols = [c for c in df_cal.columns if c not in set(meta_cols + morpho_cols + color_cols)]
    
    X_nir_raw = df_cal[nir_cols].values.astype(np.float64)
    X_morpho_raw = df_cal[morpho_cols].values.astype(np.float64)
    X_color_raw = df_cal[color_cols].values.astype(np.float64)
    y = np.array(df_cal['Variety'].astype(str))
    
    print(f"Total calibration samples: {len(y)}")
    
    # 2. Configuration Nested CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    outer_scores_linear = []
    outer_scores_rbf_nested = []
    outer_scores_ensemble = []
    
    # SNV + Savitzky-Golay est une transformation ligne par ligne (par échantillon physique individuel)
    # mathématiquement indépendante des autres échantillons, mais nous l'appliquons proprement :
    nir_preproc = ChemometricPreprocessor(apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1)
    X_nir_filtered = nir_preproc.transform(X_nir_raw)
    
    print("\nStarting Outer 5-Fold Evaluation...")
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_nir_raw, y), 1):
        # A. Split strictly train and test
        X_nir_tr, X_nir_te = X_nir_filtered[train_idx], X_nir_filtered[test_idx]
        X_morpho_tr, X_morpho_te = X_morpho_raw[train_idx].copy(), X_morpho_raw[test_idx].copy()
        X_color_tr, X_color_te = X_color_raw[train_idx].copy(), X_color_raw[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # B. Imputation strictly on train fold WITHOUT knowing the labels (Label-free imputation)
        for col_idx in range(X_morpho_tr.shape[1]):
            # Detect invalid in train
            invalid_tr = X_morpho_tr[:, col_idx] <= 0
            if np.any(invalid_tr):
                valid_median = np.median(X_morpho_tr[~invalid_tr, col_idx])
                X_morpho_tr[invalid_tr, col_idx] = valid_median
            else:
                valid_median = np.median(X_morpho_tr[:, col_idx])
            
            # Apply train median to test if invalid
            invalid_te = X_morpho_te[:, col_idx] <= 0
            if np.any(invalid_te):
                X_morpho_te[invalid_te, col_idx] = valid_median
                
        # C. BlockScaler fitted strictly on training fold only
        blocks_tr = {'nir': X_nir_tr, 'morpho': X_morpho_tr, 'color': X_color_tr}
        blocks_te = {'nir': X_nir_te, 'morpho': X_morpho_te, 'color': X_color_te}
        
        scaler = BlockScaler()
        scaler.fit(blocks_tr)
        X_tr_fused = scaler.transform(blocks_tr)
        X_te_fused = scaler.transform(blocks_te)
        
        # D. Model 1: Linear SVM baseline (C=1)
        svm_lin = SVC(kernel='linear', C=1.0, random_state=42)
        svm_lin.fit(X_tr_fused, y_tr)
        acc_lin = accuracy_score(y_te, svm_lin.predict(X_te_fused))
        outer_scores_linear.append(acc_lin)
        
        # E. Model 2: Nested CV for RBF SVM
        param_grid = {'C': [10.0, 25.0, 50.0], 'gamma': [0.01, 0.02, 0.05]}
        grid = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=42),
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_tr_fused, y_tr)
        best_rbf = grid.best_estimator_
        acc_rbf = accuracy_score(y_te, best_rbf.predict(X_te_fused))
        outer_scores_rbf_nested.append(acc_rbf)
        
        # F. Model 3: Multimodal Ensemble with ExtraTrees
        et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        et.fit(X_tr_fused, y_tr)
        
        svm_lin_prob = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        svm_lin_prob.fit(X_tr_fused, y_tr)
        
        # Soft voting
        p_rbf = best_rbf.predict_proba(X_te_fused)
        p_lin = svm_lin_prob.predict_proba(X_te_fused)
        p_et = et.predict_proba(X_te_fused)
        
        p_ensemble = (3.0 * p_rbf + 2.0 * p_lin + 1.0 * p_et) / 6.0
        y_pred_ens = best_rbf.classes_[np.argmax(p_ensemble, axis=1)]
        acc_ens = accuracy_score(y_te, y_pred_ens)
        outer_scores_ensemble.append(acc_ens)
        
        print(f"Fold {fold} - Best Params: {grid.best_params_} | Lin SVM: {acc_lin*100:.2f}% | Nested RBF: {acc_rbf*100:.2f}% | Ensemble: {acc_ens*100:.2f}%")
        
    print("\n" + "="*70)
    print("FINAL STRICT ZERO-LEAKAGE NESTED CV RESULTS")
    print("="*70)
    print(f"Linear SVM (Block-Scaled) : {np.mean(outer_scores_linear)*100:.2f}% ± {np.std(outer_scores_linear)*100:.2f}%")
    print(f"Nested RBF SVM            : {np.mean(outer_scores_rbf_nested)*100:.2f}% ± {np.std(outer_scores_rbf_nested)*100:.2f}%")
    print(f"Nested Multimodal Ensemble: {np.mean(outer_scores_ensemble)*100:.2f}% ± {np.std(outer_scores_ensemble)*100:.2f}%")
    print("="*70)

if __name__ == '__main__':
    run_strict_nested_cv()

