"""
Models & Benchmark Module - Chimiométrie 2027 Challenge
Implémentation et comparaison des modèles unimodaux et multimodaux :
- PLS-DA (Partial Least Squares Discriminant Analysis)
- SVM (Linéaire et RBF)
- Random Forest
- Fusion précoce (Early Fusion) avec mise à l'échelle des blocs
- Fusion tardive (Late Fusion / Soft Voting)
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import ChemometricPreprocessor, BlockScaler, clean_physical_outliers, extract_modalities

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")


class PLSDA(BaseEstimator, ClassifierMixin):
    """
    Partial Least Squares Discriminant Analysis (PLS-DA).
    Régression PLS standard sur la matrice indicatrice (one-hot) des classes.
    """
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pls = PLSRegression(n_components=self.n_components)
        Y_onehot = (y[:, None] == self.classes_).astype(float)
        self.pls.fit(X, Y_onehot)
        return self
        
    def predict(self, X):
        Y_pred = self.pls.predict(X)
        return self.classes_[np.argmax(Y_pred, axis=1)]
        
    def predict_proba(self, X):
        Y_pred = self.pls.predict(X)
        # Softmax pour convertir les projections PLS continues en pseudo-probabilités
        exp_Y = np.exp(Y_pred - np.max(Y_pred, axis=1, keepdims=True))
        return exp_Y / np.sum(exp_Y, axis=1, keepdims=True)


def evaluate_model_cv(model, X, y, cv=5, name="Modèle"):
    """Évalue un modèle avec validation croisée stratifiée à 5 plis."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        scores.append(acc)
        
    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    print(f"[{name:<35}] Accuracy CV ({cv}-Fold) : {mean_acc:6.2f} % (± {std_acc:4.2f} %)")
    return mean_acc, std_acc, model


def run_comprehensive_benchmark():
    """Exécute le benchmark complet sur toutes les configurations et modalités."""
    print("="*75)
    print("LANCEMENT DU BENCHMARK CHIMIOMÉTRIE 2027")
    print("="*75)
    
    # 1. Chargement et nettoyage des données
    df_raw = pd.read_excel(os.path.join(DATA_DIR, "Calibration_set_challenge.xlsx"))
    df = clean_physical_outliers(df_raw, is_train=True)
    modalities = extract_modalities(df)
    
    y = modalities['y']
    X_nir = modalities['X_nir']
    X_morpho = modalities['X_morpho']
    X_color = modalities['X_color']
    X_phys = np.hstack([X_morpho, X_color])
    
    # Prétraitements chimiométriques pour NIR
    prep_snv_sg = ChemometricPreprocessor(apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1)
    X_nir_prep = prep_snv_sg.fit_transform(X_nir)
    
    # Standardisation morpho + couleur
    scaler_phys = StandardScaler()
    X_phys_scaled = scaler_phys.fit_transform(X_phys)
    
    results = []
    
    print("\n--- 0. REPRODUCTION EXACTE DU MODÈLE DE MARCOS ---")
    X_all_raw = np.hstack([X_nir, X_phys])
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        X_all_raw, y, test_size=0.2, stratify=y, random_state=42
    )
    sc_m = StandardScaler()
    X_tr_m_sc = sc_m.fit_transform(X_tr_m)
    X_te_m_sc = sc_m.transform(X_te_m)
    svm_marcos = SVC(kernel='linear', C=1.0, random_state=42)
    svm_marcos.fit(X_tr_m_sc, y_tr_m)
    preds_m = svm_marcos.predict(X_te_m_sc)
    acc_m = accuracy_score(y_te_m, preds_m) * 100
    n_correct = int(np.sum(preds_m == y_te_m))
    print(f"[Baseline Marcos (Holdout 20%)] Accuracy : {acc_m:.2f} % ({n_correct}/350 corrects)")
    
    # Évaluation de la baseline Marcos en 5-Fold CV pour comparaison équitable
    acc_m_cv, std_m_cv, _ = evaluate_model_cv(
        SVC(kernel='linear', C=1.0, random_state=42),
        sc_m.fit_transform(X_all_raw), y, name="Baseline Marcos : SVM Lin (CV 5-Fold)"
    )
    results.append({"Modalité": "Toutes (Brut)", "Prétraitement": "StandardScaler", "Modèle": "SVM Linéaire (Marcos)", "CV_Accuracy": acc_m_cv, "CV_Std": std_m_cv})

    print("\n--- 1. MODALITÉ UNIQUE : SPECTRES NIR ---")
    # A. SVM Linéaire brut (configuration Marcos)
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='linear', C=1.0, random_state=42),
        X_nir, y, name="NIR Brut : SVM Linéaire (Base)"
    )
    results.append({"Modalité": "NIR", "Prétraitement": "Brut", "Modèle": "SVM Linéaire", "CV_Accuracy": acc, "CV_Std": std})
    
    # B. SVM Linéaire avec SNV + Savitzky-Golay (1ère dérivée)
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='linear', C=1.0, random_state=42),
        X_nir_prep, y, name="NIR SNV+SG-1d : SVM Linéaire"
    )
    results.append({"Modalité": "NIR", "Prétraitement": "SNV + SG 1ère dérivée", "Modèle": "SVM Linéaire", "CV_Accuracy": acc, "CV_Std": std})
    
    # C. SVM RBF avec SNV + Savitzky-Golay
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42),
        X_nir_prep, y, name="NIR SNV+SG-1d : SVM RBF (C=10)"
    )
    results.append({"Modalité": "NIR", "Prétraitement": "SNV + SG 1ère dérivée", "Modèle": "SVM RBF", "CV_Accuracy": acc, "CV_Std": std})
    
    # D. PLS-DA (Standard Chimiométrique)
    for n_comp in [10, 15, 20]:
        acc, std, _ = evaluate_model_cv(
            PLSDA(n_components=n_comp),
            X_nir_prep, y, name=f"NIR SNV+SG-1d : PLS-DA ({n_comp} LV)"
        )
        results.append({"Modalité": "NIR", "Prétraitement": "SNV + SG 1ère dérivée", "Modèle": f"PLS-DA ({n_comp} LV)", "CV_Accuracy": acc, "CV_Std": std})

    # E. Random Forest
    acc, std, _ = evaluate_model_cv(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        X_nir_prep, y, name="NIR SNV+SG-1d : Random Forest"
    )
    results.append({"Modalité": "NIR", "Prétraitement": "SNV + SG 1ère dérivée", "Modèle": "Random Forest", "CV_Accuracy": acc, "CV_Std": std})

    print("\n--- 2. MODALITÉS PHYSIQUES (MORPHOLOGIE & COULEUR) ---")
    # A. Morphologie seule
    scaler_m = StandardScaler()
    X_m_scaled = scaler_m.fit_transform(X_morpho)
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='rbf', C=10.0, random_state=42),
        X_m_scaled, y, name="Morphologie seule : SVM RBF"
    )
    results.append({"Modalité": "Morphologie", "Prétraitement": "StandardScaler", "Modèle": "SVM RBF", "CV_Accuracy": acc, "CV_Std": std})

    # B. Couleur seule
    scaler_c = StandardScaler()
    X_c_scaled = scaler_c.fit_transform(X_color)
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='rbf', C=10.0, random_state=42),
        X_c_scaled, y, name="Couleur seule : SVM RBF"
    )
    results.append({"Modalité": "Couleur", "Prétraitement": "StandardScaler", "Modèle": "SVM RBF", "CV_Accuracy": acc, "CV_Std": std})

    # C. Morphologie + Couleur
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='rbf', C=10.0, random_state=42),
        X_phys_scaled, y, name="Morpho + Couleur : SVM RBF"
    )
    results.append({"Modalité": "Morpho + Couleur", "Prétraitement": "StandardScaler", "Modèle": "SVM RBF", "CV_Accuracy": acc, "CV_Std": std})

    print("\n--- 3. FUSION MULTIMODALE PRÉCOCE (EARLY FUSION) ---")
    # Concaténation simple vs Block Scaling
    # Simple (Non équilibré)
    X_early_raw = np.hstack([X_nir_prep, X_phys_scaled])
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='rbf', C=10.0, random_state=42),
        X_early_raw, y, name="Early Fusion brut : SVM RBF"
    )
    results.append({"Modalité": "Early Fusion (Simple)", "Prétraitement": "SNV + Standard", "Modèle": "SVM RBF", "CV_Accuracy": acc, "CV_Std": std})

    # Block Scaled (Chimiométrique)
    block_scaler = BlockScaler()
    blocks = {
        'nir': X_nir_prep,
        'morpho': X_morpho,
        'color': X_color
    }
    block_scaler.fit(blocks)
    X_early_blocked = block_scaler.transform(blocks)
    
    acc, std, _ = evaluate_model_cv(
        SVC(kernel='linear', C=1.0, random_state=42),
        X_early_blocked, y, name="Early Fusion Block-Scaled : SVM Linéaire"
    )
    results.append({"Modalité": "Early Fusion (Block-Scaled)", "Prétraitement": "Block Scaling", "Modèle": "SVM Linéaire", "CV_Accuracy": acc, "CV_Std": std})

    acc, std, best_svm_rbf = evaluate_model_cv(
        SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42),
        X_early_blocked, y, name="Early Fusion Block-Scaled : SVM RBF"
    )
    results.append({"Modalité": "Early Fusion (Block-Scaled)", "Prétraitement": "Block Scaling", "Modèle": "SVM RBF", "CV_Accuracy": acc, "CV_Std": std})

    print("\n--- 4. FUSION TARDIVE (LATE FUSION / SOFT VOTING) ---")
    # Validation croisée manuelle pour Soft Voting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    late_scores = []
    y_true_all = []
    y_pred_all = []
    
    for train_idx, val_idx in skf.split(X_nir_prep, y):
        # Modèle 1 : NIR SVM RBF calibré (avec probabilités)
        m_nir = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
        m_nir.fit(X_nir_prep[train_idx], y[train_idx])
        prob_nir = m_nir.predict_proba(X_nir_prep[val_idx])
        
        # Modèle 2 : Morpho+Couleur SVM RBF calibré
        m_phys = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
        m_phys.fit(X_phys_scaled[train_idx], y[train_idx])
        prob_phys = m_phys.predict_proba(X_phys_scaled[val_idx])
        
        # Combinaison pondérée (0.65 NIR + 0.35 Physique)
        prob_fusion = 0.65 * prob_nir + 0.35 * prob_phys
        preds_fusion = m_nir.classes_[np.argmax(prob_fusion, axis=1)]
        
        late_scores.append(accuracy_score(y[val_idx], preds_fusion))
        y_true_all.extend(y[val_idx])
        y_pred_all.extend(preds_fusion)
        
    acc_late = np.mean(late_scores) * 100
    std_late = np.std(late_scores) * 100
    print(f"[{'Late Fusion : Soft Voting (NIR+Phys)':<35}] Accuracy CV (5-Fold) : {acc_late:6.2f} % (± {std_late:4.2f} %)")
    results.append({"Modalité": "Late Fusion (Soft Voting)", "Prétraitement": "Pondéré 65/35", "Modèle": "Ensemble SVMs", "CV_Accuracy": acc_late, "CV_Std": std_late})

    # Matrice de confusion pour la Late Fusion
    classes = sorted(np.unique(y))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Matrice de confusion - Late Fusion (Accuracy = {acc_late:.1f} %)")
    plt.xlabel("Variété Prédite")
    plt.ylabel("Variété Réelle")
    plt.tight_layout()
    out_cm = os.path.join(FIG_DIR, "fig8_confusion_matrix_best.png")
    plt.savefig(out_cm, dpi=300)
    plt.close()
    print(f"\nMatrice de confusion sauvegardée : {out_cm}")

    # Enregistrer le tableau comparatif
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by="CV_Accuracy", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(FIG_DIR, "benchmark_results.csv")
    res_df.to_csv(csv_path, index=False)
    print("\n" + "="*75)
    print("RÉCAPITULATIF DES PERFORMANCES DU BENCHMARK")
    print("="*75)
    print(res_df.to_string(index=False))
    print("="*75)
    return res_df


if __name__ == "__main__":
    run_comprehensive_benchmark()
