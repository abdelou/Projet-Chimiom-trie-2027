"""
Optimization & Final Predictions - Chimiométrie 2027 Challenge
Optimisation fine des hyperparamètres, fusion multimodale avancée,
évaluation complète et génération du fichier de soumission My_contribution.xlsx.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import ChemometricPreprocessor, BlockScaler, clean_physical_outliers, extract_modalities
from models import PLSDA

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")


def prepare_datasets():
    """Charge et prétraite les jeux de données de calibration et de prédiction."""
    print("Chargement des données de calibration et de prédiction...")
    df_cal_raw = pd.read_excel(os.path.join(DATA_DIR, "Calibration_set_challenge.xlsx"))
    df_pred_raw = pd.read_excel(os.path.join(DATA_DIR, "Prediction_set_challenge.xlsx"))
    
    # Nettoyage des anomalies physiques
    df_cal = clean_physical_outliers(df_cal_raw, is_train=True)
    df_pred = clean_physical_outliers(df_pred_raw, is_train=False)
    
    cal_mod = extract_modalities(df_cal)
    pred_mod = extract_modalities(df_pred)
    
    # Prétraitement chimiométrique des spectres NIR (SNV + Savitzky-Golay 1ère dérivée)
    prep_nir = ChemometricPreprocessor(apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1)
    X_cal_nir = prep_nir.fit_transform(cal_mod['X_nir'])
    X_pred_nir = prep_nir.transform(pred_mod['X_nir'])
    
    # Construction des blocs pour le Block Scaling
    blocks_cal = {
        'nir': X_cal_nir,
        'morpho': cal_mod['X_morpho'],
        'color': cal_mod['X_color']
    }
    blocks_pred = {
        'nir': X_pred_nir,
        'morpho': pred_mod['X_morpho'],
        'color': pred_mod['X_color']
    }
    
    block_scaler = BlockScaler()
    block_scaler.fit(blocks_cal)
    X_cal_blocked = block_scaler.transform(blocks_cal)
    X_pred_blocked = block_scaler.transform(blocks_pred)
    
    y_cal = cal_mod['y']
    pred_ids = df_pred['ID'].values
    
    print(f"Jeu de calibration : {X_cal_blocked.shape[0]} échantillons, {X_cal_blocked.shape[1]} variables après fusion par blocs.")
    print(f"Jeu de prédiction   : {X_pred_blocked.shape[0]} échantillons à classifier.")
    
    return {
        'X_cal_blocked': X_cal_blocked,
        'y_cal': y_cal,
        'X_pred_blocked': X_pred_blocked,
        'pred_ids': pred_ids,
        'cal_mod': cal_mod,
        'pred_mod': pred_mod,
        'X_cal_nir': X_cal_nir,
        'X_pred_nir': X_pred_nir
    }


def optimize_svm(data):
    """Recherche sur grille pour trouver les meilleurs hyperparamètres SVM RBF."""
    print("\n" + "="*60)
    print("OPTIMISATION DES HYPERPARAMÈTRES (GRID SEARCH SVM RBF)")
    print("="*60)
    
    X = data['X_cal_blocked']
    y = data['y_cal']
    
    param_grid = {
        'C': [1.0, 5.0, 10.0, 15.0, 25.0, 50.0],
        'gamma': ['scale', 0.005, 0.01, 0.02, 0.05]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42),
        param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    print(f"Meilleurs paramètres trouvés : {grid.best_params_}")
    print(f"Meilleure Accuracy CV (5-Fold) : {grid.best_score_ * 100:.2f} %")
    return grid.best_estimator_


def test_ensemble_voting(data, best_svm):
    """Évalue un Voting Classifier combinant SVM RBF, SVM Linéaire et Extra-Trees."""
    print("\n" + "="*60)
    print("ÉVALUATION D'UN ENSEMBLE MULTIMODAL (VOTING CLASSIFIER)")
    print("="*60)
    
    X = data['X_cal_blocked']
    y = data['y_cal']
    
    clf1 = best_svm
    clf2 = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf3 = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[('svm_rbf', clf1), ('svm_lin', clf2), ('et', clf3)],
        voting='soft',
        weights=[3, 2, 1]
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    y_true_all = []
    y_pred_all = []
    
    for train_idx, val_idx in skf.split(X, y):
        ensemble.fit(X[train_idx], y[train_idx])
        preds = ensemble.predict(X[val_idx])
        scores.append(accuracy_score(y[val_idx], preds))
        y_true_all.extend(y[val_idx])
        y_pred_all.extend(preds)
        
    acc = np.mean(scores) * 100
    std = np.std(scores) * 100
    print(f"Accuracy Ensemble Voting CV (5-Fold) : {acc:.2f} % (± {std:.2f} %)")
    
    # Rapport de classification détaillé
    print("\nRapport de classification par variété :")
    classes = sorted(np.unique(y))
    print(classification_report(y_true_all, y_pred_all, target_names=classes, digits=3))
    
    # Matrice de confusion finale
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Matrice de confusion - Modèle Optimal ({acc:.2f} %)")
    plt.xlabel("Variété Prédite")
    plt.ylabel("Variété Réelle")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "fig9_confusion_matrix_optimal.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Matrice de confusion sauvegardée : {cm_path}")
    
    return ensemble, acc


def generate_submission(model, data):
    """Génère les 700 prédictions et remplit My_contribution.xlsx."""
    print("\n" + "="*60)
    print("GÉNÉRATION DES 700 PRÉDICTIONS POUR LE CHALLENGE")
    print("="*60)
    
    X_train = data['X_cal_blocked']
    y_train = data['y_cal']
    X_test = data['X_pred_blocked']
    pred_ids = data['pred_ids']
    
    print("Entraînement du modèle final sur l'intégralité des 1 750 pois étiquetés...")
    model.fit(X_train, y_train)
    
    print("Prédiction des 700 pois anonymes...")
    predictions = model.predict(X_test)
    
    # Chargement et remplissage du template officiel My_contribution.xlsx
    sub_path = os.path.join(DATA_DIR, "My_contribution.xlsx")
    df_sub = pd.read_excel(sub_path)
    
    df_sub['Variety'] = predictions
    df_sub.to_excel(sub_path, index=False)
    print(f"Fichier de soumission complété et enregistré avec succès : {sub_path}")
    
    # Distribution des prédictions générées
    counts = pd.Series(predictions).value_counts()
    print("\nDistribution des prédictions générées sur les 700 pois de test :")
    for var, cnt in counts.items():
        print(f"  - {var:<12} : {cnt:3d} pois ({cnt/len(predictions)*100:5.1f} %)")
        
    return df_sub


if __name__ == "__main__":
    data = prepare_datasets()
    best_svm = optimize_svm(data)
    best_model, cv_acc = test_ensemble_voting(data, best_svm)
    df_sub = generate_submission(best_model, data)
    print("\nProcessus d'optimisation et prédictions terminé avec succès !")

