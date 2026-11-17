"""
Générateur de Notebooks Jupyter interactifs pour le projet Chimiométrie 2027.
"""

import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells, filepath):
    nb = {
        'cells': cells,
        'metadata': {
            'language_info': {
                'name': 'python',
                'version': '3.14'
            },
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Carnet Jupyter créé avec succès : {filepath}")


def md_cell(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.splitlines(keepends=True)
    }


def code_cell(code):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': code.splitlines(keepends=True)
    }


def create_nb1():
    cells = [
        md_cell("# 01 — Analyse Exploratoire et Prétraitements Chimiométriques 🌱\n\n**Data Challenge :** Chimiométrie 2027 (SFDS)\n**Auteurs :** Abdelouahad ALLA & Marcos FUNEZ FAMOSO (UMons)"),
        md_cell("## 1. Imports et Configuration"),
        code_cell("""import sys
import os
sys.path.append(os.path.abspath('../src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import ChemometricPreprocessor, clean_physical_outliers, extract_modalities

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline"""),
        md_cell("## 2. Chargement des Données et Nettoyage"),
        code_cell("""df_raw = pd.read_excel('../data/Calibration_set_challenge.xlsx')
print(f"Échantillons : {df_raw.shape[0]} | Colonnes : {df_raw.shape[1]}")

df = clean_physical_outliers(df_raw, is_train=True)
modalities = extract_modalities(df)

y = modalities['y']
X_nir = modalities['X_nir']
X_morpho = modalities['X_morpho']
X_color = modalities['X_color']
wavelengths = modalities['wavelengths']

print(f"Variétés disponibles (250 / classe) : {np.unique(y)}")"""),
        md_cell("## 3. Visualisation des Spectres NIR : Bruts vs Prétraités (SNV + Savitzky-Golay)"),
        code_cell("""prep = ChemometricPreprocessor(apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1)
X_nir_prep = prep.fit_transform(X_nir)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Spectres bruts
for i in range(0, len(df), 20):
    axes[0].plot(wavelengths, X_nir[i], alpha=0.3, lw=0.8)
axes[0].set_title("Spectres NIR Bruts (Diffusion et dérives optiques)")
axes[0].set_xlabel("Longueur d'onde (nm)")
axes[0].set_ylabel("Absorbance brute")

# Spectres prétraités
for i in range(0, len(df), 20):
    axes[1].plot(wavelengths, X_nir_prep[i], alpha=0.3, lw=0.8)
axes[1].set_title("Spectres Prétraités (SNV + Savitzky-Golay 1ère dérivée)")
axes[1].set_xlabel("Longueur d'onde (nm)")
axes[1].set_ylabel("Signal corrigé")

plt.tight_layout()
plt.show()"""),
        md_cell("## 4. Distributions Morphologiques et Colorimétriques"),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df, x='Variety', y='volume [mm^3]', ax=axes[0], palette='Set2')
axes[0].set_title("Volume par variété de pois (mm³)")
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(data=df, x='Variety', y='A', ax=axes[1], palette='Pastel1')
axes[1].set_title("Composante couleur A* (axe Vert-Rouge CIELAB)")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()"""),
        md_cell("## 5. Analyse de l'Effet de Lot (Batches)"),
        code_cell("""ct = pd.crosstab(df['Variety'], df['Batch'])
plt.figure(figsize=(11, 5))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Répartition des variétés par lot (Batch)")
plt.xlabel("Identifiant du Lot")
plt.ylabel("Variété")
plt.tight_layout()
plt.show()""")
    ]
    make_notebook(cells, os.path.join(NOTEBOOKS_DIR, "01_Exploration_et_Pretraitements.ipynb"))


def create_nb2():
    cells = [
        md_cell("# 02 — Modélisation Multimodale, Fusion et Prédictions Finales 🚀\n\n**Data Challenge :** Chimiométrie 2027 (SFDS)\n**Auteurs :** Abdelouahad ALLA & Marcos FUNEZ FAMOSO (UMons)"),
        md_cell("## 1. Imports et Dépendances"),
        code_cell("""import sys
import os
sys.path.append(os.path.abspath('../src'))

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import ChemometricPreprocessor, BlockScaler, clean_physical_outliers, extract_modalities
from models import PLSDA
from sopls import SOPLSClassifier"""),
        md_cell("## 2. Chargement et Normalisation Multivoie (Block Scaling)"),
        code_cell("""df_cal_raw = pd.read_excel('../data/Calibration_set_challenge.xlsx')
df_pred_raw = pd.read_excel('../data/Prediction_set_challenge.xlsx')

df_cal = clean_physical_outliers(df_cal_raw, is_train=True)
df_pred = clean_physical_outliers(df_pred_raw, is_train=False)

cal_mod = extract_modalities(df_cal)
pred_mod = extract_modalities(df_pred)

# Prétraitement NIR
prep = ChemometricPreprocessor(apply_snv=True, sg_window=15, sg_poly=2, sg_deriv=1)
X_cal_nir = prep.fit_transform(cal_mod['X_nir'])
X_pred_nir = prep.transform(pred_mod['X_nir'])

# Block Scaling (Frobenius)
blocks_cal = {'nir': X_cal_nir, 'morpho': cal_mod['X_morpho'], 'color': cal_mod['X_color']}
blocks_pred = {'nir': X_pred_nir, 'morpho': pred_mod['X_morpho'], 'color': pred_mod['X_color']}

block_scaler = BlockScaler()
block_scaler.fit(blocks_cal)
X_cal_blocked = block_scaler.transform(blocks_cal)
X_pred_blocked = block_scaler.transform(blocks_pred)

y = cal_mod['y']
print(f"Jeu d'entraînement : {X_cal_blocked.shape} | Test : {X_pred_blocked.shape}")"""),
        md_cell("## 3. Évaluation du Modèle Optimal (Soft-Voting Ensemble) en 5-Fold CV"),
        code_cell("""clf_rbf = SVC(kernel='rbf', C=50.0, gamma=0.05, probability=True, random_state=42)
clf_lin = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
clf_et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1)

ensemble = VotingClassifier(
    estimators=[('svm_rbf', clf_rbf), ('svm_lin', clf_lin), ('et', clf_et)],
    voting='soft',
    weights=[3, 2, 1]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
y_true_all, y_pred_all = [], []

for train_idx, val_idx in skf.split(X_cal_blocked, y):
    ensemble.fit(X_cal_blocked[train_idx], y[train_idx])
    preds = ensemble.predict(X_cal_blocked[val_idx])
    scores.append(accuracy_score(y[val_idx], preds))
    y_true_all.extend(y[val_idx])
    y_pred_all.extend(preds)

print(f"Accuracy Moyenne (5-Fold CV) : {np.mean(scores)*100:.2f} % (± {np.std(scores)*100:.2f} %)")"""),
        md_cell("## 4. Matrice de Confusion et Rapport Détaillé par Variété"),
        code_cell("""classes = sorted(np.unique(y))
print(classification_report(y_true_all, y_pred_all, target_names=classes, digits=3))

cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f"Matrice de Confusion Finale (Accuracy = {np.mean(scores)*100:.2f} %)")
plt.xlabel("Variété Prédite")
plt.ylabel("Variété Réelle")
plt.tight_layout()
plt.show()"""),
        md_cell("## 5. Prédictions Finales pour le Challenge (700 Pois Inconnus)"),
        code_cell("""ensemble.fit(X_cal_blocked, y)
predictions = ensemble.predict(X_pred_blocked)

sub_df = pd.read_excel('../data/My_contribution.xlsx')
sub_df['Variety'] = predictions
print("Aperçu des 10 premières prédictions :")
print(sub_df.head(10))

print("\\nDistribution des 700 prédictions :")
print(sub_df['Variety'].value_counts())""")
    ]
    make_notebook(cells, os.path.join(NOTEBOOKS_DIR, "02_Modelisation_et_Predictions.ipynb"))


if __name__ == "__main__":
    create_nb1()
    create_nb2()

