"""
Exploratory Data Analysis (EDA) - Chimiométrie 2027 Challenge
Module d'analyse exploratoire et de visualisation scientifique pour les données multimodales de pois.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 13

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Calibration_set_challenge.xlsx")


def load_dataset():
    """Charge et structure les différentes modalités du jeu de données de calibration."""
    print("Chargement du jeu de calibration...")
    df = pd.read_excel(DATA_PATH)
    
    meta_cols = ['Batch', 'Variety', 'ImageName']
    morpho_cols = ['volume [mm^3]', 'length [mm]', 'width [mm]', 'thickness [mm]']
    color_cols = ['L', 'A', 'B']
    
    # Identifier les colonnes NIR (les noms de colonnes restants représentant les longueurs d'onde)
    non_nir = set(meta_cols + morpho_cols + color_cols)
    nir_cols = [c for c in df.columns if c not in non_nir]
    wavelengths = np.array([float(c) for c in nir_cols])
    
    print(f"Échantillons : {len(df)}")
    print(f"Modalités identifiées :")
    print(f"  - Morphologie : {len(morpho_cols)} variables")
    print(f"  - Couleur : {len(color_cols)} variables")
    print(f"  - Spectres NIR : {len(nir_cols)} longueurs d'onde ({wavelengths.min():.1f} - {wavelengths.max():.1f} nm)")
    print(f"  - Variétés : {df['Variety'].nunique()} classes ({df['Variety'].unique()})")
    print(f"  - Lots (Batches) : {df['Batch'].nunique()} lots distincts")
    
    return df, meta_cols, morpho_cols, color_cols, nir_cols, wavelengths


def plot_nir_spectra(df, nir_cols, wavelengths):
    """Trace les spectres NIR bruts et les spectres moyens par variété."""
    print("Génération du graphique des spectres NIR...")
    varieties = sorted(df['Variety'].unique())
    colors = sns.color_palette("tab10", len(varieties))
    variety_color_map = dict(zip(varieties, colors))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    
    # 1. Tous les spectres individuels (échantillon aléatoire de 200 pour lisibilité)
    sample_df = df.sample(n=min(300, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        axes[0].plot(wavelengths, row[nir_cols].values, color=variety_color_map[row['Variety']], alpha=0.15, lw=0.8)
    axes[0].set_title("Spectres NIR individuels (300 échantillons bruts)")
    axes[0].set_xlabel("Longueur d'onde (nm)")
    axes[0].set_ylabel("Absorbance / Signal NIR")
    
    # 2. Spectres moyens par variété avec bande +/- 1 écart-type
    for var in varieties:
        sub = df[df['Variety'] == var][nir_cols].values
        mean_spec = np.mean(sub, axis=0)
        std_spec = np.std(sub, axis=0)
        col = variety_color_map[var]
        axes[1].plot(wavelengths, mean_spec, label=var, color=col, lw=2.0)
        axes[1].fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec, color=col, alpha=0.12)
        
    axes[1].set_title("Spectres NIR moyens par variété (± 1 écart-type)")
    axes[1].set_xlabel("Longueur d'onde (nm)")
    axes[1].legend(title="Variété", loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = os.path.join(FIG_DIR, "fig1_nir_raw_spectra.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {output_path}")


def plot_morphology_and_color(df, morpho_cols, color_cols):
    """Trace les distributions morphologiques et colorimétriques par variété."""
    print("Génération des distributions morphologiques et colorimétriques...")
    
    # Morphologie
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for i, col in enumerate(morpho_cols):
        sns.boxplot(data=df, x='Variety', y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribution : {col}")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=30)
    plt.suptitle("Caractéristiques morphologiques par variété de pois", y=1.01)
    plt.tight_layout()
    out_morpho = os.path.join(FIG_DIR, "fig2_morphology_distributions.png")
    plt.savefig(out_morpho, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Enregistré : {out_morpho}")
    
    # Couleur CIELAB
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, col in enumerate(color_cols):
        sns.boxplot(data=df, x='Variety', y=col, ax=axes[i], palette="Pastel1")
        axes[i].set_title(f"Composante couleur : {col}")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=30)
    plt.suptitle("Caractéristiques colorimétriques CIELAB par variété", y=1.02)
    plt.tight_layout()
    out_color = os.path.join(FIG_DIR, "fig3_color_distributions.png")
    plt.savefig(out_color, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Enregistré : {out_color}")


def plot_batch_structure(df):
    """Analyse la distribution des lots (Batches) par rapport aux variétés."""
    print("Génération de l'analyse des effets de lot (Batch)...")
    ct = pd.crosstab(df['Variety'], df['Batch'])
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': "Nombre d'échantillons"})
    plt.title("Répartition des variétés à travers les 12 lots (Batches)")
    plt.xlabel("Identifiant du Lot (Batch)")
    plt.ylabel("Variété de pois")
    plt.tight_layout()
    out_batch = os.path.join(FIG_DIR, "fig4_batch_distribution.png")
    plt.savefig(out_batch, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {out_batch}")


def plot_pca_analysis(df, nir_cols, morpho_cols, color_cols):
    """Effectue une ACP (PCA) sur les spectres NIR et sur les variables morpho-couleur."""
    print("Calcul et visualisation des analyses en composantes principales (PCA)...")
    varieties = sorted(df['Variety'].unique())
    palette = sns.color_palette("tab10", len(varieties))
    var_dict = dict(zip(varieties, palette))
    
    # 1. PCA sur Spectres NIR
    X_nir = df[nir_cols].values
    scaler_nir = StandardScaler(with_std=False)  # En chimiométrie, on centre souvent sans réduire pour préserver les intensités relatives
    X_nir_centered = scaler_nir.fit_transform(X_nir)
    
    pca_nir = PCA(n_components=5)
    scores_nir = pca_nir.fit_transform(X_nir_centered)
    ev_nir = pca_nir.explained_variance_ratio_ * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for var in varieties:
        mask = df['Variety'] == var
        axes[0].scatter(scores_nir[mask, 0], scores_nir[mask, 1], label=var, alpha=0.6, s=25, color=var_dict[var])
        axes[1].scatter(scores_nir[mask, 0], scores_nir[mask, 2], label=var, alpha=0.6, s=25, color=var_dict[var])
        
    axes[0].set_xlabel(f"PC1 ({ev_nir[0]:.1f} % variance)")
    axes[0].set_ylabel(f"PC2 ({ev_nir[1]:.1f} % variance)")
    axes[0].set_title("PCA NIR : Score plot PC1 vs PC2")
    axes[0].legend(title="Variété", loc="best", fontsize=8)
    
    axes[1].set_xlabel(f"PC1 ({ev_nir[0]:.1f} % variance)")
    axes[1].set_ylabel(f"PC3 ({ev_nir[2]:.1f} % variance)")
    axes[1].set_title("PCA NIR : Score plot PC1 vs PC3")
    axes[1].legend(title="Variété", loc="best", fontsize=8)
    
    plt.tight_layout()
    out_pca_nir = os.path.join(FIG_DIR, "fig5_pca_nir.png")
    plt.savefig(out_pca_nir, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {out_pca_nir}")
    
    # 2. PCA sur Morphologie + Couleur (standardisé)
    phys_cols = morpho_cols + color_cols
    X_phys = df[phys_cols].values
    scaler_phys = StandardScaler()
    X_phys_scaled = scaler_phys.fit_transform(X_phys)
    
    pca_phys = PCA(n_components=4)
    scores_phys = pca_phys.fit_transform(X_phys_scaled)
    ev_phys = pca_phys.explained_variance_ratio_ * 100
    
    plt.figure(figsize=(9, 6.5))
    for var in varieties:
        mask = df['Variety'] == var
        plt.scatter(scores_phys[mask, 0], scores_phys[mask, 1], label=var, alpha=0.65, s=30, color=var_dict[var])
        
    # Superposer les vecteurs de loadings (Biplot)
    loadings = pca_phys.components_.T * np.sqrt(pca_phys.explained_variance_)
    for j, feature in enumerate(phys_cols):
        plt.arrow(0, 0, loadings[j, 0]*2.5, loadings[j, 1]*2.5, color='darkred', alpha=0.8, width=0.03, head_width=0.1)
        plt.text(loadings[j, 0]*2.8, loadings[j, 1]*2.8, feature, color='darkred', fontsize=9, weight='bold')
        
    plt.xlabel(f"PC1 ({ev_phys[0]:.1f} % variance)")
    plt.ylabel(f"PC2 ({ev_phys[1]:.1f} % variance)")
    plt.title("Biplot PCA : Caractéristiques Physiques (Morphologie + Couleur)")
    plt.legend(title="Variété", loc="best", fontsize=8)
    plt.tight_layout()
    out_pca_phys = os.path.join(FIG_DIR, "fig6_pca_morpho_color.png")
    plt.savefig(out_pca_phys, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {out_pca_phys}")


def plot_correlations(df, morpho_cols, color_cols):
    """Matrice de corrélation de Pearson entre variables physiques."""
    print("Génération de la matrice de corrélation...")
    phys_df = df[morpho_cols + color_cols]
    corr = phys_df.corr()
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Matrice de corrélation : Morphologie et Couleur")
    plt.tight_layout()
    out_corr = os.path.join(FIG_DIR, "fig7_correlation_matrix.png")
    plt.savefig(out_corr, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {out_corr}")


def print_summary_statistics(df, morpho_cols, color_cols):
    """Affiche les statistiques récapitulatives textuelles."""
    print("\n" + "="*60)
    print("RÉSUMÉ STATISTIQUE DES VARIABLES PHYSIQUES")
    print("="*60)
    stats = df[morpho_cols + color_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    print(stats.to_string())
    print("\n" + "="*60)
    print("MOYENNES PAR VARIÉTÉ (MORPHOLOGIE & COULEUR)")
    print("="*60)
    grouped = df.groupby('Variety')[morpho_cols + color_cols].mean()
    print(grouped.to_string())
    print("="*60)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df, meta_cols, morpho_cols, color_cols, nir_cols, wavelengths = load_dataset()
    print_summary_statistics(df, morpho_cols, color_cols)
    plot_nir_spectra(df, nir_cols, wavelengths)
    plot_morphology_and_color(df, morpho_cols, color_cols)
    plot_batch_structure(df)
    plot_pca_analysis(df, nir_cols, morpho_cols, color_cols)
    plot_correlations(df, morpho_cols, color_cols)
    print("\nAnalyse exploratoire terminée avec succès ! Figures prêtes dans report/figures/")

