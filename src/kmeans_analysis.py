import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_k(X_sc_selected, k_range=range(2, 7), save_path="figures/silhouette_score.png"):
    """
    Calcule et trace le score de Silhouette et l'Inertie (Méthode du Coude).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    silhouettes = []
    inertias = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_sc_selected)
        sil = silhouette_score(X_sc_selected, cluster_labels)
        silhouettes.append(sil)
        inertias.append(kmeans.inertia_)

    best_k = k_values[np.argmax(silhouettes)]
    best_score = max(silhouettes)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Nombre de Clusters (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel("Inertie Intra-Cluster (Coude)", color=color, fontsize=11)
    ax1.plot(k_values, inertias, marker='o', color=color, linewidth=2, label="Inertie")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Score de Silhouette', color=color, fontsize=11)
    ax2.plot(k_values, silhouettes, marker='s', linestyle='--', color=color, linewidth=2, label="Silhouette")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axvline(x=best_k, color='green', linestyle=':', label=f"Pic Silhouette : k={best_k} ({best_score:.3f})")

    plt.title("Détermination du k Optimal : Inertie et Score de Silhouette", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"[RÉSULTAT] k optimal suggéré par Silhouette : k = {best_k} (Score : {best_score:.4f})")
    return best_k, silhouettes

def run_kmeans_and_profile(X_sc_selected, y_true_names, k, feature_names,
                           save_centroids=None,
                           save_distribution=None):
    """
    Entraîne le K-Means pour un k donné, interprète les centroïdes
    et trace la répartition des statuts réels par cluster sans bug d'axe vide.
    """
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(X_sc_selected)

    # 1. Heatmap des Centroïdes
    df_centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=feature_names,
        index=[f"Cluster {i}" for i in range(k)]
    )

    plt.figure(figsize=(10, 3 + k * 0.7))
    sns.heatmap(df_centroids, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Profil des Centroïdes (Moyennes Standardisées par Cluster, k={k})", fontsize=12, fontweight='bold')
    plt.xlabel("Variables Sélectionnées")
    plt.ylabel("Clusters")
    plt.tight_layout()
    if save_centroids:
        os.makedirs(os.path.dirname(save_centroids), exist_ok=True)
        plt.savefig(save_centroids, dpi=300)
    plt.show()

    # 2. Répartition des statuts réels
    df_analysis = pd.DataFrame({
        'Cluster': [f"Cluster {c}" for c in cluster_labels],
        'Statut_Reel': y_true_names
    })
    cross_tab_counts = pd.crosstab(df_analysis['Cluster'], df_analysis['Statut_Reel'])
    cross_tab_pct = pd.crosstab(df_analysis['Cluster'], df_analysis['Statut_Reel'], normalize='index') * 100

    # 3. Tracé propre du graphique empilé (sans axe vide)
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {'Dropout': '#e74c3c', 'Enrolled': '#f39c12', 'Graduate': '#2ecc71'}
    colonnes_ordonnees = [c for c in ['Dropout', 'Enrolled', 'Graduate'] if c in cross_tab_pct.columns]
    
    cross_tab_pct[colonnes_ordonnees].plot(
        kind='bar', 
        stacked=True, 
        color=[palette[c] for c in colonnes_ordonnees], 
        edgecolor='black', 
        ax=ax
    )
    ax.set_title(f"Répartition des Statuts Réels par Cluster (k={k})", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pourcentage au sein du cluster (%)")
    ax.set_xlabel("Cluster Détecté par K-Means")
    ax.legend(title="Statut Académique Réel", loc="upper right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_distribution:
        os.makedirs(os.path.dirname(save_distribution), exist_ok=True)
        plt.savefig(save_distribution, dpi=300)
    plt.show()

    return df_centroids, cross_tab_counts, cross_tab_pct