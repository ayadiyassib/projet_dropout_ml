import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_confusion_matrices(resultats, class_names=['Dropout', 'Enrolled', 'Graduate'], save_path="figures/confusion_matrices.png"):
    """Trace et sauvegarde les matrices de confusion côte à côte."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_names = ['Logistic Regression', 'SVM (Kernel RBF)']
    palettes = ['Blues', 'Greens']

    for i, name in enumerate(model_names):
        cm = resultats[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap=palettes[i], ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[i].set_title(f"Matrice de Confusion — {name}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Statut Prédit")
        axes[i].set_ylabel("Statut Réel")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[FIGURE] Sauvegardée dans : {save_path}")

def plot_multiclass_roc_curves(resultats, y_test, class_names=['Dropout', 'Enrolled', 'Graduate'], save_path="figures/roc_curves.png"):
    """
    Trace les courbes ROC multi-classes (One-vs-Rest) pour la Régression Logistique et le SVM.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Binarisation de y_test pour le calcul ROC multi-classes (3 colonnes 0/1)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_names = ['Logistic Regression', 'SVM (Kernel RBF)']
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Rouge, Bleu, Vert

    for ax_idx, name in enumerate(model_names):
        y_prob = resultats[name]['y_prob']

        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_prob[:, c])
            roc_auc = auc(fpr, tpr)
            axes[ax_idx].plot(fpr, tpr, color=colors[c], lw=2,
                              label=f"{class_names[c]} (AUC = {roc_auc:.2f})")

        # Ligne diagonale aléatoire
        axes[ax_idx].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Aléatoire (AUC = 0.50)")
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.05])
        axes[ax_idx].set_xlabel("Taux de Faux Positifs (FPR)")
        axes[ax_idx].set_ylabel("Taux de Vrais Positifs (TPR)")
        axes[ax_idx].set_title(f"Courbes ROC (OvR) — {name}", fontsize=11, fontweight='bold')
        axes[ax_idx].legend(loc="lower right")
        axes[ax_idx].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[FIGURE] Sauvegardée dans : {save_path}")