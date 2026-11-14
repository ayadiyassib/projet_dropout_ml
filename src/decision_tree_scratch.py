import numpy as np
import pandas as pd


class Node:
    """Représente un nœud interne ou une feuille de l'arbre."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Indice de la variable utilisée pour la coupure
        self.threshold = threshold      # Valeur seuil de coupure
        self.left = left                # Sous-arbre gauche
        self.right = right              # Sous-arbre droit
        self.value = value              # Classe majoritaire prédite (si feuille)

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Arbre de décision multiclasse codé STRICTEMENT avec NumPy et Pandas.
    Calcule automatiquement l'importance des variables (Mean Decrease Impurity).
    """
    def __init__(self, max_depth=6, min_samples_split=10, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.feature_importances_ = None
        self.feature_names_ = None

    def _gini(self, y):
        """Calcule l'impureté de Gini : 1 - sum(p_i^2)"""
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        return 1.0 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        """Calcule l'entropie de Shannon : - sum(p_i * log2(p_i))"""
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _impurity(self, y):
        return self._gini(y) if self.criterion == 'gini' else self._entropy(y)

    def _best_split(self, X, y):
        """Trouve la variable et le seuil offrant la meilleure réduction d'impureté."""
        n_samples, n_features = X.shape
        best_gain = -1.0
        split_feat, split_thresh = None, None

        current_impurity = self._impurity(y)

        for feat in range(n_features):
            col_values = X[:, feat]
            thresholds = np.unique(col_values)

            # Optimisation : si trop de valeurs distinctes, tester les percentiles
            if len(thresholds) > 20:
                thresholds = np.percentile(col_values, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = col_values <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                n_left, n_right = len(y_left), len(y_right)

                # Gain = Impureté parent - Moyenne pondérée de l'impureté enfants
                gain = current_impurity - (
                    (n_left / n_samples) * self._impurity(y_left)
                    + (n_right / n_samples) * self._impurity(y_right)
                )

                if gain > best_gain:
                    best_gain = gain
                    split_feat = feat
                    split_thresh = thresh

        return split_feat, split_thresh, best_gain

    def _build_tree(self, X, y, depth, total_samples):
        n_samples, n_classes = len(y), len(np.unique(y))

        # Conditions d'arrêt : profondeur max atteinte, nœud pur, ou trop peu d'échantillons
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            # Classe majoritaire
            values, counts = np.unique(y, return_counts=True)
            leaf_value = values[np.argmax(counts)]
            return Node(value=leaf_value)

        # Recherche de la meilleure coupure
        feat, thresh, gain = self._best_split(X, y)

        if gain <= 0 or feat is None:
            values, counts = np.unique(y, return_counts=True)
            return Node(value=values[np.argmax(counts)])

        # Accumulation de l'importance de la variable (pondérée par la fraction d'échantillons)
        weight = n_samples / total_samples
        self._raw_importances[feat] += weight * gain

        # Partition et descente récursive
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, total_samples)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, total_samples)

        return Node(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        """Entraîne l'arbre et calcule l'importance des variables."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_mat = np.array(X)

        y_vec = np.array(y)
        n_features = X_mat.shape[1]
        self._raw_importances = np.zeros(n_features)

        self.root = self._build_tree(X_mat, y_vec, depth=0, total_samples=len(y_vec))

        # Normalisation pour obtenir une somme égale à 1
        total_imp = np.sum(self._raw_importances)
        if total_imp > 0:
            self.feature_importances_ = self._raw_importances / total_imp
        else:
            self.feature_importances_ = np.ones(n_features) / n_features

        return self

    def _predict_one(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X):
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return np.array([self._predict_one(self.root, x) for x in X_mat])

    def get_feature_importances(self):
        """Renvoie un DataFrame trié des variables les plus importantes."""
        df_imp = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': self.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df_imp