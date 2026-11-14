import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_dataset():
    """
    Télécharge et charge le dataset 'Predict Students Dropout and Academic Success' (ID=697).
    Si indisponible via ucimlrepo, charge depuis data/raw/data.csv.
    """
    try:
        dataset = fetch_ucirepo(id=697)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        df = pd.concat([X, y], axis=1)

    except Exception:
        # Fallback si le téléchargement automatique échoue
        df = pd.read_csv("data/raw/data.csv", sep=";")
        X = df.drop(columns=["Target"])
        y = df[["Target"]]

    # Nettoyage des noms de colonnes (suppression d'espaces éventuels)
    X.columns = [c.strip().replace(" ", "_").replace("'", "") for c in X.columns]
    y = y.iloc[:, 0].astype(str).str.strip()

    return X, y


def encode_target(y):
    """
    Encode la cible multiclasse :
    - Dropout : 0 (profil à risque / décrocheur)
    - Enrolled : 1 (toujours inscrit)
    - Graduate : 2 (diplômé)
    """
    mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    inv_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    y_encoded = y.map(mapping).values
    return y_encoded, mapping, inv_mapping


def prepare_train_test(X, y_encoded, test_size=0.2, random_state=42):
    """
    Sépare les données en Train/Test de manière stratifiée (conserve la proportion des 3 classes),
    et renvoie les versions brutes (pour l'arbre) et standardisées (pour SVM/K-Means/RL).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Standardisation pour les modèles métriques
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler