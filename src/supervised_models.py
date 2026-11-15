import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def train_and_evaluate_models(X_train_sc, y_train, X_test_sc, y_test, selected_features):
    """
    Entraîne et compare la Régression Logistique et le SVM (RBF)
    sur le sous-ensemble de variables sélectionnées.
    """
    # On filtre uniquement sur les 8 variables retenues
    X_tr = X_train_sc[selected_features].values
    X_te = X_test_sc[selected_features].values

    # 1. Modèle A : Régression Logistique
    # (multi_class retiré pour compatibilité avec les versions récentes de scikit-learn)
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_tr, y_train)

    y_pred_lr = lr_model.predict(X_te)
    y_prob_lr = lr_model.predict_proba(X_te)

    # 2. Modèle B : Support Vector Machine (SVC avec Noyau RBF)
    svm_model = SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(X_tr, y_train)

    y_pred_svm = svm_model.predict(X_te)
    y_prob_svm = svm_model.predict_proba(X_te)

    # 3. Calcul des Métriques Comparatives
    resultats = {
        'Logistic Regression': {
            'model': lr_model,
            'y_pred': y_pred_lr,
            'y_prob': y_prob_lr,
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'f1_macro': f1_score(y_test, y_pred_lr, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred_lr, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
            'report': classification_report(y_test, y_pred_lr, target_names=['Dropout', 'Enrolled', 'Graduate'])
        },
        'SVM (Kernel RBF)': {
            'model': svm_model,
            'y_pred': y_pred_svm,
            'y_prob': y_prob_svm,
            'accuracy': accuracy_score(y_test, y_pred_svm),
            'f1_macro': f1_score(y_test, y_pred_svm, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred_svm, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred_svm),
            'report': classification_report(y_test, y_pred_svm, target_names=['Dropout', 'Enrolled', 'Graduate'])
        }
    }

    # Tableau comparatif synthétique
    df_compare = pd.DataFrame({
        'Modèle': ['Régression Logistique', 'SVM (Noyau RBF)'],
        'Accuracy': [resultats['Logistic Regression']['accuracy'], resultats['SVM (Kernel RBF)']['accuracy']],
        'F1-Score (Macro)': [resultats['Logistic Regression']['f1_macro'], resultats['SVM (Kernel RBF)']['f1_macro']],
        'F1-Score (Pondéré)': [resultats['Logistic Regression']['f1_weighted'], resultats['SVM (Kernel RBF)']['f1_weighted']]
    })

    return resultats, df_compare