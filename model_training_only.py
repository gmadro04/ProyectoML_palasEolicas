import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from config import config

# === RUTAS ===
INPUT_PATH = os.path.join(config.BASE_PATH, "processed_data", "wind_turbine_dataset_visual_only.csv")
OUTPUT_PATH = os.path.join(config.BASE_PATH, "outputs", "models_visual_only")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === CARGA DE DATOS ===
print(f"Cargando datos desde: {INPUT_PATH}\n")
df = pd.read_csv(INPUT_PATH)

# Seleccionar características visuales únicamente
feature_cols = [
    'intensity_mean', 'intensity_std', 'intensity_median', 'intensity_skew', 'intensity_kurtosis',
    'texture_contrast_mean', 'texture_contrast_std', 'texture_dissimilarity_mean', 'texture_dissimilarity_std',
    'texture_homogeneity_mean', 'texture_homogeneity_std', 'texture_energy_mean', 'texture_energy_std',
    'texture_correlation_mean', 'texture_correlation_std'
]

X = df[feature_cols]
y = df['is_defective']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print("División de datos:")
print(f"- Entrenamiento: {len(y_train)} muestras")
print(f"- Prueba: {len(y_test)} muestras")
print(f"- Proporción de clases (train): {y_train.mean():.2f}")
print(f"- Proporción de clases (test): {y_test.mean():.2f}\n")

# === DEFINICIÓN DE MODELOS Y PIPES ===
models = {
    'logistic_regression': (LogisticRegression(max_iter=1000), {
        'logistic_regression__C': [0.1, 1, 10],
        'logistic_regression__penalty': ['l1', 'l2'],
        'logistic_regression__solver': ['liblinear'],
        'logistic_regression__class_weight': ['balanced']
    }),
    'rf': (RandomForestClassifier(), {
        'rf__n_estimators': [100],
        'rf__max_depth': [None, 10, 20],
        'rf__class_weight': ['balanced']
    }),
    'knn': (KNeighborsClassifier(), {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }),
    'mlp': (MLPClassifier(max_iter=500), {
        'mlp__hidden_layer_sizes': [(50,), (100,)],
        'mlp__activation': ['relu'],
        'mlp__alpha': [0.0001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01]
    }),
    'svm': (SVC(probability=True), {
        'svm__C': [1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__class_weight': ['balanced']
    })
}

# === ENTRENAMIENTO ===
print("Iniciando entrenamiento de modelos...\n")

for name, (clf, param_grid) in models.items():
    print(f"=== Entrenando {name.upper()} ===")
    pipe = Pipeline([
        (name, clf),
    ]) if name == 'knn' else Pipeline([
        ('scaler', StandardScaler()),
        (name, clf),
    ])

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Mejor AUC-ROC (CV): {grid.best_score_:.4f}\n")

    # Evaluación en test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}\n")

    # Guardar modelo
    model_path = os.path.join(OUTPUT_PATH, f"{name}_model_visual_only.pkl")
    joblib.dump(best_model, model_path)
    print(f"Modelo guardado en: {model_path}\n")
