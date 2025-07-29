import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import joblib
import seaborn as sns

# === RUTAS ===
INPUT_PATH = os.path.join(config.BASE_PATH, "processed_data", "wind_turbine_dataset_visual_only.csv")
OUTPUT_MODEL_PATH = os.path.join(config.BASE_PATH, "outputs", "models_visual_only")
OUTPUT_PLOT_PATH = os.path.join(config.BASE_PATH, "outputs", "plots_visual_only")
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

# === CARGA DE DATOS ===
df = pd.read_csv(INPUT_PATH)
feature_cols = [
    'intensity_mean', 'intensity_std', 'intensity_median', 'intensity_skew', 'intensity_kurtosis',
    'texture_contrast_mean', 'texture_contrast_std', 'texture_dissimilarity_mean', 'texture_dissimilarity_std',
    'texture_homogeneity_mean', 'texture_homogeneity_std', 'texture_energy_mean', 'texture_energy_std',
    'texture_correlation_mean', 'texture_correlation_std'
]

X = df[feature_cols]
y = df['is_defective']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === DEFINICIÓN DE MODELOS Y PIPES ===
models = {
    'logistic_regression': (LogisticRegression(max_iter=1000), {
        'logistic_regression__C': [0.1, 1],
        'logistic_regression__penalty': ['l2'],
        'logistic_regression__solver': ['liblinear'],
        'logistic_regression__class_weight': ['balanced']
    }),
    'rf': (RandomForestClassifier(), {
        'rf__n_estimators': [100],
        'rf__max_depth': [None, 10],
        'rf__class_weight': ['balanced']
    }),
    'knn': (KNeighborsClassifier(), {
        'knn__n_neighbors': [3],
        'knn__weights': ['uniform'],
        'knn__metric': ['euclidean']
    }),
    'mlp': (MLPClassifier(max_iter=200), {
        'mlp__hidden_layer_sizes': [(50,)],
        'mlp__activation': ['relu'],
        'mlp__alpha': [0.0001],
        'mlp__learning_rate_init': [0.001]
    }),
    'svm': (SVC(probability=True), {
        'svm__C': [1],
        'svm__kernel': ['linear'],
        'svm__class_weight': ['balanced']
    })
}

# === ENTRENAMIENTO Y VISUALIZACIÓN ===
for name, (clf, param_grid) in models.items():
    print(f"\n=== Entrenando {name.upper()} ===")
    pipe = Pipeline([(name, clf)]) if name == 'knn' else Pipeline([
        ('scaler', StandardScaler()),
        (name, clf),
    ])

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # === Métricas
    auc_score = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(cm)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {auc_score:.4f}")

    # === Guardar modelo
    joblib.dump(best_model, os.path.join(OUTPUT_MODEL_PATH, f"{name}_model_visual_only.pkl"))

    # === Gráficas
    results_df = pd.DataFrame(grid.cv_results_)

    # 1. Matriz de Confusión
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {name.upper()}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_PATH, f"{name}_confusion_matrix.png"))
    plt.close()

    # 2. AUC - entrenamiento vs test
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['mean_train_score'], label='Train AUC')
    plt.plot(results_df['mean_test_score'], label='Test AUC')
    plt.fill_between(range(len(results_df)),
                     results_df['mean_test_score'] - results_df['std_test_score'],
                     results_df['mean_test_score'] + results_df['std_test_score'],
                     alpha=0.2)
    plt.title(f'Train vs Test AUC - {name.upper()}')
    plt.xlabel('Combinación de hiperparámetros')
    plt.ylabel('AUC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_PATH, f"{name}_train_test_auc.png"))
    plt.close()

    # 3. Curva de pérdida solo para MLP
    if name == 'mlp':
        loss_curve = best_model.named_steps['mlp'].loss_curve_
        plt.figure(figsize=(7, 4))
        plt.plot(loss_curve, label="Training Loss")
        plt.title(f'Curva de Pérdida - {name.upper()}')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOT_PATH, f"{name}_loss_curve.png"))
        plt.close()
