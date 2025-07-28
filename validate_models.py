import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.impute import SimpleImputer

# Configuración inicial
plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)

def load_data_and_model():
    """Carga los datos y el modelo entrenado"""
    # Cargar dataset procesado
    data_path = r'C:\Users\GMADRO04\Documents\PROYECTOML\processed_data\1wind_turbine_dataset.csv'
    df = pd.read_csv(data_path)
    
    # Cargar modelo entrenado
    model_path = r'C:\Users\GMADRO04\Documents\PROYECTOML\outputs\models\rf_model.pkl'
    model = joblib.load(model_path)
    
    return df, model

def check_features(model, X):
    """Verifica que las características coincidan con las usadas en el entrenamiento"""
    if hasattr(model, 'feature_names_in_'):
        missing = set(model.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan características requeridas: {missing}")
        return X[list(model.feature_names_in_)]
    return X

def evaluate_false_negatives(model, df):
    """Identifica y analiza falsos negativos"""
    # Obtener características del modelo
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        features = [col for col in df.columns 
                   if col not in ['image_path', 'source_directory', 'is_defective', 
                                'defect_metadata', 'defect_type']
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    # Predecir probabilidades
    X = check_features(model, df[features])
    y = df['is_defective']
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Identificar falsos negativos
    df['prediction'] = y_pred
    df['probability'] = y_proba
    false_negatives = df[(df['is_defective'] == 1) & (df['prediction'] == 0)]
    
    print(f"\nTotal de muestras defectuosas: {len(df[df['is_defective']==1])}")
    print(f"Falsos negativos encontrados: {len(false_negatives)}")
    print(f"Tasa de falsos negativos: {len(false_negatives)/len(df[df['is_defective']==1]):.2%}")
    
    return false_negatives.sort_values('probability', ascending=False), features

def plot_defect_analysis(sample, features):
    """Visualiza un caso particular con sus características"""
    img = Image.open(sample['image_path'])
    mask_path = sample['image_path'].replace('.jpg', '_mask.png')
    
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original\nEtiqueta: {'Defectuosa' if sample['is_defective'] else 'Sana'}")
    
    # Máscara si existe
    plt.subplot(1, 3, 2)
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        plt.imshow(mask, cmap='gray')
        plt.title('Máscara de defecto')
    else:
        plt.imshow(img)
        plt.title('Sin máscara disponible')
    
    # Heatmap de características
    plt.subplot(1, 3, 3)
    features_series = sample[features]
    sns.barplot(x=features_series.values, y=features_series.index, palette='viridis')
    plt.title('Valores de características clave')
    plt.xlabel('Valor')
    plt.tight_layout()
    plt.show()
    
    print(f"\nProbabilidad predicha: {sample['probability']:.2%}")
    print(f"Características principales:")
    print(features_series.sort_values(ascending=False).head(5))

def plot_feature_space(df, features):
    """Visualización del espacio de características con manejo de versiones de TSNE"""
    # Prepara los datos
    X = df[features].copy()
    y = df['is_defective']
    preds = df['prediction']
    
    # 1. Manejo de valores NaN
    print("\nPreprocesando datos...")
    print(f"Valores NaN encontrados: {X.isna().sum().sum()}")
    
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Escalar datos
    X_scaled = StandardScaler().fit_transform(X_imputed)
    
    # 2. Reducción de dimensionalidad - Versión compatible
    print("Aplicando reducción dimensional...")
    try:
        # Intentar con parámetros modernos primero
        from sklearn.manifold import TSNE
        try:
            # Para versiones recientes de scikit-learn
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
            X_reduced = tsne.fit_transform(X_scaled)
            method_name = 't-SNE'
        except TypeError:
            # Para versiones antiguas
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                X_reduced = tsne.fit_transform(X_scaled)
                method_name = 't-SNE'
            except Exception as e:
                print(f"Error con t-SNE: {str(e)}")
                raise
        
    except Exception as e:
        print(f"No se pudo aplicar t-SNE: {str(e)}")
        print("Usando PCA como alternativa...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        method_name = 'PCA'
    
    # 3. Crear el gráfico
    plt.figure(figsize=(12, 8))
    
    # Graficar puntos según su estado real y predicción
    for (true_label, pred_label, marker, color, label) in [
        (1, 1, 'o', 'green', 'Verdaderos Positivos'),
        (1, 0, 'x', 'red', 'Falsos Negativos'), 
        (0, 0, 's', 'blue', 'Verdaderos Negativos'),
        (0, 1, '^', 'orange', 'Falsos Positivos')
    ]:
        mask = (y == true_label) & (preds == pred_label)
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                    marker=marker, color=color, label=label)
    
    plt.title(f'Espacio de características ({method_name})')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_optimal_threshold(model, X, y):
    """Encuentra el umbral óptimo para minimizar falsos negativos"""
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
    
    # Encontrar umbral que maximiza sensibilidad (minimiza falsos negativos)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr, label='Tasa de Verdaderos Positivos')
    plt.plot(thresholds, fpr, label='Tasa de Falsos Positivos')
    plt.axvline(optimal_threshold, color='r', linestyle='--')
    plt.title('Selección de Umbral Óptimo')
    plt.xlabel('Umbral de Probabilidad')
    plt.ylabel('Tasa')
    plt.legend()
    plt.show()
    
    return optimal_threshold

def main():
    try:
        # 1. Cargar datos y modelo
        print("Cargando datos y modelo...")
        df, model = load_data_and_model()
        
        # 2. Evaluar falsos negativos
        print("\nEvaluando falsos negativos...")
        false_negatives, features = evaluate_false_negatives(model, df)
        
        # 3. Visualizar casos problemáticos
        if len(false_negatives) > 0:
            print("\nAnalizando casos problemáticos...")
            for idx, row in false_negatives.head(5).iterrows():
                plot_defect_analysis(row, features)
        else:
            print("\nNo se encontraron falsos negativos para analizar")
        
        # 4. Visualizar espacio de características
        print("\nVisualizando espacio de características...")
        plot_feature_space(df, features)
        
        # 5. Optimizar umbral de decisión
        print("\nOptimizando umbral de decisión...")
        X = check_features(model, df[features])
        y = df['is_defective']
        optimal_thresh = find_optimal_threshold(model, X, y)
        
        print(f"\nUmbral óptimo para minimizar falsos negativos: {optimal_thresh:.2f}")
        
        # Re-evaluar con nuevo umbral
        df['new_prediction'] = (model.predict_proba(X)[:, 1] > optimal_thresh).astype(int)
        new_fn = len(df[(df['is_defective'] == 1) & (df['new_prediction'] == 0)])
        print(f"Falsos negativos con nuevo umbral: {new_fn}")

    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()