import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import joblib
from config import config

class ModelTrainer:
    def __init__(self, data_path):
        """Inicializa el entrenador con validación de datos"""
        try:
            self.data = pd.read_csv(data_path)
            
            # Verificar columnas esenciales
            required_columns = ['is_defective']
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
            
            # Seleccionar solo columnas numéricas para características
            self.feature_columns = [col for col in self.data.columns 
                                  if col not in ['image_path', 'source_directory', 'is_defective', 'defect_metadata', 'defect_type']
                                  and pd.api.types.is_numeric_dtype(self.data[col])]
            
            print("\nColumnas de características seleccionadas:")
            print(self.feature_columns)
            
        except Exception as e:
            raise ValueError(f"Error inicializando ModelTrainer: {str(e)}")

    def prepare_data(self):
        """Prepara datos para entrenamiento con validación"""
        try:
            X = self.data[self.feature_columns]
            y = self.data['is_defective']
            
            # Manejar valores nulos
            X = X.fillna(X.mean())
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.3, 
                stratify=y,
                random_state=42
            )
            
            print("\nDivisión de datos:")
            print(f"- Entrenamiento: {X_train.shape[0]} muestras")
            print(f"- Prueba: {X_test.shape[0]} muestras")
            print(f"- Proporción de clases (train): {y_train.mean():.2f}")
            print(f"- Proporción de clases (test): {y_test.mean():.2f}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise ValueError(f"Error preparando datos: {str(e)}")

    def train_models(self, X_train, y_train):
        """Entrena modelos con búsqueda de hiperparámetros"""
        models = {
            'logistic_regression': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('logistic_regression', LogisticRegression(random_state=42))
                ]),
                'params': {
                    'logistic_regression__C': [0.001, 0.01, 0.1, 1, 10],
                    'logistic_regression__penalty': ['l1', 'l2'],
                    'logistic_regression__solver': ['liblinear'],
                    'logistic_regression__class_weight': ['balanced', None]
                }
            },
            'mlp': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(random_state=42, max_iter=500))
                ]),
                'params': {
                    'mlp__hidden_layer_sizes': [(50,), (100,)],
                    'mlp__activation': ['relu', 'tanh'],
                    'mlp__alpha': [0.0001, 0.001],
                    'mlp__learning_rate_init': [0.001, 0.01]
                }
            },
            'knn': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())
                ]),
                'params': {
                    'knn__n_neighbors': [3, 5, 7],
                    'knn__weights': ['uniform', 'distance'],
                    'knn__metric': ['euclidean', 'manhattan']
                }
            },
            'rf': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestClassifier(random_state=42))
                ]),
                'params': {
                    'rf__n_estimators': [100, 200],
                    'rf__max_depth': [None, 10],
                    'rf__class_weight': ['balanced', None]
                }
            },
            'svm': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(probability=True, random_state=42))
                ]),
                'params': {
                    'svm__C': [0.1, 1, 10],
                    'svm__kernel': ['linear', 'rbf', 'sigmoid'],
                    'svm__class_weight': ['balanced']
                }
            }
        }
        
        best_models = {}
        
        for name, config in models.items():
            try:
                print(f"\n=== Entrenando {name.replace('_', ' ').title()} ===")
                
                # Configurar GridSearchCV
                grid = GridSearchCV(
                    config['pipeline'],
                    config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Entrenar modelo
                grid.fit(X_train, y_train)
                
                # Mostrar resultados
                print("\nResultados de búsqueda:")
                print(f"Mejores parámetros: {grid.best_params_}")
                print(f"Mejor AUC-ROC (CV): {grid.best_score_:.4f}")
                
                # Guardar mejor modelo
                best_models[name] = grid.best_estimator_
                
            except Exception as e:
                print(f"\nError entrenando {name}: {str(e)}")
                continue
        
        return best_models

    def evaluate_models(self, models, X_test, y_test):
        """Evalúa modelos en conjunto de prueba"""
        results = {}
        
        for name, model in models.items():
            try:
                print(f"\n=== Evaluando {name.replace('_', ' ').title()} ===")
                
                # Hacer predicciones
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(X_test)
                
                # Generar reportes
                print("\nReporte de Clasificación:")
                print(classification_report(y_test, y_pred))
                
                print("\nMatriz de Confusión:")
                print(confusion_matrix(y_test, y_pred))
                
                roc_auc = roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else 0.5
                print(f"\nAUC-ROC: {roc_auc:.4f}")
                
                # Guardar resultados
                results[name] = {
                    'model': model,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'roc_auc': roc_auc
                }
                
                # Guardar modelo
                self._save_model(model, name)
                
            except Exception as e:
                print(f"\nError evaluando {name}: {str(e)}")
                continue
        
        return results

    def _save_model(self, model, model_name):
        """Guarda el modelo entrenado"""
        try:
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            model_path = os.path.join(config.MODEL_SAVE_PATH, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"\nModelo guardado en: {model_path}")
        except Exception as e:
            print(f"\nError guardando modelo {model_name}: {str(e)}")

if __name__ == "__main__":
    try:
        # 1. Cargar datos procesados
        data_path = os.path.join(config.PROCESSED_DATA_PATH, '1wind_turbine_dataset.csv')
        print(f"\nCargando datos desde: {data_path}")
        
        trainer = ModelTrainer(data_path)
        
        # 2. Preparar datos
        X_train, X_test, y_train, y_test = trainer.prepare_data()
        
        # 3. Entrenar modelos
        print("\nIniciando entrenamiento de modelos...")
        models = trainer.train_models(X_train, y_train)
        
        # 4. Evaluar modelos
        if models:
            print("\nEvaluando modelos en conjunto de prueba...")
            results = trainer.evaluate_models(models, X_test, y_test)
        else:
            print("\nNo se entrenaron modelos exitosamente.")
            
    except Exception as e:
        print(f"\nError en el proceso principal: {str(e)}")