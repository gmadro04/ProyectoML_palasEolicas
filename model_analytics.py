import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from config import config
import os

class ModelAnalyzer:
    def __init__(self, data_path, model_dir):
        """Inicializa el analizador con datos y modelos guardados"""
        # Cargar datos y eliminar columnas duplicadas
        self.data = pd.read_csv(data_path)
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        
        self.model_dir = model_dir
        self.models = {}
        
        # Cargar modelos guardados
        self._load_models()
        
        # Verificar que exista la columna target
        if 'is_defective' not in self.data.columns:
            raise ValueError("La columna objetivo 'is_defective' no existe en los datos")
        
        # Seleccionar características numéricas excluyendo rutas y metadatos
        self.numeric_columns = [col for col in self.data.columns 
                                if pd.api.types.is_numeric_dtype(self.data[col]) 
                                and col != 'is_defective']
        
        # Preparar datos
        self.X = self.data[self.numeric_columns].fillna(self.data[self.numeric_columns].mean())
        self.y = self.data['is_defective']

    def _load_models(self):
        """Carga todos los modelos .pkl del directorio"""
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                model_name = file.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(os.path.join(self.model_dir, file))
                print(f"Modelo cargado: {model_name}")

    def plot_feature_correlations(self):
        """Visualiza correlaciones entre características numéricas y target"""
        try:
            # Crear DataFrame solo con características numéricas y target
            analysis_data = self.data[self.numeric_columns + ['is_defective']].copy()
            
            # Calcular matriz de correlación
            corr_matrix = analysis_data.corr()
            
            # Obtener correlaciones con el target
            target_corr = corr_matrix[['is_defective']].drop('is_defective', errors='ignore')
            
            # Ordenar por valor absoluto de correlación
            target_corr['abs_corr'] = target_corr['is_defective'].abs()
            target_corr = target_corr.sort_values('abs_corr', ascending=False).drop('abs_corr', axis=1)
            
            # Configurar gráfico
            plt.figure(figsize=(10, max(6, len(target_corr)*0.25)))
            sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0,
                        fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})
            plt.title('Correlación de características con is_defective')
            plt.tight_layout()
            
            # Guardar gráfico
            output_path = os.path.join(config.ANALYSIS_OUTPUT_PATH, 'feature_correlations.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Guardar datos
            target_corr.to_csv(os.path.join(config.ANALYSIS_OUTPUT_PATH, 'feature_correlations.csv'))
            
            print(f"Análisis de correlaciones guardado en {output_path}")
            
        except Exception as e:
            print(f"Error en plot_feature_correlations: {str(e)}")
            raise

    def analyze_feature_importance(self):
        """Calcula importancia de características para cada modelo"""
        os.makedirs(config.ANALYSIS_OUTPUT_PATH, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                print(f"\nAnalizando importancia para {model_name}...")
                
                # Calcular importancia por permutación
                result = permutation_importance(
                    model, self.X, self.y, 
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Crear DataFrame con resultados
                importance_df = pd.DataFrame({
                    'feature': self.numeric_columns,
                    'importance_mean': result.importances_mean,
                    'importance_std': result.importances_std
                }).sort_values('importance_mean', ascending=False)
                
                # Limitar a las 20 características más importantes para visualización
                top_features = importance_df.head(20)
                
                # Configurar gráfico
                plt.figure(figsize=(10, max(6, len(top_features)*0.3)))
                ax = sns.barplot(
                    x='importance_mean', 
                    y='feature', 
                    data=top_features,
                    hue='feature',
                    palette='viridis',
                    dodge=False,
                    legend=False
                )
                
                # Añadir barras de error manualmente
                for i, (_, row) in enumerate(top_features.iterrows()):
                    ax.errorbar(
                        x=row['importance_mean'],
                        y=i,
                        xerr=row['importance_std'],
                        fmt='none',
                        c='black',
                        capsize=3
                    )
                
                plt.title(f'Top 20 características importantes - {model_name}')
                plt.xlabel('Disminución en precisión al permutar')
                plt.tight_layout()
                
                # Guardar gráfico
                output_path = os.path.join(config.ANALYSIS_OUTPUT_PATH, f'feature_importance_{model_name}.png')
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                # Guardar CSV
                importance_df.to_csv(
                    os.path.join(config.ANALYSIS_OUTPUT_PATH, f'feature_importance_{model_name}.csv'),
                    index=False
                )
                
                print(f"Importancia de características guardada en {output_path}")
                
            except Exception as e:
                print(f"Error analizando {model_name}: {str(e)}")

    def plot_model_comparison(self):
        """Compara métricas entre modelos"""
        try:
            metrics = []
            
            for model_name, model in self.models.items():
                try:
                    y_pred = model.predict(self.X)
                    y_proba = model.predict_proba(self.X)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calcular métricas básicas
                    model_metrics = {
                        'Modelo': model_name.replace('_', ' ').title(),
                        'Accuracy': model.score(self.X, self.y)
                    }
                    
                    # Añadir ROC AUC si está disponible
                    if y_proba is not None:
                        model_metrics['ROC AUC'] = roc_auc_score(self.y, y_proba)
                    
                    metrics.append(model_metrics)
                    
                except Exception as e:
                    print(f"Error evaluando {model_name}: {str(e)}")
                    continue
            
            # Crear DataFrame con métricas
            metrics_df = pd.DataFrame(metrics)
            
            # Verificar si hay métricas para graficar
            if not metrics_df.empty:
                # Configurar gráfico
                fig, axes = plt.subplots(1, 2, figsize=(14, 6)) if 'ROC AUC' in metrics_df.columns else plt.subplots(1, 1, figsize=(7, 6))
                
                # Gráfico de Accuracy
                sns.barplot(
                    x='Accuracy',
                    y='Modelo',
                    data=metrics_df,
                    ax=axes[0] if 'ROC AUC' in metrics_df.columns else axes,
                    hue='Modelo',
                    palette='Blues_d',
                    legend=False
                )
                axes[0].set_title('Comparación de Accuracy') if 'ROC AUC' in metrics_df.columns else axes.set_title('Comparación de Accuracy')
                axes[0].set_xlim(0, 1) if 'ROC AUC' in metrics_df.columns else axes.set_xlim(0, 1)
                
                # Gráfico de ROC AUC si existe
                if 'ROC AUC' in metrics_df.columns:
                    sns.barplot(
                        x='ROC AUC',
                        y='Modelo',
                        data=metrics_df,
                        ax=axes[1],
                        hue='Modelo',
                        palette='Greens_d',
                        legend=False
                    )
                    axes[1].set_title('Comparación de ROC AUC')
                    axes[1].set_xlim(0, 1)
                
                plt.tight_layout()
                
                # Guardar gráfico
                output_path = os.path.join(config.ANALYSIS_OUTPUT_PATH, 'model_comparison.png')
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                # Guardar CSV
                metrics_df.to_csv(
                    os.path.join(config.ANALYSIS_OUTPUT_PATH, 'model_comparison.csv'),
                    index=False
                )
                
                print(f"Comparación de modelos guardada en {output_path}")
            else:
                print("No hay métricas válidas para comparar modelos")
            
        except Exception as e:
            print(f"Error en plot_model_comparison: {str(e)}")

if __name__ == "__main__":
    try:
        # Configuración inicial
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        os.makedirs(config.ANALYSIS_OUTPUT_PATH, exist_ok=True)
        
        print("Iniciando análisis post-entrenamiento...")
        
        # Inicializar analizador
        analyzer = ModelAnalyzer(
            data_path=os.path.join(config.PROCESSED_DATA_PATH, '1wind_turbine_dataset.csv'),
            model_dir=config.MODEL_SAVE_PATH
        )
        
        # Ejecutar análisis
        print("\n1. Analizando correlaciones...")
        analyzer.plot_feature_correlations()
        
        print("\n2. Calculando importancia de características...")
        analyzer.analyze_feature_importance()
        
        print("\n3. Comparando modelos...")
        analyzer.plot_model_comparison()
        
        print("\nAnálisis completado exitosamente. Resultados en:")
        print(config.ANALYSIS_OUTPUT_PATH)
        
    except Exception as e:
        print(f"\nError durante el análisis: {str(e)}")