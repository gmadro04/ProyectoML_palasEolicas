# ProyectoML: Clasificación de Defectos en Palas Eólicas

Este proyecto implementa un pipeline completo de Machine Learning clásico para detectar defectos en imágenes de palas de turbinas eólicas. Utiliza un conjunto de imágenes con máscaras de defectos y archivos `.json` con anotaciones, aplicando técnicas de extracción de características, entrenamiento de modelos y evaluación con métricas interpretables.

## 📂 Estructura del Proyecto

```bash
ProyectoML_palasEolicas/
├── config.py # Configuraciones generales del proyecto
├── image_processor.py # Procesamiento de imágenes y extracción de características
├── main_pipeline.py # Ejecución del pipeline de preprocesamiento
├── model_training.py # Entrenamiento y evaluación de modelos ML
├── saved_models/ # Modelos entrenados (pkl)
├── processed_data/ # Dataset procesado en formato CSV
├── notebooks/ # Notebooks de visualización y análisis
│ └── visualizacion_modelos_palaseolicas_fusionado.ipynb
├── requirements.txt # Lista de dependencias
└── README.md # Este archivo
```

## 🧪 Técnicas empleadas

- Redimensionamiento de imágenes con preservación del aspecto
- Aplicación de máscaras binarias para extraer la región de interés
- Extracción de características:
  - Intensidad (media, desviación, skewness, kurtosis)
  - Textura (matriz GLCM: contraste, homogeneidad, energía, etc.)
  - Geometría (área, excentricidad, perímetro)
- Modelos entrenados:
  - Random Forest
  - Support Vector Machine (SVM)
  - Regresión Logística
  - Multi-Layer Perceptron (MLP)
  - K-Nearest Neighbors (KNN)
- Evaluación con:
  - Matriz de confusión
  - Clasification report
  - AUC-ROC
  <!-- - Visualización de bordes de decisión (con PCA) -->

## 🖥️ Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt 
py -m pip install -r requirements.txt 
```