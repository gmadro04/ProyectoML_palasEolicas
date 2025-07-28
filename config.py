import os
import numpy as np

class Config:
    # Rutas base
    BASE_PATH = r'C:\Users\GMADRO04\Documents\PROYECTOML'
    
    # Directorios de datos (ambos blades)
    RAW_DATA_PATHS = [
        os.path.join(BASE_PATH, '3_blade_1_15_with_labeldata'),
        os.path.join(BASE_PATH, '3_blade_16_30_with_labeldata')
    ]
    
    # Directorio para datos procesados
    PROCESSED_DATA_PATH = os.path.join(BASE_PATH, 'processed_data')
    
    # Configuración de procesamiento
    TARGET_IMAGE_SIZE = 1024      # Tamaño para redimensionar imágenes
    GLCM_DISTANCES = [1, 3, 5]    # Distancias para GLCM
    GLCM_ANGLES = [0, np.pi/4, np.pi/2]  # Ángulos para GLCM
    
    # Configuración de visualización
    VISUALIZATION_PATH = os.path.join(BASE_PATH, 'outputs', 'visualizations')
    
    # Configuración de modelos
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'outputs', 'models')

    # Configuración de análisis Directorio para guardar los análisis
    ANALYSIS_OUTPUT_PATH = os.path.join(BASE_PATH, 'outputs', 'analysis')
    
    # Configuración de paralelismo
    N_WORKERS = max(1, os.cpu_count() - 1)  # Núcleos a usar para procesamiento paralelo

config = Config()