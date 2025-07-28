"""
Codigo de preparacion de datos para el modelo de deteccion de defectos en turbinas eolicas
curación de datos, limpieza y transformacion de los mismos.
"""
import os
import json
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class WindTurbineDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data = []
        self.labels = []
        self.features = []
        
    def load_and_process_data(self):
        """Recorre el directorio y procesa imágenes y anotaciones"""
        for root, dirs, files in os.walk(self.base_path):
            if 'mask' in dirs:
                # Esta es una carpeta con imágenes
                self._process_image_folder(root)
                
    def _process_image_folder(self, folder_path):
        """Procesa una carpeta con imágenes y máscaras"""
        mask_path = os.path.join(folder_path, 'mask')
        
        for file in os.listdir(folder_path):
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(folder_path, file)
                json_path = os.path.join(folder_path, file.replace('.jpg', '.json'))
                
                # Determinar si es defectuosa (tiene JSON)
                is_defective = os.path.exists(json_path)
                
                # Cargar imagen y máscara
                img = Image.open(img_path)
                mask = self._load_mask(mask_path, file)
                
                # Extraer características
                features = self._extract_features(img, mask)
                
                # Almacenar datos
                self.data.append({
                    'image_path': img_path,
                    'mask_path': mask,
                    'features': features,
                    'is_defective': is_defective,
                    'defect_details': self._load_defect_details(json_path) if is_defective else None
                })
                
    def _load_mask(self, mask_folder, img_file):
        """Carga la máscara correspondiente a una imagen"""
        mask_file = img_file.replace('.jpg', '_mask.png')
        mask_path = os.path.join(mask_folder, mask_file)
        return mask_path if os.path.exists(mask_path) else None
    
    def _load_defect_details(self, json_path):
        """Carga los detalles del defecto desde el JSON"""
        with open(json_path) as f:
            return json.load(f)
    
    def _extract_features(self, img, mask):
        """Extrae características relevantes de la imagen"""
        # Convertir a array numpy
        img_array = np.array(img)
        
        # Aquí implementarías la extracción de características
        features = {
            'mean_intensity': np.mean(img_array),
            'std_intensity': np.std(img_array),
            # Agregar más características según análisis exploratorio
        }
        return features
    
    def to_dataframe(self):
        """Convierte los datos a DataFrame para análisis"""
        return pd.DataFrame(self.data)