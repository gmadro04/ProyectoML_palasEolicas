import os
import json
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import pandas as pd
from config import config

class ImageProcessor:
    def __init__(self):
        self.feature_names = []

    def process_single_image(self, img_path, mask_path=None):
        """Procesa una imagen y su máscara correspondiente"""
        try:
            with Image.open(img_path) as img:
                # Redimensionar manteniendo aspecto
                img_resized = self.resize_with_aspect(img, config.TARGET_IMAGE_SIZE)
                img_array = np.array(img_resized.convert('L'))
                
                # Procesar máscara si existe
                mask_array = None
                if mask_path and os.path.exists(mask_path):
                    with Image.open(mask_path) as mask:
                        mask_resized = mask.resize(img_resized.size, Image.Resampling.NEAREST)
                        mask_array = np.array(mask_resized.convert('L'))
                
                return img_array, mask_array
        except Exception as e:
            print(f"Error procesando {img_path}: {str(e)}")
            return None, None

    def resize_with_aspect(self, image, max_dimension):
        """Redimensiona imagen manteniendo relación de aspecto"""
        width, height = image.size
        if max(width, height) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    def extract_features(self, img_array, mask_array, defect_meta=None):
        """Extrae características combinadas de imagen y metadatos"""
        features = {}
        
        # 1. Características básicas de intensidad
        features.update(self._extract_intensity_features(img_array, mask_array))
        
        # 2. Características de textura
        features.update(self._extract_texture_features(img_array, mask_array))
        
        # 3. Características geométricas
        if mask_array is not None:
            features.update(self._extract_geometric_features(img_array, mask_array))
        
        # 4. Características de defectos (si existen)
        if defect_meta:
            features.update(self._extract_defect_features(defect_meta))
        
        # Guardar nombres de características
        if not self.feature_names:
            self.feature_names = list(features.keys())
            
        return features

    def _extract_intensity_features(self, img_array, mask_array):
        """Extrae características de intensidad"""
        roi = img_array[mask_array > 0] if mask_array is not None else img_array.flatten()
        roi = roi[roi > 0] if mask_array is not None else roi
        
        return {
            'intensity_mean': float(np.nanmean(roi)),
            'intensity_std': float(np.nanstd(roi)),
            'intensity_median': float(np.nanmedian(roi)),
            'intensity_skew': float(pd.Series(roi).skew()),
            'intensity_kurtosis': float(pd.Series(roi).kurtosis())
        }

    def _extract_texture_features(self, img_array, mask_array):
        """Extrae características de textura usando GLCM"""
        try:
            img_ubyte = img_as_ubyte(img_array)
            mask_ubyte = (mask_array > 0).astype(np.uint8) if mask_array is not None else None
            
            glcm = graycomatrix(
                img_ubyte,
                distances=config.GLCM_DISTANCES,
                angles=config.GLCM_ANGLES,
                levels=256,
                symmetric=True,
                normed=True
            )
            
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            features = {}
            
            for prop in props:
                values = graycoprops(glcm, prop)
                features[f'texture_{prop}_mean'] = float(np.mean(values))
                features[f'texture_{prop}_std'] = float(np.std(values))
                
            return features
        except Exception as e:
            print(f"Error en textura: {str(e)}")
            return {f'texture_{prop}_mean': 0.0 for prop in props}

    def _extract_geometric_features(self, img_array, mask_array):
        """Extrae características geométricas de la máscara"""
        try:
            binary_mask = mask_array > 0
            props = regionprops(binary_mask.astype(int), intensity_image=img_array)
            
            if not props:
                return {}
                
            main_region = max(props, key=lambda x: x.area)
            
            return {
                'region_area': float(main_region.area),
                'region_perimeter': float(main_region.perimeter),
                'region_eccentricity': float(main_region.eccentricity),
                'region_solidity': float(main_region.solidity)
            }
        except:
            return {}

    def _extract_defect_features(self, defect_meta):
        """Extrae características de los metadatos de defectos"""
        return {
            'defect_count': defect_meta['defect_count'],
            'relative_area': defect_meta['relative_area'],
            'defect_center_x': defect_meta['bounding_box']['center_x_norm'],
            'defect_center_y': defect_meta['bounding_box']['center_y_norm'],
            'defect_width': defect_meta['bounding_box']['width_norm'],
            'defect_height': defect_meta['bounding_box']['height_norm'],
            'has_erosion': defect_meta['has_erosion'],
            'has_coating': defect_meta['has_coating']
        }

    def load_defect_metadata(self, json_path):
        """Carga y procesa metadatos de defectos desde JSON"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Procesar formas de defectos
            defect_count = len(data['shapes'])
            main_defect_type = self._get_main_defect_type(data['shapes'])
            
            # Calcular bounding box y área
            bbox = self._calculate_bounding_box(data['shapes'])
            bbox_norm = self._normalize_bbox(bbox, data['imageWidth'], data['imageHeight'])
            relative_area = self._calculate_relative_area(data['shapes'], data['imageWidth'], data['imageHeight'])
            
            return {
                'defect_count': defect_count,
                'main_defect_type': main_defect_type,
                'relative_area': relative_area,
                'bounding_box': {**bbox, **bbox_norm},
                'has_erosion': int('erosion' in str(main_defect_type).lower()),
                'has_coating': int('coating' in str(main_defect_type).lower())
            }
        except Exception as e:
            print(f"Error procesando {json_path}: {str(e)}")
            return None

    def _get_main_defect_type(self, shapes):
        """Obtiene el tipo principal de defecto"""
        if not shapes:
            return None
        return shapes[0]['label'].split(';')[0].strip()

    def _calculate_bounding_box(self, shapes):
        """Calcula la bounding box que contiene todos los defectos"""
        if not shapes:
            return None
            
        all_points = [point for shape in shapes for point in shape['points']]
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        return {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'center_x': (min(x_coords) + max(x_coords)) / 2,
            'center_y': (min(y_coords) + max(y_coords)) / 2
        }

    def _normalize_bbox(self, bbox, img_width, img_height):
        """Normaliza las coordenadas de la bounding box"""
        if not bbox:
            return {}
            
        return {
            'x_min_norm': bbox['x_min'] / img_width,
            'x_max_norm': bbox['x_max'] / img_width,
            'y_min_norm': bbox['y_min'] / img_height,
            'y_max_norm': bbox['y_max'] / img_height,
            'center_x_norm': bbox['center_x'] / img_width,
            'center_y_norm': bbox['center_y'] / img_height,
            'width_norm': (bbox['x_max'] - bbox['x_min']) / img_width,
            'height_norm': (bbox['y_max'] - bbox['y_min']) / img_height
        }

    def _calculate_relative_area(self, shapes, img_width, img_height):
        """Calcula el área relativa de los defectos"""
        if not shapes:
            return 0.0
            
        total_area = 0.0
        for shape in shapes:
            if shape['shape_type'] == 'polygon':
                polygon = shape['points']
                area = 0.5 * abs(sum(
                    (polygon[i][0] * polygon[i+1][1] - polygon[i+1][0] * polygon[i][1])
                    for i in range(-1, len(polygon)-1)
                ))
                total_area += area
                
        return total_area / (img_width * img_height)

    def plot_defects(self, img_path, json_path, save_path=None):
        """Genera visualización de defectos superpuestos en la imagen"""
        try:
            # Cargar imagen
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Cargar anotaciones
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.imshow(img_array)
            
            # Dibujar cada defecto
            for shape in annotations['shapes']:
                if shape['shape_type'] == 'polygon':
                    polygon = shape['points']
                    path = Path(polygon)
                    patch = patches.PathPatch(
                        path, 
                        facecolor='red', 
                        edgecolor='yellow',
                        lw=2, 
                        alpha=0.4
                    )
                    ax.add_patch(patch)
                    
                    # Etiqueta
                    centroid = np.mean(polygon, axis=0)
                    ax.text(
                        centroid[0], centroid[1], 
                        shape['label'], 
                        color='white', 
                        fontsize=10, 
                        bbox=dict(facecolor='black', alpha=0.7)
                    )
            
            plt.title(f"Defectos en {os.path.basename(img_path)}")
            plt.axis('off')
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error generando visualización para {img_path}: {str(e)}")