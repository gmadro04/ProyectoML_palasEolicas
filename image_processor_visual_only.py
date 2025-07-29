import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
import pandas as pd
from config import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json


class ImageProcessorVisualOnly:
    def __init__(self):
        self.feature_names = []

    def process_single_image(self, img_path, mask_path=None):
        try:
            with Image.open(img_path) as img:
                img_resized = self.resize_with_aspect(img, config.TARGET_IMAGE_SIZE)
                img_array = np.array(img_resized.convert('L'))

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

    def extract_features(self, img_array, mask_array):
        features = {}
        features.update(self._extract_intensity_features(img_array, mask_array))
        features.update(self._extract_texture_features(img_array, mask_array))
        if mask_array is not None:
            features.update(self._extract_geometric_features(img_array, mask_array))

        if not self.feature_names:
            self.feature_names = list(features.keys())

        return features

    def _extract_intensity_features(self, img_array, mask_array):
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
        try:
            img_ubyte = img_as_ubyte(img_array)
            graycom = graycomatrix(
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
                values = graycoprops(graycom, prop)
                features[f'texture_{prop}_mean'] = float(np.mean(values))
                features[f'texture_{prop}_std'] = float(np.std(values))
            return features
        except Exception as e:
            print(f"Error en textura: {str(e)}")
            return {f'texture_{prop}_mean': 0.0 for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}

    def _extract_geometric_features(self, img_array, mask_array):
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
        
    def plot_defects(self, sample_paths_labels, output_dir=None):
        """
        Muestra imágenes procesadas con máscara aplicada y su etiqueta (defectuosa o no).
    
        Args:
        sample_paths_labels (list): Lista de tuplas (ruta_imagen, ruta_mascara, etiqueta)
        output_dir (str, optional): Si se proporciona, guarda las imágenes en esa ruta
        """
        os.makedirs(output_dir, exist_ok=True) if output_dir else None

        for idx, (img_path, mask_path, label) in enumerate(sample_paths_labels):
            try:
                img = Image.open(img_path).convert('L')
                img_np = np.array(img)

                if mask_path is not None and os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask_np = np.array(mask) > 0
                    masked_img = img_np * mask_np
                else:
                    masked_img = img_np

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(img_np, cmap='gray')
                plt.title('Original')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(masked_img, cmap='gray')
                plt.title(f'Máscara aplicada\nEtiqueta: {"Defectuosa" if label == 1 else "Sana"}')
                plt.axis('off')

                if output_dir:
                    save_path = os.path.join(output_dir, f"sample_{idx}_defecto_{label}.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
            except Exception as e:
                print(f"Error generando imagen {img_path}: {e}")

