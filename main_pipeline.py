import os
import json
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import joblib
from hashlib import md5
from config import config
from image_processor import ImageProcessor

class WindTurbinePipeline:
    def __init__(self):
        self.processor = ImageProcessor()
        os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(config.VISUALIZATION_PATH, exist_ok=True)

    def run(self):
        """Ejecuta el pipeline completo"""
        print("Iniciando procesamiento de dataset...")
        print(f"Directorios a procesar: {config.RAW_DATA_PATHS}")
        
        # Verificar que existen los directorios
        for path in config.RAW_DATA_PATHS:
            if not os.path.exists(path):
                raise ValueError(f"Directorio no encontrado: {path}")
        
        # 1. Explorar directorio y recopilar rutas
        image_data = self._explore_directory()
        
        # 2. Procesamiento paralelo de imágenes
        print("\nProcesando imágenes y extrayendo características...")
        processed_data = self._parallel_process_images(image_data)
        
        # 3. Crear DataFrame
        df = pd.DataFrame(processed_data)
        
        # 4. Generar visualizaciones de ejemplo
        self._generate_sample_visualizations(image_data)
        
        # 5. Guardar resultados
        output_file = os.path.join(config.PROCESSED_DATA_PATH, '1wind_turbine_dataset.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\nProcesamiento completado. Datos guardados en: {output_file}")
        print(f"Resumen:\n- Total imágenes: {len(df)}\n- Defectuosas: {df['is_defective'].sum()}")
        
        return df

    def _explore_directory(self):
        """Explora recursivamente todos los directorios de blades"""
        image_data = []
        
        for data_path in config.RAW_DATA_PATHS:
            if not os.path.exists(data_path):
                print(f"Advertencia: Directorio no encontrado - {data_path}")
                continue
                
            print(f"\nExplorando directorio: {data_path}")
            
            for root, dirs, files in os.walk(data_path):
                if 'mask' in dirs:
                    mask_dir = os.path.join(root, 'mask')
                    for file in files:
                        if file.lower().endswith('.jpg'):
                            img_path = os.path.join(root, file)
                            mask_path = os.path.join(mask_dir, file.replace('.jpg', '_mask.png'))
                            json_path = os.path.join(root, file.replace('.jpg', '.json'))
                            
                            image_data.append({
                                'img_path': img_path,
                                'mask_path': mask_path if os.path.exists(mask_path) else None,
                                'json_path': json_path if os.path.exists(json_path) else None,
                                'source_dir': data_path  # Guardamos de qué directorio proviene
                            })
        
        print(f"\nTotal de imágenes encontradas: {len(image_data)}")
        return image_data

    def _parallel_process_images(self, image_data):
        """Procesa imágenes en paralelo"""
        with Pool(processes=config.N_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_item, image_data),
                total=len(image_data),
                desc="Procesando imágenes"
            ))
        
        return [r for r in results if r is not None]

    def _process_single_item(self, item):
        """Procesa un solo item (imagen + máscara + json)"""
        try:
            # Procesar imagen y máscara
            img_array, mask_array = self.processor.process_single_image(
                item['img_path'], item['mask_path'])
            
            if img_array is None:
                return None
            
            # Determinar si es defectuosa
            is_defective = os.path.exists(item['json_path']) if item['json_path'] else False
            
            # Cargar metadatos de defectos si existen
            defect_meta = None
            if is_defective:
                defect_meta = self.processor.load_defect_metadata(item['json_path'])
            
            # Extraer características
            features = self.processor.extract_features(img_array, mask_array, defect_meta)
            
            # Estructurar resultado
            result = {
                'image_path': item['img_path'],
                'source_directory': os.path.basename(item['source_dir']),  # Nombre del directorio fuente
                'is_defective': int(is_defective),
                **features
            }
            
            # Añadir metadatos si existen
            if defect_meta:
                result.update({
                    'defect_type': defect_meta['main_defect_type'],
                    'defect_metadata': json.dumps(defect_meta)  # Guardar como string JSON
                })
            
            return result
            
        except Exception as e:
            print(f"\nError procesando {item['img_path']}: {str(e)}")
            return None

    def _generate_sample_visualizations(self, image_data, samples_per_dir=3):
        """Genera visualizaciones balanceadas de ambos directorios"""
        defective_samples = []
        
        # Muestras del primer directorio
        dir1_samples = [item for item in image_data 
                        if item['source_dir'] == config.RAW_DATA_PATHS[0] and
                        item['json_path'] and os.path.exists(item['json_path'])][:samples_per_dir]
        
        # Muestras del segundo directorio
        dir2_samples = [item for item in image_data 
                        if item['source_dir'] == config.RAW_DATA_PATHS[1] and
                        item['json_path'] and os.path.exists(item['json_path'])][:samples_per_dir]
        
        defective_samples = dir1_samples + dir2_samples
        
        print(f"\nGenerando visualizaciones para {len(defective_samples)} imágenes de ejemplo...")
        
        for item in defective_samples:
            try:
                # Crear subdirectorio por blade
                dir_name = os.path.basename(item['source_dir'])
                save_dir = os.path.join(config.VISUALIZATION_PATH, dir_name)
                
                save_path = os.path.join(
                    save_dir,
                    os.path.basename(item['img_path'].replace('.jpg', '_defects.png')))
                
                self.processor.plot_defects(item['img_path'], item['json_path'], save_path)
            except Exception as e:
                print(f"Error generando visualización para {item['img_path']}: {str(e)}")

if __name__ == "__main__":
    pipeline = WindTurbinePipeline()
    dataset = pipeline.run()