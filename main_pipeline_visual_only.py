import os
import json
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import joblib
from config import config
from image_processor_visual_only import ImageProcessorVisualOnly as ImageProcessor

class WindTurbineVisualOnlyPipeline:
    def __init__(self):
        self.processor = ImageProcessor()
        self.output_path = os.path.join(config.PROCESSED_DATA_PATH, 'wind_turbine_dataset_visual_only.csv')
        self.visualization_path = config.VISUALIZATION_PATH + '_visual_only'
        os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(self.visualization_path, exist_ok=True)

    def run(self):
        print("[VISUAL ONLY] Iniciando procesamiento del dataset...")
        print(f"Directorios a procesar: {config.RAW_DATA_PATHS}")

        for path in config.RAW_DATA_PATHS:
            if not os.path.exists(path):
                raise ValueError(f"Directorio no encontrado: {path}")

        image_data = self._explore_directory()
        print("\nProcesando imágenes y extrayendo características visuales...")
        processed_data = self._parallel_process_images(image_data)

        df = pd.DataFrame(processed_data)
        self._generate_sample_visualizations(image_data)

        df.to_csv(self.output_path, index=False)
        print(f"\n[VISUAL ONLY] Procesamiento completado. Datos guardados en: {self.output_path}")
        print(f"Resumen:\n- Total imágenes: {len(df)}\n- Defectuosas: {df['is_defective'].sum()}")

        return df

    def _explore_directory(self):
        image_data = []
        for data_path in config.RAW_DATA_PATHS:
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
                                'source_dir': data_path
                            })

        print(f"\nTotal de imágenes encontradas: {len(image_data)}")
        return image_data

    def _parallel_process_images(self, image_data):
        with Pool(processes=config.N_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_item, image_data),
                total=len(image_data),
                desc="Procesando imágenes (visual)"
            ))
        return [r for r in results if r is not None]

    def _process_single_item(self, item):
        try:
            img_array, mask_array = self.processor.process_single_image(
                item['img_path'], item['mask_path'])

            if img_array is None:
                return None

            is_defective = os.path.exists(item['json_path']) if item['json_path'] else False

            features = self.processor.extract_features(img_array, mask_array)

            result = {
                'image_path': item['img_path'],
                'source_directory': os.path.basename(item['source_dir']),
                'is_defective': int(is_defective),
                **features
            }
            return result

        except Exception as e:
            print(f"\nError procesando {item['img_path']}: {str(e)}")
            return None

    def _generate_sample_visualizations(self, image_data, samples_per_dir=3):
        defective_samples = []

        dir1_samples = [item for item in image_data 
                        if item['source_dir'] == config.RAW_DATA_PATHS[0] and
                        item['json_path'] and os.path.exists(item['json_path'])][:samples_per_dir]

        dir2_samples = [item for item in image_data 
                        if item['source_dir'] == config.RAW_DATA_PATHS[1] and
                        item['json_path'] and os.path.exists(item['json_path'])][:samples_per_dir]

        defective_samples = dir1_samples + dir2_samples

        print(f"\nGenerando visualizaciones (visual only) para {len(defective_samples)} imágenes de ejemplo...")

        # Crear lista de tuplas (img_path, mask_path, label)
        sample_tuples = []
        for item in defective_samples:
            label = 1  # Son defectuosas ya que tienen json
            mask = item['mask_path'] if item['mask_path'] and os.path.exists(item['mask_path']) else None
            sample_tuples.append((item['img_path'], mask, label))

        self.processor.plot_defects(sample_tuples, output_dir=self.visualization_path)


if __name__ == "__main__":
    pipeline = WindTurbineVisualOnlyPipeline()
    dataset = pipeline.run()
