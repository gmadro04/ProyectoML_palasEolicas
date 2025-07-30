import pandas as pd
import os
import json
from PIL import Image
import numpy as np

# === Configuración ===
IMG_DIR = r"C:\Users\GMADRO04\Documents\PROYECTOML\processed_data\defectuosas"
METADATA_PATH = os.path.join(IMG_DIR, "metadata.json")
OUTPUT_CSV = r"C:\Users\GMADRO04\Documents\PROYECTOML\processed_data\defectuosas\features_defectuosas.csv"

# === Función para obtener histograma RGB ===
def get_rgb_histogram(image_path, bins=32):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    histogram = []
    for channel in range(3):
        hist, _ = np.histogram(image_array[:, :, channel], bins=bins, range=(0, 256), density=True)
        histogram.extend(hist)
    return histogram

# === Cargar metadatos y procesar imágenes ===
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

features = []
for item in metadata:
    img_file = item["img_file"]
    img_path = os.path.join(IMG_DIR, img_file)

    if os.path.exists(img_path):
        hist = get_rgb_histogram(img_path)
        hist.append(img_file)  # para rastreo
        features.append(hist)

# === Crear DataFrame y guardar ===
df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(96)] + ["img_file"])
df.to_csv(OUTPUT_CSV, index=False)

print(f" ----------- Archivo guardado en: {OUTPUT_CSV} ----------- ")
