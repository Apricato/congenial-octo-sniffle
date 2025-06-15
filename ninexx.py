import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize

# Carpetas
folder_ruido    = 'image_ruido'    # Solo para quitar ruido
folder_imagenes = 'images'         # Para aplicar el resto de transformaciones
folder_limpias  = 'limpias'        # Donde se guarda el resultado de quitar ruido
folder_procesadas = 'procesadas'  # Donde se guarda el resto del procesamiento

os.makedirs(folder_limpias, exist_ok=True)
os.makedirs(folder_procesadas, exist_ok=True)

kernel = np.ones((3, 3), np.uint8)

records = []

# --- 1. Quitar ruido (solo usando image_ruido) ---
for fn in sorted(os.listdir(folder_ruido)):
    if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue

    img = Image.open(os.path.join(folder_ruido, fn)).convert('L')
    arr = np.array(img)
    _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY)

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(folder_limpias, fn), cleaned * 255)

print("✅ Imágenes sin ruido guardadas en carpeta 'limpias/'")

# --- 2. Aplicar operaciones al conjunto real (desde 'images/') ---
for fn in sorted(os.listdir(folder_imagenes)):
    if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue

    path = os.path.join(folder_imagenes, fn)
    img = Image.open(path).convert('L')
    arr = np.array(img)
    _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY)
    original_pixels = np.sum(binary)

    # Cerradura (suavizar bordes)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(folder_procesadas, f"1_cerradura_{fn}"), closed * 255)
    pixels_closed = np.sum(closed)

    # Rellenar huecos
    filled = binary_fill_holes(closed).astype(np.uint8)
    cv2.imwrite(os.path.join(folder_procesadas, f"2_relleno_{fn}"), filled * 255)
    pixels_filled = np.sum(filled)

    # Esqueletización
    skeleton = skeletonize(filled).astype(np.uint8)
    cv2.imwrite(os.path.join(folder_procesadas, f"3_esqueleto_{fn}"), skeleton * 255)
    pixels_skeleton = np.sum(skeleton)

    records.append({
        'imagen': fn,
        'original_1pix': int(original_pixels),
        'cerradura_1pix': int(pixels_closed),
        'relleno_1pix': int(pixels_filled),
        'esqueleto_1pix': int(pixels_skeleton)
    })

# Crear tabla de resultados
df = pd.DataFrame(records)
print("\n📊 Resultados de transformaciones morfológicas (desde 'images/'):")
print(df.to_markdown(index=False))
df.to_csv("reporte_morfologia.csv", index=False)

print("\n✅ Procesamiento completo: \n- Imágenes sin ruido en 'limpias/'\n- Resultados en 'procesadas/'\n- Tabla exportada a 'reporte_morfologia.csv'")
