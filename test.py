import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import os
import math

# Datos de ejemplo (tus imágenes y 1-pixels)
one_pixel_results = [
    ('crown-1.gif', 160763),
    ('device0-1.gif', 79096),
    ('device8-1.gif', 88636),
    ('flatfish-1.gif', 98532),
    ('flower_six.gif', 71589),
    ('frog-1.gif', 41779),
    ('heart.gif', 116621),
    ('tomato.gif', 28279),
    ('triangle.gif', 84957),
    ('turtle-1.gif', 25343),
]

# Directorios
carpeta = 'images'  # carpeta donde están las imágenes originales
output_dir = 'escaladas'  # carpeta donde guardaremos las escaladas
os.makedirs(output_dir, exist_ok=True)

# Promedio de pixeles 1 (int)
promedio_1pix = int(sum(area for _, area in one_pixel_results) / len(one_pixel_results))
print(f"Promedio de 1-pixeles objetivo: {promedio_1pix}")

# Tamaño lienzo común (puedes cambiarlo según tus imágenes)
canvas_width = 703
canvas_height = 693

def binarizar(imagen):
    # Convierte imagen en escala de grises a matriz 0 y 1
    arr = np.array(imagen)
    return (arr > 128).astype(np.uint8)

def calcular_factor_escala(area_actual, area_deseada):
    return math.sqrt(area_deseada / area_actual)

for nombre_archivo, area_actual in one_pixel_results:
    ruta_entrada = os.path.join(carpeta, nombre_archivo)
    
    # Leer y convertir a escala de grises
    img = Image.open(ruta_entrada).convert('L')
    
    # Binarizar
    bin_img = binarizar(img)
    
    # Calcular factor de escala
    factor_a = calcular_factor_escala(area_actual, promedio_1pix)
    
    # Escalar con zoom - nearest neighbor para mantener 0 y 1
    escalado = zoom(bin_img, factor_a, order=0)
    
    # Asegurar que sigue binario (por si acaso)
    escalado = (escalado > 0.5).astype(np.uint8)
    
    # Crear lienzo blanco (0) para centrar
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # Calcular posiciones para centrar
    y_offset = (canvas_height - escalado.shape[0]) // 2
    x_offset = (canvas_width - escalado.shape[1]) // 2
    
    # Insertar imagen escalada en el lienzo
    canvas[y_offset:y_offset+escalado.shape[0], x_offset:x_offset+escalado.shape[1]] = escalado
    
    # Contar 1s después de escalar y centrar
    total_1s = np.sum(canvas)
    
    # Guardar imagen resultante (convertir a 0-255 para PNG)
    img_result = Image.fromarray(canvas * 255)
    ruta_salida = os.path.join(output_dir, f"escalada_{nombre_archivo}")
    img_result.save(ruta_salida)
    
    print(f"{nombre_archivo}: factor α = {factor_a:.3f}, 1-pixeles tras escalado: {total_1s}, diferencia con promedio: {abs(promedio_1pix - total_1s)}")
