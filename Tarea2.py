# PARTE 3: Escalamiento

from PIL import Image, ImageOps
import numpy as np
import os
import math

# Valores ya recolectados:
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

# Promedio de píxeles 1 (como pide el enunciado)
promedio = 79559 
print(f"Promedio de 1-pixeles objetivo: {promedio}")

carpeta = "images"
output_dir = "escaladas"
os.makedirs(output_dir, exist_ok=True)

# Dimensiones finales (más grande de todas)
final_w, final_h = 623, 558

# Escalar y centrar cada imagen
for nombre_archivo, pixeles_1 in one_pixel_results:
    ruta = os.path.join(carpeta, nombre_archivo)
    img = Image.open(ruta).convert('L')
    
    # Binarizar
    arr = np.array(img)
    binaria = np.where(arr < 128, 0, 1)
    
    # Calcular factor de escala: α = sqrt(promedio / pixeles actuales)
    alpha = math.sqrt(promedio / pixeles_1)
    
    # Redimensionar (manteniendo proporción)
    nuevo_ancho = int(img.width * alpha)
    nuevo_alto = int(img.height * alpha)
    img_redimensionada = img.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    
    # Centrar en imagen final de 703x693
    lienzo = Image.new("L", (final_w, final_h), color=0)  # fondo negro
    offset_x = (final_w - nuevo_ancho) // 2
    offset_y = (final_h - nuevo_alto) // 2
    lienzo.paste(img_redimensionada, (offset_x, offset_y))

    # Guardar en carpeta "escaladas"
    salida = os.path.join(output_dir, nombre_archivo)
    lienzo.save(salida)
    
    print(f"{nombre_archivo} → Factor a = {alpha:.4f} ")
