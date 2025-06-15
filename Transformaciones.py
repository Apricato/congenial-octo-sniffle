'''
_______________________________________________________________________
PRACTICA NUMERO 2 - GRAFICACIÓN
TANIA LÓPEZ IBARRA , ID:336673
FECHA: 25/05/2025
_______________________________________________________________________
UNIVERSIDAD AUTONOMA DEL ESTADO DE AGUASCALIENTES
INGENIERIA EN COMPUTACION INTELIGENTE
_______________________________________________________________________
'''

import numpy as np  # Para operaciones matemáticas
from PIL import Image, ImageDraw
import os
import math
import cv2  # Para cálculo de momentos
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.morphology import opening, closing, disk, skeletonize
from scipy.ndimage import binary_fill_holes

# PARTE 1: Cargar los archivos necesarios de las imágenes
carpeta = "images"
output_dir = "escaladas"
os.makedirs(output_dir, exist_ok=True)  # Crear carpeta de salida si no existe

one_pixel_results = []

# PARTE 2: Calcular los uno-píxeles disponibles en la imagen
def count_pixel(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L')
    arr = np.array(img)
    binaria = np.where(arr < 128, 0, 1)
    cantidad = np.sum(binaria)
    return cantidad

# Recolectar información de cada imagen
for archive in os.listdir(carpeta):
    if archive.endswith(('.png', '.gif')):
        ruta = os.path.join(carpeta, archive)
        cantidad = count_pixel(ruta)
        one_pixel_results.append((archive, cantidad))
        print(f"{archive}: {cantidad} píxeles del objeto (1-pixel)")

# PARTE 3: Escalamiento

# Valores ya recolectados:
one_pixel_e = [
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

# Promedio de píxeles 1 
promedio = 79559 
print(f"Promedio de 1-pixeles objetivo: {promedio}")

carpeta = "images"
output_dir = "escaladas"
os.makedirs(output_dir, exist_ok=True)

# Dimensiones finales (más grande de todas)
final_w, final_h = 623, 558

# Escalar y centrar cada imagen
for nombre_archivo, pixeles_1 in one_pixel_e:
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

# PARTE 4: Momentos Normalizados ηpq
def calcular_eta_pq(binaria, max_p=2, max_q=2):
    coords = np.argwhere(binaria == 1)
    if len(coords) == 0:
        return {}
   

   #Calcular el centroide
    x = coords[:, 1]
    y = coords[:, 0]

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    #Momento
    def mu_pq(p, q):
        return np.sum((x - x_bar) ** p * (y - y_bar) ** q)

    mu00 = mu_pq(0, 0)
    resultados = {}
   

   #las 9 por cada imagen
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p + q >= 1:
                mu = mu_pq(p, q)
                gamma = (p + q) / 2 + 1
                eta = mu / (mu00 ** gamma) if mu00 != 0 else 0
                resultados[(p, q)] = eta
    return resultados

# Reportar momentos normalizados ηpq para imágenes originales y escaladas
print("\n--- Momentos Normalizados ηpq ---")

for archivo, _ in one_pixel_results:
    # Imagen original
    ruta_original = os.path.join(carpeta, archivo)
    img = Image.open(ruta_original).convert('L')
    arr = np.array(img)
    binaria = np.where(arr < 128, 1, 0)
    eta_original = calcular_eta_pq(binaria)

    # Imagen escalada
    ruta_escalada = os.path.join(output_dir, f"{archivo}")
    img_s = Image.open(ruta_escalada).convert('L')
    arr_s = np.array(img_s)
    binaria_s = np.where(arr_s < 128, 1, 0)
    eta_escalada = calcular_eta_pq(binaria_s)

    print(f"\nImagen: {archivo}")
    print("ηpq antes del escalamiento:")
    for key in sorted(eta_original):
        print(f"  η{key[0]}{key[1]} = {eta_original[key]:.6f}")
    print("ηpq después del escalamiento:")
    for key in sorted(eta_escalada):
        print(f"  η{key[0]}{key[1]} = {eta_escalada[key]:.6f}")



#PARTE 6: Obtención de contornos de la figura 

contour_output_dir = "contornos"
os.makedirs(contour_output_dir, exist_ok=True)

# Función para obtener el contorno usando erosión
def obtener_contorno(binaria):
    kernel = np.ones((3, 3), np.uint8) #kernel  
    erosionada = cv2.erode(binaria.astype(np.uint8), kernel, iterations=1)
    contorno = binaria - erosionada
    return contorno

# Procesar las imágenes escaladas para obtener los contornos
for archivo in os.listdir("escaladas"):
    if archivo.endswith(('.png', '.gif')):
        ruta = os.path.join("escaladas", archivo)
        imagen = Image.open(ruta).convert("L")
        arr = np.array(imagen)
        binaria = np.where(arr < 128, 1, 0).astype(np.uint8)
        
        contorno = obtener_contorno(binaria)

        # Guardar el contorno como imagen
        imagen_contorno = Image.fromarray((contorno * 255).astype(np.uint8))
        ruta_guardado = os.path.join(contour_output_dir, f"contorno_{archivo}")
        imagen_contorno.save(ruta_guardado)

#PARTE 7: 

# === PARÁMETROS ===
input_folder  = 'images'       # Carpeta con tus .gif binarios
output_folder = 'trasladadas'  # Carpeta donde se guardarán las imágenes trasladadas
os.makedirs(output_folder, exist_ok=True)

shift_x, shift_y = 50, 50      # Ajusta estos valores para mayor o menor desplazamiento

# === PROCESAMIENTO y CÁLCULO ===
records = []

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith('.gif'):
        continue

    # 1) Leer y binarizar la imagen original
    img = Image.open(os.path.join(input_folder, filename)).convert('L')
    arr = np.array(img)
    _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY)

    # 2) Trasladar píxel a píxel
    translated = np.zeros_like(binary)
    coords = np.column_stack(np.where(binary > 0))
    shifted = coords.copy()
    shifted[:, 1] += shift_x  # X += shift_x
    shifted[:, 0] += shift_y  # Y += shift_y
    # Filtrar coordenadas válidas
    valid = (
        (shifted[:, 0] >= 0) & (shifted[:, 0] < binary.shape[0]) &
        (shifted[:, 1] >= 0) & (shifted[:, 1] < binary.shape[1])
    )
    shifted = shifted[valid]
    translated[shifted[:, 0], shifted[:, 1]] = 1

    # Guardar la imagen trasladada
    base = os.path.splitext(filename)[0]
    cv2.imwrite(
        os.path.join(output_folder, f"trasladada_{base}.png"),
        translated * 255
    )

    # 3) Centro de masa y momentos μ_pq para p,q=0,1,2
    y_t, x_t = shifted[:, 0], shifted[:, 1]
    x_cm = x_t.mean()
    y_cm = y_t.mean()

    moments = {}
    for p in range(3):
        for q in range(3):
            moments[(p, q)] = np.sum((x_t - x_cm)**p * (y_t - y_cm)**q)

    # 4) Almacenar registro
    record = {
        'imagen': filename,
        'x_cm': round(x_cm, 2),
        'y_cm': round(y_cm, 2),
    }
    for (p, q), val in moments.items():
        record[f'mu_{p}{q}'] = round(val, 2)
    records.append(record)

# === CREAR TABLA con pandas ===
df = pd.DataFrame(records)

# Orden de columnas
cols = [
    'imagen', 'x_cm', 'y_cm',
    'mu_00', 'mu_01', 'mu_10', 'mu_11', 'mu_02', 'mu_20', 'mu_12', 'mu_21', 'mu_22'
]
df = df[cols]

# Imprimir tabla en Markdown
try:
    print(df.to_markdown(index=False))
except ImportError:
    pd.set_option('display.max_columns', None)
    print(df.to_string(index=False))

print(f"\nDesplazamiento aplicado: {shift_x}px en X, {shift_y}px en Y")




#PARTE 8: CALCULO DE MOMENTOS TRAS ROTAS

input_folder  = 'escaladas'    # Carpeta con imágenes escaladas (binary 0/1)
output_folder = 'rotadas'      # Carpeta para imágenes rotadas
os.makedirs(output_folder, exist_ok=True)

theta = 45  # Ángulo de rotación (grados)

def central_moments(binary):
    # Coordenadas de píxeles=1
    ys, xs = np.nonzero(binary)
    x_bar, y_bar = xs.mean(), ys.mean()
    mu = {}
    for p in range(4):
        for q in range(4):
            if p+q <= 3:
                mu[(p,q)] = ((xs - x_bar)**p * (ys - y_bar)**q).sum()
    return mu

# Guardar resultados
records = []
for fn in sorted(os.listdir(input_folder)):
    if not fn.lower().endswith(('.png', '.gif', '.jpg', '.jpeg', '.bmp')):
        continue
    # Leer y umbralizar
    img = Image.open(os.path.join(input_folder, fn)).convert('L')
    arr = np.array(img)
    _, bin0 = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY)
    
    # momentos originales
    mu0 = central_moments(bin0)
    # invariantes antes
    phi1_0 = mu0[(2,0)] + mu0[(0,2)]
    phi2_0 = (mu0[(2,0)] - mu0[(0,2)])**2 + 4*(mu0[(1,1)]**2)
    phi3_0 = (mu0[(3,0)] - 3*mu0[(1,2)])**2 + (3*mu0[(2,1)] - mu0[(0,3)])**2

    # rotar alrededor del centro de la imagen
    h, w = bin0.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), theta, 1.0)
    rot_raw = cv2.warpAffine((bin0*255).astype(np.uint8), M, (w, h), borderValue=0)
    _, bin1 = cv2.threshold(rot_raw, 127, 1, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_folder, f"rotada_{os.path.splitext(fn)[0]}.png"), bin1*255)
    
    # momentos rotados
    mu1 = central_moments(bin1)
    phi1_1 = mu1[(2,0)] + mu1[(0,2)]
    phi2_1 = (mu1[(2,0)] - mu1[(0,2)])**2 + 4*(mu1[(1,1)]**2)
    phi3_1 = (mu1[(3,0)] - 3*mu1[(1,2)])**2 + (3*mu1[(2,1)] - mu1[(0,3)])**2
    
    records.append({
        'imagen':      fn,
        'phi1_before': phi1_0,
        'phi1_after':  phi1_1,
        'phi2_before': phi2_0,
        'phi2_after':  phi2_1,
        'phi3_before': phi3_0,
        'phi3_after':  phi3_1,
    })

# Tabla con pandas
df = pd.DataFrame(records)[[
    'imagen',
    'phi1_before','phi1_after',
    'phi2_before','phi2_after',
    'phi3_before','phi3_after'
]].round(2)

print(df.to_markdown(index=False))
print(f"\nRotación aplicada: {theta}°")

#PARTE 9 APLICAR OPERADORES MORFOLOGICOS A LAS IMAGENES ESCALADAS?


# === PARÁMETROS ===
folder_ruido     = 'image_ruido'   # Solo para quitar ruido
folder_general   = 'escaladas'     # Para suavizado, relleno y esqueleto
output_root      = 'morfologia'    # Carpeta raíz para salidas

# === OPERADORES MORFOLÓGICOS ===
operations = {
    'ruido':     {
        'folder': folder_ruido,
        'func':   lambda img: opening(img, disk(1))
    },
    'suavizado': {
        'folder': folder_general,
        'func':   lambda img: closing(img, disk(1))
    },
    'relleno':   {
        'folder': folder_general,
        'func':   lambda img: binary_fill_holes(img).astype(np.uint8)
    },
    'esqueleto': {
        'folder': folder_general,
        'func':   lambda img: skeletonize(img).astype(np.uint8)
    }
}

# === CREAR CARPETAS DE SALIDA ===
for op_name in operations:
    os.makedirs(os.path.join(output_root, op_name), exist_ok=True)

# === PROCESAR OPERACIÓN POR OPERACIÓN ===
for op_name, op_info in operations.items():
    folder_in = op_info['folder']
    op_func   = op_info['func']
    
    for filename in sorted(os.listdir(folder_in)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # Leer imagen como binaria
        img_path = os.path.join(folder_in, filename)
        img_gray = np.array(Image.open(img_path).convert('L'))
        binary   = (img_gray > 128).astype(np.uint8)

        # Aplicar operador
        result   = op_func(binary)
        out_img  = (result * 255).astype(np.uint8)
        
        # Guardar siempre como .png
        name, _ = os.path.splitext(filename)
        out_path = os.path.join(output_root, op_name, f"{name}.png")
        Image.fromarray(out_img).save(out_path)

        print(f"✅ {filename} procesada con '{op_name}' y guardada como '{out_path}'")
