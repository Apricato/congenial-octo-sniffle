import numpy as np
from PIL import Image
import os
import math
import cv2
from skimage.measure import moments_central, moments_hu
from scipy.ndimage import center_of_mass, rotate
from skimage.morphology import erosion, dilation, square, disk

# === CONFIGURACIÓN ===
carpeta = "homework/images"
output_dir = "homework/escaladas"
matriz_dir = "homework/matrices_binarias"
contornos_dir = "homework/contornos"
rotadas_dir = "homework/rotadas"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(matriz_dir, exist_ok=True)
os.makedirs(contornos_dir, exist_ok=True)
os.makedirs(rotadas_dir, exist_ok=True)

# === FUNCIONES AUXILIARES ===
def binarizar_imagen(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    binaria = np.where(arr < 128, 0, 1)
    return binaria

def calcular_factor_escala(area_original, area_deseada):
    return math.sqrt(area_deseada / area_original)

def escalar_imagen(imagen, factor):
    nuevo_tamano = (int(imagen.width * factor), int(imagen.height * factor))
    return imagen.resize(nuevo_tamano, resample=Image.BILINEAR)

def guardar_matriz_txt(nombre_base, matriz):
    path_txt = os.path.join(matriz_dir, f"{nombre_base}.txt")
    np.savetxt(path_txt, matriz.astype(int), fmt='%d')

def obtener_contorno(binaria):
    kernel = np.ones((3, 3), dtype=np.uint8)
    dil = cv2.dilate(binaria.astype(np.uint8), kernel, iterations=1)
    ero = cv2.erode(binaria.astype(np.uint8), kernel, iterations=1)
    return dil - ero

def calcular_invariantes(matriz):
    return moments_hu(moments_central(matriz, center=(matriz.shape[0] // 2, matriz.shape[1] // 2)))

def aplicar_operaciones_morfologicas(matriz):
    erosionada = erosion(matriz, square(3))
    dilatada = dilation(matriz, square(3))
    return erosionada, dilatada

# === PROCESAMIENTO PRINCIPAL ===
one_pixel_results = []
matrices_binarias = []

# Paso 1 y 2: Cargar imágenes y calcular cantidad de 1-pixels
for archivo in os.listdir(carpeta):
    if archivo.endswith(('.png', '.gif')):
        ruta = os.path.join(carpeta, archivo)
        matriz = binarizar_imagen(ruta)
        area = np.sum(matriz)
        one_pixel_results.append((archivo, area))
        matrices_binarias.append((archivo, matriz))

# Paso 3: Escalamiento uniforme
promedio_area = sum([area for _, area in one_pixel_results]) / len(one_pixel_results)
for archivo, area in one_pixel_results:
    factor = calcular_factor_escala(area, promedio_area)
    ruta = os.path.join(carpeta, archivo)
    imagen = Image.open(ruta)
    imagen_escalada = escalar_imagen(imagen, factor)
    imagen_escalada.save(os.path.join(output_dir, f"escalada_{archivo}"))

# Paso 4: Invariantes de Hu
invariantes_resultados = []
for nombre, matriz in matrices_binarias:
    inv = calcular_invariantes(matriz)
    invariantes_resultados.append((nombre, inv))

# Paso 5-6: Contorno y guardado
for nombre, matriz in matrices_binarias:
    contorno = obtener_contorno(matriz)
    Image.fromarray((contorno * 255).astype(np.uint8)).save(os.path.join(contornos_dir, f"contorno_{nombre}"))

# Paso 7: Centro de masa
centros_de_masa = []
for nombre, matriz in matrices_binarias:
    centro = center_of_mass(matriz)
    centros_de_masa.append((nombre, centro))

# Paso 8: Rotación + invariantes antes y después
rot_invariantes = []
for nombre, matriz in matrices_binarias:
    rotada = rotate(matriz.astype(float), angle=45, reshape=False)
    before = calcular_invariantes(matriz)
    after = calcular_invariantes(rotada)
    rot_invariantes.append((nombre, before, after))
    Image.fromarray((rotada > 0.5).astype(np.uint8) * 255).save(os.path.join(rotadas_dir, f"rotada_{nombre}"))

# Paso 9: Operaciones morfológicas
morf_resultados = []
for nombre, matriz in matrices_binarias:
    ero, dil = aplicar_operaciones_morfologicas(matriz)
    morf_resultados.append((nombre, ero, dil))

# Paso 10: Conclusiones (se colocarán manualmente)

# === GUARDAR MATRICES BINARIAS ===
for nombre, matriz in matrices_binarias:
    base = os.path.splitext(nombre)[0]
    guardar_matriz_txt(base, matriz)

# Output parcial (puedes imprimir estas listas si quieres inspeccionarlas luego)
(one_pixel_results, invariantes_resultados, centros_de_masa, rot_invariantes)


