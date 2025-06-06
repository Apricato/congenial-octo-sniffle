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
from PIL import Image
import os
import math
import cv2  # Para cálculo de momentos
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
def calcular_factor_escala(area_original, area_deseada):
    return math.sqrt(area_deseada / area_original)

promedio_area = sum([area for _, area in one_pixel_results]) / len(one_pixel_results)
print(f"\nPromedio de área objetivo: {promedio_area:.2f}")

for nombre_archivo, area_actual in one_pixel_results:
    factor_a = calcular_factor_escala(area_actual, promedio_area)
    ruta_entrada = os.path.join(carpeta, nombre_archivo)
    imagen = Image.open(ruta_entrada)
    nuevo_tamano = (
        int(imagen.width * factor_a),
        int(imagen.height * factor_a)
    )
    imagen_escalada = imagen.resize(nuevo_tamano, resample=Image.NEAREST)
    ruta_salida = os.path.join(output_dir, f"escalada_{nombre_archivo}")
    imagen_escalada.save(ruta_salida)
    print(f"{nombre_archivo}: Escalado con factor a = {factor_a:.3f}, nuevo tamaño: {nuevo_tamano}")

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
    ruta_escalada = os.path.join(output_dir, f"escalada_{archivo}")
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

# PARTE 5: 


def graficar_pixeles_binarios(imagen_binaria, nombre):
    h, w = imagen_binaria.shape
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for y in range(h):
        for x in range(w):
            if imagen_binaria[y, x] == 255:
                rect = Rectangle((x, h - y - 1), 1, 1, edgecolor='black', facecolor='none', linewidth=0.5)
                ax.add_patch(rect)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Contorno de píxeles '1' en {nombre}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def calcular_momento_central(p, q, imagen_binaria, x_cm, y_cm):
    indices_y, indices_x = np.where(imagen_binaria == 255)
    if len(indices_x) == 0:
        return 0.0  # No píxeles blancos
    dx = indices_x - x_cm
    dy = indices_y - y_cm
    return np.sum((dx ** p) * (dy ** q))

input_dir = "binarias"

print("=== Punto 5: Visualizando contornos de los píxeles 1 ===")

for archivo in os.listdir(input_dir):
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        ruta = os.path.join(input_dir, archivo)

        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        _, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        print(f"\n📄 escalada_imagen_{archivo}")

        indices_y, indices_x = np.where(binaria == 255)
        if len(indices_x) == 0:
            print("❗ Imagen sin píxeles blancos. Saltando.")
            continue

        x_cm = np.mean(indices_x)
        y_cm = np.mean(indices_y)
        print(f"Centro de masa original: x_cm = {x_cm:.2f}, y_cm = {y_cm:.2f}")

        momentos_tras = {}
        for p in range(3):
            for q in range(3):
                clave = f"mu{p}{q}"
                momentos_tras[clave] = calcular_momento_central(p, q, binaria, x_cm, y_cm)

        print("=== Momentos centrales μpq tras traslación ===")
        for p in range(3):
            for q in range(3):
                clave = f"mu{p}{q}"
                valor = momentos_tras.get(clave, 0.0)
                print(f"μ{p}{q} (trasladada): {valor:.5e}")

        print(f"📄 Mostrando contornos para: {archivo}")
        graficar_pixeles_binarios(binaria, archivo)


#PARTE 6: Obtención de contornos de la figura seis

contour_output_dir = "contornos"
os.makedirs(contour_output_dir, exist_ok=True)

# Función para obtener el contorno usando erosión
def obtener_contorno(binaria):
    kernel = np.ones((3, 3), np.uint8)
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

input_dir = "escaladas" 

def trasladar_imagen(img, dx, dy):
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    trasladada = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return trasladada

print("=== Momentos centrales μpq tras traslación ===")

for archivo in os.listdir(input_dir):
    if archivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        ruta = os.path.join(input_dir, archivo)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        _, binaria = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

        # Calcular centro de masa original
        momentos = cv2.moments(binaria)
        if momentos["m00"] == 0:
            continue  # Evita división por cero si la imagen está vacía
        x_cm = momentos["m10"] / momentos["m00"]
        y_cm = momentos["m01"] / momentos["m00"]

        # Trasladar imagen (por ejemplo, mover +30 px en x, +20 px en y)
        img_trasladada = trasladar_imagen(binaria.astype(np.uint8)*255, 30, 20)
        img_trasladada_bin = np.where(img_trasladada > 127, 1, 0).astype(np.uint8)

        # Calcular momentos centrales tras traslado
        momentos_tras = cv2.moments(img_trasladada_bin)

        print(f"\n📄 {archivo}")
        print(f"Centro de masa original: x_cm = {x_cm:.2f}, y_cm = {y_cm:.2f}")
        for p in range(3):
            for q in range(3):
                if p + q <= 2:
                    clave = f"mu{p}{q}"
                    print(f"μ{p}{q} (trasladada): {momentos_tras[clave]:.5e}")






#PARTE 8: CALCULO DE MOMENTOS TRAS ROTAS

# --- Función para rotar una imagen binaria ---
def rotar_imagen(imagen, angulo):
    h, w = imagen.shape
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotada = cv2.warpAffine(imagen, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotada

# --- Función para calcular los 3 primeros momentos de Hu explícitamente ---
def calcular_phi_momentos(binaria):
    m = cv2.moments(binaria)

    mu20 = m['mu20']
    mu02 = m['mu02']
    mu11 = m['mu11']
    mu30 = m['mu30']
    mu12 = m['mu12']
    mu21 = m['mu21']
    mu03 = m['mu03']

    phi1 = mu20 + mu02
    phi2 = (mu20 - mu02)**2 + 4 * mu11**2
    phi3 = (mu30 - 3 * mu12)**2 + (3 * mu21 - mu03)**2

    return phi1, phi2, phi3

# --- Ruta de las imágenes (ajústala a tu caso) ---

#PARTE 9 APLICAR OPERADORES MORFOLOGICOS A LAS IMAGENES ESCALADAS?

input_dir = 'escaladas'
output_dir = 'morfologia'
os.makedirs(output_dir, exist_ok=True)

# Definir el kernel estructurante
kernel = np.ones((3, 3), np.uint8)

# Procesar cada imagen en el directorio de entrada
for archivo in os.listdir(input_dir):
    if archivo.endswith(('.png', '.gif')):
        ruta_entrada = os.path.join(input_dir, archivo)
        ruta_salida = os.path.join(output_dir, archivo)

        # Leer la imagen en escala de grises
        imagen = cv2.imread(ruta_entrada, cv2.IMREAD_GRAYSCALE)

        # Binarizar la imagen
        _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

        # a) Quitar ruido: Apertura (erosión seguida de dilatación)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

        # b) Suavizar bordes: Cierre (dilatación seguida de erosión)
        cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)

        # c) Rellenar huecos: Dilatación adicional
        rellenada = cv2.dilate(cierre, kernel, iterations=1)

        # d) Encontrar esqueletos: Transformada de distancia y umbralización
    
        distancia = cv2.distanceTransform(rellenada, cv2.DIST_L2, 5)
        _, esqueleto = cv2.threshold(distancia, 0.4 * distancia.max(), 255, 0)
        esqueleto = np.uint8(esqueleto)

        # Guardar las imágenes procesadas
        nombre_base = os.path.splitext(archivo)[0]
        cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_apertura.png"), apertura)
        cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_cierre.png"), cierre)
        cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_rellenada.png"), rellenada)
        cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_esqueleto.png"), esqueleto)

        print(f"Procesamiento completo para {archivo}")





