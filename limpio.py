import numpy as np  # Para operaciones matemáticas
from PIL import Image
import os
import math

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

# PARTE 3: Escalamiento (sin centrar en lienzo negro)

# Valores recolectados manualmente:
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

# Promedio objetivo
promedio = 79559 
print(f"\nPromedio de 1-pixeles objetivo: {promedio}")

# Escalar solo el objeto sin añadir lienzo negro
for nombre_archivo, pixeles_1 in one_pixel_e:
    ruta = os.path.join(carpeta, nombre_archivo)
    img = Image.open(ruta).convert('L')
    
    arr = np.array(img)
    binaria = np.where(arr < 128, 0, 1)

    # Recortar el objeto (bounding box)
    coords = np.argwhere(binaria == 1)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    objeto_recortado = arr[y_min:y_max+1, x_min:x_max+1]
    img_objeto = Image.fromarray(objeto_recortado.astype(np.uint8))

    # Calcular factor de escala
    alpha = math.sqrt(promedio / pixeles_1)

    nuevo_ancho = int(img_objeto.width * alpha)
    nuevo_alto = int(img_objeto.height * alpha)
    img_escalada = img_objeto.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    # Guardar directamente sin agregar márgenes
    salida = os.path.join(output_dir, nombre_archivo)
    img_escalada.save(salida)

    print(f"{nombre_archivo} → Factor a = {alpha:.4f}")

# PARTE 4: Momentos Normalizados ηpq
def calcular_eta_pq(binaria, max_p=2, max_q=2):
    coords = np.argwhere(binaria == 1)
    if len(coords) == 0:
        return {}

    # Calcular el centroide
    x = coords[:, 1]
    y = coords[:, 0]
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    # Momento central
    def mu_pq(p, q):
        return np.sum((x - x_bar) ** p * (y - y_bar) ** q)

    mu00 = mu_pq(0, 0)
    resultados = {}
    
    # Calcular ηpq
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p + q >= 1:
                mu = mu_pq(p, q)
                gamma = (p + q) / 2 + 1
                eta = mu / (mu00 ** gamma) if mu00 != 0 else 0
                resultados[(p, q)] = eta
    return resultados

# PARTE 5: Comparación de momentos
print("\n--- Comparación de Momentos Normalizados ηpq ---")

for archivo, _ in one_pixel_results:
    # Imagen original
    ruta_original = os.path.join(carpeta, archivo)
    img = Image.open(ruta_original).convert('L')
    arr = np.array(img)
    binaria = np.where(arr < 128, 1, 0)
    eta_original = calcular_eta_pq(binaria)

    # Imagen escalada
    ruta_escalada = os.path.join(output_dir, archivo)
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
