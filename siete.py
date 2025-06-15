import os
import numpy as np
from PIL import Image
import cv2

input_folder = 'images'
output_folder = 'trasladadas'
os.makedirs(output_folder, exist_ok=True)

tabla_resultados = []

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.gif'):
        path = os.path.join(input_folder, filename)
        
        # Abrir con PIL
        pil_img = Image.open(path)
        
        # Convertir a escala de grises y luego a numpy array
        gray_img = pil_img.convert('L')
        img_np = np.array(gray_img)
        
        # Binarizar (0,1)
        _, binary = cv2.threshold(img_np, 127, 1, cv2.THRESH_BINARY)
        
        # Calcular centro de masa imagen original
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) == 0:
            continue  # No hay píxeles blancos
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        x_cm = np.mean(x_coords)
        y_cm = np.mean(y_coords)

        # Trasladar imagen
        shift_x, shift_y = 10, 10
        trasladada = np.zeros_like(binary)
        for y, x in zip(y_coords, x_coords):
            new_x = x + shift_x
            new_y = y + shift_y
            if 0 <= new_x < binary.shape[1] and 0 <= new_y < binary.shape[0]:
                trasladada[new_y, new_x] = 1

        # Calcular centro de masa imagen trasladada (ajustado)
        coords_t = np.column_stack(np.where(trasladada > 0))
        y_coords_t, x_coords_t = coords_t[:, 0], coords_t[:, 1]
        if len(coords_t) == 0:
            continue  # Por si acaso no quedan píxeles tras traslado
        x_cm_t = np.mean(x_coords_t)
        y_cm_t = np.mean(y_coords_t)

        # Guardar imagen trasladada como PNG
        salida_path = os.path.join(output_folder, f"trasladada_{filename[:-4]}.png")
        cv2.imwrite(salida_path, trasladada * 255)

        # Calcular momentos centrales μ_pq con respecto al centro de masa trasladado
        mu = {}
        for p in range(3):
            for q in range(3):
                suma = 0.0
                for y, x in zip(y_coords_t, x_coords_t):
                    suma += ((x - x_cm_t) ** p) * ((y - y_cm_t) ** q)
                mu[(p, q)] = suma

        tabla_resultados.append({
            'imagen': filename,
            'x_cm': x_cm_t,
            'y_cm': y_cm_t,
            'mu_00': mu[(0, 0)],
            'mu_01': mu[(0, 1)],
            'mu_10': mu[(1, 0)],
            'mu_11': mu[(1, 1)],
            'mu_02': mu[(0, 2)],
            'mu_20': mu[(2, 0)],
            'mu_12': mu[(1, 2)],
            'mu_21': mu[(2, 1)],
            'mu_22': mu[(2, 2)],
        })

# Mostrar resultados
print(f"{'Imagen':<20} {'x_cm':<8} {'y_cm':<8} {'mu_00':<15} {'mu_01':<15} {'mu_10':<15} {'mu_11':<15} {'mu_02':<15} {'mu_20':<15} {'mu_12':<15} {'mu_21':<15} {'mu_22':<15}")
for fila in tabla_resultados:
    print(f"{fila['imagen']:<20} {fila['x_cm']:<8.2f} {fila['y_cm']:<8.2f} {fila['mu_00']:<15.2f} {fila['mu_01']:<15.2f} {fila['mu_10']:<15.2f} {fila['mu_11']:<15.2f} {fila['mu_02']:<15.2f} {fila['mu_20']:<15.2f} {fila['mu_12']:<15.2f} {fila['mu_21']:<15.2f} {fila['mu_22']:<15.2f}")
