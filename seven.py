import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd

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
