

import numpy as np  # Para operaciones matemáticas
from PIL import Image, ImageDraw
import os
import math
import cv2  # Para cálculo de momentos
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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