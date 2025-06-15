import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_folder = 'images'
output_folder = 'binarias'
cell_size = 20
pink_color = '#ff69b4'

os.makedirs(output_folder, exist_ok=True)

def plot_and_save_image(binary_image, title, output_path):
    rows, cols = binary_image.shape
    # Ajustamos figsize para evitar error de memoria (dividimos para reducir tamaño)
    fig, ax = plt.subplots(figsize=(cols * cell_size / 100, rows * cell_size / 100))
    
    ax.set_xlim(0, cols * cell_size)
    ax.set_ylim(0, rows * cell_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Dibujar cuadrícula
    for y in range(rows + 1):
        ax.axhline(y * cell_size, color='lightgray', linewidth=0.5)
    for x in range(cols + 1):
        ax.axvline(x * cell_size, color='lightgray', linewidth=0.5)

    # Encontrar contornos (asegurarse de que es uint8)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        polygon = plt.Polygon(
            [(point[0][0] * cell_size, point[0][1] * cell_size) for point in cnt],
            edgecolor=pink_color,
            facecolor='none',
            linewidth=2.0
        )
        ax.add_patch(polygon)

    ax.set_title(title)
    ax.axis('off')

    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.gif')):
        filepath = os.path.join(image_folder, filename)
        print(f"Leyendo imagen: {filepath}")

        try:
            pil_img = Image.open(filepath).convert('L')  # Escala de grises
            img = np.array(pil_img)
        except Exception as e:
            print(f'Error al abrir la imagen {filepath}: {e}')
            continue

        print(f"Imagen cargada con shape: {img.shape}")

        ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        if binary is None:
            print("Error: binary es None después del threshold")
            continue

       

        output_path = os.path.join(output_folder, f'contornos_{filename}.png')
        plot_and_save_image(binary, f'Contornos: {filename}', output_path)

        print(f'Guardado: {output_path}')
