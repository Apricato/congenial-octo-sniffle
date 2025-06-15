import os
import numpy as np
from PIL import Image
from skimage.morphology import opening, closing, disk, skeletonize
from scipy.ndimage import binary_fill_holes

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
