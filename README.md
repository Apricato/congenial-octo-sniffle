# 📘 README - TAREA 2: GRAFICACIÓN

**Universidad Autónoma del Estado de Aguascalientes**  
**Ingeniería en Computación Inteligente**  
**Asignatura:** Graficación  
**Profesor:** M.C. Hermilo Sánchez Cruz  
**Alumno:** Tania López Ibarra — ID: 336673  
**Fecha de entrega:** 25/05/2025  

---

## 📁 ESTRUCTURA DE CARPETAS

```
📁 images/ → Imágenes originales del dataset MPEG7 (.gif, .png)  
📁 image_ruido/ → Versión ruidosa de las imágenes para prueba de limpieza  
📁 escaladas/ → Imágenes escaladas para igualar la cantidad de píxeles 1  
📁 cuadriculadas/ → Imágenes con celdas cuadradas dibujadas donde hay píxeles 1  
📁 contornos/ → Contornos obtenidos por vecindad-8 (por erosión)  
📁 trasladadas/ → Imágenes trasladadas + tabla con centroide y momentos  
📁 rotadas/ → Imágenes rotadas 45° + tabla con momentos de Hu antes/después  
📁 morfologia/  
├── ruido/ → Imágenes tras aplicar apertura para quitar ruido (solo image_ruido)  
├── suavizado/ → Imágenes con cierre (closing) para suavizar bordes  
├── relleno/ → Imágenes con huecos internos rellenados  
└── esqueleto/ → Imágenes con esqueletización  

📄 readme.txt → Este archivo
```


## ⚙️ REQUISITOS PARA EJECUTAR

- Python 3.8 o superior
- Librerías necesarias:

pip install numpy pillow opencv-python pandas matplotlib scikit-image

Mostrar siempre los detalles

---

## 📝 DESCRIPCIÓN DE LOS PROGRAMAS

### 1. Conteo de píxeles 1
- Desde `images/`, binariza y cuenta píxeles del objeto (valor 1).

### 2. Escalamiento
- Escala imágenes para igualar el área (número de 1-pixeles).
- Centra en lienzo común. Resultado: `escaladas/`.

### 3. Momentos normalizados ηpq
- Calcula ηpq antes y después de escalar. Confirma invariancia de escala.

### 4. Celdas rosas
- Dibuja celdas sobre píxeles 1 (color rosa). Resultado: `cuadriculadas/`.

### 5. Contornos
- Usa vecindad-8 mediante erosión y resta. Resultado: `contornos/`.

### 6. Traslación
- Desplaza figuras, calcula nuevo centroide y momentos μpq. Resultado: `trasladadas/`.

### 7. Rotación
- Rota imágenes 45°. Calcula momentos de Hu antes/después. Resultado: `rotadas/`.

### 8. Morfología
- Aplica operadores sobre `escaladas/` y `image_ruido/`.
- **ruido** (apertura): limpia imágenes ruidosas
- **suavizado** (cierre): suaviza bordes
- **relleno**: rellena huecos
- **esqueleto**: genera esqueletos

Resultados guardados en: `morfologia/` con subcarpetas por operación.

---

---

## 📌 NOTAS

- Las imágenes `.gif` se convierten a `.png` para compatibilidad con OpenCV.
- ηpq permanece casi constante tras escalamiento → ✔ invariante
- Momentos de Hu estables ante rotación → ✔ invariante
- Las operaciones morfológicas cumplen con su función:
- Apertura limpia ruido
- Cierre suaviza bordes
- Relleno cierra huecos internos
- Esqueleto reduce a líneas centrales

---

## ✅ CONCLUSIÓN

Esta práctica aplica procesamiento binario, análisis morfológico, transformaciones geométricas y momentos invariantes para caracterizar objetos. Los resultados validan la teoría y refuerzan el uso de morfología matemática en visión computacional.
