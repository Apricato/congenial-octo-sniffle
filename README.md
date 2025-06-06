





### CONOCIMIENTO TEORICO DE LA TAREA 2: GRAFICACIÓN ###
__________________________________________________________

## Descripción

Este proyecto procesa 10 imágenes binarias de objetos tomados del dataset MPEG-7. Las actividades cubren procesamiento básico, morfología matemática, y análisis de invariantes.

--
## Descripción

1. **Selección de imágenes binarizadas** (blanco y negro)

No tiene chiste, son imagenes en formato .gif

2. **Conteo de píxeles del objeto** (`1` o `0`, dependiendo del formato)


donde los pixeles cero corresponden al pixel negro.
Pregunta los uno pixeles se entienden como los pixeles prendidos, aunque en general en el tratamiento de imagenes deberia entenderse que los pixeles 1 son los negros y los blancos representan el cero.

3. **Escalado para igualar área del objeto**

El valor de a o el factor de escala serra a el cual representa o corresponde a  :

![alt text](image.png)
a= raiz de area deseada sobre area actual
donde el are deseada es el area promedio sobre el area actual la cual es la cantidad de uno pixeles disponibles en la imagen


4. **Cálculo de invariantes de escala** `η_pq`
5. **Obtención de contornos** (vecindad-8)
6. **Cálculo del centro de masa y momentos centrales**
7. **Rotaciones y momentos de Hu**
8. **Aplicación de operadores morfológicos**:
   - Eliminación de ruido
   - Suavizado de bordes
   - Relleno de huecos
   - Esqueletización
9. **Análisis y conclusiones**
