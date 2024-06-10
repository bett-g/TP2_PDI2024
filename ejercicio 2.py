import matplotlib.pyplot as plt
from os.path import basename
from math import floor
import numpy as np
import imutils
import cv2

def calcular_factor_forma(img):
   '''
   Recibe una sub-imagen y devuelve su factor de forma
   '''
   ext_cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   area = cv2.contourArea(ext_cont[0])
   perimeter = cv2.arcLength(ext_cont[0], True)
   rho = 4 * np.pi * area /(perimeter ** 2)
   return rho

def contar_circulos(imagen):
   '''
   Recibe una imagen y devuelve la cantidad de círculos que hay en ella.
   '''
   imagen = cv2.medianBlur(imagen, 7)

   circles = cv2.HoughCircles(imagen,
                              cv2.HOUGH_GRADIENT,
                              1, 20,
                              param1=50, param2=50,
                              minRadius=20, maxRadius=50)

   n = 0

   if isinstance(circles, np.ndarray):
      n = len(circles[0])

   return n

def graficar_caja(img, stats, color, box=True, text=None,
                  thickness=3, fontScale=10):
   '''
   Función para graficar un boundig box 
   sobre una imagen dada.
   '''

   left   = stats[cv2.CC_STAT_LEFT]
   top    = stats[cv2.CC_STAT_TOP]
   width  = stats[cv2.CC_STAT_WIDTH]
   height = stats[cv2.CC_STAT_HEIGHT]

   if box:
      cv2.rectangle(img, 
                    (left, top), 
                    (left + width,top + height),
                    color=color, thickness=thickness)
      
   if text:
      cv2.putText(img, text, 
                  (left, top - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=fontScale,
                  color=color,
                  thickness=thickness)

def obtener_sub_imagen(img, stats):
    '''
    Recibe una imagen y stats sobre un área de interés,
    devuelve la sub-imagen correspondiente a esa área.
    '''

    coor_h = stats[cv2.CC_STAT_LEFT] 
    coor_v = stats[cv2.CC_STAT_TOP]

    ancho  = stats[cv2.CC_STAT_WIDTH]   
    largo  = stats[cv2.CC_STAT_HEIGHT]

    return img[coor_v:coor_v + largo, coor_h: coor_h + ancho]

def rellenar(img):
   '''
   Recibe una imagen binaria, devuelve la misma imagen con
   las formas huecas rellenas.
   '''
   img_flood_fill = img.copy().astype('uint8')
   h, w = img.shape[:2]
   mask = np.zeros((h+2, w+2), np.uint8)
   cv2.floodFill(img_flood_fill, mask, (0,0), 255)
   img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)
   img_fh = img | img_flood_fill_inv
   return img_fh 

def procesar_patente(path, plot=False):
    '''
    Recibe el path a una imagen, devuelve una tupla con dos 
    elementos donde el primero es la imagen original con la 
    patente segmentada y el segundo es la sub-imagen de la patente.
    '''
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    K_SIZE_GAUSSIAN_BLUR = (1, 21)

    # Blur y detección de bordes
    blur = cv2.GaussianBlur(img, K_SIZE_GAUSSIAN_BLUR, 0)
    # plt.imshow(blur, cmap='gray'), plt.show()
    img_canny = cv2.Canny(blur, 250, 300)
    # plt.imshow(img_canny, cmap='gray'), plt.show()

    # Morfología - Operación de cierre
    elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
    img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)

    # Componente conectados, filtramos por área
    _, labels, stats, _  = cv2.connectedComponentsWithStats(img_cierre)

    AREA_MIN = 400
    filtro = (
        (stats[:, cv2.CC_STAT_AREA] >= AREA_MIN) & 
        (stats[:, cv2.CC_STAT_HEIGHT] < stats[:, cv2.CC_STAT_WIDTH]))

    labels_filtrado = np.argwhere(filtro).flatten().tolist()

    # Nos quedamos con el último índice donde está la patente
    idx_patente = labels_filtrado[-1]
    stats_patente = stats[idx_patente]

    # Creación de la máscara para segmentar la patente
    sub_imagen = imutils.obtener_sub_imagen(img, stats_patente)
    mascara = (labels == idx_patente).astype('uint8') * 255

    K_ANCHO =  15 # floor(sub_imagen.shape[0]  * 0.20)  # 15
    K_LARGO =  3  # floor(sub_imagen.shape[1]  * 0.15)  # 3

    elemento_dil = cv2.getStructuringElement(
                        cv2.MORPH_RECT, 
                        (K_ANCHO, K_LARGO))

    img_dil = cv2.dilate(mascara, elemento_dil)

    pataente_segmentada = np.bitwise_and(img, img_dil)

    if plot:
        plt.figure()
        plt.imshow(pataente_segmentada, cmap='gray')
        plt.title(basename(path))
        plt.show()

    return pataente_segmentada, sub_imagen

def umbralar(img_tupla):
    '''
    Devuelve un umbral para segmentar las letras de la patente.
    '''

    mediana = np.median(img_tupla[1]) * 0.50
    suma = 0

    encontrado = False

    while not encontrado:
        umbralada = (img_tupla[0] > (mediana + suma)).astype('uint8')
        cantidad_letras = contar_letras(umbralada)

        if cantidad_letras != 6:
            suma += 1
        elif (mediana + suma) >= 255:
            mediana =  0
            suma    = -1
            encontrado = True
        else:
            encontrado = True
    
    return mediana + suma

def es_letra(stats, area_max=100, area_min=15):
    '''
    Determina si una sub-imagen es una letra en base a sus stats
    '''
    return (stats[cv2.CC_STAT_AREA] < area_max \
            and stats[cv2.CC_STAT_AREA] > area_min) and \
            (stats[cv2.CC_STAT_WIDTH] < stats[cv2.CC_STAT_HEIGHT])

def contar_letras(img):
    '''
    Recibe una imagen umbralada y cuenta cuantas letras tiene
    '''

    n, _, stats, _ = cv2.connectedComponentsWithStats(img)
    
    letras = 0
    for i in range(1, n):
        if es_letra(stats[i]):
            letras += 1

    return letras

def segmentar_caracteres(img_tupla, box=False):
    '''
    Segmenta los caracteres de una sub-imagen buscando un umbral,
    devuelve una imagen binaria con los 6 caracteres. 
    '''

    umbral = umbralar(img_tupla)
    umbralada = (img_tupla[0] > umbral).astype('uint8')

    n, labels, stats, _ = cv2.connectedComponentsWithStats(umbralada)

    img_caracteres = np.zeros_like(img_tupla[0] > umbral, dtype='uint8')

    for i in range(1, n):
        if es_letra(stats[i]):
            img_caracteres[labels == i] = 255
            if box:
                imutils.graficar_caja(img_caracteres, stats[i], 100, thickness=1)

    return img_caracteres

def main():
    PATENTES = [f'img/img{i:02}.png' for i in range(1, 13)]
    
    recortes = list()

    for patente in PATENTES:
        recortes.append(procesar_patente(patente))

    for recorte in recortes:
        plt.figure()
        plt.imshow(
                segmentar_caracteres(recorte, box=True),
                cmap='gray')
        plt.show()

main()

# Gráficas para el informe
# =========================

path = 'img/img01.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray'), plt.show()
blur = cv2.GaussianBlur(img, (1, 21), 0)
img_canny = cv2.Canny(blur, 250, 300)

plt.subplot(131)
plt.title('Imagen en escala de grises')
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.title('Imagen con blur')
plt.imshow(blur, cmap='gray')
plt.subplot(133)
plt.title('Detección de bordes con Canny')
plt.imshow(img_canny, cmap='gray')
plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)

plt.subplot(121)
plt.title('Canny')
plt.imshow(img_canny, cmap='gray')
plt.subplot(122)
plt.title('Morfología - Operación de clausura')
plt.imshow(img_cierre, cmap='gray')
plt.show()

img1_tupla = procesar_patente(path)

plt.subplot(121)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title('Segmentada')
plt.imshow(img1_tupla[0], cmap='gray')
plt.show()


plt.subplot(121)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title('Caracteres segmentados')
plt.imshow(segmentar_caracteres(img1_tupla), cmap='gray')
plt.show()