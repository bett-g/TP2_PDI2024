import matplotlib.pyplot as plt
import numpy as np
import cv2

# Función para mostrar una imagen
def mostrar_imagen(titulo, img, cmap='gray'):
    plt.figure()
    plt.title(titulo)
    plt.imshow(img, cmap=cmap)
    plt.show()

# Función para obtener una subimagen a partir de una imagen y sus estadísticas
def sub_img(img, stats):
    # Coordenadas horizontales y verticales, ancho y largo de la subimagen
    coor_h = stats[cv2.CC_STAT_LEFT]
    coor_v = stats[cv2.CC_STAT_TOP]
    ancho = stats[cv2.CC_STAT_WIDTH]
    largo = stats[cv2.CC_STAT_HEIGHT]
    return img[coor_v:coor_v + largo, coor_h: coor_h + ancho]

# Muestra las distintas etapas de procesamiento para la detección y segmentación de la patente
def procesar_patente(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mostrar_imagen('Etapas Ejericio a) - Imagen Original', img)
    
    K_SIZE_GAUSSIAN_BLUR = (1, 21)
    # Aplicar filtro de desenfoque gaussiano
    blur = cv2.GaussianBlur(img, K_SIZE_GAUSSIAN_BLUR, 0)
    mostrar_imagen('Imagen con Blur', blur)

    # Detectar bordes utilizando el algoritmo de Canny
    img_canny = cv2.Canny(blur, 250, 300)
    mostrar_imagen('Detección de Bordes con Canny', img_canny)

    # Realizar operación de cierre morfológico
    elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
    img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)
    mostrar_imagen('Morfología - Operación de Cierre', img_cierre)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(img_cierre)
    AREA_MIN = 400
    # Filtrar componentes conectados basados en su área y relación de aspecto
    filtro = (
        (stats[:, cv2.CC_STAT_AREA] >= AREA_MIN) & 
        (stats[:, cv2.CC_STAT_HEIGHT] < stats[:, cv2.CC_STAT_WIDTH]))
    labels_filtrado = np.argwhere(filtro).flatten().tolist()
    idx_patente = labels_filtrado[-1]
    stats_patente = stats[idx_patente]
    sub_imagen = sub_img(img, stats_patente)
    mostrar_imagen('Sub-Imagen de la Patente', sub_imagen)

    mascara = (labels == idx_patente).astype('uint8') * 255
    K_ANCHO =  15
    K_LARGO =  3
    elemento_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (K_ANCHO, K_LARGO))
    # Dilatar la máscara
    img_dil = cv2.dilate(mascara, elemento_dil)
    mostrar_imagen('Máscara Dilatada', img_dil)
    
    # Segmentar la patente utilizando la máscara dilatada
    patente_segmentada = np.bitwise_and(img, img_dil)
    mostrar_imagen('Resultado final: Patente Segmentada', patente_segmentada)

    return patente_segmentada, sub_imagen

# Determinar si una región de interés es una letra
def letra(stats, area_max=100, area_min=15):
    return (stats[cv2.CC_STAT_AREA] < area_max and stats[cv2.CC_STAT_AREA] > area_min) and (stats[cv2.CC_STAT_WIDTH] < stats[cv2.CC_STAT_HEIGHT])

# Contar el número de letras en una imagen
def contar_letras(img):
    n, _, stats, _ = cv2.connectedComponentsWithStats(img)
    letras = 0
    for i in range(1, n):
        if letra(stats[i]):
            letras += 1
    return letras

# Calcular el umbral óptimo para binarizar una imagen
def umbral_optimo(img_tupla):
    mediana = np.median(img_tupla[1]) * 0.50
    suma = 0
    encontrado = False
    while not encontrado:
        umbralada = (img_tupla[0] > (mediana + suma)).astype('uint8')
        cantidad_letras = contar_letras(umbralada)
        if cantidad_letras != 6:
            suma += 1
        elif (mediana + suma) >= 255:
            mediana = 0
            suma = -1
            encontrado = True
        else:
            encontrado = True
    return mediana + suma

# Dibujar un cuadro alrededor de una región de interés
def region_interes(img, stats, color, box=True, text=None, thickness=3, fontScale=10):
    left = stats[cv2.CC_STAT_LEFT]
    top = stats[cv2.CC_STAT_TOP]
    width = stats[cv2.CC_STAT_WIDTH]
    height = stats[cv2.CC_STAT_HEIGHT]
    if box:
        cv2.rectangle(img, (left, top), (left + width, top + height), color=color, thickness=thickness)
    if text:
        cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=color, thickness=thickness)

# Segmentar caracteres en una región de interés
def segmentar_caract(img_tupla, box=False):
    umbral = umbral_optimo(img_tupla)
    umbralada = (img_tupla[0] > umbral).astype('uint8')
    mostrar_imagen('Imagen Umbralada', umbralada)
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(umbralada)
    img_caracteres = np.zeros_like(img_tupla[0] > umbral, dtype='uint8')

    for i in range(1, n):
        if letra(stats[i]):
            img_caracteres[labels == i] = 255
            if box:
                region_interes(img_caracteres, stats[i], 100, thickness=1)
    mostrar_imagen('Caracteres Segmentados con Cuadros', img_caracteres)
    
    return img_caracteres

# Procesar caracteres en una imagen de patente
def procesar_caracteres(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mostrar_imagen('Etapas Ejericio b) - Imagen Original', img)
    
    K_SIZE_GAUSSIAN_BLUR = (1, 21)
    # Aplicar filtro de desenfoque gaussiano
    blur = cv2.GaussianBlur(img, K_SIZE_GAUSSIAN_BLUR, 0)
    mostrar_imagen('Imagen con Blur', blur)

    # Detectar bordes utilizando el algoritmo de Canny
    img_canny = cv2.Canny(blur, 250, 300)
    mostrar_imagen('Detección de Bordes con Canny', img_canny)

    # Realizar operación de cierre morfológico.
    elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
    img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)
    mostrar_imagen('Morfología - Operación de Cierre', img_cierre)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(img_cierre)
    AREA_MIN = 400
    # Filtrar componentes conectados basados en su área y relación de aspecto
    filtro = (
        (stats[:, cv2.CC_STAT_AREA] >= AREA_MIN) & 
        (stats[:, cv2.CC_STAT_HEIGHT] < stats[:, cv2.CC_STAT_WIDTH]))
    labels_filtrado = np.argwhere(filtro).flatten().tolist()

    idx_patente = labels_filtrado[-1]
    stats_patente = stats[idx_patente]
    sub_imagen = sub_img(img, stats_patente)
    
    mascara = (labels == idx_patente).astype('uint8') * 255
    K_ANCHO =  15
    K_LARGO =  3
    elemento_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (K_ANCHO, K_LARGO))
    # Dilatar la máscara.
    img_dil = cv2.dilate(mascara, elemento_dil)
    mostrar_imagen('Máscara Dilatada', img_dil)
    
    # Segmentar la patente utilizando la máscara dilatada.
    patente_segmentada = np.bitwise_and(img, img_dil)
    mostrar_imagen('Patente Segmentada', patente_segmentada)

    img_tupla = (patente_segmentada, sub_imagen)
    img_caracteres = segmentar_caract(img_tupla, box=True)
    mostrar_imagen('Resultado final: Caracteres Segmentados con Cuadros', img_caracteres)

# Menú principal
def main():
    PATENTES_A = ['img/img01.png']
    PATENTES_B = ['img/img01.png']
    
    # Procesar patentes para el ejercicio A
    print("Ejercicio A:")
    for patente in PATENTES_A:
        print(f"Procesando patente: {patente}")
        procesar_patente(patente)
    
    # Procesar patentes para el ejercicio B
    print("\nEjercicio B:")
    for patente in PATENTES_B:
        print(f"Procesando patente: {patente}")
        procesar_caracteres(patente)

main()


"""
PRUEBAS VARIAS PATENTES:
def main():
    PATENTES_A = [f'img/img{i:02}.png' for i in range(1, 13)]
    PATENTES_B = [f'img/img{i:02}.png' for i in range(1, 13)]
    
    # Procesar patentes para el ejercicio A
    print("Ejercicio A:")
    for patente in PATENTES_A:
        print(f"Procesando patente: {patente}")
        procesar_patente(patente)
    
    # Procesar patentes para el ejercicio B
    print("\nEjercicio B:")
    for patente in PATENTES_B:
        print(f"Procesando patente: {patente}")
        procesar_caracteres(patente)

# Ejecutar la función principal
main()
"""