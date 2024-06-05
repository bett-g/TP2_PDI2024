import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def detect_circles(gray_image):
    # Aplicar filtro Gaussiano para eliminar ruido
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Detectar bordes usando Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar círculos en la imagen mediante el algoritmo de detección de círculos de Hough
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=50)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    else:
        return None

# Ruta de la imagen
image_path = 'img/placa.png'

# Verificar si el archivo existe
if not os.path.exists(image_path):
    print("Error: La ruta del archivo es incorrecta o el archivo no existe.")
else:
    # Leer la imagen en escala de grises
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print("Error: No se pudo leer la imagen. Verifica la ruta y el formato del archivo.")
    else:
        # Aplicar filtro Gaussiano para eliminar ruido
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Detectar bordes usando Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Dilatar los bordes para cerrar contornos incompletos
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Aplicar operación de cierre para cerrar pequeños agujeros dentro de los contornos
        closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

        # Detectar contornos en la imagen
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar contornos filtrados en la imagen original
        contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # Detectar círculos en la imagen contorneada
        detected_circles = detect_circles(closed_edges)
        
        # Visualizar círculos detectados
        if detected_circles is not None:
            for circle in detected_circles:
                x, y, radius = circle
                cv2.circle(contour_image, (x, y), radius, (0, 255, 0), 2)

        # Visualizar imagen con contornos y círculos detectados
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title('Contornos y círculos detectados')
        plt.axis('off')
        plt.show()

        # Clasificación de componentes electrónicos y conteo de resistencias
        capacitor_sizes = {"pequeño": 0, "mediano": 0, "grande": 0}
        resistor_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # Contador de resistores
            if  350 <= area < 615: 
                resistor_count += 1

        # Clasificación de capacitadores por tamaño
        capacitor_sizes = {"pequeño": 0, "mediano": 0, "grande": 0}
        for circle in detected_circles:
            x, y, radius = circle
            area = np.pi * (radius ** 2)
            if 2000 <= area < 10000: 
                # Clasificar capacitadores por tamaño
                if radius < 30:
                    capacitor_sizes["pequeño"] += 1
                elif 30 <= radius < 40:
                    capacitor_sizes["mediano"] += 1
                else:
                    capacitor_sizes["grande"] += 1

        # Imprimir resultados
        print("Clasificación de capacitores por tamaño:")
        for size, count in capacitor_sizes.items():
            print(f"{size.capitalize()}: {count}")
        print(f"Cantidad de resistencias eléctricas: {resistor_count}")