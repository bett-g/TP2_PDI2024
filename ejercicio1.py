import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen de la placa
imagen_placa = cv2.imread("img/placa.png")

# Convertir la imagen de BGR a escala de grises
gray_image = cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2GRAY)

# Aplicar suavizado a la imagen
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Crear una copia de la imagen para dibujar los contornos
imagen_contornos_resistores = imagen_placa.copy()

# Cargar la imagen de la placa
imagen_placa = cv2.imread("img/placa.png")

# Convertir la imagen de BGR a HSV
hsv_image = cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2HSV)

# Definir diccionario de colores
COLORS = {
    'gray': (128, 128, 128),  # gris
    'silver': (192, 192, 192)  # plateado
}

# Definir rangos de colores en HSV ajustados
lower_gray = np.array([0, 0, 50])
upper_gray = np.array([200, 50, 150])

lower_silver = np.array([0, 0, 150])
upper_silver = np.array([180, 50, 255])

# Función para visualizar rangos de colores en la imagen original
def visualizar_rangos_color(imagen, lower_color, upper_color, color_name):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), COLORS[color_name], 2)
    cv2.putText(imagen, f'{color_name.upper()} Range', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[color_name], 2)

# Visualizar rangos de colores en la imagen original
imagen_placa_color_ranges = imagen_placa.copy()
visualizar_rangos_color(imagen_placa_color_ranges, lower_gray, upper_gray, 'gray')
visualizar_rangos_color(imagen_placa_color_ranges, lower_silver, upper_silver, 'silver')

# Mostrar la imagen con los rangos de colores visualizados
plt.imshow(cv2.cvtColor(imagen_placa_color_ranges, cv2.COLOR_BGR2RGB))
plt.title('Rangos de colores identificados')
plt.axis('off')
plt.show()

# Segmentar componentes basados en color
mask_chip = cv2.inRange(hsv_image, lower_gray, upper_gray)
mask_capacitor = cv2.inRange(hsv_image, lower_silver, upper_silver)

kernel = np.ones((5, 5), np.uint8)

# Aplicar preprocesamiento (suavizado) antes de Canny
blurred_image = cv2.medianBlur(imagen_placa, ksize=8)

# Aplicar el algoritmo de Canny para la detección de bordes
edges_resistencia = cv2.Canny(blurred_image, 0.20*255, 0.40*255)
dilated_edges = cv2.dilate(edges_resistencia, kernel, iterations=1)

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
f_mg = cv2.morphologyEx(edges_resistencia, cv2.MORPH_GRADIENT, kernel)

# Mostrar los bordes detectados por Canny
plt.imshow(f_mg, cmap='gray')
plt.title('Bordes detectados por Canny')
plt.axis('off')
plt.show()

# Función para post-procesar las máscaras
def post_process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Post-procesar las máscaras
mask_chip = post_process_mask(mask_chip)
mask_capacitor = post_process_mask(mask_capacitor)

# Encontrar los contornos en las máscaras
contornos_resistencia, _ = cv2.findContours(f_mg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Encontrar los contornos en los bordes detectados por Canny (para chips y capacitores)
contornos_chip, _ = cv2.findContours(mask_chip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_capacitor, _ = cv2.findContours(mask_capacitor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostrar la imagen con los contornos detectados y clasificados
def mostrar_contornos(imagen, contornos_resistencia, contornos_chip, contornos_capacitor):
    for cnt in contornos_resistencia:
        area = cv2.contourArea(cnt)
        if 2000 < area < 2800:  # Ajusta los umbrales según tus necesidades
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imagen, 'RESISTENCIA', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Dibujar contornos rectangulares para chips
    for cnt in contornos_chip:
        area = cv2.contourArea(cnt)
        if area < 90000 or area > 100000:  
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imagen, 'CHIP', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dibujar contornos circulares para capacitores
    for cnt in contornos_capacitor:
        area = cv2.contourArea(cnt)
        if area < 7500 or area > 130000: 
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(imagen, center, radius, (255, 0, 0), 2)
        cv2.putText(imagen, 'CAPACITOR', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Crear una copia de la imagen original para dibujar los contornos detectados
imagen_contornos_clasificados = imagen_placa.copy()

# Dibujar los contornos clasificados en la imagen copiada
mostrar_contornos(imagen_contornos_clasificados, contornos_resistencia, contornos_chip, contornos_capacitor)

# Mostrar la imagen con los contornos detectados y clasificados
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(imagen_contornos_clasificados, cv2.COLOR_BGR2RGB))
plt.title('Componentes electrónicos detectados y clasificados')
plt.axis('off')
plt.show()

# Clasificar capacitores por tamaño
def classify_capacitors_by_size(capacitors):
    small = []
    medium = []
    large = []

    for cnt in capacitors:
        area = cv2.contourArea(cnt)
        radius = np.sqrt(area / np.pi)
        if 45 <= radius < 80:
            small.append(cnt)
        elif 80 <= radius <= 140:
            medium.append(cnt)
        elif 140 <= radius < 200:
            large.append(cnt)
    return small, medium, large

def clasificar_resistencia(contornos_resistencia):
    resistor_count = 0
    for contour in contornos_resistencia:
            area = cv2.contourArea(contour)
            # Contador de resistores
            if  2000 < area < 2800: 
                resistor_count += 1

    return resistor_count

# Clasificar los capacitores
small_capacitors, medium_capacitors, large_capacitors = classify_capacitors_by_size(contornos_capacitor)

# Dibujar los capacitores clasificados en la imagen
output_capacitors_size = imagen_placa.copy()

for cnt in small_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(output_capacitors_size, center, radius, (255, 0, 0), 2)
    cv2.putText(output_capacitors_size, 'PEQUENIO', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

for cnt in medium_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(output_capacitors_size, center, radius, (0, 255, 0), 2)
    cv2.putText(output_capacitors_size, 'MEDIANO', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

for cnt in large_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(output_capacitors_size, center, radius, (0, 0, 255), 2)
    cv2.putText(output_capacitors_size, 'GRANDE', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Mostrar la imagen con los capacitores clasificados
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output_capacitors_size, cv2.COLOR_BGR2RGB))
plt.title('Capacitores clasificados por tamaño')
plt.axis('off')
plt.show()

# Contar las resistencias eléctricas
resistor_count = clasificar_resistencia(contornos_resistencia)
#resistor_count = len(resistor_count)

print(f"Cantidad de resistencias eléctricas:",resistor_count)

small_count = len(small_capacitors)
medium_count = len(medium_capacitors)
large_count = len(large_capacitors)
print(f"Cantidad de capacitores pequeños: {small_count}")
print(f"Cantidad de capacitores medianos: {medium_count}")
print(f"Cantidad de capacitores grandes: {large_count}")
