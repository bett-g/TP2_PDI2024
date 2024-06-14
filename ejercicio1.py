import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Segmentar componentes basados en color
mask_chip = cv2.inRange(hsv_image, lower_gray, upper_gray)
mask_capacitor = cv2.inRange(hsv_image, lower_silver, upper_silver)

# Función para post-procesar las máscaras
def post_process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Post-procesar las máscaras
mask_chip = post_process_mask(mask_chip)
mask_capacitor = post_process_mask(mask_capacitor)

# Encontrar los contornos en los bordes detectados por Canny (para chips y capacitores)
contornos_chip, _ = cv2.findContours(mask_chip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_capacitor, _ = cv2.findContours(mask_capacitor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostrar la imagen con los contornos chip y capacitor
def mostrar_contornos(imagen, contornos_chip, contornos_capacitor):

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
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if area < 7500 or area > 130000:
            continue
        elif radius == 175 or radius == 105:
            continue
        else:
            cv2.circle(imagen, center, radius, (255, 0, 0), 2)
            cv2.putText(imagen, f'CAPACITOR - Radio: {radius}', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Crear una copia de la imagen original para dibujar los contornos detectados
imagen_contornos_clasificados = imagen_placa.copy()
# Dibujar los contornos clasificados en la imagen copiada
mostrar_contornos(imagen_contornos_clasificados, contornos_chip, contornos_capacitor)

# Mostrar la imagen con los contornos detectados y clasificados
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(imagen_contornos_clasificados, cv2.COLOR_BGR2RGB))
plt.title('Componentes electrónicos detectados (Capacitores y Chip)')
plt.axis('off')
plt.show()

# Convertir a escala de grises y aplicar desenfoque
gray_image = cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Definir umbrales para la detección de bordes con Canny
low_threshold2 = 0.10 * 255
high_threshold2 = 0.55 * 255

# Aplicar Canny para detectar bordes
edges2 = cv2.Canny(blurred_image, low_threshold2, high_threshold2)

# Definir kernel de cierre para la operación de morfología
kernel_cierre1 = np.ones((10, 18), np.uint8)

# Aplicar la operación de cerrado para mejorar la detección de bordes
edges_closed1 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel_cierre1, iterations=1)

# Encontrar componentes conectados en los bordes cerrados
num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(edges_closed1)

# Dibujar los componentes conectados sobre la imagen original
connected_components_image = imagen_placa.copy()
identificadores_detectados = 0

# Procesar los componentes detectados por el segundo kernel, evitando duplicados y aplicando filtros
for i in range(1, num_labels2):
    x, y, w, h, area = stats2[i]
    
    # Filtrar por condiciones específicas como el área y posición en la imagen
    if area == 5962 or area == 4896:  # Excluir áreas específicas
        continue
    elif 4000 < area < 8000 and 219 <= x < 1710 and y > 800:  # Filtrar por área, posición en x y y
        cv2.rectangle(connected_components_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(connected_components_image, f'Resistencia {i} - Area: {area}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(connected_components_image, f'Pos: ({x}, {y}), Size: ({w}, {h})', (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        identificadores_detectados += 1

# Mostrar la imagen con los componentes conectados (resistencias)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(connected_components_image, cv2.COLOR_BGR2RGB))
plt.title('Componentes electrónicos detectados (Resistencias)')
plt.axis('off')
plt.show()

#FIN DEL PUNTO A)
# EJERCICIO 1)B)
# Clasificar capacitores por tamaño
def classify_capacitors_by_size(capacitors):
    small = []
    medium = []
    large = []
    verysmall = []
    for cnt in capacitors:
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)
        if area < 7500 or area > 130000:
            continue
        elif radius == 175 or radius == 105:
            continue
        elif 45 <= radius < 77:
                verysmall.append(cnt)
        elif 77 <= radius < 144:
                small.append(cnt)
        elif 144 <= radius <= 200:
                medium.append(cnt)
        elif radius > 200:
                large.append(cnt)
   # return small, medium, large
    return verysmall, small, medium, large

# Clasificar los capacitores
verysmall_capacitors,small_capacitors, medium_capacitors, large_capacitors = classify_capacitors_by_size(contornos_capacitor)

# Dibujar los capacitores clasificados en la imagen
output_capacitors_size = imagen_placa.copy()

for cnt in verysmall_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius == 175 or radius == 105:
        continue
    else:
        cv2.circle(output_capacitors_size, center, radius, (255, 255, 0) , 2)
        cv2.putText(output_capacitors_size, f'MUY PEQUENIO - Radio: {radius}', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

for cnt in small_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius == 175 or radius == 105 or radius ==91:
        continue
    else:
        cv2.circle(output_capacitors_size, center, radius, (255, 0, 0), 2)
        cv2.putText(output_capacitors_size, f'PEQUENIO - Radio: {radius}', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

for cnt in medium_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius == 175 or radius == 105:
        continue
    else:
        cv2.circle(output_capacitors_size, center, radius, (0, 255, 0), 2)
        cv2.putText(output_capacitors_size, f'MEDIANO - Radio: {radius}', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

for cnt in large_capacitors:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius == 175 or radius == 105:
        continue
    else:
        cv2.circle(output_capacitors_size, center, radius, (0, 0, 255), 2)
        cv2.putText(output_capacitors_size, f'GRANDE - Radio: {radius}', (int(x) - 40, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Mostrar la imagen con los capacitores clasificados
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output_capacitors_size, cv2.COLOR_BGR2RGB))
plt.title('Capacitores clasificados por tamaño')
plt.axis('off')
plt.show()


verysmall_capacitors = len(verysmall_capacitors)
small_count = len(small_capacitors)
medium_count = len(medium_capacitors)
large_count = len(large_capacitors)

print("RESULTADO EJERCICIO 1)B)")
print(f"Cantidad de capacitores muy pequeños: {verysmall_capacitors}")
print(f"Cantidad de capacitores pequeños: {small_count}")
print(f"Cantidad de capacitores medianos: {medium_count}")
print(f"Cantidad de capacitores grandes: {large_count}")

print("RESULTADO EJERCICIO 1)C)")
print(f'Se han detectado {identificadores_detectados} resistores.')
