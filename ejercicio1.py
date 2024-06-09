import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen de la placa
imagen_placa = cv2.imread("img/placa.png")

# Convertir la imagen de BGR a HSV
hsv_image = cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2HSV)

# Definir diccionario de colores
COLORS = {
    'yellow': (0, 255, 255),  # amarillo
    'gray': (128, 128, 128),   # gris
    'silver': (192, 192, 192)  # plateado
}

# Definir rangos de colores en HSV ajustados
###AGREGAR LA PARTE ROJA... SOLO SE IDENTIFICA EL AMARILLO PARA LA RESISTENCIAS
# OTRA SOLUCION: UNIR LOS COMPONENTES DETECTADOS COMO RESISTENCIA
lower_yellow = np.array([15, 50, 50])
upper_yellow = np.array([35, 255, 255])

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
visualizar_rangos_color(imagen_placa_color_ranges, lower_yellow, upper_yellow, 'yellow')
visualizar_rangos_color(imagen_placa_color_ranges, lower_gray, upper_gray, 'gray')
visualizar_rangos_color(imagen_placa_color_ranges, lower_silver, upper_silver, 'silver')

# Mostrar la imagen con los rangos de colores visualizados
plt.imshow(cv2.cvtColor(imagen_placa_color_ranges, cv2.COLOR_BGR2RGB))
plt.title('Rangos de colores identificados')
plt.axis('off')
plt.show()

# Segmentar componentes basados en color
mask_resistencia = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
mask_chip = cv2.inRange(hsv_image, lower_gray, upper_gray)
mask_capacitor = cv2.inRange(hsv_image, lower_silver, upper_silver)

# Función para post-procesar las máscaras
def post_process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Post-procesar las máscaras
mask_resistencia = post_process_mask(mask_resistencia)
mask_chip = post_process_mask(mask_chip)
mask_capacitor = post_process_mask(mask_capacitor)

# Encontrar los contornos en las máscaras
contornos_resistencia, _ = cv2.findContours(mask_resistencia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_chip, _ = cv2.findContours(mask_chip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_capacitor, _ = cv2.findContours(mask_capacitor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostrar la imagen con los contornos detectados
def mostrar_contornos(contornos, imagen, color, etiqueta, umbral_minimo_area, umbral_maximo_area, es_circulo=False):
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < umbral_minimo_area or area > umbral_maximo_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if es_circulo:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(imagen, center, radius, color, 2)
            cv2.putText(imagen, etiqueta, (int(x)-40, int(y)-radius-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.rectangle(imagen, (x, y), (x+w, y+h), color, 2)
            cv2.putText(imagen, etiqueta, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Dibujar contornos
imagen_placa_contornos = imagen_placa.copy()
mostrar_contornos(contornos_resistencia, imagen_placa_contornos, (0, 0, 255), 'RESISTENCIA', 2000, 7000)
mostrar_contornos(contornos_chip, imagen_placa_contornos, (0, 255, 0), 'CHIP', 90000, 100000)
mostrar_contornos(contornos_capacitor, imagen_placa_contornos, (255, 0, 0), 'CAPACITOR', 7500, 130000, es_circulo=True)

# Mostrar las máscaras de cada componente
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mask_resistencia, cmap='gray')
plt.title('Máscara de Resistencia')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mask_chip, cmap='gray')
plt.title('Máscara de Chip')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(mask_capacitor, cmap='gray')
plt.title('Máscara de Capacitor')
plt.axis('off')

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(imagen_placa_contornos, cv2.COLOR_BGR2RGB))
plt.title('Componentes electrónicos detectados y clasificados')
plt.axis('off')

plt.show()
