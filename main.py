import cv2
import mediapipe as mp
import numpy as np
import platform
import urllib.request
from apicat import get_url_cat
import time
import math
import os

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Crear carpeta para guardar imágenes descargadas
save_folder = "imagenes_guardadas"
os.makedirs(save_folder, exist_ok=True)

# Detectar el sistema operativo
sistema = platform.system()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sistema == "Windows" else cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Configuración de imágenes
image_list = []  # Lista de imágenes descargadas
current_image_index = 0  # Índice de la imagen actual
max_width, max_height = 480, 360  # Tamaño máximo para la ventana

# Función para redimensionar imagen manteniendo la relación de aspecto
def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
    else:
        new_width, new_height = width, height
    return cv2.resize(image, (new_width, new_height))

# Descargar imagen de gato
def fetch_cat_image():
    global image_list
    image_url = get_url_cat()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    req = urllib.request.Request(image_url, headers=headers)
    try:
        image_data = urllib.request.urlopen(req).read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is not None:
            redim_image = resize_image(image, max_width, max_height)
            image_list.append(redim_image)
            print(redim_image.shape)
            return redim_image
        else:
            print("Error al decodificar la imagen.")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
    except Exception as e:
        print(f"Error desconocido: {e}")

def descargar_imagen(image):
    timestamp = int(time.time())
    filename = os.path.join(save_folder, f"imagen_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    print(f"Imagen guardada en {filename}")

"""
#############################################################
Sección de código para agregar funciones de manejo de imagen 
y detección de gestos.
#############################################################
"""

# Función para cambiar a la siguiente imagen
def next_image():
    global current_image_index, image
    if current_image_index < len(image_list) - 1:
        current_image_index += 1
    else:
        new_image = fetch_cat_image()
        if new_image is not None:
            current_image_index += 1
    image = image_list[current_image_index]

# Función para cambiar a la imagen anterior
def previous_image():
    global current_image_index, image
    if current_image_index > 0:
        current_image_index -= 1
    image = image_list[current_image_index]

def detect_sign(hand_landmarks):
    """ 
    Detecta distintos tipos de gestos y devuelve el nombre del gesto detectado.
    """

    # Obtener las posiciones de los dedos en el espacio 2D (x, y)
    x_thumb = hand_landmarks.landmark[4].x
    x_index = hand_landmarks.landmark[8].x
    x_middle = hand_landmarks.landmark[12].x
    x_ring = hand_landmarks.landmark[16].x
    x_pinky = hand_landmarks.landmark[20].x
    
    y_thumb = hand_landmarks.landmark[4].y
    y_index = hand_landmarks.landmark[8].y
    y_middle = hand_landmarks.landmark[12].y
    y_ring = hand_landmarks.landmark[16].y
    y_pinky = hand_landmarks.landmark[20].y

    # Lógica para el gesto "right" (pulgar apuntando a la derecha)
    # El pulgar debe estar apuntando hacia la derecha, mientras que los demás dedos están cerrados
    if x_thumb > max(x_index, x_middle, x_ring, x_pinky) + 0.05:
        # Verificamos que el pulgar esté un poco más hacia la derecha
        # Y nos aseguramos de que no esté apuntando hacia arriba (y_thumb no debe ser más bajo que los otros dedos)
        if y_thumb < y_index and y_thumb < y_middle and y_thumb < y_ring and y_thumb < y_pinky:
            return "right"

    # Lógica para el gesto "left" (pulgar apuntando a la izquierda)
    # El pulgar debe estar apuntando hacia la izquierda, mientras que los demás dedos están cerrados
    if x_thumb < min(x_index, x_middle, x_ring, x_pinky) - 0.05:
        # Verificamos que el pulgar esté un poco más hacia la izquierda
        # Y nos aseguramos de que no esté apuntando hacia arriba (y_thumb no debe ser más bajo que los otros dedos)
        if y_thumb < y_index and y_thumb < y_middle and y_thumb < y_ring and y_thumb < y_pinky:
            return "left"

    # Detectar gesto de "like" (pulgar arriba, otros dedos abajo)
    # El pulgar debe estar claramente apuntando hacia arriba, y no a un lado
    if y_thumb < y_index and y_thumb < y_middle and y_thumb < y_ring and y_thumb < y_pinky:
        # Aseguramos que el pulgar esté apuntando hacia arriba, no a un lado (evitar confusión con "right" o "left")
        if abs(x_thumb - x_index) < 0.05 and abs(x_thumb - x_middle) < 0.05 and abs(x_thumb - x_ring) < 0.05 and abs(x_thumb - x_pinky) < 0.05:
            return "like"

    # Agregar más gestos aquí según sea necesario

    return None  # Si no se detecta ningún gesto

def is_hand_open(hand_landmarks, threshold=0.03, min_fingers=2):
    """
    Verifica si la mano está abierta considerando la separación de los dedos en los ejes X e Y.
    Se considera abierta si al menos `min_fingers` de los dedos medio, anular y meñique están separados.
    """
    fingers = [(12, 10), (16, 14), (20, 18)]  # Medio, anular y meñique
    open_fingers = sum(
        abs(hand_landmarks.landmark[p].y - hand_landmarks.landmark[b].y) > threshold and
        abs(hand_landmarks.landmark[p].x - hand_landmarks.landmark[b].x) > 0.02
        for p, b in fingers
    )
    return open_fingers >= min_fingers

def rotate_and_zoom_image_by_hand(image, hand_landmarks):
    """
    Rota y hace zoom en la imagen según la inclinación de la mano.
    """
    if not is_hand_open(hand_landmarks):
        return image
    
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    # Calcular el ángulo de rotación
    v_x = middle_tip.x - wrist.x
    v_y = middle_tip.y - wrist.y
    angle = math.degrees(math.atan2(v_y, v_x))
    
    # Ajustar el ángulo para que 0° sea la posición natural
    angle = (angle + 90) % 360  # Corregimos la orientación inicial
    if angle > 180:
        angle -= 360
    
    # Calcular zoom basado en la distancia entre pulgar e índice
    thumb_index_distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    zoom_factor = max(0.5, min(2.0, 1 + (thumb_index_distance - 0.1) * 10))
    
    # Aplicar transformación
    rows, cols, _ = image.shape
    M_rotation = cv2.getRotationMatrix2D((cols // 2, rows // 2), -angle, zoom_factor)
    return cv2.warpAffine(image, M_rotation, (cols, rows))


# Descargar la primera imagen
image = fetch_cat_image()
if image is None:
    print("No se pudo obtener la imagen inicial.")
    exit()

# Variable para almacenar el tiempo de la última detección del gesto
last_gesture_time = 0
gesture_cooldown = 2  # Segundos de espera entre gestos

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        rotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_time = time.time()
                gesture_swipe = detect_sign(hand_landmarks)
                
                if current_time - last_gesture_time > gesture_cooldown:
                    if gesture_swipe == "right":
                        next_image()
                        last_gesture_time = current_time
                    elif gesture_swipe == "left":
                        previous_image()
                        last_gesture_time = current_time
                    elif gesture_swipe == "like":
                        descargar_imagen(image)
                        last_gesture_time = current_time
                rotated_image = rotate_and_zoom_image_by_hand(image, hand_landmarks)
        
        cv2.imshow("Captura de Video", frame)
        cv2.imshow("Imagen Rotada", rotated_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
