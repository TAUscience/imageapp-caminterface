import cv2
import mediapipe as mp
import numpy as np
import platform
import urllib.request
from apicat import get_url_cat
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

# Función para detectar gestos (AGREGAR MÁS GESTOS AQUÍ SEGÚN SEA NECESARIO)
def detect_sign(hand_landmarks):
    """ 
    Agregar aquí la lógica para detectar distintos tipos de gestos.
    Regresar una cadena de texto con el nombre del gesto detectado.
    """

    #Lógica para detectar el gesto de cambio de imagen
    x_thumb = hand_landmarks.landmark[4].x
    x_index = hand_landmarks.landmark[8].x
    if x_thumb < x_index - 0.1:
        return "right"
    elif x_thumb > x_index + 0.1:
        return "left"
    
    #Agregar más gestos aquí según sea necesario
    

    return None  # Si no se detecta ningún gesto

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
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener el tiempo actual
                current_time = time.time()
                
                # Detectar el gesto
                gesture_swipe = detect_sign(hand_landmarks)

                # Si ha pasado suficiente tiempo desde la última detección, ejecutar la acción
                if current_time - last_gesture_time > gesture_cooldown:
                    """ 
                    Agregar aquí la lógica para ejecutar acciones basadas en los gestos detectados.
                    """
                    if gesture_swipe == "right":
                        next_image() # Función para cambiar a la siguiente imagen
                        last_gesture_time = current_time

                    elif gesture_swipe == "left":
                        previous_image() # Función para cambiar a la imagen anterior
                        last_gesture_time = current_time
                        
                    #Agregar más acciones aquí según sea necesario
                        
                # Aquí van los gestos que no requieren tiempo de espera
                if gesture_swipe is not None:
                    if gesture_swipe == "otrogesto":
                        print("Otro gesto")
        
        cv2.imshow("Captura de Video", frame)
        cv2.imshow("Imagen de Gato", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
