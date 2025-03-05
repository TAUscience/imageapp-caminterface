import cv2
import mediapipe as mp
import numpy as np
import math
import urllib.request
from apicat import get_url_cat

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Tamaño máximo para la imagen
max_width, max_height = 480, 360

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
    image_url = get_url_cat()
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(image_url, headers=headers)
    try:
        image_data = urllib.request.urlopen(req).read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is not None:
            return resize_image(image, max_width, max_height)
        else:
            print("Error al decodificar la imagen.")
            return None
    except Exception as e:
        print(f"Error al obtener la imagen: {e}")
        return None

# Cargar imagen inicial desde API
image = fetch_cat_image()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener coordenadas clave
                wrist = hand_landmarks.landmark[0]
                index_tip = hand_landmarks.landmark[8]
                
                # Calcular vector de dirección
                v_x = index_tip.x - wrist.x
                v_y = index_tip.y - wrist.y
                
                # Calcular ángulo con el eje X y ajustar signo
                angle = -math.degrees(math.atan2(v_y, v_x))
                if angle < 0:
                    angle += 360  # Normalizar ángulo entre 0 y 360°
                print(f"Ángulo de rotación: {angle:.2f}°")

                # Rotar la imagen según el ángulo detectado
                if image is not None:
                    rows, cols, _ = image.shape
                    M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, 1)
                    rotated_image = cv2.warpAffine(image, M, (cols, rows))
                    
                    cv2.imshow("Imagen Rotada", rotated_image)
        
        cv2.imshow("Captura de Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()