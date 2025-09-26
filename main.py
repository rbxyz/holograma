import cv2
import numpy as np
import mediapipe as mp

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Lista para armazenar trajetória do movimento
trajetoria = []

# Função para criar efeito holograma (4 cópias rotacionadas)
def render_holograma(frame, objeto):
    h, w, _ = frame.shape
    obj_h, obj_w, _ = objeto.shape
    centro_y, centro_x = h // 2, w // 2

    # Adiciona as 4 cópias em posições diferentes
    # Topo
    if centro_y - obj_h - 50 > 0:
        frame[50:50+obj_h, centro_x-obj_w//2:centro_x+obj_w//2] = objeto
    # Direita
    if centro_x + obj_w + 50 < w:
        frame[centro_y-obj_h//2:centro_y+obj_h//2, w-obj_w-50:w-50] = cv2.rotate(objeto, cv2.ROTATE_90_CLOCKWISE)
    # Baixo
    if centro_y + obj_h + 50 < h:
        frame[h-obj_h-50:h-50, centro_x-obj_w//2:centro_x+obj_w//2] = cv2.rotate(objeto, cv2.ROTATE_180)
    # Esquerda
    if centro_x - obj_w - 50 > 0:
        frame[centro_y-obj_h//2:centro_y+obj_h//2, 50:50+obj_w] = cv2.rotate(objeto, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

# Captura da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Conversão para RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha mão
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Coordenada da ponta do dedo indicador (landmark 8)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            trajetoria.append((x, y))

    # Desenha trajetória verde
    for i in range(1, len(trajetoria)):
        cv2.line(frame, trajetoria[i-1], trajetoria[i], (0, 255, 0), 2)

    # Recorta objeto do movimento para holograma
    if trajetoria:
        min_x = min([p[0] for p in trajetoria])
        max_x = max([p[0] for p in trajetoria])
        min_y = min([p[1] for p in trajetoria])
        max_y = max([p[1] for p in trajetoria])
        objeto = frame[min_y:max_y, min_x:max_x].copy()
        if objeto.size != 0:
            frame = render_holograma(frame, objeto)

    cv2.imshow("Holograma Mao", frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
