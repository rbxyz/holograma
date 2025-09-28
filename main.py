import cv2
import numpy as np
import mediapipe as mp
import os
import sys

# init do mediapipe hans
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

trajetoria = []

PINCH_THRESHOLD = 0.05

# Função para calcular distância euclidiana entre dois pontos
def calcular_distancia(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

pid_file = "/tmp/camera_pinch.pid"
if os.path.exists(pid_file):
    print("Aviso: Já há uma instância rodando. Fechando...")
    sys.exit(0)
with open(pid_file, 'w') as f:
    f.write(str(os.getpid()))

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera!")
        exit()

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Erro: Falha ao capturar frame da câmera")
                break
            frame = cv2.flip(frame, 1)
            if frame is None or frame.size == 0:
                print("Erro: Frame inválido recebido da câmera")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            pinch_ativo = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Desenha mão
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detecta gesto de pinça (distância entre polegar e indicador)
                    polegar = hand_landmarks.landmark[4]  # Ponta do polegar
                    indicador = hand_landmarks.landmark[8]  # Ponta do indicador
                    distancia = calcular_distancia(polegar, indicador)

                    if distancia < PINCH_THRESHOLD:
                        pinch_ativo = True
                        h, w, _ = frame.shape
                        x_polegar = int(polegar.x * w)
                        y_polegar = int(polegar.y * h)
                        x_indicador = int(indicador.x * w)
                        y_indicador = int(indicador.y * h)

                        cv2.circle(frame, (x_polegar, y_polegar), 10, (255, 0, 255), -1)
                        cv2.circle(frame, (x_indicador, y_indicador), 10, (255, 0, 255), -1)

                        trajetoria.append((x_indicador, y_indicador))

                        if len(trajetoria) > 100:
                            trajetoria.pop(0)

            cor_trajetoria = (255, 0, 0) if pinch_ativo else (0, 255, 0)
            for i in range(1, len(trajetoria)):
                cv2.line(frame, trajetoria[i-1], trajetoria[i], cor_trajetoria, 3)

            # Mostra status da pinça na tela
            status_texto = "PINCA ATIVA - DESENHANDO" if pinch_ativo else "PINCA INATIVA"
            cor_texto = (255, 0, 0) if pinch_ativo else (0, 255, 0)
            cv2.putText(frame, status_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_texto, 2)

            try:
                if not cv2.getWindowProperty("Camera com Pinça", cv2.WND_PROP_VISIBLE):
                    cv2.namedWindow("Camera com Pinça", cv2.WINDOW_NORMAL)
            except:
                cv2.namedWindow("Camera com Pinça", cv2.WINDOW_NORMAL)

            cv2.imshow("Camera com Pinça", frame)

            # Tecla 'q' para sair (otimizado para ~33 FPS)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("\nInterrupção do usuário detectada.")
            break
        except Exception as e:
            print(f"Erro durante processamento: {e}")
            break

finally:
    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
    except:
        pass
