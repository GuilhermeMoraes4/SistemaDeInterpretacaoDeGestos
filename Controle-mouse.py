import cv2
import numpy as np
import math
import pyautogui  # Biblioteca para controlar o mouse e o teclado

# Inicialize a captura de vídeo
cap = cv2.VideoCapture(0)

# Defina algumas variáveis úteis
controlando_mouse = False  # Variável para controlar se o mouse está sendo controlado

while True:
    try:
        # Capturar frame por frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # Definir ROI (Região de Interesse) para a detecção da mão
        roi = frame[100:300, 400:600]

        # Desenhar um retângulo ao redor da ROI
        cv2.rectangle(frame, (400, 100), (600, 300), (0, 255, 0), 0)

        # Converter a ROI para o espaço de cores HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Definir intervalo de cores da pele para detecção
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Criar máscara para segmentar a mão
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # Encontrar contornos da mão na máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar o maior contorno
        cnt = max(contours, key=lambda x: cv2.contourArea(x), default=None)

        if cnt is not None:
            # Detectar os defeitos de convexidade na mão
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            # Loop através dos defeitos de convexidade
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                # Calcular os ângulos e distâncias dos defeitos de convexidade
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # Se o ângulo é menor que 90 graus, desenhar um círculo na ponta do dedo
                if angle <= 90:
                    cv2.circle(roi, far, 5, [0, 0, 255], -1)

                    # Se não estivermos controlando o mouse, habilitar o controle do mouse
                    if not controlando_mouse:
                        controlando_mouse = True

                # Mapear a posição da mão para a posição do cursor do mouse
                largura_tela, altura_tela = pyautogui.size()
                x = int(far[0] * largura_tela / 200)  # Mapear x para a largura da tela
                y = int(far[1] * altura_tela / 200)  # Mapear y para a altura da tela

                # Mover o cursor do mouse para a nova posição
                pyautogui.moveTo(x, y)
        else:
            # Se não houver mão detectada, desativar o controle do mouse
            controlando_mouse = False

        # Exibir imagens na tela
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    except Exception as e:
        print("Erro:", e)

    # Detectar tecla ESC para sair do loop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Fechar todas as janelas e liberar a captura de vídeo
cv2.destroyAllWindows()
cap.release()
