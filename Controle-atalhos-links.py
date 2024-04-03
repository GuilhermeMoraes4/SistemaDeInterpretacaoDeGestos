import cv2
import numpy as np
import math
import psutil
import webbrowser
import os

cap = cv2.VideoCapture(0)
abas_abertas = []  # Lista para armazenar os URLs das abas abertas
controlando_mouse = False  # Variável para controlar se o mouse está sendo controlado

# Função para verificar se a aba com o link específico já está aberta no navegador
def verificar_aba_aberta(url):
    for process in psutil.process_iter():
        try:
            if "chrome" in process.name().lower() and url in process.cmdline():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

while True:
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        roi = frame[100:300, 400:600]

        cv2.rectangle(frame, (400, 100), (600, 300), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull = cv2.convexHull(cnt)
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
        arearatio = ((areahull - areacnt) / areacnt) * 100

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        l = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            d = (2 * ar) / a
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90 and d > 30:
                l += 1

        l += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Esperando dados', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0 = Bloco de Notas', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    url = "notepad.exe"
                    if not verificar_aba_aberta(url):
                        os.system("start " + url)  # Abrir o bloco de notas
                elif arearatio < 17.5:
                    cv2.putText(frame, '', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, '1 = Spotify', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    url = "https://open.spotify.com/intl-pt"
                    if url not in abas_abertas and not verificar_aba_aberta(url):
                        webbrowser.open(url)  # Abrir no navegador padrão
                        abas_abertas.append(url)

        elif l == 2:
            cv2.putText(frame, '2 = PT', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            url = "https://www.youtube.com/@ptbrasil/"
            if url not in abas_abertas and not verificar_aba_aberta(url):
                webbrowser.open(url)  # Abrir no navegador padrão
                abas_abertas.append(url)
        elif l == 3:
            if arearatio < 27:
                cv2.putText(frame, '3 = Linkedin', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                url = "https://www.linkedin.com/in/guilherme-moraes040500//"
                if url not in abas_abertas and not verificar_aba_aberta(url):
                    webbrowser.open(url)  # Abrir no navegador padrão
                    abas_abertas.append(url)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except Exception as e:
        print("Erro:", e)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
