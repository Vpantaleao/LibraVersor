import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
def identificar_letra_libras(lm, w, h):
    def d(p1, p2):
        return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

    tamanho_palma = d(lm[0], lm[9])

    ind_aberto = d(lm[8], lm[5]) > tamanho_palma * 0.65
    med_aberto = d(lm[12], lm[9]) > tamanho_palma * 0.65
    ane_aberto = d(lm[16], lm[13]) > tamanho_palma * 0.65
    min_aberto = d(lm[20], lm[17]) > tamanho_palma * 0.65

    ind_fechado = d(lm[8], lm[5]) < tamanho_palma * 0.55
    med_fechado = d(lm[12], lm[9]) < tamanho_palma * 0.55
    ane_fechado = d(lm[16], lm[13]) < tamanho_palma * 0.55
    min_fechado = d(lm[20], lm[17]) < tamanho_palma * 0.55

    pol_junto_indicador = d(lm[4], lm[6]) < tamanho_palma * 0.45
    pol_na_frente_dedos = d(lm[4], lm[10]) < tamanho_palma * 0.45 or d(lm[4], lm[14]) < tamanho_palma * 0.45
    pol_toca_indicador = d(lm[4], lm[8]) < tamanho_palma * 0.35

    if ind_fechado and med_fechado and ane_fechado and min_fechado:
        if pol_junto_indicador and (lm[4].y < lm[6].y):
            return "A"

    if ind_fechado and med_fechado and ane_fechado and min_fechado:
        if pol_na_frente_dedos:
            return "E"

    if ind_aberto and med_aberto and ane_aberto and min_aberto:
        if d(lm[4], lm[9]) < tamanho_palma * 0.6:
            return "B"

    if ind_aberto and med_fechado and ane_fechado and min_fechado:
        if d(lm[4], lm[12]) < tamanho_palma * 0.5:
            return "D"

    if med_aberto and ane_aberto and min_aberto and ind_fechado:
        if pol_toca_indicador:
            return "F"

    dist_pol_ind = d(lm[4], lm[8])
    if (tamanho_palma * 0.4 < dist_pol_ind < tamanho_palma * 1.1):
        if not (ind_aberto and med_aberto and ane_aberto and min_aberto) and not (ind_fechado and med_fechado):
            return "C"

    return ""
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

def main():
    point_style = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=8)
    connection_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=7)
    with mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.7,
    ) as hands:
        while True:
            attempt = 0
            sucess, img = cap.read()
            while not sucess and attempt < 5:
                time.sleep(0.2)
                sucess, img = cap.read()
                attempt += 1
            if not sucess:
                print('Falha em ler o frame')
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
            letra_detectada = ""
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks,
                          mp_hands.HAND_CONNECTIONS,
                          landmark_drawing_spec=point_style,
                        connection_drawing_spec=connection_style
                    )
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
                    lm = hand_landmarks.landmark

                    letra_detectada = identificar_letra_libras(lm, w, h)

            if letra_detectada != "":
                cv2.rectangle(img, (20, 20), (260, 120), (0, 0, 0), -1)
                cv2.putText(img, f"Letra: {letra_detectada}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
            cv2.imshow('Imagem', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()