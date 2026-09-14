import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

point_style = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=8)
connection_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=7)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def extract_landmarks(landmarks):
    """Extrai os 21 pontos da mão (x, y, z) e normaliza em relação ao punho (ponto 0)."""
    coords = []
    base_x = landmarks.landmark[0].x
    base_y = landmarks.landmark[0].y
    base_z = landmarks.landmark[0].z

    for lm in landmarks.landmark:
        coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

    return np.array(coords).reshape(1, -1)


def main():
    with mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.7,
    ) as hands:
        while True:
            sucess, img = cap.read()
            if not sucess:
                print('Falha em ler o frame')
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            status_text = "Nenhuma mao detectada"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=point_style,
                        connection_drawing_spec=connection_style
                    )

                    features = extract_landmarks(hand_landmarks)

                    status_text = f"Features extraidas: {features.shape[1]} dims (RF Pronto)"

            cv2.putText(
                img,
                status_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow('LibraVersor', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()