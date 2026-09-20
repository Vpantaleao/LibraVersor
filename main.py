import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import collections
import pickle

MODEL_PATH = "modelo_gestos.p"
SEQUENCE_LENGTH = 30

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

clf = model_data["model"]
labels_dict = model_data["labels_dict"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

point_style = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=8)
connection_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=7)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def extract_landmarks(landmarks):
    """Extrai os 21 pontos da mão (x, y, z) e normaliza em relação ao punho (ponto 0)."""
    if landmarks is None:
        return np.zeros(63)

    coords = []
    base_x = landmarks.landmark[0].x
    base_y = landmarks.landmark[0].y
    base_z = landmarks.landmark[0].z

    for lm in landmarks.landmark:
        coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return np.array(coords)


def main():
    sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

    current_gesture = "Aguardando..."
    confidence = 0.0
    
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
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)


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
            else:
                features = np.zeros(63)

            sequence_buffer.append(features)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Transforma a janela (30, 63) no vetor de entrada (1, 1890)
                input_data = np.array(sequence_buffer).flatten().reshape(1, -1)

                probabilities = clf.predict_proba(input_data)[0]
                best_class_idx = np.argmax(probabilities)
                confidence = probabilities[best_class_idx]

                # Só atualiza a tela se o modelo tiver certeza razoável (> 70%)
                if confidence > 0.70:
                    current_gesture = labels_dict[best_class_idx]
                else:
                    current_gesture = "Indefinido"

            # Interface Visual (HUD)
            cv2.rectangle(img, (0, 0), (450, 90), (30, 30, 30), -1)
            cv2.putText(
                img,
                f"Gesto: {current_gesture.upper()}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"Confiança: {confidence * 100:.1f}%",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )


            cv2.imshow('LibraVersor', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()