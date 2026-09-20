import csv
import os
import time
import cv2
import mediapipe as mp
import numpy as np

# Inicialização do MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATASET_FILE = "dataset_gestos.csv"
SAMPLES_PER_GESTURE = 100

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def extract_landmarks(landmarks):
    coords = []
    base_x = landmarks.landmark[0].x
    base_y = landmarks.landmark[0].y
    base_z = landmarks.landmark[0].z

    for lm in landmarks.landmark:
        coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return coords


def save_to_csv(data, label):
    file_exists = os.path.isfile(DATASET_FILE)
    with open(DATASET_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Cabeçalho: 63 features + 1 coluna de classe
            header = [f"coord_{i}" for i in range(63)] + ["label"]
            writer.writerow(header)
        for row in data:
            writer.writerow(row + [label])


def main():
    with mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        print("\n--- COLETOR DE DADOS LIBRAS ---")
        gesture_name = input(
            "Digite o nome do gesto que vai gravar (ex: 'oi', 'obrigado', 'a'): "
        ).strip()
        print(
            f"Posicione sua mão. Quando estiver pronto, pressione 'G' na janela do vídeo para gravar {SAMPLES_PER_GESTURE} frames."
        )

        recording = False
        collected_samples = []

        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if recording:
                        features = extract_landmarks(hand_landmarks)
                        collected_samples.append(features)

            # Informações visuais na tela
            if recording:
                status_text = f"Gravando '{gesture_name}': {len(collected_samples)}/{SAMPLES_PER_GESTURE}"
                color = (0, 0, 255)  # Vermelho
                if len(collected_samples) >= SAMPLES_PER_GESTURE:
                    save_to_csv(collected_samples, gesture_name)
                    print(f"\n[OK] {SAMPLES_PER_GESTURE} amostras de '{gesture_name}' salvas com sucesso!")
                    recording = False
                    collected_samples = []
            else:
                status_text = f"Pronto para gravar '{gesture_name}'. Pressione 'G' para iniciar."
                color = (0, 255, 0)  # Verde

            cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Gravador de Dataset", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("g") and not recording:
                recording = True
                collected_samples = []
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()