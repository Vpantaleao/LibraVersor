import os
import time
import cv2
import mediapipe as mp
import numpy as np

# Configurações de Captura
DATA_PATH = os.path.join("dados_gestos")
NO_SEQUENCES = 30       # Quantidade de vezes que você repetirá o gesto
SEQUENCE_LENGTH = 30    # Quantidade de frames por sequência (~1 segundo)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def extract_landmarks(landmarks):
    """Extrai 21 pontos (x, y, z) normalizados em relação ao punho (ponto 0)."""
    if landmarks is None:
        return np.zeros(63)
    
    base_x = landmarks.landmark[0].x
    base_y = landmarks.landmark[0].y
    base_z = landmarks.landmark[0].z

    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return np.array(coords)


def main():
    print("\n" + "="*50)
    print("GRAVADOR DE GESTOS DINÂMICOS (SEQUÊNCIAS)")
    print("="*50)
    gesture_name = input("Nome do gesto que vai gravar (ex: 'oi', 'obrigado'): ").strip().lower()
    
    gesture_dir = os.path.join(DATA_PATH, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        print("\nPressione a tecla 'ESPAÇO' na janela de vídeo para começar a gravar as sequências.")

        ready_to_start = False
        while not ready_to_start:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            cv2.putText(img, f"Gesto: '{gesture_name}'. Pressione ESPACO para iniciar!", 
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gravador Dinamico", img)
            if cv2.waitKey(1) & 0xFF == 32:  # Barra de espaço
                ready_to_start = True

        # Loop de gravação das sequências
        for seq in range(NO_SEQUENCES):
            # 1. Pausa e aviso entre as repetições
            start_time = time.time()
            while time.time() - start_time < 1.5:
                success, img = cap.read()
                if not success:
                    break
                img = cv2.flip(img, 1)
                cv2.putText(img, f"PREPARE-SE: Repeticao {seq + 1}/{NO_SEQUENCES}", 
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(img, f"Faca o movimento continuo em 1.5s...", 
                            (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow("Gravador Dinamico", img)
                cv2.waitKey(1)

            # 2. Gravação contínua dos 30 frames
            sequence_data = []
            for frame_num in range(SEQUENCE_LENGTH):
                success, img = cap.read()
                if not success:
                    break
                img = cv2.flip(img, 1)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        keypoints = extract_landmarks(hand_landmarks)
                else:
                    keypoints = np.zeros(63)

                sequence_data.append(keypoints)

                # Feedback visual de gravação ativa
                cv2.putText(img, f"GRAVANDO '{gesture_name}' [{frame_num + 1}/{SEQUENCE_LENGTH}]", 
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Gravador Dinamico", img)
                cv2.waitKey(1)

            # Salva o array de formato (30, 63)
            save_path = os.path.join(gesture_dir, f"{seq}.npy")
            np.save(save_path, np.array(sequence_data))
            print(f"[OK] Repetição {seq + 1}/{NO_SEQUENCES} gravada.")

        print(f"\nGravação concluída para o gesto '{gesture_name}'!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()