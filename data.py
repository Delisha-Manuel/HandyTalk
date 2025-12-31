import cv2
import numpy as np
import mediapipe as mp
import os
import time

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
numSamples = 300
dataPath = os.path.join(os.getcwd(), "data")
captureInterval = 0.01

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

for letter in letters:
    os.makedirs(os.path.join(dataPath, letter), exist_ok=True)

cap = cv2.VideoCapture(1)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    for letter in letters:
        letter_dir = os.path.join(dataPath, letter)

        count = 0
        collected = 0
            
        print(f"\nGet ready to record gesture: {letter}")
        time.sleep(2)

        last_capture = 0

        while collected < numSamples:
            ret, frame = cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)

                current_time = time.time()
                if current_time - last_capture > captureInterval:
                    file_path = os.path.join(dataPath, letter, f"{count}.npy")
                    np.save(file_path, landmarks)
                    print(f"Saved {file_path}")
                    count += 1
                    collected += 1
                    last_capture = current_time

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"{letter}: {count}/{numSamples}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ASL Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()