from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import joblib
import numpy as np
import mediapipe as mp
import cv2

app = Flask(__name__)

model = joblib.load("asl_model.pkl")["model"]
gestures = joblib.load("asl_model.pkl")["gestures"]

# initialize mediapipe hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

def get_landmarks(image: Image.Image):
    # convert image
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = hands_detector.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # store hand landmarks
        landmarks = []
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks).reshape(1, -1)
    
    return None

# home route
@app.route('/')
def home():
    return render_template('index.html')

# translate route
@app.route('/translator')
def translator():
    return render_template('translator.html')

# predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"letter": None, "handDetected": False})
    
    file = request.files['file']
    try:
        image = Image.open(file.stream).convert("RGB")
    except:
        return jsonify({"letter": None, "handDetected": False})

    landmarks = get_landmarks(image)
    if landmarks is None:
        return jsonify({"letter": None, "handDetected": False})

    pred_idx = model.predict(landmarks)[0]
    predicted_letter = gestures[pred_idx]

    return jsonify({"letter": predicted_letter, "handDetected": True})

if __name__ == '__main__':
    app.run(debug=True)