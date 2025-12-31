import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

data_path = "../data" 
model_path = "asl_model.pkl"

gestures = sorted(os.listdir(data_path))
hand_data = []
labels = []

for index, gesture in enumerate(gestures):
    folder = os.path.join(data_path, gesture)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            landmarks = np.load(os.path.join(folder, file))
            hand_data.append(landmarks)
            labels.append(index)

hand_data = np.array(hand_data)
labels = np.array(labels)
print(f"Loaded dataset: {hand_data.shape[0]} samples, {hand_data.shape[1]} features each")

handTrain, handTest, labelTrain, labelTest = train_test_split(
    hand_data, labels, test_size=0.30, random_state=40, stratify=labels
)

classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=40)
classifier.fit(handTrain, labelTrain)

prediction = classifier.predict(handTest)
accuracyScore = accuracy_score(labelTest, prediction)
print(f"Test accuracy: {accuracyScore:.4f}")

print(confusion_matrix(labelTest, prediction))
print(classification_report(labelTest, prediction))

joblib.dump({"model": classifier, "gestures": gestures}, model_path)
print(f"Trained model saved to {model_path}")