const video = document.getElementById("video");
const predictionDiv = document.getElementById("prediction");
const currentWordSection = document.getElementById("current-word");

const toggleButton = document.getElementById("toggle-button");
const addButton = document.getElementById("add-button");
const spaceButton = document.getElementById("space-button");
const clearButton = document.getElementById("clear-button");

let recognizing = false;
let lastLetter = "";
let currentWord = "";
let intervalId = null;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();
    } catch (err) {
        console.error("Camera error:", err);
        alert("Could not access the camera. Please check permissions.");
    }
}

async function sendFrame() {
    if (!recognizing) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg"));
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();

        if (data.handDetected && data.letter) {
            lastLetter = data.letter;
            predictionDiv.textContent = data.letter;
        } else {
            predictionDiv.textContent = "No hand detected";
        }
    } catch {
        predictionDiv.textContent = "Error";
    }
}

toggleButton.onclick = async () => {
    recognizing = !recognizing;
    toggleButton.textContent = recognizing ? "Stop Recognition" : "Start Recognition";

    if (recognizing) {
        if (!video.srcObject) await startCamera();
        intervalId = setInterval(sendFrame, 300);
    } else {
        clearInterval(intervalId);
    }
};

addButton.onclick = () => {
    if (lastLetter) currentWord += lastLetter;
    currentWordSection.textContent = "Current Word: " + currentWord;
};

spaceButton.onclick = () => {
    currentWord += " ";
    currentWordSection.textContent = "Current Word: " + currentWord;
};

clearButton.onclick = () => {
    lastLetter = "";
    currentWord = "";
    predictionDiv.textContent = "No hand detected";
    currentWordSection.textContent = "Current Word: ";
};