import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_model_video_data.h5')

# Emotion labels
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define target image size (should match training)
img_size = (48, 48)

# Start webcam
cap = cv2.VideoCapture(0)

print("🔍 Starting real-time emotion detection. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face using OpenCV Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, img_size)
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = model.predict(reshaped_face)
        emotion_index = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_index]

        # Display result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow(r'C:\Users\jhasr\Downloads\archive\emotion_model_video_data.h5', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
