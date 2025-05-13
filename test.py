import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model = load_model(r'C:\Users\jhasr\Downloads\archive\emotion_model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion classes (must match the order used during model training)
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---- USER INPUT ----
# Provide the path to the input image
image_path = r'C:\Users\jhasr\Downloads\archive\img1.png'  # <-- Change this to your image path
# ---------------------

# Read the image from disk
#image = cv2.imread(image_path)
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()
else:
    print(f"Image loaded successfully: {image_path}")


# Check if the image is loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()
else:
    print(f"Image loaded successfully: {image_path}")

# Convert the image to grayscale (needed for face detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

# Handle the case where no faces are detected
if len(faces) == 0:
    print("No faces detected in the image.")
    # Optionally display the original image without annotations
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Loop through the detected faces
for (x, y, w, h) in faces:
    # Extract the face ROI (region of interest) from grayscale image
    face = gray_image[y:y+h, x:x+w]
    
    # Preprocess the face image: resize to 48x48 pixels
    face = cv2.resize(face, (48, 48))
    # Normalize pixel values to [0,1]
    face = face.astype('float32') / 255.0
    
    # Prepare the face image for the model: reshape to (1, 48, 48, 1)
    face = np.expand_dims(face, axis=0)   # shape becomes (1, 48, 48)
    face = np.expand_dims(face, axis=-1)  # shape becomes (1, 48, 48, 1)
    
    # Predict the emotion for this face
    prediction = emotion_model.predict(face)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_classes[emotion_index]

    # Draw a rectangle around the face in the original color image
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Put the emotion label above the rectangle
    cv2.putText(image, emotion_label, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the output image with bounding boxes and emotion labels
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)  # Wait indefinitely for a key press
cv2.destroyAllWindows()  # Close the window after a key press
