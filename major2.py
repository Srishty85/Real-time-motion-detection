import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Paths
video_folder = r"C:\Users\jhasr\Downloads\archive\video_data"  # Path to RAVDESS video folder
video_output = r"C:\Users\jhasr\Downloads\archive\video_output"  # Folder to save frames

# Map emotion numbers to labels
emotion_dict = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

# Ensure output folder exists
if not os.path.exists(video_output):
    os.makedirs(video_output)

# Frame extraction function (modified to check if frames already exist)
def extract_frames(video_path, emotion_label, target_size=(48, 48), max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"❌ Skipping {video_path} (no frames)")
        return

    print(f"▶ Processing {video_path} ({total_frames} total frames)")
    frame_step = max(1, total_frames // max_frames)
    count = 0

    # Make per-emotion subfolder if not exists
    emotion_folder = os.path.join(video_output, emotion_label)
    os.makedirs(emotion_folder, exist_ok=True)

    # Check if frames already exist
    existing_frames = os.listdir(emotion_folder)
    if existing_frames:
        print(f"⚠️ Frames already exist for {emotion_label}. Skipping extraction.")
        return

    for i in range(0, total_frames, frame_step):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read frame {i} from {video_path}")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)

        # Save frame
        frame_filename = f"{os.path.basename(video_path).replace('.mp4','')}_frame{i}.jpg"
        save_path = os.path.join(emotion_folder, frame_filename)
        cv2.imwrite(save_path, resized)
        print(f"✅ Saved: {save_path}")
        count += 1
        if count >= max_frames:
            break

    cap.release()

# Process all videos (this part stays the same)
def process_videos(input_dir, max_frames=30):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_label = emotion_dict.get(emotion_code, "unknown")
                    extract_frames(file_path, emotion_label, max_frames=max_frames)

# Run frame extraction
process_videos(video_folder)
print(f"\n✅ All frames processed and saved in: {video_output}")

# --- Now, add the new part for loading and preprocessing the frames ---

# Define image size and paths
img_size = (48, 48)  # Resize frames to 48x48
emotion_dict = {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7}

# Function to load and preprocess frames
def load_data():
    images = []
    labels = []
    
    for emotion in os.listdir(video_output):
        emotion_folder = os.path.join(video_output, emotion)
        
        if os.path.isdir(emotion_folder):
            label = emotion_dict.get(emotion, None)
            if label is not None:  # Ensure valid label
                for frame_file in os.listdir(emotion_folder):
                    if frame_file.endswith('.jpg'):
                        frame_path = os.path.join(emotion_folder, frame_file)
                        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                        image = cv2.resize(image, img_size)  # Resize image to target size
                        images.append(image)
                        labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0

    return images, labels

# Load data
X, y = load_data()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data to include a channel dimension (grayscale image)
X_train = X_train.reshape(X_train.shape[0], img_size[0], img_size[1], 1)
X_val = X_val.reshape(X_val.shape[0], img_size[0], img_size[1], 1)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# --- Now, build and train the model ---

# Build CNN model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output for fully connected layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))  # 8 classes for emotions

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model
model.save('emotion_model_video_data.h5')

# Display training history
# Plot accuracy and loss curves
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()
import sys
sys.exit()
