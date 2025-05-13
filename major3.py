import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os 
import time

# Paths
train_archive_dir = r"C:\Users\jhasr\Downloads\archive\train"
validation_archive_dir = r"C:\Users\jhasr\Downloads\archive\test"

# Data generators
train_archivegen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_archivegen = ImageDataGenerator(rescale=1./255)

train_generator = train_archivegen.flow_from_directory(
    train_archive_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_archivegen.flow_from_directory(
    validation_archive_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preview one batch
img, label = train_generator.__next__()


from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

model_path = r"C:\Users\jhasr\Downloads\archive\emotion_model.h5"

if os.path.exists(model_path):
    print("✅ Loading saved model from 50 epochs...")

    # ✅ Step 1: Clear the previous TensorFlow graph
    K.clear_session()

    # ✅ Step 2: Load the model without its original compile state
    model = load_model(model_path, compile=False)

    # ✅ Step 3: Recompile for eager execution
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

else:
    print("🔄 No saved model found. Creating a new model...")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Count images
print("Counting training and validation images...")
num_train_imgs = sum([len(files) for r, d, files in os.walk(train_archive_dir) if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)])
num_test_imgs = sum([len(files) for r, d, files in os.walk(validation_archive_dir) if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)])

print(f"Training images: {num_train_imgs}")
print(f"Validation images: {num_test_imgs}")

# Calculate safe steps
steps_per_epoch = max(1, num_train_imgs // 32)
validation_steps = max(1, num_test_imgs // 32)

# Training
print("Starting training...")
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1
)

print("Training finished at:", time.strftime('%X'))
print("Total training time (s):", time.time() - start_time)

model.save("emotion_model.h5")
print("📦 Model saved at:", os.path.join(os.getcwd(), model_path))

