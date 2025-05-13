from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
import os 


#train_archive_dir = 'archive/train/'
#validation_archive_dir = 'archive/test/'
train_archive_dir = r"C:\Users\jhasr\Downloads\archive\train"
validation_archive_dir = r"C:\Users\jhasr\Downloads\archive\test"


train_archivegen = ImageDataGenerator(
                       rescale = 1./255,
                       rotation_range = 30,
                       shear_range = 0.3,
                       zoom_range = 0.3,
                       horizontal_flip = True,
                       fill_mode = 'nearest')

validation_archivegen = ImageDataGenerator(rescale = 1./255)

train_generator = train_archivegen.flow_from_directory(
                     train_archive_dir,
                     color_mode = 'grayscale',
                     target_size = (48,48),
                     batch_size = 32,
                     class_mode = 'categorical',
                     shuffle = True)

validation_generator = validation_archivegen.flow_from_directory(
                              validation_archive_dir,
                              color_mode = 'grayscale',
                              target_size = (48,48),
                              batch_size = 32,
                              class_mode = 'categorical',
                              shuffle = True)   


class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__()


model = Sequential()

model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu', input_shape = (48,48,1)))

model.add(Conv2D(64,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

train_path = r"C:\Users\jhasr\Downloads\archive\train"
test_path = r"C:\Users\jhasr\Downloads\archive\test"

import time  # Add this near your imports if it's not already there

# Count images again to be sure
print("Counting training and validation images...")
num_train_imgs = 0
for root, dirs, files in os.walk(train_path): 
    num_train_imgs += len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

num_test_imgs = 0
for root, dirs, files in os.walk(test_path): 
    num_test_imgs += len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

print(f"Training images: {num_train_imgs}")
print(f"Validation images: {num_test_imgs}")

# Set safe steps per epoch
steps_per_epoch = max(1, num_train_imgs // 32)
validation_steps = max(1, num_test_imgs // 32)

# Train the model
print("Starting training...")
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # increasing the number of epochs for better accuracy
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1
)

print("Training finished at:", time.strftime('%X'))
print("Total training time (s):", time.time() - start_time)

model.save("emotion_model.h5")

import os
print("Model saved at:", os.getcwd())
