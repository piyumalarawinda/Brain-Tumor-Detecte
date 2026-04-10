import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = []
labels = []

# Load images
for category in ["yes", "No"]:
    path = "dataset/" + category
    label = 1 if category == "yes" else 0
    
    for img in os.listdir(path):
        img_path = path + "/" + img
        
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128,128))
        
        data.append(image)
        labels.append(label)

# Convert to numpy
data = np.array(data) / 255.0
labels = np.array(labels)

# Model
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(data, labels, epochs=10, batch_size=32)

# Save model
model.save("model.h5")

print("Training Done ✅")