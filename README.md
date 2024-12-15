# Face-Mask-Detection

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.image import load_img

# Paths to the dataset
data_dir = "C:/Users/HP/Downloads/data"  # Update with your dataset path

# Data Preprocessing and Augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% validation split
)

train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Adjusted for VGG16
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Model Development with Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping]
)

# Plotting Training Results
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the Model
model.save('face_mask_detector_vgg16.h5')

# Evaluate the Model
val_generator.reset()
predictions = model.predict(val_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Classification Report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# F1 Score
f1 = f1_score(true_classes, predicted_classes)
print(f"F1 Score: {f1:.4f}")

# Display Random Images from Test Dataset
def display_random_test_images(generator, num_images=2):
    fig, ax = plt.subplots(2, num_images, figsize=(12, 8))
    categories = list(generator.class_indices.keys())
    for i, category in enumerate(categories):
        category_indices = np.where(generator.classes == generator.class_indices[category])[0]
        selected_indices = np.random.choice(category_indices, num_images, replace=False)
        for j, idx in enumerate(selected_indices):
            img_path = generator.filepaths[idx]
            img = load_img(img_path, target_size=(224, 224))
            ax[i, j].imshow(img)
            ax[i, j].axis('off')
            ax[i, j].set_title(category.replace('_', ' ').capitalize())
    plt.tight_layout()
    plt.show()

display_random_test_images(val_generator)
