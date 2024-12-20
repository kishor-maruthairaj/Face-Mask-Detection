1. Imports



#os, random: File path management and randomness.

#numpy: Numerical operations.

#matplotlib, seaborn: Visualization tools.

#ImageDataGenerator: Used for image augmentation and preprocessing.

#VGG16: A pre-trained model for transfer learning.

#Model layers: To build a neural network.


#Callbacks: Improve training through early stopping, model checkpointing, and learning rate adjustments.

#Evaluation metrics: classification_report, confusion_matrix, and f1_score.

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.image import load_img

# Paths to the dataset
data_dir = "C:/Users/HP/Downloads/data"  # Update with your dataset path

#rescale=1./255: Normalizes image pixel values to [0, 1].

#Augmentation includes rotation, zoom, flipping, and shifts to make the model robust to variations.

#validation_split=0.2: Splits the dataset into 80% training and 20% validation.

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


#Reads images from data_dir.

#target_size=(224, 224): Resizes images to match VGG16 input dimensions.

#class_mode='binary': For a binary classification problem (mask/no mask).


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


#VGG16: A pre-trained model trained on ImageNet data.

#include_top=False: Removes the top layers (fully connected layers) to use it as a feature extractor.


# Model Development with Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Unfreeze some layers for fine-tuning

#Freezes the first 15 layers and fine-tunes the rest to adapt VGG16 features to the new dataset.

# Fine-tune from the 15th layer onwards
for layer in base_model.layers[:15]:
    layer.trainable = False

#GlobalAveragePooling2D: Reduces the feature maps into a vector.

#Dense(128): Adds a dense layer with 128 neurons and ReLU activation.

#Dropout(0.5): Prevents overfitting.

#Dense(1): Final binary output layer with sigmoid activation.


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
#Optimizer: Adam with a learning rate of 0.001.

#Loss Function: Binary cross-entropy since it is a binary classification task.

#Metrics: Accuracy.

# Model Summary
model.summary()

# Callbacks
#EarlyStopping: Stops training if validation loss doesn't improve after 5 epochs.

#ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.

#ModelCheckpoint: Saves the best model based on validation accuracy.

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_face_mask_detector.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
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

# Load the Best Model
from tensorflow.keras.models import load_model
best_model = load_model('best_face_mask_detector.keras')

# Evaluate the Model
val_generator.reset()
predictions = best_model.predict(val_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Classification Report

classification_report: Displays precision, recall, and F1-score for each class.
confusion_matrix: Plots a confusion matrix to visualize true vs predicted labels.

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# F1 Score
f1 = f1_score(true_classes, predicted_classes)
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Inference Function
def predict_mask(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = best_model.predict(img_array)
    return "Mask" if prediction > 0.5 else "No Mask"

# Test the Inference Function
image_path = "C:/Users/HP/Downloads/test_image.jpg"  # Replace with your test image path
result = predict_mask(image_path)
print(f"Prediction for the test image: {result}")

# Display Random Images from Validation Dataset
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
