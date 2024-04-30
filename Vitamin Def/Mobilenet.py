import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
# Define image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 5

# Define directories for training and testing data
train_data_dir = "dataset/train"
test_data_dir = "dataset/test"

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255  # Normalize pixel values
)

# Data augmentation for testing images (only rescale)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators for training and testing
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False  # No need to shuffle for evaluation
)

# Define the MobileNet base model
base_model = MobileNet(
    input_shape=(img_height, img_width, 3),  # Adjust input shape
    include_top=False,  # Exclude the fully-connected layers
    weights='imagenet'  # Pre-trained on ImageNet
)

# Freeze the base model layers
base_model.trainable = False

# Create the classification head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(14, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=8,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Evaluate the model on the test set
model.evaluate(test_generator)

# Save the model
# model.save("model/Mobilenet1.h5")

# Plot training history
# plt.style.use("ggplot")
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(history.history['accuracy'], color='b', label="Training Accuracy")
# ax[0].plot(history.history['val_accuracy'], color='r', label="Validation Accuracy")
# legend = ax[0].legend(loc='best', shadow=True)

# ax[1].plot(history.history['loss'], color='b', label="Training Loss")
# ax[1].plot(history.history['val_loss'], color='r', label="Validation Loss")
# legend = ax[1].legend(loc='best', shadow=True)
# plt.savefig("model/mobilenet_acc.png")
# Get the true labels of the test data
true_labels = test_generator.classes

# Make predictions on the test data
predictions = model.predict(test_generator)

# Convert predictions from probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)