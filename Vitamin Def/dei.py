from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the image data using Keras ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(128, 128),
                                                 batch_size=6,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(128, 128),
                                            batch_size=6,
                                            class_mode='categorical')

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(training_set, test_set)

# Evaluate the model
accuracy = model.score(test_set)
print("Accuracy:", accuracy)

# Save the model if needed
