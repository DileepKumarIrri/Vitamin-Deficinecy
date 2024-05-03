from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator here
from keras.models import load_model

# Create the decision tree classifier
model = DecisionTreeClassifier()

# Load and preprocess the image data
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(128, 128),
                                                 batch_size=12,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(128, 128),
                                            batch_size=12,
                                            class_mode='categorical')

print("\nTraining the decision tree model...\n")
X_train, y_train = [], []

# Process the training data
for i in range(len(training_set)):
    batch_X, batch_y = training_set[i]
    batch_X = batch_X.reshape(batch_X.shape[0], -1)  # Flatten the images
    X_train.append(batch_X)
    y_train.append(batch_y.argmax(axis=1))

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# Train the decision tree model
model.fit(X_train, y_train)
history=model.fit(training_set,epochs=75,validation_data = test_set,verbose = 1)
print("\nTesting the decision tree model...\n")
X_test, y_test = [], []

# Process the test data
for i in range(len(test_set)):
    batch_X, batch_y = test_set[i]
    batch_X = batch_X.reshape(batch_X.shape[0], -1)  # Flatten the images
    X_test.append(batch_X)
    y_test.append(batch_y.argmax(axis=1))

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

model.evaluate(test_set)

model.save(r"model\CNN.h5")

plt.style.use("ggplot")
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['accuracy'], color='b', label="Training Accuracy")
ax[0].plot(history.history['val_accuracy'], color='r',label="Validation Accuracy")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['loss'], color='b', label="Training loss")
ax[1].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[1])
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig(r"model/CNN_acc.png")
from sklearn.metrics import accuracy_score

# Assuming you have the predicted labels stored in 'y_pred'
y_pred = model.predict(X_test)

# Convert one-hot encoded labels to class labels
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
