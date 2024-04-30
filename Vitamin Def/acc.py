from sklearn.metrics import accuracy_score
from keras.models import load_model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
# Load the saved MobileNet model
model = load_model("model/Mobilenet.h5")

# Load and preprocess the test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(224, 224),  # MobileNet input size
                                            batch_size=12,
                                            class_mode='categorical',
                                            shuffle=False)  # Make sure to set shuffle to False

# Predict the labels for the test set
y_pred_prob = model.predict(test_set)
y_pred = np.argmax(y_pred_prob, axis=1)

# Get the true labels
y_true = test_set.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
