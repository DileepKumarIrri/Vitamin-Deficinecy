import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Define image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 32

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
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No need to shuffle for evaluation
)

# Define the ResNet50 base model
def create_resnet_model():
    resnet_model = ResNet50(
        input_shape=(img_height, img_width, 3),  # Adjust input shape
        include_top=False,  # Exclude the fully-connected layers
        weights='imagenet'  # Pre-trained on ImageNet
    )
    resnet_model.trainable = False

    model = Sequential([
        resnet_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(14, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the VGG16 base model
def create_vgg_model():
    vgg_model = VGG16(
        input_shape=(img_height, img_width, 3),  # Adjust input shape
        include_top=False,  # Exclude the fully-connected layers
        weights='imagenet'  # Pre-trained on ImageNet
    )
    vgg_model.trainable = False

    model = Sequential([
        vgg_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(14, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap Keras models as scikit-learn compatible classifiers
resnet_classifier = KerasClassifier(build_fn=create_resnet_model, epochs=8, batch_size=batch_size, verbose=0)
vgg_classifier = KerasClassifier(build_fn=create_vgg_model, epochs=8, batch_size=batch_size, verbose=0)

# Combine the models into a voting ensemble
ensemble = VotingClassifier(estimators=[('resnet', resnet_classifier), ('vgg', vgg_classifier)], voting='hard')

# Train the ensemble
ensemble.fit(train_generator, train_generator.classes)

# Evaluate the ensemble
predictions = ensemble.predict(test_generator)
true_labels = test_generator.classes
accuracy = accuracy_score(true_labels, predictions)
print("Ensemble Accuracy:", accuracy)
