from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D,GlobalAveragePooling2D,Activation
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=96, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=96, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(14, activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
# print("\nTraining the data...\n")
# training_set = train_datagen.flow_from_directory('dataset\\train',
#                                                  target_size=(128,128),
#                                                  batch_size=12,
#                                                  class_mode='categorical'
#                                                  )

training_set = train_datagen.flow_from_directory('dataset\\train',
                                                 target_size=(128,128),
                                                 batch_size=12,
                                                 class_mode='categorical',
                                                 classes=['class1', 'class2', ... , 'class14'])


# test_set = test_datagen.flow_from_directory('dataset\\test',
#                                             target_size=(128,128),
#                                             batch_size=12,
#                                             class_mode='categorical'
#
#
#                                             )

test_set = test_datagen.flow_from_directory('dataset\\test',
                                            target_size=(128,128),
                                            batch_size=12,
                                            class_mode='categorical',
                                            classes=['class1', 'class2', ... , 'class14'])

print("\n Testing the data.....\n")

accuracy = model.evaluate_generator(test_set, steps=test_set.n // test_set.batch_size)[1]
print("Accuracy:", accuracy)

history=model.fit_generator(training_set,steps_per_epoch=20,epochs=75,validation_data = test_set,verbose = 1)
model.evaluate(test_set)

model.save(r"model\CNN1.h5")

plt.style.use("ggplot")
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['accuracy'], color='b', label="Training Accuracy")
ax[0].plot(history.history['val_accuracy'], color='r',label="Validation Accuracy")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['loss'], color='b', label="Training loss")
ax[1].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[1])
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig(r"model/CNN_acc.png")

