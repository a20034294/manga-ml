# Import Keras libraries and packages
from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.layers import Conv2D  # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D  # Pooling
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense  # Fully Connected Networks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers

import numpy as np
import os
import cv2
from DataGenerator import DataGenerator

# initializing CNN
model = Sequential()
model.add(
    Conv2D(
        128,
        5,
        2,
        input_shape=(128, 128, 1),
        activation='relu',
        padding='same'))
model.add(Conv2D(128, 5, 2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, 3, 2, activation='relu', padding='same'))
model.add(Conv2D(256, 3, 2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(
    Dense(
        units=1024,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(
    Dense(
        units=1024,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=91, activation='softmax'))

model.summary()

list_IDs = []
labels = []

for i in range(1, 91):
    dirs = os.listdir('face/' + str(i))
    for file in dirs:
        list_IDs.append('face/' + str(i) + '/' + file)
        one_hot = to_categorical(i, 91)
        labels.append(one_hot)

# Shuffle IDs
from sklearn.utils import shuffle
list_IDs_shu, labels_shu = shuffle(list_IDs, labels, random_state=0)

from sklearn.model_selection import train_test_split
seed = 7 # 確保相同的驗證集分割
IDs_train, IDs_val, labels_train, labels_val = train_test_split(list_IDs_shu, labels_shu, test_size=0.2, random_state=seed)


# Generators
# Parameters
params = {'dim': (128, 128),
          'batch_size': 128,
          'n_classes': 91,
          'n_channels': 1,
          'shuffle': True}
training_generator = DataGenerator(IDs_train, labels_train, **params)
validation_generator = DataGenerator(IDs_val, labels_val, **params)


# 定義訓練方式
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['categorical_accuracy'])

# 開始訓練
train_history = model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=1,
    epochs=20,
    verbose=2,
    shuffle=True)

model.save('manga_model.h5')
model.save_weights('manga_model_weights.h5')
