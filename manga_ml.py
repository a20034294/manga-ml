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

# initializing CNN
model = Sequential()
model.add(
    Conv2D(
        128,
        5,
        2,
        input_shape=(256, 256, 1),
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
model.add(Dense(units=31, activation='sigmoid'))

model.summary()

x = []
y = []
for i in range(1, 31):
    dirs = os.listdir('face/' + str(i))
    for file in dirs:
        x.append(cv2.imread('face/' + str(i) + '/' + file))
        one_hot = to_categorical(i, 31)
        y.append(one_hot)

    #break

XX = np.array(x)
XXX = XX[:, :, :, 0]
XXX = XXX[:, :, :, np.newaxis]
print(XXX)
X = XXX / 255

Y = np.array(y)

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=0)

print(X.shape)
print(Y.shape)

# 定義訓練方式
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['categorical_accuracy'])

# 開始訓練
train_history = model.fit(
    x=X,
    y=Y,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    verbose=2,
    shuffle=True)

model.save('manga_model.h5')
model.save_weights('manga_model_weights.h5')