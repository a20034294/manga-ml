from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D  # Pooling
from tensorflow.keras.layers import Dropout, Input, Activation, Dense, Flatten, Add, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, multi_gpu_model

from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l2

import tensorflow as tf

class MangaModel:
    def __init__(self):
        input = Input(shape=(128, 128, 1))

        x = Conv2D(16, 3, 1, activation=LeakyReLU(), padding='same')(input)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(256, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        
        x = Dense(units=512, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        encoded = Dense(units=128, activation=LeakyReLU(), name='encoded')(x)
        x = Dense(units=128, activation=LeakyReLU())(encoded)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=128, activation=LeakyReLU())(x)
        x = Dense(units=4096, activation=LeakyReLU())(x)
        x = Reshape((4, 4, 256))(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(256, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, 3, 1, activation=LeakyReLU(), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, 3, 1, activation=LeakyReLU(), padding='same')(x)

        decoded = Conv2D(1, 3, 1, activation='sigmoid', padding='same')(x)

        model = Model(inputs = input, outputs = decoded)

        self.model = model
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

