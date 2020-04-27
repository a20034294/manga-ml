from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.layers import Conv2D  # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D  # Pooling
from tensorflow.keras.layers import Dropout, Input, Activation, Dense, Flatten, Add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l2


class MangaModel:
    def __init__(self):
        input = Input(shape=(128, 128, 1))
        conv1 = Conv2D(32, 4, 2, activation=LeakyReLU(), padding='same')(input)

        conv2_1 = Conv2D(32, 3, 1, activation=LeakyReLU(), padding='same')(conv1)
        conv2_2 = Conv2D(32, 3, 1, activation=LeakyReLU(), padding='same')(conv2_1)
        sum2 = Add()([conv1, conv2_2])
        act2 = Activation(LeakyReLU())(sum2)

        conv3_1 = Conv2D(32, 3, 1, activation=LeakyReLU(), padding='same')(act2)
        conv3_2 = Conv2D(32, 3, 1, activation=LeakyReLU(), padding='same')(conv3_1)
        sum3 = Add()([act2, conv3_2])
        act3 = Activation(LeakyReLU())(sum3)

        conv4 = Conv2D(64, 4, 2, activation=LeakyReLU(), padding='same')(act3)

        conv5_1 = Conv2D(64, 3, 1, activation=LeakyReLU(), padding='same')(conv4)
        conv5_2 = Conv2D(64, 3, 1, activation=LeakyReLU(), padding='same')(conv5_1)
        sum5 = Add()([conv4, conv5_2])
        act5 = Activation(LeakyReLU())(sum5)

        conv6_1 = Conv2D(64, 3, 1, activation=LeakyReLU(), padding='same')(act5)
        conv6_2 = Conv2D(64, 3, 1, activation=LeakyReLU(), padding='same')(conv6_1)
        sum6 = Add()([act5, conv6_2])
        act6 = Activation(LeakyReLU())(sum6)

        conv7 = Conv2D(128, 4, 2, activation=LeakyReLU(), padding='same')(act6)

        conv8_1 = Conv2D(128, 3, 1, activation=LeakyReLU(), padding='same')(conv7)
        conv8_2 = Conv2D(128, 3, 1, activation=LeakyReLU(), padding='same')(conv8_1)
        sum8 = Add()([conv7, conv8_2])
        act8 = Activation(LeakyReLU())(sum8)

        conv9_1 = Conv2D(128, 3, 1, activation=LeakyReLU(), padding='same')(act8)
        conv9_2 = Conv2D(128, 3, 1, activation=LeakyReLU(), padding='same')(conv9_1)
        sum9 = Add()([act8, conv9_2])
        act9 = Activation(LeakyReLU())(sum9)

        conv10 = Conv2D(256, 4, 2, activation=LeakyReLU(), padding='same')(act9)
        '''
        conv11_1 = Conv2D(256, 3, 1, activation=LeakyReLU(), padding='same')(conv10)
        conv11_2 = Conv2D(256, 3, 1, activation=LeakyReLU(), padding='same')(conv11_1)
        sum11 = Add()([conv10, conv11_2])
        act11 = Activation(LeakyReLU())(sum11)

        conv12_1 = Conv2D(256, 3, 1, activation=LeakyReLU(), padding='same')(act11)
        conv12_2 = Conv2D(256, 3, 1, activation=LeakyReLU(), padding='same')(conv12_1)
        sum12 = Add()([act11, conv12_2])
        act12 = Activation(LeakyReLU())(sum12)

        conv13 = Conv2D(512, 4, 2, activation=LeakyReLU(), padding='same')(act12)

        conv14_1 = Conv2D(512, 3, 1, activation=LeakyReLU(), padding='same')(conv13)
        conv14_2 = Conv2D(512, 3, 1, activation=LeakyReLU(), padding='same')(conv14_1)
        sum14 = Add()([conv13, conv14_2])

        conv15_1 = Conv2D(512, 3, 1, activation=LeakyReLU(), padding='same')(sum14)
        conv15_2 = Conv2D(512, 3, 1, activation=LeakyReLU(), padding='same')(conv15_1)
        sum15 = Add()([sum14, conv15_2])

        conv16 = Conv2D(1024, 3, 2, activation=LeakyReLU(), padding='same')(sum15)
        '''
        flat = Flatten()(conv10)
        den1 = Dense(units=1024, activation=LeakyReLU())(flat)
        den1_drop = Dropout(rate=0.4)(den1)
        den2 = Dense(units=1024, activation=LeakyReLU())(den1_drop)
        den2_drop = Dropout(rate=0.4)(den2)
        den3 = Dense(units=91, activation='softmax')(den2_drop)
        self.model = Model(inputs = input, outputs = den3)

