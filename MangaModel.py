from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.layers import Conv2D  # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D  # Pooling
from tensorflow.keras.layers import Dropout, Input, Average, Activation, Dense, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l2


class MangaModel:
    def __init__(self):
        input = Input(shape=(128, 128, 1))
        conv1 = Conv2D(32, 4, 2, activation='relu', padding='same')(input)

        conv2_1 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv1)
        conv2_2 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv2_1)
        sum2 = Average()([conv1, conv2_2])
        act2 = Activation('relu')(sum2)

        conv3_1 = Conv2D(32, 3, 1, activation='relu', padding='same')(act2)
        conv3_2 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv3_1)
        sum3 = Average()([act2, conv3_2])
        act3 = Activation('relu')(sum3)

        conv4 = Conv2D(64, 4, 2, activation='relu', padding='same')(act3)

        conv5_1 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv4)
        conv5_2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv5_1)
        sum5 = Average()([conv4, conv5_2])
        act5 = Activation('relu')(sum5)

        conv6_1 = Conv2D(64, 3, 1, activation='relu', padding='same')(act5)
        conv6_2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv6_1)
        sum6 = Average()([act5, conv6_2])
        act6 = Activation('relu')(sum6)

        conv7 = Conv2D(128, 4, 2, activation='relu', padding='same')(act6)

        conv8_1 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv7)
        conv8_2 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv8_1)
        sum8 = Average()([conv7, conv8_2])
        act8 = Activation('relu')(sum8)

        conv9_1 = Conv2D(128, 3, 1, activation='relu', padding='same')(act8)
        conv9_2 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv9_1)
        sum9 = Average()([act8, conv9_2])
        act9 = Activation('relu')(sum9)

        conv10 = Conv2D(256, 4, 2, activation='relu', padding='same')(act9)
        """
        conv11_1 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv10)
        conv11_2 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv11_1)
        sum11 = Average()([conv10, conv11_2])
        act11 = Activation('relu')(sum11)

        conv12_1 = Conv2D(256, 3, 1, activation='relu', padding='same')(act11)
        conv12_2 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv12_1)
        sum12 = Average()([act11, conv12_2])
        act12 = Activation('relu')(sum12)

        conv13 = Conv2D(512, 4, 2, activation='relu', padding='same')(act12)

        conv14_1 = Conv2D(512, 3, 1, activation='relu', padding='same')(conv13)
        conv14_2 = Conv2D(512, 3, 1, activation='relu', padding='same')(conv14_1)
        sum14 = Average()([conv13, conv14_2])

        conv15_1 = Conv2D(512, 3, 1, activation='relu', padding='same')(sum14)
        conv15_2 = Conv2D(512, 3, 1, activation='relu', padding='same')(conv15_1)
        sum15 = Average()([sum14, conv15_2])

        conv16 = Conv2D(1024, 3, 2, activation='relu', padding='same')(sum15)
        """
        flat = Flatten()(conv10)
        den1 = Dense(units=1024, activation='relu')(flat)
        den1_drop = Dropout(rate=0.4)(den1)
        den2 = Dense(units=1024, activation='relu')(den1_drop)
        den2_drop = Dropout(rate=0.4)(den2)
        den3 = Dense(units=512, activation='relu')(den2_drop)
        den3_drop = Dropout(rate=0.1)(den3)
        den4 = Dense(units=256, activation='relu')(den3_drop)
        den4_drop = Dropout(rate=0.1)(den4)
        den5 = Dense(units=256, activation='relu')(den4_drop)
        den5_drop = Dropout(rate=0.1)(den5)
        den6 = Dense(units=91, activation='softmax')(den5_drop)
        self.model = Model(inputs = input, outputs = den6)
