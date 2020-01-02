
from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D  # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D  # Pooling
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense  # Fully Connected Networks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers

import os
import numpy as np
import cv2


model = load_model('manga_model.h5')
model.load_weights('manga_model.h5')
x = []
y = []
for i in range(7, 8):
    dirs = os.listdir('face/' + str(i))
    for file in dirs:
        x = np.array(cv2.imread('face/' + str(i) + '/' + file))
        x = x[:, :, 0]
        x = x[np.newaxis, :, :, np.newaxis]
        x = x / 255
        print(model.predict(x))
        one_hot = to_categorical(i, 31)
        break

    #break



