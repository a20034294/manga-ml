import numpy as np
import os
import cv2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers
from DataGenerator import DataGenerator
from tensorflow.keras.utils import to_categorical
from MangaModel import MangaModel


model = MangaModel().model
model.summary()

list_IDs = []
labels = []

f = open('face_training_set.txt')
lines = f.readlines()
for line in lines:
    data = line.split(' ')
    if int(data[1]) == 91:
        break
    list_IDs.append(data[0])
    one_hot = to_categorical(int(data[1]), 91)
    labels.append(one_hot)

# Shuffle IDs
from sklearn.utils import shuffle
list_IDs_shu, labels_shu = shuffle(list_IDs, labels, random_state=None)

from sklearn.model_selection import train_test_split
seed = 9487 # 確保相同的驗證集分割
IDs_train, IDs_val, labels_train, labels_val = train_test_split(list_IDs_shu, labels_shu, test_size=0.2, random_state=seed)

print(len(IDs_train))
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
    workers=32,
    epochs=5,
    verbose=2,
    shuffle=True)

model.save('manga_model.h5')
model.save_weights('manga_model_weights.h5')
