
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import MangaModelAE

import os
import numpy as np

list_IDs = []
labels = []

f = open('face_test_set.txt')
lines = f.readlines()
for line in lines:
    data = line.split(' ')
    if int(data[1]) == 91:
        break
    list_IDs.append(data[0])
    one_hot = to_categorical(int(data[1]), 91)
    labels.append(one_hot)

model = MangaModelAE.MangaModel().model
model.load_weights('manga_model_weights.h5')
encoder = Model(inputs=model.input, outputs=model.get_layer('dense').output)

# Shuffle IDs
from sklearn.utils import shuffle
list_IDs_shu, labels_shu = shuffle(list_IDs, labels, random_state=None)

from DataGenerator import DataGenerator
params = {'dim': (128, 128),
          'batch_size': len(labels),
          'n_classes': 91,
          'n_channels': 1,
          'shuffle': True}
generator = DataGenerator(list_IDs_shu, labels_shu, **params)
x, y = generator.__getitem__(0)
pred = model.predict(x)
pred_en = encoder.predict(x)

pred_255 = pred * 255;
x_255 = x * 255;
from PIL import Image
for i in range(0, 10):

    inp = np.zeros((128, 128, 3), dtype = np.uint8)
    inp[:, :, 0] = pred_255[i][:, :, 0]
    inp[:, :, 1] = pred_255[i][:, :, 0]
    inp[:, :, 2] = pred_255[i][:, :, 0]
    # print(np.shape(inp))
    # print(inp)
    img = Image.fromarray(inp)
    img.save('testimg/' + str(np.argmax(y[i])) + '-' + str(i) + '.jpg')

    inp = np.zeros((128, 128, 3), dtype = np.uint8)
    inp[:, :, 0] = x_255[i][:, :, 0]
    inp[:, :, 1] = x_255[i][:, :, 0]
    inp[:, :, 2] = x_255[i][:, :, 0]
    # print(np.shape(inp))
    # print(inp)
    img = Image.fromarray(inp)
    img.save('testimg/' + str(np.argmax(y[i])) + '-' + str(i) + 'input.jpg')


ans = []
for i in range(0, len(y)):
    ans.append(np.argmax(y[i]))

print(ans)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
print(np.shape(pred_en))

