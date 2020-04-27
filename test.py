
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers
import MangaModel

import os
import numpy as np

list_IDs = []
labels = []

f = open('face_test_set.txt')
lines = f.readlines()
for line in lines:
    data = line.split(' ')
    if int(data[1]) == 21:
        break
    list_IDs.append(data[0])
    one_hot = to_categorical(int(data[1]), 91)
    labels.append(one_hot)

model = MangaModel.MangaModel().model
model.load_weights('manga_model_weights.h5')

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

ans = []
ac = 0
wa = 0
for i in range(0, len(y)):
    ans.append(np.argmax(y[i]))
    if np.argmax(pred[i]) == np.argmax(y[i]):
        ac = ac + 1
    else:
        wa = wa + 1
print('ac: ', ac)
print('wa: ', wa)
print('acc: ', ac / (wa + wa))

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
pred_np = np.array(pred)
print(np.shape(pred_np))
tsne.fit_transform(pred_np)
# print(tsne.embedding_)

X = tsne.embedding_[:, 0]
Y = tsne.embedding_[:, 1]

import pandas as pd
data = pd.DataFrame({
    'X': X,
    'Y': Y,
    'ans': ans
})
print(data)
import matplotlib.pyplot as plt
import seaborn as sns

g = sns.lmplot(x="X", y="Y", hue="ans", data=data, scatter_kws={"s": 10} ,fit_reg=False, palette='bright')
plt.savefig("tsne.png")

