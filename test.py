
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
model.load_weights('manga_model_weights.h5')
x = []
y = [0.] * 91

pred_list = []
ans_list = []

for i in range(1, 31):
    cnt = 0
    dirs = os.listdir('face/' + str(i))
    for file in dirs:
        if file[0] == '.':
            continue

        cnt = cnt + 1
        if cnt % 10 != 0:
            continue
        x = np.array(cv2.imread('face/' + str(i) + '/' + file))
        x2 = x[:, :, 0]
        x3 = x2[np.newaxis, :, :, np.newaxis]
        x4 = x3 / 255
        pred = model.predict(x4)

        pred_list.append(pred[0])
        # ans_list.append(i)
        for idx, num in enumerate(pred[0]):
            if num == max(pred[0]):
                if idx == i:
                    ans_list.append(1)
                else:
                    ans_list.append(0)

    print(i)


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
pred_np = np.array(pred_list)
print(np.shape(pred_np))
tsne.fit_transform(pred_np)
# print(tsne.embedding_)

X = tsne.embedding_[:, 0]
Y = tsne.embedding_[:, 1]

import pandas as pd
data = pd.DataFrame({
    'X': X,

    'Y': Y,
    'ans': ans_list
})
print(data)
import matplotlib.pyplot as plt
import seaborn as sns

g = sns.lmplot(x="X", y="Y", hue="ans", data=data, scatter_kws={"s": 10} ,fit_reg=False, palette='bright')
plt.savefig("tsne.png")

'''
        y = y + pred
        cnt = cnt + 1
        for idx, num in enumerate(pred[0]):
            if num == max(pred[0]):
                print(idx)
        print()

    y = y / cnt
    y = y[0]
    print('第' + str(i))
    if max(y) > 0.1:
        print(max(y))

        # print(y)
        print()


    #break
'''


