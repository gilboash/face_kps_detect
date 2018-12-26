import sys
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from skimage import io, transform
from nolearn.lasagne import BatchIterator



EPOCHS = 5
FTRAIN = 'data/face_kps/training.csv'
FTEST = 'data/face_kps/test.csv'
MODEL = 'data/weights.pt'
LEARNING_RATE = 0.001
ADJUST_LEARNING_RATE = 0
ALLOW_TRANSFORMS = 0

DEBUG = 0
PLOT = 0


def plot_sample(x, y, axis, reshape=True):
        if reshape==True:
            img = x.reshape(96, 96)
        else:
            img = x
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def plot_face_kps_16_batch(face_dataset,starting_idx):
    fig = plt.figure()
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):#len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(sample['image'], sample['landmarks'], ax)
    plt.show()














