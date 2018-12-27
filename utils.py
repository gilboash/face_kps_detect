import sys
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from skimage import io, transform
from nolearn.lasagne import BatchIterator


SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        num_ele=4,
        unique_id=1,
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        num_ele=2,
        unique_id=2,
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        num_ele=6,
        unique_id=3,
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        num_ele=2,
        unique_id=4,
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        num_ele=8,
        unique_id=5,
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        num_ele=8,
        unique_id=6,
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]



EPOCHS = 300
FTRAIN = 'data/face_kps/training.csv'
FTEST = 'data/face_kps/test.csv'
MODEL = 'data/weights.pt'
LEARNING_RATE = 0.001
ADJUST_LEARNING_RATE = 0
ALLOW_TRANSFORMS = 1
SETTINGS=SPECIALIST_SETTINGS#None
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









