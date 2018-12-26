import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import sys
import os

from torch.utils.data import Dataset, DataLoader
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from skimage import io, transform
from nolearn.lasagne import BatchIterator


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None,cols=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = read_csv(os.path.expanduser(csv_file))
        self.landmarks_frame['Image'] = self.landmarks_frame['Image'].apply(lambda im: np.fromstring(im, sep=' '))
        if cols:  # get a subset of columns
            df = df[list(cols) + ['Image']]
        print(self.landmarks_frame.count())  # prints the number of values for each column
        self.landmarks_frame = self.landmarks_frame.dropna()  # drop all rows that have missing values in them

        X = np.vstack(self.landmarks_frame['Image'].values) / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)

        y = self.landmarks_frame[self.landmarks_frame.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
        self.X = X
        self.y = y
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return ( len(self.landmarks_frame))

    def __getitem__(self, idx):
        sample = {'image': self.X[idx], 'landmarks': self.y[idx]}
        if self.transform:
            sample['image'] = sample['image'].reshape(96, 96)
            sample = self.transform(sample)

        return sample











