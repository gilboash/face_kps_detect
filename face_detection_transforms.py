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

from face_detection_dataset import FaceLandmarksDataset
import utils
import copy



class NoneTransform(object):
    def __call__(self, sample):
        Xb, yb = sample['image'], sample['landmarks']
        return {'image': Xb, 'landmarks': yb}
        
class FlipHorizontal(object):
    flip_indices_default = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]
    def __init__(self, flip_indices=None):
        if flip_indices==None:
            self.flip_indices=self.flip_indices_default
        else:
            self.flip_indices = flip_indices



    def __call__(self, sample):
        Xb, yb = sample['image'], sample['landmarks']
        
        # Flip half of the images in this batch at random:
        orig_shape = Xb.shape[::2][0]
        Xb = Xb.reshape(-1, 1, orig_shape,orig_shape ) 
        Xb = Xb[:, :, :,::-1] - np.zeros_like(Xb)

        Xb = Xb.reshape(orig_shape,orig_shape) 

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[::2] = yb[::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[a], yb[b] = (
                    yb[b], yb[a])

        return {'image': Xb, 'landmarks': yb}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        for i in range(len(landmarks)/2):
            x = (landmarks[2*i] * 48 + 48)*new_w/w
            y = (landmarks[2*i+1] * 48 + 48)*new_h/h

            landmarks[2*i] = (x - 48)/48 
            landmarks[2*i+1] = (y-48)/48 

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']


        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        print image.shape
        #landmarks = landmarks - [left, top]
        print new_h
        print new_w
        print top
        print left
        for i in range(len(landmarks)/2):
            x = (landmarks[2*i] * 48 + 48) - left
            y = (landmarks[2*i+1] * 48 + 48) - top

            landmarks[2*i] = (x - 48)/48 
            landmarks[2*i+1] = (y-48)/48 

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = image.reshape(96, 96)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}




def plot_sample(x, y, axis, reshape=True):
        if reshape==True:
            img = x.reshape(96, 96)
        else:
            img = x
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


if __name__== "__main__":

        face_dataset = FaceLandmarksDataset(csv_file=utils.FTRAIN,
                                        root_dir='data/faces/')
        horiz = FlipHorizontal()
        none_t = NoneTransform()
        scale = Rescale(30)
        #crop = RandomCrop(50)
        composed = transforms.Compose([
                                       FlipHorizontal(),Rescale(200),ToTensor()])


        fig = plt.figure()
        sample = face_dataset[65]
        for i, tsfrm in enumerate([none_t,horiz,scale]):
            copy_sample = copy.deepcopy(sample)
            copy_sample['image'] = copy_sample['image'].reshape(96, 96)
            transformed_sample = tsfrm(copy_sample)

            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title(type(tsfrm).__name__)
            #show_landmarks(**transformed_sample)
            plot_sample(transformed_sample['image'], transformed_sample['landmarks'], ax,False)

        plt.show()

        sys.exit(1)











