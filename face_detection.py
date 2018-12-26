import copy 
import random

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

from face_detection_net import Net, SPECIALIST_SETTINGS


EPOCHS = 5
FTRAIN = 'data/face_kps/training.csv'
FTEST = 'data/face_kps/test.csv'
LEARNING_RATE = 0.001

DEBUG = 0
PLOT = 0


#todo
#1 - handle partial labels training. potientially adding confidence level for each label value?
#2 - once 1 is done, start training with crop transform, and crop and scale transforms
#3 - arrange methods and classes in seperate classes, unify routines from train and run
class NoneTransform(object):
    def __call__(self, sample):
        Xb, yb = sample['image'], sample['landmarks']
        return {'image': Xb, 'landmarks': yb}
        
class FlipHorizontal(object):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]


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

def adjust_learning_rate(optimizer, epoch,init_lr=LEARNING_RATE,init_momentum=0.9):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.9 ** (epoch // 1))
    momentum = init_momentum * (1.01 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = momentum
    print "lr is %f mem is %f" %(lr,momentum)
    return lr, momentum


def fit_specialists():
    specialists = OrderedDict()

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)

        model = clone(net)
        model.output_num_units = y.shape[1]
        model.batch_iterator_train.flip_indices = setting['flip_indices']
        # set number of epochs relative to number of training examples:
        model.max_epochs = int(1e7 / y.shape[0])
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])

        print("Training model for columns {} for {} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model

    with open('net-specialists.pickle', 'wb') as f:
        # we persist a dictionary with all models:
        pickle.dump(specialists, f, -1)

if __name__== "__main__":
    face_dataset = FaceLandmarksDataset(csv_file=FTRAIN,
                                        root_dir='data/faces/')

    face_datase_trans = FaceLandmarksDataset(csv_file=FTRAIN,
                                        root_dir='data/faces/',
                                        transform=FlipHorizontal())

    #debug different transform routines and combinations
    if DEBUG==1:
        horiz = FlipHorizontal()
        none_t = NoneTransform()
        scale = Rescale(30)
        #crop = RandomCrop(50)
        composed = transforms.Compose([
                                       FlipHorizontal(),Rescale(200),ToTensor()])


        fig = plt.figure()
        sample = face_datase_trans[65]
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
    net = Net()

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    lr, momentum = adjust_learning_rate(optimizer,0,init_lr=LEARNING_RATE,init_momentum=0.9)
    print "Start training! num ephocs %d size of dataset %d" %(EPOCHS,len(face_dataset))
    #train
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        lr, momentum = adjust_learning_rate(optimizer,epoch,init_lr=lr,init_momentum=momentum)
        for i, data in enumerate(face_dataset):
            # get the inputs
            #inputs, labels = data
            if random.randint(0,1) == 1:
                data = face_datase_trans[i]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            resized_input = data['image'].reshape(-1, 1, 96, 96)
            _tensor = torch.from_numpy(resized_input)

            outputs = net(_tensor)
            resized_exp = torch.from_numpy(data['landmarks']).reshape(1,30)
            if DEBUG==1:
                print (outputs)
                print (outputs.shape)
                print (outputs.dtype)
                print (resized_exp)
                print (resized_exp.shape)
                print (resized_exp.dtype)
            loss = criterion(outputs, resized_exp)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    face_dataset_test = FaceLandmarksDataset(csv_file=FTEST,
                                        root_dir='data/faces/')
    #start running model
    #from some reason, test.csv doesnt have output labels. thats weird, should further look into it
    if PLOT==1:
        fig = plt.figure()
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(
                    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    plot_index=0
    #correct = 0
    #total = 0
    with torch.no_grad():
        for data in face_dataset_test:
            resized_input = data['image'].reshape(-1, 1, 96, 96)
            _tensor = torch.from_numpy(resized_input)

            outputs = net(_tensor)
            if PLOT==1 and plot_index<16:   
                ax = fig.add_subplot(4, 4,plot_index+ 1, xticks=[], yticks=[])
                with torch.no_grad():
                    plot_sample(data['image'],(outputs).reshape(30,1) , ax)
                plot_index=plot_index+1
            #resized_exp = torch.from_numpy(data['landmarks']).reshape(1,30)

            #_, predicted = torch.max(outputs.data, 1)
            #total += resized_exp.size(0)
            #correct += (predicted == resized_exp).sum().item()

    if PLOT==1:
        plt.show()
    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    torch.save(net, 'data/weights.pt')

    sys.exit(1)











