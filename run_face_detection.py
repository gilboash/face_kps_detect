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



FTEST = 'data/face_kps/test.csv'
MODEL = 'data/weights.pt'
PLOT = 1


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = read_csv(os.path.expanduser(csv_file))
        self.landmarks_frame['Image'] = self.landmarks_frame['Image'].apply(lambda im: np.fromstring(im, sep=' '))

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
            sample = self.transform(sample)

        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 11 * 11, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 30)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


face_dataset_test = FaceLandmarksDataset(csv_file=FTEST,
                                    root_dir='data/faces/')
net = Net()

net = torch.load(MODEL)
net.eval()


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

sys.exit(1)










