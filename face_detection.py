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

from face_detection_net import Net, Net_Train , train_subsets, run_subsets
from face_detection_dataset import FaceLandmarksDataset
from utils import plot_sample, plot_face_kps_16_batch
import utils



#todo
#1 - handle partial labels training. potientially adding confidence level for each label value?
#2 - once 1 is done, start training with crop transform, and crop and scale transforms



if __name__== "__main__":

    specialists = train_subsets(settings=utils.SETTINGS)

    #test
    face_dataset_test = FaceLandmarksDataset(csv_file=utils.FTEST,
                                        root_dir='data/faces/')
    #start running model
    #from some reason, test.csv doesnt have output labels. thats weird, should further look into it
    if utils.PLOT==1:
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

            outputs = run_subsets(_tensor,specialists=specialists, settings=utils.SETTINGS)
            if utils.PLOT==1 and plot_index<16:   
                ax = fig.add_subplot(4, 4,plot_index+ 1, xticks=[], yticks=[])
                with torch.no_grad():
                    plot_sample(data['image'],(outputs).reshape(30,1) , ax)
                plot_index=plot_index+1
            #resized_exp = torch.from_numpy(data['landmarks']).reshape(1,30)

            #_, predicted = torch.max(outputs.data, 1)
            #total += resized_exp.size(0)
            #correct += (predicted == resized_exp).sum().item()

    if utils.PLOT==1:
        plt.show()
    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))

    sys.exit(1)











