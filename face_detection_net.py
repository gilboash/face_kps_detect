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

import random
from collections import OrderedDict

from face_detection_dataset import FaceLandmarksDataset
from face_detection_transforms import FlipHorizontal,Rescale,RandomCrop,ToTensor,NoneTransform
import utils



class Net(nn.Module):
    def __init__(self,num_output_units=30):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.do1   = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.do2   = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.do3   = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 11 * 11, 500)
        self.do4   = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, num_output_units)

    def forward(self, x):
        #ended up not using dropout
        x = (self.pool1(F.relu(self.conv1(x))))
        x = (self.pool2(F.relu(self.conv2(x))))
        x = (self.pool3(F.relu(self.conv3(x))))
        x = x.view(-1, 128 * 11 * 11)
        x = (F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def adjust_learning_rate(optimizer, epoch,init_lr=utils.LEARNING_RATE,init_momentum=0.9):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.9 ** (epoch // 1))
    momentum = init_momentum * (1.01 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = momentum
    print "lr is %f mem is %f" %(lr,momentum)
    return lr, momentum

def Net_Train(net,cols=None,target_pt_fn=None,flip_indices=None, num_output_units=30):

    #composed = transforms.Compose([FlipHorizontal(flip_indices=flip_indices),Rescale(192),RandomCrop(96)])
    composed = FlipHorizontal(flip_indices=flip_indices)
    face_dataset = FaceLandmarksDataset(csv_file=utils.FTRAIN,
                                        root_dir='data/faces/',cols=cols)

    face_datase_trans = FaceLandmarksDataset(csv_file=utils.FTRAIN,
                                        root_dir='data/faces/',
                                        transform=composed,cols=cols)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=utils.LEARNING_RATE, momentum=0.9)

    if utils.ADJUST_LEARNING_RATE==1:
        lr, momentum = adjust_learning_rate(optimizer,0,init_lr=utils.LEARNING_RATE,init_momentum=0.9)
    print "Start training! num ephocs %d size of dataset %d" %(utils.EPOCHS,len(face_dataset))
    #train
    for epoch in range(utils.EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        if utils.ADJUST_LEARNING_RATE==1:
            lr, momentum = adjust_learning_rate(optimizer,epoch,init_lr=lr,init_momentum=momentum)
        for i, data in enumerate(face_dataset):
            # zero the parameter gradients
            optimizer.zero_grad()
            # get the inputs
            #inputs, labels = data
            if utils.ALLOW_TRANSFORMS==1 and random.randint(0,1) == 1:
                data = face_datase_trans[i]
            # forward + backward + optimize
            resized_input = data['image'].reshape(-1, 1, 96, 96)
            _tensor = torch.from_numpy(resized_input)
            outputs = net(_tensor)
            resized_exp = torch.from_numpy(data['landmarks']).reshape(1,num_output_units)
            loss = criterion(outputs, resized_exp)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    #save model to target weights.pt file
    if target_pt_fn!=None:
        torch.save(net, target_pt_fn )


def train_subsets(settings=None):

    if settings==None:
        net = Net()
        Net_Train(net,target_pt_fn=utils.MODEL)
        specialists=net
    else:
        specialists =  OrderedDict()
        for setting in settings:
            cols = setting['columns']
            model = Net(setting['num_ele'])
            #max_epochs = int(1e7 / Net(setting['num_ele'])

            weight_fn = "data/weights%d.pt" %setting['unique_id']
            Net_Train(model,cols=cols,target_pt_fn=weight_fn, flip_indices=setting['flip_indices'], num_output_units=setting['num_ele'])
            #model.fit(X, y)
            specialists[cols] = model

    return specialists

def run_subsets(_tensor,specialists=None,settings=None,load_from_file=False):
    if settings==None:
        if load_from_file==True:
            weight_fn = utils.MODEL 
            net = torch.load(weight_fn)
            net.eval()
        else:
            net = specialists 
        outputs = net(_tensor)
    else:
        outputs = torch.Tensor(0,0)
        for setting in settings:
            cols = setting['columns']
            #max_epochs = int(1e7 / Net(setting['num_ele'])

            if load_from_file==True:
                weight_fn = "data/weights%d.pt" %setting['unique_id']
                net = torch.load(weight_fn)
                net.eval()
            else:
                net = specialists[cols] 

            outputs = torch.cat((outputs,net(_tensor)),1)
         
    return outputs
        




