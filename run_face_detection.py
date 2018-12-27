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
from pandas import DataFrame
from sklearn.utils import shuffle
from datetime import datetime


from face_detection_net import Net, Net_Train , run_subsets
from face_detection_dataset import FaceLandmarksDataset
from utils import plot_sample, plot_face_kps_16_batch
import utils


face_dataset_test = FaceLandmarksDataset(csv_file=utils.FTEST,
                                    root_dir='data/faces/')


#start running model
#from some reason, test.csv doesnt have output labels. thats weird, should further look into it
fig = plt.figure()
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
plot_index=0
#correct = 0
#total = 0
with torch.no_grad():
    output_list = []
    numpy_out = np.empty((len(face_dataset_test), 0))
    for data in face_dataset_test:
        resized_input = data['image'].reshape(-1, 1, 96, 96)
        _tensor = torch.from_numpy(resized_input)

        outputs = run_subsets(_tensor,settings=utils.SETTINGS,load_from_file=True)

        this_output = outputs.numpy() 
        numpy_out =  this_output[0] * 48 + 48
        numpy_out = numpy_out.clip(0, 96)
        output_list.append(this_output[0])
        if plot_index<16:   
            ax = fig.add_subplot(4, 4,plot_index+ 1, xticks=[], yticks=[])
            with torch.no_grad():
                plot_sample(data['image'],(outputs).reshape(30,1) , ax)
            plot_index=plot_index+1

    
    #plt.show()
    columns = ()
    for cols in utils.SPECIALIST_SETTINGS:
        columns += cols['columns']
    print columns.index('mouth_left_corner_x')
    #df = DataFrame(numpy_out, columns=columns)

    lookup_table = read_csv(utils.FLOOKUP)
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            output_list[row.ImageId - 1][columns.index(row.FeatureName)],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))

sys.exit(1)











