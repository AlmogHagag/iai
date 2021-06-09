import pandas as pd
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import csv
import funcOper as fu


class DigitsLandmarksDataset(Dataset):
    """Digits Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('idx = ', idx)
        # print('self.root_dir = ', self.root_dir)
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        # print('img_name = ', img_name)
        image = io.imread(img_name)
        image = np.float32(image)
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        # print('image = ', image.shape)
        # print('image = ', type(image))

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks[0]
        # print('landmarks = ', landmarks)
        landmarks = landmarks.astype('float')
        # print('image.shape[1] = ',image.shape[1])
        landmarks = landmarks / image.shape[1]
        # print('landmarks float = ', landmarks)
        sample = {'image': image, 'landmarks': landmarks}

        # print('sample type = ', type(sample))
        # print('sample[image] type = ', type(sample['image']))

        if self.transform:
            sample = self.transform(sample)

        return sample


# Create the model class
class CNN_Detection(nn.Module):
    def __init__(self):
        super(CNN_Detection, self).__init__()
        # Same Padding = [(filter size - 1) / 2] (Same Padding--> input size = output size)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # The output size of each of the 8 feature maps is
        # [(input_size - filter_size + 2(padding) / stride) +1] --> [(100-3+2(1)/1)+1] = 100 (padding type is same)
        # Batch normalization
        self.batchnorm1 = nn.BatchNorm2d(8)
        # RELU
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # After max pooling, the output of each feature map is now 100/2 = 50
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Output size of each of the 32 feature maps remains 50
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # After max pooling, the output of each feature map is 50/2 = 25
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2)
        # Output size of each of the 32 feature maps remains 50
        self.batchnorm3 = nn.BatchNorm2d(32)

        # The output size of each of the 32 feature maps is
        # [(input_size - filter_size + 2(padding) / stride) +1] --> [[(25-3+2(2)]/2)+1] = 14
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # 14 / 2 = 7

        # Flatten the feature maps. You have 32 feature maps, each of them is of size 7x7 --> 32*7*7 = 1568
        self.fc1 = nn.Linear(in_features=1568, out_features=600)
        self.droput = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(in_features=600, out_features=2)

    def forward(self, x):
        # print('forward')
        # print('x type = ', x.type)
        # print('x shape = ', x.shape)

        out = self.cnn1(x)  # 100 x 100
        out = self.batchnorm1(out)  # 100 x 100
        out = self.relu(out)  # 100 x 100
        out = -self.maxpool1(-out)  # 100 x 100 -> 50 x50

        out = self.cnn2(out)  # 50 x 50
        out = self.batchnorm2(out)  # 50 x 50
        out = self.relu(out)  # 50 x 50
        out = -self.maxpool2(-out)  # 50 x 50 -> 25 x 25
        # print('cnn3 x shape = ', out.shape)
        out = self.cnn3(out)  # 14 x 14
        out = self.batchnorm3(out)  # 14 x 14
        out = self.relu(out)  # 14 x 14
        # print('out cnn to full cnn x shape = ', out.shape)

        out = -self.maxpool2(-out)  # 14 x 14  -> 7 x 7
        # print('out pool to full cnn x shape = ', out.shape)
        # Now we have to flatten the output. This is where we apply the feed forward neural network as learned before!
        # It will take the shape (batch_size, 1568) = (100, 1568)
        out = out.view(-1, 1568)
        # Then we forward through our fully connected layer
        out = self.fc1(out)  # 1568 -> 600
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)  # 600 -> 2
        return out


if __name__ == '__main__':
    batch_size = 16
    digits_dataSet = DigitsLandmarksDataset(csv_file='data_xy.csv', root_dir='data/IAI')
    digits_dataSet_test = DigitsLandmarksDataset(csv_file='dataTest_xy.csv', root_dir='data/test/IAITest')
    print('digits_dattaset.__len__() ', digits_dataSet.__len__())

    dataloader = DataLoader(digits_dataSet, batch_size=batch_size, shuffle=True)
    dataloaderTest = DataLoader(digits_dataSet_test, batch_size=batch_size, shuffle=True)

    # fu.TrainNet(dataloader, batch_size, dataloaderTest)
    # exit()
    modelTest = CNN_Detection()
    modelTest.load_state_dict(torch.load('CNN_MNIST_detection_good.pth'))
    MSE = fu.testModelVec(digits_dataSet_test, modelTest)

    pos, image = fu.testModel(digits_dataSet_test, 77, 'CNN_MNIST_detection_good')
    print('search in pos =', pos)

    plt.figure(1)
    plt.imshow(image)
    plt.plot(pos[0], pos[1], 'r.')

    patch = image[int(pos[1] - 28 / 2):int(pos[1] + 28 / 2), int(pos[0] - 28 / 2):int(pos[0] + 28 / 2)]
    plt.figure(2)
    plt.imshow(patch)
    plt.show()

    exit()

