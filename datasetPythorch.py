import pandas as pd
# from __future__ import print_function, division
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
        self.droput = nn.Dropout(p=0.5)
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


def testModel(digits_dattaset, index):

    sample = digits_dattaset.__getitem__(index)
    inputs = torch.tensor(sample['image'])
    labels = torch.tensor(sample['landmarks'])
    inputs2 = np.reshape(inputs, (1, 1, inputs.shape[1], inputs.shape[2]))

    print('inputs2 shape = ', inputs2.shape)

    image = inputs
    image = image.numpy()
    image = np.reshape(image, (image.shape[1], image.shape[2]))
    print('image shape = ', image.shape)

    print("For one iteration, Test:")
    print("second Input Shape:", inputs.shape)
    print("second Labels Shape:", labels.shape)

    modelTest = CNN_Detection()
    modelTest.load_state_dict(torch.load('CNN_MNIST_detection.pth'))
    modelTest.eval()
    outputs = modelTest(inputs2)

    outputs = outputs.float()
    labels = labels.float()
    print('outputs =', outputs*100)
    print('labels  = ', labels*100)


if __name__ == '__main__':
    batch_size = 16

    digits_dattaset = DigitsLandmarksDataset(csv_file='data_xy.csv', root_dir='data/IAI')
    print('digits_dattaset.__len__() ', digits_dattaset.__len__())

    dataloader = DataLoader(digits_dattaset, batch_size=batch_size, shuffle=True)
    testModel(digits_dattaset, 50)

    exit()
    print('dataloader = ', type(dataloader))
    print('dataloader = ', len(dataloader))

    model = CNN_Detection()
    CUDA = torch.cuda.is_available()
    print('CUDA = ', CUDA)
    if CUDA:
        model = model.cuda()

    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    iteration = 0
    correct_nodata = 0
    correct_data = 0

    for i, sample in enumerate(dataloader):

        if iteration == 1:
            print('break')
            break

        print('ind image = ', i, 'sample = ', type(sample))
        inputs = torch.tensor(sample['image'])
        labels = torch.tensor(sample['landmarks'])

        inputs = Variable(inputs)
        labels = Variable(labels)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        print("For one iteration, this is what happens:")
        print("second Input Shape:", inputs.shape)
        print("second Labels Shape:", labels.shape)

        output = model(inputs)
        print("Outputs Shape", output.shape)

        correct_nodata += (output == labels).sum()
        print("output Predictions: ", output)
        print("\n\nlabels Predictions: ", labels)
        print("\n\nCorrect Predictions: ", correct_nodata)

        iteration += 1

    # exit()

    # Training the CNN
    num_epochs = 5

    # Define the lists to store the results of loss and accuracy
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # Training
    for epoch in range(num_epochs):
        # Reset these below variables to 0 at the begining of every epoch
        correct = 0
        iterations = 0
        iter_loss = 0.0

        model.train()  # Put the network into training mode

        for i, sample in enumerate(dataloader):

            print('epoch = ', epoch, ' i = ', i, 'from = ', len(dataloader))
            inputs = torch.tensor(sample['image']).clone().detach()
            labels = torch.tensor(sample['landmarks']).clone().detach()
            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
            outputs = model(inputs)
            outputs = outputs.float()
            labels = labels.float()
            # print('model label')
            # print(type(outputs))
            # print(type(labels))

            loss = loss_fn(outputs, labels)
            iter_loss += loss.item()  # Accumulate the loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            # Record the correct predictions for training data
            correct += ((outputs - labels) ** 2).sum().item()
            # print('correct = ', correct)
            iterations += 1

        # Record the training loss
        train_loss.append(iter_loss / iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct // len(dataloader)))

        # Testing
        loss = 0.0
        correct = 0
        iterations = 0

        model.eval()  # Put the network into evaluation mode

        for i, sample_eval in enumerate(dataloader):

            print(i, len(dataloader))
            inputs = torch.tensor(sample_eval['image']).clone().detach()
            labels = torch.tensor(sample_eval['landmarks']).clone().detach()

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            outputs = outputs.float()
            labels = labels.float()

            loss = loss_fn(outputs, labels)  # Calculate the loss
            loss += loss.item()  # Accumulate the loss

            iterations += 1

        # Record the Testing loss
        test_loss.append(loss / iterations)
        # Record the Testing accuracy
        test_accuracy.append((100 * correct // len(dataloader)))

        print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
              .format(epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1],
                      test_loss[-1], test_accuracy[-1]))

    # Run this if you want to save the model
    torch.save(model.state_dict(), 'CNN_MNIST_detection.pth')

    # Loss
    f = plt.figure(figsize=(10, 10))
    plt.plot(train_loss * 100, label='Training Loss')
    plt.plot(test_loss, label='Testing Loss')
    plt.legend()
    plt.show()

    # Accuracy
    f = plt.figure(figsize=(10, 10))
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(test_accuracy, label='Testing Accuracy')
    plt.legend()
    plt.show()
    # transforms_ori = transforms.Compose([transforms.ToTensor(),
    # transforms.Normalize((0,), (1,))])
    # transformed_dataset = DigitsLandmarksDataset(csv_file='data_xy.csv',
    #                                            root_dir='data/IAI',transform=transforms_ori)
