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
import datasetPythorch as iai





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

    modelTest = iai.CNN_Detection()
    modelTest.load_state_dict(torch.load('CNN_MNIST_detection.pth'))
    modelTest.eval()
    outputs = modelTest(inputs2)

    outputs = outputs.float()
    labels = labels.float()
    print('outputs =', outputs * 100)
    print('labels  = ', labels * 100)


def TrainNet(dataloader):
    model = iai.CNN_Detection()
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
