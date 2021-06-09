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


def testModelVec(digits_dattaset, modelTest):
    mseTotal = 0
    failCount = 0
    modelTest.eval()
    for index in range(len(digits_dattaset)):
        sample = digits_dattaset.__getitem__(index)
        inputs = torch.tensor(sample['image'])
        labels = torch.tensor(sample['landmarks'])
        inputs2_reshape = np.reshape(inputs, (1, 1, inputs.shape[1], inputs.shape[2]))
        outputs = modelTest(inputs2_reshape)

        outputs = outputs.float().detach().numpy()
        labels = labels.float().detach().numpy()
        currentMse = np.sum(((labels - outputs) * 100) ** 2) ** 0.5
        # print('mse = ', currentMse)
        mseTotal += currentMse
        if currentMse > 20:
            failCount += 1

    print('mse = ', mseTotal / len(digits_dattaset))
    print('Count fail = ', failCount)
    return mseTotal


def testModel(digits_dattaset, index, strPath):
    sample = digits_dattaset.__getitem__(index)
    inputs = torch.tensor(sample['image'])
    labels = torch.tensor(sample['landmarks'])
    inputs2 = np.reshape(inputs, (1, 1, inputs.shape[1], inputs.shape[2]))

    print("For one iteration, Test:")
    print("second Input Shape:", inputs.shape)
    print("second Labels Shape:", labels.shape)

    modelTest = iai.CNN_Detection()
    modelTest.load_state_dict(torch.load(strPath + '.pth'))
    modelTest.eval()
    outputs = modelTest(inputs2)

    outputs = outputs.float()
    labels = labels.float()
    print('outputs =', outputs * 100)
    print('labels  = ', labels * 100)
    labels_np = labels.numpy() * 100
    image = inputs.numpy()
    image = np.reshape(image, (image.shape[1], image.shape[2]))
    return labels_np, image


def TrainNet(dataloader, batch_size, dataloaderTest):
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

        for i, sample_eval in enumerate(dataloaderTest):

            print(i, len(dataloaderTest))
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
        test_accuracy.append((100 * correct // len(dataloaderTest)))

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


def createDataSetsAndFolders(statisticDataSet, landScape, savePath, debug_flag):
    # savePath = {'DATA': rootFolder_DATA, 'labels': rootFolder_labels, 'csv': rootFolder_csv}
    # landScape = {'image': largeImage, 'mean': mean_largeImage, 'std': std_largeImage}
    # statisticDataSet = {'dataSet': dataSet, 'mean': mean_gray, 'std': stddev_gray, 'csvLabels': xy_list}

    largeImage = landScape['image']
    mean_largeImage = landScape['mean']
    std_largeImage = landScape['std']

    dataSet = statisticDataSet['dataSet']
    xy_list = statisticDataSet['csvLabels']

    stddev_gray = statisticDataSet['std']
    mean_gray = statisticDataSet['mean']

    rootFolder_DATA = savePath['DATA']
    rootFolder_labels = savePath['labels']
    rootFolder = savePath['csv']

    print('create ', int(len(dataSet) / 1), 'images')
    # exit()
    for i in range(int(len(dataSet) / 1)):
        y = np.random.randint(0, largeImage.shape[0] - 28, 1)
        x = np.random.randint(0, largeImage.shape[1] - 28, 1)
        print('create i ', i, 'from ', len(dataSet))

        largeImageCopy = largeImage.copy()
        # print('largeImageCopy.shape = ', largeImageCopy.shape)
        # print('x = ', x[0], ' y = ', y[0])
        xy_list.append([str(i) + '.jpg', x[0] + 28 / 2, y[0] + 28 / 2])

        temp = largeImageCopy[y[0]:y[0] + 28, x[0]:x[0] + 28]

        random_image = dataSet[i][0].numpy() * stddev_gray + mean_gray
        random_image = random_image.reshape(28, 28)
        largeImageCopy[y[0]:y[0] + 28, x[0]:x[0] + 28] = random_image

        largeImageCopy = std_largeImage * (largeImageCopy + mean_largeImage)
        largeImageCopy = largeImageCopy / np.max(largeImageCopy) * 255
        largeImageCopy = np.floor(largeImageCopy)
        largeImageCopy = np.uint8(largeImageCopy)

        segImage = np.zeros(largeImageCopy.shape)
        segImage[largeImageCopy == 89] = 255
        segImage = np.uint8(segImage)

        # num_labels, labels_im = cv2.connectedComponents(segImage)
        # output = cv2.connectedComponentsWithStats(segImage)
        # (numLabels, labels, stats, centroids) = output
        # area = stats[:, cv2.CC_STAT_AREA]
        # area[0] = 0
        # iMax = np.argmax(area)
        # segImage = np.zeros(largeImageCopy.shape)
        # segImage[labels == iMax] = 255

        if debug_flag:
            plt.figure(1)
            plt.imshow(random_image)
            plt.figure(86)
            plt.imshow(segImage)
            plt.figure(186)
            plt.imshow(largeImageCopy)
            # plt.plot(centroids[iMax][0], centroids[iMax][1], 'r.')

        im = Image.fromarray(largeImageCopy)

        # exit()
        if im.mode == "F":
            im = im.convert('RGB')
        im.save(rootFolder_DATA + '/' + str(i) + '.jpg')

        # im = Image.fromarray(segImage)
        # if im.mode == "F":
        # im = im.convert('RGB')
        # im.save(rootFolder_labels + '/seg_' + str(i) + '.jpg')
        # plt.show()

    print(rootFolder + '_xy.csv')
    with open(rootFolder + '_xy.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        for xy in xy_list:
            write.writerow(xy)
