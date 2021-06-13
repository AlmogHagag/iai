import pandas as pd
# from __future__ import print_function, division
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
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
import cnnDigitsClass as digCl


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


def testModel(digits_dattaset, index, strPath, strPathDigits):
    sample = digits_dattaset.__getitem__(index)
    inputs = torch.tensor(sample['image'])
    labels = torch.tensor(sample['landmarks'])
    inputs2 = np.reshape(inputs, (1, 1, inputs.shape[1], inputs.shape[2]))

    modelTest = iai.CNN_Detection()
    modelTest.load_state_dict(torch.load(strPath + '.pth'))
    modelTest.eval()
    # print('full image tensor = ', inputs2.shape)
    outputs = modelTest(inputs2)

    esLocNumpy = outputs.detach().numpy()
    esLocNumpy = esLocNumpy * 100
    # print('esLocNumpy = ', esLocNumpy)
    patchTorch = torch.zeros(inputs2.shape[0], inputs2.shape[1], 28, 28)
    loc = esLocNumpy
    loc[loc > 86] = 86
    loc[loc < 14] = 14

    for iPatch in range(esLocNumpy.shape[0]):
        patchTorch[iPatch, 0, :, :] = inputs2[iPatch, :,
                                      int(loc[iPatch, 1] - 28 / 2):int(loc[iPatch, 1] + 28 / 2),
                                      int(loc[iPatch, 0] - 28 / 2):int(loc[iPatch, 0] + 28 / 2)]
    modelClessifer = digCl.CNN()
    modelClessifer.load_state_dict(torch.load(strPathDigits + '.pth'))
    modelClessifer.eval()
    # print('patchTorch = ', type(patchTorch))
    # print('shape = ', patchTorch.shape)

    outputDigits = modelClessifer(patchTorch)
    _, predicted_nodata = torch.max(outputDigits, 1)
    predicted_nodata = predicted_nodata.numpy()[0]
    # print('\n\npredicted_nodata = ', predicted_nodata)
    outputDigits = outputDigits.float()
    labels = labels.float()
    # print('\n\noutputDigits =', outputDigits * 100)

    # print('labels  = ', labels * 100)
    labels_np = labels.numpy() * 100
    image = inputs.numpy()
    image = np.reshape(image, (image.shape[1], image.shape[2]))

    return labels_np, image, (predicted_nodata != labels_np[2]).sum()


def TrainNetclesfier(dataloader, batch_size, dataloaderTest, modelDetection):
    model = digCl.CNN()
    CUDA = torch.cuda.is_available()
    print('CUDA = ', CUDA)
    if CUDA:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    iteration = 0
    correct_nodata = 0
    correct_data = 0
    modelDetection.eval()

    for i, sample in enumerate(dataloader):

        if iteration == 1:
            print('break')
            break

        print('ind image = ', i, 'sample = ', type(sample))
        inputs = torch.tensor(sample['image']).clone().detach()
        labels = torch.tensor(sample['landmarks']).clone().detach()
        esLoc = modelDetection(inputs)
        esLocNumpy = esLoc.detach().numpy()

        inputs = Variable(inputs)
        labels = Variable(labels)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # print('esLoc = ', esLoc)
        print(type(input))
        image = inputs.numpy()
        print(image.shape)
        image = image[0, 0, :, :]
        image = np.reshape(image, (100, 100))
        print(image.shape)

        print(type(image))
        print(esLocNumpy.shape)
        esLocNumpy = esLocNumpy * 100
        patch = image[int(esLocNumpy[0, 1] - 28 / 2):int(esLocNumpy[0, 1] + 28 / 2),
                int(esLocNumpy[0, 0] - 28 / 2):int(esLocNumpy[0, 0] + 28 / 2)]

        patchTorch = torch.zeros(inputs.shape[0], inputs.shape[1], 28, 28)
        for iPatch in range(esLocNumpy.shape[0]):
            print(iPatch, ' esLocNumpy = ', esLocNumpy[iPatch, :])
            patchTorch[iPatch, 0, :, :] = inputs[iPatch, :,
                                     int(esLocNumpy[iPatch, 1] - 28 / 2):int(esLocNumpy[iPatch, 1] + 28 / 2),
                                     int(esLocNumpy[iPatch, 0] - 28 / 2):int(esLocNumpy[iPatch, 0] + 28 / 2)]

        # plt.figure(9)
        # for iPatch in range(esLocNumpy.shape[0]):
        #     plt.imshow(np.reshape(patchTorch[iPatch, 0, :, :].numpy(), (28, 28)))
        #     plt.show()

        output = model(patchTorch)
        _, predicted_nodata = torch.max(output, 1)

        print("Outputs Shape", output.shape)
        print('labels = ', labels[:, 2] * 100)
        correct_nodata += (predicted_nodata == labels[:, 2]).sum()
        print("output Predictions: ", output)
        print("\n\nlabels Predictions: ", labels)
        print("\n\nCorrect Predictions error: ", correct_nodata)

        iteration += 1

    # exit()
    # Training the CNN
    num_epochs = 7

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
            print('ind image = ', i, 'from = ', len(dataloader))
            inputs = torch.tensor(sample['image']).clone().detach().requires_grad_(True)
            labels = torch.tensor(sample['landmarks']).clone().detach().requires_grad_(True)
            esLoc = modelDetection(inputs)
            esLocNumpy = esLoc.detach().numpy()

            inputs = Variable(inputs)
            labels = Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # print(type(image))
            # print(esLocNumpy.shape)
            esLocNumpy = (esLocNumpy * 100).astype(int)
            patchTorch = torch.zeros(inputs.shape[0], inputs.shape[1], 28, 28)
            for iPatch in range(esLocNumpy.shape[0]):
                # print(esLocNumpy[iPatch, :])
                loc = (esLocNumpy[iPatch, :])
                loc[loc > 86] = 86
                loc[loc < 14] = 14

                if (loc-esLocNumpy[iPatch, :]).sum()>0:
                    print(esLocNumpy[iPatch, :])
                    print('Loc= ', loc)
                    # exit()

                patchTorch[iPatch, 0, :, :] = inputs[iPatch, :, int(esLocNumpy[iPatch, 1] - 28 / 2):int(esLocNumpy[iPatch, 1] + 28 / 2),
                                         int(esLocNumpy[iPatch, 0] - 28 / 2):int(esLocNumpy[iPatch, 0] + 28 / 2)]
            labels = labels[:, 2] * 100
            labels = labels.long()

            # plt.figure(9)
            # for iPatch in range(esLocNumpy.shape[0]):
            #     plt.imshow(np.reshape(patchTorch[iPatch, 0, :, :].numpy(), (28, 28)))
            #     print('label = ', labels[iPatch])
            #     plt.show()

            optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
            outputs = model(patchTorch)

            loss = loss_fn(outputs, labels)
            iter_loss += loss.item()  # Accumulate the loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
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

        for i, sample in enumerate(dataloaderTest):

            print('testing ind image = ', i, 'from = ', len(dataloader))
            inputs = torch.tensor(sample['image']).clone().detach().requires_grad_(True)
            labels = torch.tensor(sample['landmarks']).clone().detach().requires_grad_(True)
            esLoc = modelDetection(inputs)
            esLocNumpy = esLoc.detach().numpy()

            # Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # image = inputs.numpy()
            # image = image[0, 0, :, :]
            # image = np.reshape(image, (image.shape[2], image.shape[3]))

            esLocNumpy = (esLocNumpy * 100).astype(int)
            patchTorch = torch.zeros(inputs.shape[0], inputs.shape[1], 28, 28)
            for iPatch in range(esLocNumpy.shape[0]):
                # print(esLocNumpy[iPatch, :])
                loc = (esLocNumpy[iPatch, :])
                loc[loc > 86] = 86
                loc[loc < 14] = 14

                if (loc - esLocNumpy[iPatch, :]).sum() > 0:
                    print(esLocNumpy[iPatch, :])
                    print('Loc= ', loc)
                    # exit()

                patchTorch[iPatch, 0, :, :] = inputs[iPatch, :,
                                              int(esLocNumpy[iPatch, 1] - 28 / 2):int(esLocNumpy[iPatch, 1] + 28 / 2),
                                              int(esLocNumpy[iPatch, 0] - 28 / 2):int(esLocNumpy[iPatch, 0] + 28 / 2)]
            labels = labels[:, 2] * 100
            labels = labels.long()

            optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
            outputs = model(patchTorch)

            loss = loss_fn(outputs, labels)  # Calculate the loss
            loss += loss.item()  # Accumulate the loss
            print('loss = ', loss)
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            print('test percent correct = ', correct/iterations)
            iterations += 1

        # Record the Testing loss
        test_loss.append(loss / iterations)
        # Record the Testing accuracy
        test_accuracy.append((100 * correct // len(dataloaderTest)))

        print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
              .format(epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))

    # Run this if you want to save the model
    torch.save(model.state_dict(), 'CNN_MNIST_from_patch2.pth')

    # Loss
    f = plt.figure(num=1, figsize=(10, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Testing Loss')
    plt.legend()
    plt.show()

    # Accuracy
    f1 = plt.figure(num=2, figsize=(50, 50))
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(test_accuracy, label='Testing Accuracy')
    plt.legend()
    plt.show()


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
        print('labels = ', labels[:, 0:2])
        correct_nodata += (output == labels[:, 0:2]).sum()
        print("output Predictions: ", output)
        print("\n\nlabels Predictions: ", labels)
        print("\n\nCorrect Predictions error: ", correct_nodata)

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
            labels = labels[:, 0:2]
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
            labels = labels[:, 0:2]

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


def createDataSetsAndFolders(statisticDataSet, landScape, savePath, debug_flag, ADD_NOISE=False):
    # savePath = {'DATA': rootFolder_DATA, 'labels': rootFolder_labels, 'csv': rootFolder_csv}
    # landScape = {'image': largeImage, 'mean': mean_largeImage, 'std': std_largeImage}
    # statisticDataSet = {'dataSet': dataSet, 'mean': mean_gray, 'std': stddev_gray, 'csvLabels': xy_list}

    largeImage = landScape['image']
    mean_largeImage = landScape['mean']
    std_largeImage = landScape['std']

    dataSet = statisticDataSet['dataSet']
    xy_list = statisticDataSet['csvLabels']
    xy_list_patchNum = statisticDataSet['csvLabelsPatchOnNum']

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

        temp = largeImageCopy[y[0]:y[0] + 28, x[0]:x[0] + 28]

        random_image = dataSet[i][0].numpy() * stddev_gray + mean_gray
        if ADD_NOISE:
            im_pil = Image.fromarray(random_image)
            rotated = im_pil.rotate(np.random.randint(0, 360))
            random_image = np.asarray(rotated)
            mean = 0
            row, col, ch = random_image.shape
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            random_image = random_image + gauss
            random_image = cv2.flip(random_image, 0)
            random_image = cv2.flip(random_image, 1)
            random_image = cv2.resize(random_image, (20, 20))


        numberOnPatch = dataSet[i][1].numpy()
        xy_list.append([str(i) + '.jpg', x[0] + 28 / 2, y[0] + 28 / 2, numberOnPatch])

        xy_list_patchNum.append([numberOnPatch])
        # print('numberOnPatch = ', numberOnPatch)
        # plt.figure(1)
        # plt.imshow(np.reshape(random_image, (random_image.shape[1], random_image.shape[2])))
        # plt.title('random_image from set')
        # plt.show()

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

    print(rootFolder + 'patchNum_xy.csv')
    with open(rootFolder + 'patchNum_xy.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        for xy in xy_list_patchNum:
            write.writerow(xy)
