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

debug_flag = False
rootFolder_csv = '/home/almog/PycharmProjects/rcnnSimpleDigites/data'
rootFolder_DATA = '/home/almog/PycharmProjects/rcnnSimpleDigites/data/IAI'
rootFolder_labels = '/home/almog/PycharmProjects/rcnnSimpleDigites/data/labels'


rootFolder_csv_test = '/home/almog/PycharmProjects/rcnnSimpleDigites/dataTest'
rootFolder_DATA_test = '/home/almog/PycharmProjects/rcnnSimpleDigites/data/test/IAITest'
rootFolder_labels_test = '/home/almog/PycharmProjects/rcnnSimpleDigites/data/test/labelsTest'

size_mnist = 28

# onlyfiles = [f for f in listdir(rootFolder_csv) if isfile(join(rootFolder_csv, f))]
# for name in onlyfiles:
#     print(name)

largeImage = Image.open("/home/almog/PycharmProjects/rcnnSimpleDigites/data/landScape.jpeg")
largeImage = ImageOps.grayscale(largeImage)
(width, height) = (largeImage.width, largeImage.height)
sizeImage = np.min([width, height])

largeImage = largeImage.crop((0, 0, sizeImage, sizeImage))
(width, height) = (largeImage.width // 8, largeImage.height // 8)
print(width, height)
largeImage = largeImage.resize((width, height))

largeImage = np.asarray(largeImage)
std_largeImage = np.max(largeImage)
largeImage = largeImage / np.max(largeImage)
mean_largeImage = np.mean(largeImage)
largeImage = largeImage - np.mean(largeImage)

print('largeImage shape = ', largeImage.shape)
plt.figure(99)
plt.imshow(largeImage)

# Specify the Mean and standard deviation of all the pixels in the MNIST dataset. They are precomputed
mean_gray = 0.1307
stddev_gray = 0.3081
batch_size = 100

# Transform the images to tensors
# Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn)
# for n channels, this transform will normalize each channel of the input torch.Tensor
# i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

transforms_ori = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((mean_gray,), (stddev_gray,))])


# Load our data set
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms_ori,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms_ori)

random_image = train_dataset[20][0].numpy() * stddev_gray + mean_gray

print("\n\nsize train_dataset ", train_dataset.__sizeof__())
print("type train_dataset len =", len(train_dataset))
xy_listPatchOnNumber = []
xy_list = []

landScape = {'image': largeImage, 'mean': mean_largeImage, 'std': std_largeImage}

savePath = {'DATA': rootFolder_DATA, 'labels': rootFolder_labels, 'csv': rootFolder_csv}
statisticDataSet = {'dataSet': train_dataset, 'mean': mean_gray, 'std': stddev_gray, 'csvLabels': xy_list,
                    'csvLabelsPatchOnNum': xy_listPatchOnNumber}
fu.createDataSetsAndFolders(statisticDataSet, landScape, savePath, debug_flag)

xy_listPatchOnNumber = []
xy_list = []
#
savePath_test = {'DATA': rootFolder_DATA_test, 'labels': rootFolder_labels_test, 'csv': rootFolder_csv_test}
statisticDataSet = {'dataSet': test_dataset, 'mean': mean_gray, 'std': stddev_gray, 'csvLabels': xy_list,
                    'csvLabelsPatchOnNum': xy_listPatchOnNumber}
fu.createDataSetsAndFolders(statisticDataSet, landScape, savePath_test, debug_flag)


exit()
