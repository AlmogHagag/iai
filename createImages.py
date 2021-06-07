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

debug_flag = False
rootFolder = '/home/almog/PycharmProjects/rcnnSimpleDigites/data'
rootFolder_DATA = '/home/almog/PycharmProjects/rcnnSimpleDigites/data/IAI'
rootFolder_labels = '/home/almog/PycharmProjects/rcnnSimpleDigites/data/labels'
size_mnist = 28

onlyfiles = [f for f in listdir(rootFolder) if isfile(join(rootFolder, f))]
for name in onlyfiles:
    print(name)

largeImage = Image.open("/home/almog/PycharmProjects/rcnnSimpleDigites/data/landScape.jpeg")
largeImage = ImageOps.grayscale(largeImage)
(width, height) = (largeImage.width, largeImage.height )
sizeImage = np.min([width, height])
print('sizeImage = ', sizeImage)
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
# plt.show()

# Specify the Mean and standard deviation of all the pixels in the MNIST dataset. They are precomputed
mean_gray = 0.1307
stddev_gray = 0.3081
batch_size = 100
epochs = 10

# Transform the images to tensors
# Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn)
# for n channels, this transform will normalize each channel of the input torch.Tensor
# i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

transforms_ori = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((mean_gray,), (stddev_gray,))])

transforms_photo = transforms.Compose([transforms.Resize((28, 28)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((mean_gray,), (stddev_gray,))])

# Load our data set
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms_ori,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms_ori)

print("type train_dataset ", type(train_dataset))
# print("dir(train_dataset)) ", dir(train_dataset))

print("\n\nsize train_dataset ", train_dataset.__sizeof__())
# print("\n\nsize train_dataset ", train_dataset.size)

random_image = train_dataset[20][0].numpy() * stddev_gray + mean_gray
# plt.imshow(random_image.reshape(28, 28), cmap='gray')

label = train_dataset[20][1]
print('lable for the image', label)
print('type(largeImage) = ', type(largeImage))

print('y= ', largeImage.shape)

y = np.random.randint(random_image.shape[0], largeImage.shape[0], 10)
print('train_dataset type= ', type(train_dataset))
# Make the data set iterable
train_load = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

print('size of train images ', len(train_load))
test_load = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)

train_dataset_IAI = []

MNISTimage = random_image

print("type train_dataset len =", len(train_dataset))
xy_list = []

for i in range(int(len(train_dataset)/4)):
    y = np.random.randint(0, largeImage.shape[0] - 28, 1)
    x = np.random.randint(0, largeImage.shape[1] - 28, 1)

    largeImageCopy = largeImage.copy()
    print('largeImageCopy.shape = ', largeImageCopy.shape)
    print('x = ', x[0], ' y = ', y[0])
    xy_list.append([str(i) + '.jpg', x[0]+28/2, y[0]+28/2])

    temp = largeImageCopy[y[0]:y[0] + 28, x[0]:x[0] + 28]

    random_image = train_dataset[i][0].numpy() * stddev_gray + mean_gray
    random_image = random_image.reshape(28, 28)
    print('random_image.shape =', random_image.shape)
    largeImageCopy[y[0]:y[0] + 28, x[0]:x[0] + 28] = random_image
    plt.figure(1)
    plt.imshow(random_image)
    plt.figure(11)
    largeImageCopy = std_largeImage * (largeImageCopy + mean_largeImage)
    largeImageCopy = largeImageCopy/np.max(largeImageCopy)*255
    largeImageCopy = np.floor(largeImageCopy)
    largeImageCopy = np.uint8(largeImageCopy)

    segImage = np.zeros(largeImageCopy.shape)

    segImage[largeImageCopy == 89] = 255
    segImage = np.uint8(segImage)

    num_labels, labels_im = cv2.connectedComponents(segImage)
    output = cv2.connectedComponentsWithStats(segImage)
    (numLabels, labels, stats, centroids) = output
    area = stats[:, cv2.CC_STAT_AREA]
    area[0] = 0
    iMax = np.argmax(area)
    segImage = np.zeros(largeImageCopy.shape)
    segImage[labels == iMax] = 255

    if debug_flag:
        plt.figure(86)
        plt.imshow(segImage)
        plt.figure(186)
        plt.imshow(largeImageCopy)
        plt.plot(centroids[iMax][0], centroids[iMax][1], 'r.')

    im = Image.fromarray(largeImageCopy)
    print('\n\n\n largeImageCopy shape = ', largeImageCopy.shape)
    # exit()
    if im.mode == "F":
        im = im.convert('RGB')
    im.save(rootFolder_DATA + '/' + str(i) + '.jpg')

    im = Image.fromarray(segImage)
    if im.mode == "F":
        im = im.convert('RGB')
    im.save(rootFolder_labels + '/seg_' + str(i) + '.jpg')
    # plt.show()

with open(rootFolder + '_xy.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    for xy in xy_list:
        write.writerow(xy)
