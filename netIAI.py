# import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


rootFolder = '/home/almog/PycharmProjects/rcnnSimpleDigites/data'
onlyfiles = [f for f in listdir(rootFolder) if isfile(join(rootFolder, f))]
for name in onlyfiles:
    print(name)
largeImage = Image.open("/home/almog/PycharmProjects/rcnnSimpleDigites/data/zidane.jpg")
# largeImage = cv2.imread('/home/almog/PycharmProjects/rcnnSimpleDigites/data/zidane.jpg')
largeImage = np.asarray(largeImage)

print('largeImage shape = ', largeImage.shape)
# cv2.imshow('largeImage', largeImage)
# cv2.waitKey(0)
plt.imshow(largeImage)
# plt.show()

# plt.show()


def pasteInImage(largeImage, MNISTimage):

     y = np.random.randint(MNISTimage.shape[0], largeImage.shape[0], 1)
     x = np.random.randint(MNISTimage.shape[1], largeImage.shape[1], 1)


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
print("dir(train_dataset)) ", dir(train_dataset))

print("\n\nsize train_dataset ", train_dataset.__sizeof__())
# print("\n\nsize train_dataset ", train_dataset.size)
print("\n\nsize train_dataset.data.shape ", train_dataset.data.shape)

random_image = train_dataset[20][0].numpy() * stddev_gray + mean_gray
plt.imshow(random_image.reshape(28, 28), cmap='gray')

label = train_dataset[20][1]
print('lable for the image', label)
print('type(largeImage) = ', type(largeImage))

print('y= ', largeImage.shape)

y = np.random.randint(random_image.shape[0], largeImage.shape[0], 10)
print('train_dataset type= ', type(train_dataset))
exit()
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
y = np.random.randint(MNISTimage.shape[0], largeImage.shape[0], 10)
print(y)
plt.show()


