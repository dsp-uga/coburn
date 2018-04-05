"""
Tiramisu is a deep fully-convolutional network for image segmentation: https://arxiv.org/abs/1611.09326
It uses DenseNets (https://arxiv.org/abs/1608.06993) for encoding/decoding but has a ladder structure similar to
U-Net.


The Github user baldassarreFe (https://github.com/baldassarreFe) has an open-source implementation of Tiramisu that is
pip installable!  We use it in this experiment to create the network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms
from torch.autograd import Variable
from torch.utils.data import dataloader
from dense import FCDenseNet103
from coburn.data import loader, preprocess


def train(input):
    dataset = loader.load('train', base_dir=input)

    SIZE = 128  # images will be resized to SIZE x SIZE before being fed into the network

    transforms = []
    # resize the images to be 512 x 512 each
    resize_transform = preprocess.UniformResize(SIZE, SIZE)
    transforms.append(resize_transform)

    # convert the time series into an image with one channel
    variance_transform = preprocess.Variance()
    transforms.append(variance_transform)

    # convert the input to a torch Tensor
    transforms.append(preprocess.ToArray())
    transforms.append(torchvision.transforms.ToTensor())

    # compose the transforms
    transform = torchvision.transforms.Compose(transforms)
    dataset.set_transform(transform)

    # create a transform that preprocesses the target masks
    # mask_transform = torchvision.transforms.Compose([preprocess.MaskToSegMap(), preprocess.ResizeMask(SIZE, SIZE), torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose([preprocess.ResizeMask(SIZE, SIZE)])
    dataset.set_mask_transform(mask_transform)

    # Create a Tiramisu network from the dense package
    # in_channels is the number of channels in the input images
    # out_channels is the number of classes
    net = FCDenseNet103(in_channels=1, out_channels=3).cuda()

    # train the network
    LR = 1e-4  # learning rate
    N_EPOCHS = 2  # number of epochs to train
    BATCH_SIZE = 2
    torch.cuda.manual_seed(0)
    optimizer = optim.SGD(net.parameters(), lr=LR)
    criterion = nn.NLLLoss2d().cuda()

    dataset_loader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, N_EPOCHS + 1):
        for idx, data in enumerate(dataset_loader):
            X = Variable(data[0].cuda())
            target = Variable(data[1].cuda()).long()
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(X)  # generate prediction
            loss = criterion(output, target)  # compute loss
            loss.backward()  # backpropagate
            optimizer.step()  # update weights


def test():
    pass


def main(input='./data', output='./results/tiramisu', mode='both'):
    """
    The `dense` package from baldassarreFe provides us with a canned implementation of Tiramisu.
    `in_channels` specify the number of channels in the input image.  `out_channels` specifies the number of classes
    """
    # net = FCDenseNet103(in_channels=1, out_channels=3)
    # print("Tiramisu!")
    train(input)
