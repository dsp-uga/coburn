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
from coburn.data import loader, preprocess, postprocess

SIZE = 256  # images will be resized to SIZE x SIZE before being fed into the network
# create tranform for input data
transforms = []
resize_transform = preprocess.UniformResize(SIZE, SIZE)
transforms.append(resize_transform)

# convert the time series into an image with one channel
variance_transform = preprocess.Variance()
transforms.append(variance_transform)

# convert the input to a torch Tensor
transforms.append(preprocess.ToArray())
transforms.append(torchvision.transforms.ToTensor())

# compose the transforms
data_transform = torchvision.transforms.Compose(transforms)


def train(input, epochs=200, learning_rate=1e-4):
    dataset = loader.load('train', base_dir=input)
    dataset.set_transform(data_transform)

    # create a transform that preprocesses the target masks
    mask_transform = torchvision.transforms.Compose([preprocess.ResizeMask(SIZE, SIZE)])
    dataset.set_mask_transform(mask_transform)

    # Create a Tiramisu network from the dense package
    # in_channels is the number of channels in the input images
    # out_channels is the number of classes
    net = FCDenseNet103(in_channels=1, out_channels=3).cuda()

    # train the network
    LR = learning_rate
    N_EPOCHS = epochs  # maximum number of epochs to train
    BATCH_SIZE = 3
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

        print("Epoch %d Finished" % epoch)
        # save the trained network
        torch.save(net, 'networks/tiramisu.torchnet')
    print("Done training!")


def test(input, output):
    # loading the testing data
    dataset = loader.load('test', base_dir=input)
    dataset.set_transform(data_transform)

    # load the network
    net = torch.load('networks/tiramisu.torchnet')

    for idx in range(0, len(dataset)):
        image, mask = dataset[idx]  # mask will be None
        image = Variable(image.unsqueeze(0).cuda())
        net_out = net(image).data.cpu()
        output_arr = net_out.numpy()[0]

        # the output is a 3 x SIZE x SIZE array that gives confidence values for each class in the three channels
        # we need to change these into a segmentation map
        segmap = np.empty((SIZE, SIZE))
        for row in range(0, SIZE):
            for col in range(0, SIZE):
                vals = output_arr[:, row, col]
                index = np.argmax(vals)
                segmap[row, col] = index

        original_size = dataset.get_original_size(idx)
        orig_size_transform = preprocess.ResizeMask(original_size[0], original_size[1])
        segmap = orig_size_transform.__call__(segmap)

        hash = dataset.get_hash(idx)
        postprocess.export_as_png(segmap, output, hash)

    tar_path = postprocess.make_tar(output)

    print("Done!")
    print("Results written to %s" % tar_path)


def main(input='./data', output='./results/tiramisu', mode='both', epochs=200, learning_rate=1e-4):
    """
    The `dense` package from baldassarreFe provides us with a canned implementation of Tiramisu.
    `in_channels` specify the number of channels in the input image.  `out_channels` specifies the number of classes
    """
    if mode == 'train' or mode == 'both':
        train(input, epochs=epochs, learning_rate=learning_rate)
    if mode == 'test' or mode == 'both':
        test(input, output)
