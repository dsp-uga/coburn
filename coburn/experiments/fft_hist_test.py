"""
This experiment shows an example of how the loader and preprocess modules can be used to load a dataset and
do some simple preprocessing (NEEDS cuda)
"""
from coburn.data import preprocess, loader, fft_features
from torchvision.transforms import Compose
import torchvision.transforms as tvt
import torch
import numpy as np


def main():
    # these samples will be automatically downloaded if they are not found locally
    dataset = loader.load(samples=['4bad52d5ef5f68e87523ba40aa870494a63c318da7ec7609e486e62f7f7a25e8',
                                   'a7e37600a431fa6d6023514df87cfc8bb5ec028fb6346a10c2ececc563cc5423',
                                   '70a6300a00dbac92be9238252ee2a75c86faf4729f3ef267688ab859eed1cc60'])
    resize_transform = preprocess.Resize(dataset, 640, 480)
    cuda_transform = fft_features.MakeCUDA()
    fft_transform = tvt.Lambda(lambda x: torch.stft(x, fft_size=128, axis=0))
    submean_tranform = tvt.Lambda(lambda x: x.sub(torch.mean(x, dim=0)))

    transforms = Compose([resize_transform, cuda_transform, fft_transform, submean_tranform])
    dataset.set_transform(transforms)
    for i in range(len(dataset)):
        print(dataset[i].type())
        print(dataset[i].shape)
