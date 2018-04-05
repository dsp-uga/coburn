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
    fft_transform = fft_features.Frequency(n=128)
    cuda_transform = tvt.Lambda(lambda x: torch.from_numpy(x).float().cuda())
    submean_tranform = tvt.Lambda(lambda x: x.sub(torch.mean(x, dim=0)))
    flat_transform = tvt.Lambda(lambda x: x.view(307200, -1))
    svd_transform = tvt.Lambda(lambda x: fft_features.PCA(x, k=10))
    reshape = tvt.Lambda(lambda x: x.view(640, 480, -1))

    transforms = Compose([resize_transform,
                            fft_transform,
                            cuda_transform,
                            submean_tranform,
                            flat_transform,
                            reshape])

    dataset.set_transform(transforms)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample.shape)
