"""
This experiment is to load a dataset, generate frequency energy
"""
from coburn.data import preprocess, loader, fft_features
from torchvision.transforms import Compose
import torchvision.transforms as tvt
import torch
import numpy as np


def main():
    # these samples will be automatically downloaded if they are not found locally
    dataset = loader.load(samples='test')
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
                            svd_transform,
                            reshape])
    dataset.set_transform(transforms)
