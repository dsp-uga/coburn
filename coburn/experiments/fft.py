"""
This experiment is to load a dataset, generate frequency energy
"""
from coburn.data import preprocess, loader, fft_features, postprocess
from torchvision.transforms import Compose
import torchvision.transforms as tvt
import torch
import numpy as np


def main(input='./data', output='./results/fft_dom', k =10, dom_frequency=11):
    # these samples will be automatically downloaded if they are not found locally
    dataset = loader.load(samples='test')
    resize_transform = preprocess.Resize(dataset, 640, 480)
    fft_transform = fft_features.Frequency(n=128)
    cuda_transform = tvt.Lambda(lambda x: torch.from_numpy(x).float().cuda())
    submean_tranform = tvt.Lambda(lambda x: x.sub(torch.mean(x, dim=0)))
    flat_transform = tvt.Lambda(lambda x: x.view(307200, -1))
    svd_transform = tvt.Lambda(lambda x: fft_features.PCA(x, k=10))
    reshape = tvt.Lambda(lambda x: x.view(640, 480, -1))
    max_freq = tvt.Lambda(lambda x: x.argmax(dim=0))

    transforms = Compose([fft_transform,
                            cuda_transform,
                            submean_tranform])
    dataset.set_transform(transforms)

    for i in range(0, len(dataset)):
        img, target = dataset[i]
        hash = dataset.get_hash(i)

        # create cilia mask based on grayscale variance thresholding
        mask = np.zeros(img.shape)

        frequency_range = img in [10, 11, 12]
        mask[frequency_range] = 2

        postprocess.export_as_png(mask, output, hash)

    tar_path = postprocess.make_tar(output)
    print("Done!")
    print("Results written to %s" % tar_path)
