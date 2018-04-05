import numpy as np
from .Transform import Transform
import torch
import torchvision.transforms as tvt


class MakeCUDA(Transform):
    """
    Gets image array and turns into Pytorch CUDA Float Tensor:
    """
    def __call__(self, images):
        return torch.from_numpy(images.toarray()).float().cuda()

class Frequency(Transform):
    """
    Turns numpy image series into fourier feature series
    """
    def __init__(self, n=128):
        self.n = n
    def __call__(self, images):
        np_images = images.toarray()
        sp = np.fft.fft(np_images, n=self.n, axis=0)
        return sp.real

def PCA(data, k=2):
    """
    Dimensionality reduction for fourier features in flattened dataself.

    Args:
        data: 2D input
        k: number of principle components to extract
    """
    # adapted from web sources
    # svd
    torch.cuda.empty_cache()
    s = torch.cuda.stream()
    with torch.cuda.stream(s):
        X = torch.t(data)
        U,S,V = torch.svd(X)
        out = torch.mm(X,U[:,:k])
    return out
