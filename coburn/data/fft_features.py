import numpy as np
from .Transform import Transform
import torch
import torchvision.transforms as tvt


class MakeCUDA(Transform):
    def __call__(self, images):
        return torch.from_numpy(images.toarray()).float().cuda()

class Frequency(Transform):
    def __init__(self, n=128):
        self.n = n
    def __call__(self, images):
        np_images = images.toarray()
        sp = np.fft.fft(np_images, n=self.n, axis=0)
        return sp.real

def PCA(data, k=2):
    # adapted from web sources
    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])
