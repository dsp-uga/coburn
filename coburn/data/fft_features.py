import numpy as np
from .Transform import Transform
import torch

class Frequency(Transform):
    def __call__(self, images):
        return torch.from_numpy(images.toarray()).float().cuda()
