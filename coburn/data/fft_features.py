import numpy as np
from .Transform import Transform
import torch

class Frequency(Transform):
    def __call__(self, images):
        torch.set_default_tensor_type("torch.FloatTensor")
        return torch.from_numpy(images.toarray()).cuda()
