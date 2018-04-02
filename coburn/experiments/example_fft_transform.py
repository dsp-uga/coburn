"""
This experiment shows an example of how the loader and preprocess modules can be used to load a dataset and
do some simple preprocessing
"""
from coburn.data import preprocess, loader, fft_features
from torchvision.transforms import Compose
from showit import tile
from matplotlib import pyplot as plt



def main():
    # these samples will be automatically downloaded if they are not found locally
    dataset = loader.load(samples=['4bad52d5ef5f68e87523ba40aa870494a63c318da7ec7609e486e62f7f7a25e8',
                                   'a7e37600a431fa6d6023514df87cfc8bb5ec028fb6346a10c2ececc563cc5423',
                                   '70a6300a00dbac92be9238252ee2a75c86faf4729f3ef267688ab859eed1cc60'])

    # compose the Mean and Variance transforms (this composition is rather meaningless, but is a good example of how this works)
    # note that order matters!  The transforms will be applied in order
    fft_transform = fft_features.Frequency()
    composed_transform = Compose([fft_transform])
    dataset.set_transform(composed_transform)

    fft_images = list()
    for i in range(len(dataset)):
        sample = dataset[i]  # mean transform has already been applied!
        fft_images.append(sample)

    tile(fft_images)
    plt.show()
