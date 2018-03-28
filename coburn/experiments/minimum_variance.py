"""
Segments cilia based on variance thresholding
Optimal variance threshold is found using a random sampling of examples from the labelled training data.
"""

from coburn.data import preprocess, loader
from torchvision.transforms import Compose
from showit import tile

# def tune():


def main(input='./data', output='./results', threshold=5):
    # preprocessing
    variance_transform = preprocess.Variance()

    transform = Compose([variance_transform])

    # load data
    print("Loading dataset...")
    dataset = loader.random_sample(10, base_dir=input, seed=1234)
    dataset.set_transform(transform)

    print(dataset[0].toarray())
    print(dataset.get_mask(0))

