import torch.utils.data
from torchvision.transforms import Compose
import numpy as np
import thunder as td
import skimage.io
import os
from .Transform import Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, hashes, base_dir="data", transform=None):
        """
        The Dataset class represents a dataset composed of thunder.Series objects.
        Items in the dataset are automatically streamed from disk as they are requested

        Check out the Thunder documentation for more info about working with Series objects:
        https://github.com/thunder-project/thunder

        :param hashes: array,
            A list of movie hashes to load into this dataset
        :param base_dir: str,
            The directory where the datasets can be found.
        :param transform: coburn.data.Transform,
            Transform object which will be applied to each series in this dataset
            Multiple transforms can be composed using the torchvision.transform.Compose method call
        """
        self.hashes = hashes
        self.base_dir = base_dir
        if transform is not None:
            assert(isinstance(transform, Transform) or isinstance(transform, Compose))
        self.transform = transform

    def __len__(self):
        """
        Get the number of items in this dataset
        :return: int, Number of items in the dataset
        """
        return len(self.hashes)

    def __getitem__(self, idx):
        """
        Get an item from this dataset by its index.
        If specified, the transform function will automatically be applied to the fetched item

        :param idx: nonnegative int,
            The index of the item to get.  Should be in the interval [0, __len__ )
        :return: object,
            If a transform is specified, then the result of calling self.transform.__call__(series) on the series from
                the specified index
            If no transform is specified for this Dataset, then thunder.Series
        """
        assert(0 <= idx < self.__len__())
        hash = self.hashes[idx]
        path_to_images = os.path.join(self.base_dir, hash, 'images')
        data = td.images.frompng(path_to_images)

        if self.transform is not None:
            data = self.transform.__call__(data)

        return data

    def get_hash(self, idx):
        assert(0 <= idx < self.__len__())
        return self.hashes[idx]

    def get_original_size(self, idx):
        """
        Gets the original size of the images in the movie at the specified index
        :param idx: int, the index of the movie in this dataset whose original size you want to retrieve
        :return: a tuple (W, H) specifying the original size of the images
        """
        hash = self.hashes[idx]
        path_to_images = os.path.join(self.base_dir, hash, 'images')
        data = td.images.frompng(path_to_images).toarray()
        return data[0].shape

    def get_mask(self, idx):
        """
        Gets the ground-truth mask for the data point at the specified index
        :param idx: int, The index of the item in this dataset whose mask you want to load
        :return: an greyscale skimage if a mask was found, None otherwise
        """
        hash = self.get_hash(idx)
        mask_path = os.path.join(self.base_dir, hash, 'mask.png')
        if os.path.exists(mask_path):
            return skimage.io.imread(mask_path, as_grey=True)
        else:
            return None

    def compute_score(self, idx, mask):
        """
        Compares the given mask with the ground truth image for dataset[idx] using the intersection-over-union score.
        The caller must ensure that the given mask has the same dimensions as the ground-truth mask.

        :param idx: the example in this dataset whose ground truth mask will be used
        :param mask: the mask to compare against
        :return: intersection-over-union score if ground truth is available for idx, 0 otherwise
        """
        ground_truth = self.get_mask(idx)
        if ground_truth is None:
            return 0
        else:
            assert(ground_truth.shape == mask.shape)
            intersection = np.logical_and(mask == 2, ground_truth == 2)
            intersection = np.count_nonzero(intersection)

            union = np.logical_or(mask == 2, ground_truth == 2)
            union = np.count_nonzero(union)

            return intersection / union

    def set_transform(self, transform):
        """
        Sets the transform that will be applied to items loaded from this dataset
        :param transform: coburn.data.Transform object
        :return: None
        """
        assert(isinstance(transform, Transform) or isinstance(transform, Compose))
        self.transform = transform

