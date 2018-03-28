import torch.utils.data
import thunder as td
import skimage.data
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
            assert(isinstance(transform, Transform))
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

    def get_mask(self, idx):
        """
        Gets the ground-truth mask for the data point at the specified index
        :param idx: int, The index of the item in this dataset whose mask you want to load
        :return: an greyscale skimage if a mask was found, None otherwise
        """
        hash = self.get_hash(idx)
        mask_path = os.path.join(self.base_dir, hash, 'mask.png')
        if os.path.exists(mask_path):
            return skimage.data.load(mask_path, as_grey=True)
        else:
            return None

    def set_transform(self, transform):
        """
        Sets the transform that will be applied to items loaded from this dataset
        :param transform: coburn.data.Transform object
        :return: None
        """
        assert(isinstance(transform, Transform))
        self.transform = transform

