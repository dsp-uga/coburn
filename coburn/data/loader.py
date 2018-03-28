"""
The loader module is responsible for downloading datasets and loading/streaming them into memory
"""

import os
import io
import tarfile
import numpy as np
import urllib.request
from .Dataset import Dataset


BASE = "https://storage.googleapis.com/uga-dsp/project4/"

# read the manifest files whenever this module is imported
with open("manifests/train.txt", 'r') as train_manifest:
    TRAINING_MANIFEST = list(map(lambda s: s.strip(), train_manifest.readlines()))

with open("manifests/test.txt", 'r') as test_manifest:
    TESTING_MANIFEST = list(map(lambda s: s.strip(), test_manifest.readlines()))


def _download(movie=None, save_location="data"):
    """
    Downloads a cilia movie from the GCP bucket and saves it locally
    The movie will be saved in a directory `<save_location>/<hash>`
    The directory will contain all the images from the movie.  If the movie is from the training set, then the directory
        will also contain the mask file

    :param movie: the hash of the movie to download
    :param save_location: the directory where the downloaded movie will be saved
    :return: the directory where the movie was saved
    """
    if movie is None:
        return None

    # download the movie from GCP
    movie_url = BASE + "data/" + movie + ".tar"
    response = urllib.request.urlopen(movie_url)
    movie_data = response.read()

    # treat the downloaded bytes as a tar file
    movie_file_object = io.BytesIO(movie_data)
    tarred_movie = tarfile.open(fileobj=movie_file_object)

    # extract the images to the save directory
    save_path = os.path.join(save_location, movie)
    images_subdir = os.path.join(save_path, 'images')
    if not os.path.exists(images_subdir):
        os.makedirs(images_subdir)

    for image in tarred_movie.getmembers():
        image_name = os.path.basename(image.name)
        image_file = tarred_movie.extractfile(image)
        image_bytes = image_file.read()
        image_path = os.path.join(images_subdir, image_name)
        with open(image_path, 'wb') as outfile:
            outfile.write(image_bytes)

    tarred_movie.close()

    # download the mask if this movie came from training data
    has_mask = movie in TRAINING_MANIFEST
    if has_mask:
        mask_url = BASE + "masks/" + movie + ".png"
        mask_filepath = os.path.join(save_location, movie, "mask.png")
        response = urllib.request.urlopen(mask_url)
        mask_data = response.read()
        with open(mask_filepath, 'wb') as mask_file:
            mask_file.write(mask_data)

    return save_path


def load(samples='all', base_dir='data', skip_cached=True):
    """
    Loads the specified examples into a coburn.data.Dataset object
    The `hashes` argument can any of the following:
        - the keyword 'all', which loads every sample in both the training and testing sets
        - the keyword 'train', which loads every sample in the training set
        - the keyword 'test', which loads every sample in the test set
        - an array of hashes, which loads each sample whose hash is in the array

    :param samples: array of str or one of ['all', 'train', 'test']
    :param base_dir: the local directory where the datasets will be stored
    :param skip_cached: if set, samples found in `base_dir` will not be downloaded again
    :return: coburn.data.Dataset object representing the set of samples
    """
    if isinstance(samples, str):
        if samples == 'all':
            samples = TRAINING_MANIFEST + TESTING_MANIFEST
        elif samples == 'train':
            samples = TRAINING_MANIFEST
        elif samples == 'test':
            samples = TESTING_MANIFEST

    # ensure all the datasets from `samples` are downloaded
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    cache = [sample for sample in os.listdir(base_dir)]
    for movie_hash in samples:
        if not skip_cached or not movie_hash in cache:
            _download(movie_hash, base_dir)

    # create a dataset for these samples
    dataset = Dataset(samples, base_dir)
    return dataset


def random_sample(n=1, type='train', base_dir='data', seed=None):
    """
    Randomly samples n movies from the train or test set.
    Somewhat useful for testing models on a small scale.

    :param n: the number of movies to sample
    :param type: 'train' to sample from the training set, 'test' to sample from the testing set, 'all' to sample from
                    the combined training and testing sets
    :param seed: float, seed for the RNG that selects the samples
    :return: coburn.data.Dataset object representing the set of samples
    """
    set = TRAINING_MANIFEST
    if type == 'test':
        set = TESTING_MANIFEST
    elif type == 'all':
        set += TESTING_MANIFEST

    if seed is not None:
        np.random.seed(seed)

    samples = np.random.choice(set, n)
    return load(samples, base_dir=base_dir)
