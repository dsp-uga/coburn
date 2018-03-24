"""
    The loader module is responsible for downloading datasets and loading/streaming them into memory
"""

import os
import io
import tarfile

import urllib.request


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
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image in tarred_movie.getmembers():
        image_name = os.path.basename(image.name)
        image_file = tarred_movie.extractfile(image)
        image_bytes = image_file.read()
        image_path = os.path.join(save_path, image_name)
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

