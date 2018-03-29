"""
The postprocess module provides utility functions for post-processing segmentation maps and writing them to files
"""

import numpy as np
from skimage.io import imsave
import os.path
import glob
import tarfile
import warnings


def export_as_png(mask, output_dir, filename):
    """
    Exports a cilia mask as a png file
    :param mask: the cilia mask to output
    :return: the path of the output file
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ensure the mask is in 8-bit unsigned int format
        mask = mask.astype(np.uint8)

        # ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # write the file
        file_path = os.path.join(output_dir, filename + '.png')
        imsave(file_path, mask)

        return file_path


def make_tar(input_dir, output_dir=None, pattern='*.png', filename='submission.tar'):
    """
    Compresses every file in the input_dir matching pattern into a tar archive
    then saves the archive to <output_dir>/<filename>.
    If output_dir is None, then it will be written to <input_dir>/<filename>

    :param input_dir: the directory where the input files reside
    :param output_dir: the directory where the output should be saved, or None to save it to the input_dir
    :param pattern: the mattern that a file must match to be included in the archive (supports asterisks as wildcards)
    :param filename: the name of the output archive file
    :return: file path of the tar archive
    """
    if output_dir is None:
        output_dir = input_dir
    file_pattern = os.path.join(input_dir, pattern)
    output_file = os.path.join(output_dir, filename)
    tar = tarfile.open(output_file, 'w')
    for filename in glob.glob(file_pattern):
        base_name = os.path.basename(filename)  # file name without the directory prefix
        tar.add(filename, arcname=base_name)
    tar.close()

    return output_file

