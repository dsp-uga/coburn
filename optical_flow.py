"""
parameter values for Farneback optical flow taken from here:
https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import os


def optical_flow(images):
    """
    images should be a numpy array or similar.
    returns a numpy array of images
    """
    # create an appropriately sized array to append flow values into
    flows = np.empty((0, images.shape[1], images.shape[2], 2))
    i = 0
    while(i < images.shape[0]-1):
        flow = cv2.calcOpticalFlowFarneback(
            images[i], images[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows = np.append(flows, [flow], axis=0)
        i += 1
    return flows


if __name__ == "__main__":
    directory = "/home/layton/Downloads/data/007f736aedbc4ca67989f8ca62f1bbeb447ad76698351fe387923963ee50e5ae"
    files = [directory+"/"+x for x in os.listdir(directory)]
    frames = io.imread_collection(files)

    flows = optical_flow(np.array(frames))

    for i in range(10):
        plt.imshow(flows[i, :, :, 0])
        plt.show()
