"""
Segments cilia based on variance thresholding
Optimal variance threshold is found using a random sampling of examples from the labelled training data.
"""

from torchvision.transforms import Compose
from showit import tile, image
from matplotlib import pyplot as plt
import numpy as np
from coburn.data import preprocess, loader


# tunes the threshold hyperparameter using the training set
def tune():
    # load the training set
    dataset = loader.load('train')
    num_samples = len(dataset)

    # compute variance along the time dimension
    variance_transform = preprocess.Variance()
    transform = Compose([variance_transform])
    dataset.set_transform(transform)

    # compute the optimal threshold for each sample
    optimal_thresholds = np.zeros(num_samples)
    optimal_scores = np.zeros(num_samples)
    for idx in range(0, num_samples):
        sample_image = dataset[idx].toarray()
        # for each image, find the threshold which maximizes intersection-over-union score
        optimal_threshold = 0.8
        optimal_score = 0.0
        for threshold in np.arange(1.0, 15.1, 0.1):
            thresholding = sample_image > threshold  # indices where the variance is greater than the threshold
            mask = np.zeros(sample_image.shape)
            mask[thresholding] = 2  # the value 2 indicated cilia
            score = dataset.compute_score(idx, mask)
            if score > optimal_score:
                optimal_score = score
                optimal_threshold = threshold

        # record the optimal threshold for this movie
        optimal_thresholds[idx] = optimal_threshold
        optimal_scores[idx] = optimal_score

    # average the optimal thresholds that were found
    print("Average optimal threshold: %0.4f" % np.mean(optimal_thresholds))
    print("Variance: %0.4f" % np.var(optimal_thresholds))

    print("Average score: %0.4f" % np.mean(optimal_scores))
    print("Variance: %0.4f" % np.var(optimal_scores))


def main(input='./data', output='./results', threshold=9.05):
    # preprocessing
    # variance_transform = preprocess.Variance()
    #
    # transform = Compose([variance_transform])
    #
    # # load data
    # print("Loading dataset...")
    # dataset = loader.random_sample(10, base_dir=input, seed=1234)
    # dataset.set_transform(transform)
    # img = dataset[1].toarray()
    # threshold_indices = img > threshold
    # mask = np.zeros(img.shape)
    # mask[threshold_indices] = 2
    # ground_truth = dataset.get_mask(1)
    #
    # print(dataset.compute_score(1, mask))
    #
    # # dataset.compute_score(0, mask)
    # tile([mask, ground_truth])
    # plt.show()
    tune()


