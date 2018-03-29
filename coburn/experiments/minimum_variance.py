"""
Segments cilia based on variance thresholding
Optimal variance threshold is found using a random sampling of examples from the labelled training data.
"""

from torchvision.transforms import Compose
import numpy as np
from coburn.data import preprocess, postprocess, loader


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


def main(input='./data', output='./results/min_var', threshold=9.05):
    # load the test data
    print("Loading Data...")
    dataset = loader.load('test', base_dir=input)

    # compute variance along the time axis
    variance_transform = preprocess.Variance()
    transform = Compose([variance_transform])
    dataset.set_transform(transform)

    # segment each image and write it to the results directory
    print("Segmenting images...")
    for idx in range(0, len(dataset)):
        img = dataset[idx].toarray()
        hash = dataset.get_hash(idx)

        # create cilia mask based on grayscale variance thresholding
        mask = np.zeros(img.shape)
        thresholding = img >= threshold
        mask[thresholding] = 2

        postprocess.export_as_png(mask, output, hash)

    tar_path = postprocess.make_tar(output)

    print("Done!")
    print("Results written to %s" % tar_path)
