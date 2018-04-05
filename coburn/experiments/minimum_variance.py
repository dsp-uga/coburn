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
    optimal_filter_sizes = np.zeros(num_samples)
    optimal_scores = np.zeros(num_samples)
    for idx in range(0, num_samples):
        sample_image = dataset[idx]

        # for each image, find parameters which maximize intersection-over-union score
        optimal_threshold = 0.8
        optimal_filter_size = 1
        optimal_score = 0.0
        for threshold in np.arange(1.0, 15.1, 0.1):
            for filter_size in np.arange(1, 5, 1):
                image = sample_image.median_filter(size=filter_size).toarray()
                thresholding = image > threshold  # indices where the variance is greater than the threshold
                mask = np.zeros(image.shape)
                mask[thresholding] = 2  # the value 2 indicated cilia
                score = dataset.compute_score(idx, mask)
                if score > optimal_score:
                    optimal_score = score
                    optimal_threshold = threshold
                    optimal_filter_size = filter_size

        # record the optimal threshold for this movie
        optimal_thresholds[idx] = optimal_threshold
        optimal_filter_sizes[idx] = optimal_filter_size
        optimal_scores[idx] = optimal_score

    # average the optimal parameters that were found
    print("Average optimal threshold: %0.4f" % np.mean(optimal_thresholds))
    print("Variance: %0.4f" % np.var(optimal_thresholds))

    print("Average optimal filter size: %0.4f" % np.mean(optimal_filter_sizes))
    print("Variance: %0.4f" % np.var(optimal_filter_sizes))

    print("Average score: %0.4f" % np.mean(optimal_scores))
    print("Variance: %0.4f" % np.var(optimal_scores))


def main(input='./data', output='./results/min_var', threshold=None, filter_size=4):
    # load the test data
    print("Loading Data...")
    dataset = loader.load('test', base_dir=input)

    # compute variance along the time axis
    transforms = []
    variance_transform = preprocess.Variance()
    transforms.append(variance_transform)

    if filter_size > 0:
        med_filter_transform = preprocess.MedianFilter(size=filter_size)
        transforms.append(med_filter_transform)

    transform = Compose(transforms)
    dataset.set_transform(transform)

    # segment each image and write it to the results directory
    print("Segmenting images...")
    for idx in range(0, len(dataset)):
        img, target = dataset[idx]
        img = img.toarray()
        hash = dataset.get_hash(idx)

        # create cilia mask based on grayscale variance thresholding
        mask = np.zeros(img.shape)
        if threshold is None:
            # if the threshold is not specified, then we use the mean variance
            threshold = np.mean(img)

        thresholding = img >= threshold
        mask[thresholding] = 2

        postprocess.export_as_png(mask, output, hash)

    tar_path = postprocess.make_tar(output)

    print("Done!")
    print("Results written to %s" % tar_path)
