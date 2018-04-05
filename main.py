import argparse
import coburn


def info(args):
    """
    Print system info
    """
    import sys
    print('Python version:')
    print(sys.version)


def main():
    parser = argparse.ArgumentParser(
        description='Coburn: Cilia Segmentation',
        argument_default=argparse.SUPPRESS,
    )
    subcommands = parser.add_subparsers()

    # coburn info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    # coburn download <movie_hash> <save_location>
    def download_movies(movies, output):
        for hash in movies:
            coburn.data.loader._download(hash, output)
    cmd = subcommands.add_parser('download', description='Dataset Downloader', argument_default=argparse.SUPPRESS)
    cmd.add_argument('movies',  help='the hashes of the movies you want to download', nargs='+')
    cmd.add_argument('--output', '-o', default="./data",
                     help='the directory where the movie will be saved [DEFAULT: ./data]')
    cmd.set_defaults(func=download_movies)

    # example script that does some preprocessing and shows the resultant images
    cmd = subcommands.add_parser('example', argument_default=argparse.SUPPRESS,
                                 description='Load a small dataset and display the mean images')
    cmd.set_defaults(func=coburn.experiments.example_compose_transforms.main)

    # segmentation using grayscale variance thresholding
    cmd = subcommands.add_parser('minimum-variance',  argument_default=argparse.SUPPRESS,
                                 description='Segment the testing set using a minimum variance threshold')
    cmd.add_argument('--input', '-i', default="./data",
                     help='The directory where the dataset can be found.  It will be downloaded to this location if it'
                          'is not found. [DEFAULT: ./data]')
    cmd.add_argument('--output', '-o', default="./results/min_var",
                     help='The directory where the image masks will be saved. [DEFAULT: ./results/min_var]')
    cmd.add_argument('--threshold', '-t', default=8.75, type=float,
                     help='Variance threshold.  Pixels with variance higher than this threshold will be marked as cilia'
                          ' [DEFAULT: Use mean variance]')
    cmd.add_argument('--filter_size', '-s', default=4, type=int,
                     help='Size of the Median Filter to apply before thresholding.  0 if you do not want to apply a '
                          'median filter [DEFAULT: 4]')
    cmd.set_defaults(func=coburn.experiments.minimum_variance.main)

    # segmentation using grayscale variance thresholding
    cmd = subcommands.add_parser('fft_test',  argument_default=argparse.SUPPRESS,
                                 description='Segment the testing set using a minimum variance threshold')
    cmd.set_defaults(func=coburn.experiments.fft_hist_test.main)

    # segmentation using ciliary beat frequency
    cmd = subcommands.add_parser('fft',  argument_default=argparse.SUPPRESS,
                                 description='Segment the testing set using a minimum variance threshold')
    cmd.add_argument('--input', '-i', default="./data",
                     help='The directory where the dataset can be found.  It will be downloaded to this location if it'
                          'is not found. [DEFAULT: ./data]')
    cmd.add_argument('--output', '-o', default="./results/min_var",
                     help='The directory where the image masks will be saved. [DEFAULT: ./results/min_var]')
    cmd.add_argument('--k', '-k', default=10, type=int,
                     help='Number of dimensions to retain in dimensionality reduction step'
                          ' [DEFAULT: Use 10]')
    cmd.add_argument('--dom_frequency', '-d', default=11, type=float,
                     help='Frequency threshold.  Pixels with variance higher than this threshold will be marked as cilia'
                          ' [DEFAULT: Use mean variance]')
    cmd.set_defaults(func=coburn.experiments.fft.main)
    # Each subcommand gives an `args.func`.
    # Call that function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
