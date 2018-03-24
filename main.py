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

    cmd = subcommands.add_parser('example', description='Load a small dataset and display the mean images', argument_default=argparse.SUPPRESS)
    cmd.set_defaults(func=coburn.experiments.example_transform.main)

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