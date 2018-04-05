# Coburn

Segmentation of cilia videos

Coburn is a project that attempts to segment microscopic videos of human cells into regions of 
cilia and non-cilia.  Coburn was completed over the course of three weeks in Spring 2018 as part of the "Data
Science Practicum Class" at the University of Georgia.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This project uses [Conda](https://conda.io/docs/) to manage dependencies.  
Conda is all you need to get started.

### Installing

The environment.yml file is used by Conda to create a virtual environment that includes all the project's dependencies (including Python!)

Navigate to the project directory and run the following command

`conda env create -f environment.yml`

This will create a virtual environment named "coburn". Activate the virtual environment with the following command

`source activate coburn`

After the environment has been activated, the program can be run as follows:

`python main.py <options>`

To display program args and help, run

`python main.py -h`

Note that the datasets are automatically downloaded as required, so the 
first time running experiments may be a little slow since the entire dataset will be downloaded.

### Minimum Variance Model

The simplest model thresholds the input movies by their variance, marking a pixel as cilia if 
its variance exceeds some threshold value.  This experiment can be run as follows:


`python main.py minimum-variance <options>`

Run `python main.py minimum-variance -h` for a complete description of optional parameters.

## Built With

* [Python 3.6](https://www.python.org/)
* [Conda](https://conda.io/docs/)
* [PyTorch](http://pytorch.org/)

## Contributing

The `master` branch of this repo is write-protected.  Every pull request must pass a code review before being merged.
Other than that, there are no specific guidelines for contributing.
If you see something that can be improved, please send us a pull request!

## Authors

* [Vibodh Fenani](https://github.com/vibodh01)
* [Layton Hayes](https://github.com/minimum-LaytonC)
* [Zach Jones](https://github.com/zachdj)
* [Raj Sivakumar](https://github.com/raj-sivakumar)

See the [contributors file](CONTRIBUTORS.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Dr. Shannon Quinn](https://github.com/magsol) for providing the data set and project guidance
* [U-Net](https://arxiv.org/abs/1505.04597) - Neural Net architecture used in this project for segmentation

