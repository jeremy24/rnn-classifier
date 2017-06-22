# rnn-classifier
The initial base for this code is from [this project](https://github.com/sherjilozair/char-rnn-tensorflow).

Initially this project is a generative character based model.
The goal is to build a character based classifier out of a RNN and learn some along the way.



## Requirements
1. Tensorflow 1.2 with GPU support (I use a version built from source on Ubuntu 16.04)
2. Python 3
3. At least one Nvidia GPU
4. The launch script using the fish shell but you can just run the python command if you don't want to install fish.

## Setup
Get TensorFlow installed [Here](https://www.tensorflow.org/install/install_linux).
Make sure you install the version with GPU support.
I am using a version of it built from source and then installed into a Python 3 virtual env.


## Usage
I am currently working on getting inference to work with the new seq2seq decoders.

1. Have an input file named input.txt at the project root, this will be fed to the model.
2. Run it with <code> ./train_model [model dir] [true/false] </code>  <b>(if true, previous dir will be overwritten)</b>
3. If you don't have fish you can run it with
    <code> cat run_train | bash </code>
4. Or you can directly run the python command in run_train
5. Run <code>python train.py --help </code> to see all available arguments.



## Disclaimer
This project is a work in progress and mainly for the purpose of allowing me to experiment with the new TensorFlow 1.2 API and RNNs.  This means it will almost definitely have a lot bugs and issues.
