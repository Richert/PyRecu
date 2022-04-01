PyRecu
======

PyRecu is a Python library for recurrent neural network (RNN) modeling and reservoir computing (RC), developed by Richard Gast.
It is very much a work-in-process kind of open-source project that everyone is welcome to contribute to.

Basic Features
--------------

# 1. rnn
- module for the forward simulation of RNN dynamics
- create the network data of your RC workflow
- runtime optimization via `Numba`, the behavior of which can be fully controlled by the user
- provide your own function implementing the integration of the RNN or use one of the provided RNNs in `pyrecu.neural_models`
- provide your own function decorators and function decorator arguments for runtime optimization

# 2. readout
- module for model fitting based on previously simulated network data
- train and test the readout layer of your RC workflow
- use ridge regression and cross-validation features 

Installation
------------

To install PyRecu, clone this repository and run the following line from the directory in which the repository was cloned:
```
python setup.py install
```
This will also install the requirements of the software listed below.

# Dependencies
- numpy
- numba
- scikit-learn

Contact
-------

If you have any questions, want to contribute to the software, or just get in touch, feel free to post an issue or contact [Richard Gast](https://www.richardgast.me).
