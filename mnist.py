'''mnist.py
Loads and preprocesses the MNIST dataset
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import os
import numpy as np


def get_mnist(N_val, path='data/mnist'):
    '''Load and preprocesses the MNIST dataset (train and test sets) located on disk within `path`.

    Parameters:
    -----------
    N_val: int. Number of data samples to reserve from the training set to form the validation set. As usual, each
    sample should be in EITHER training or validation sets, NOT BOTH.
    path: str. Path in working directory where MNIST dataset files are located.

    Returns:
    -----------
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels)
    '''
    x_train_whole = preprocess_mnist(np.load(path+"/x_train.npy"))
    y_train_whole = np.load(path+"/y_train.npy")
    x_test = preprocess_mnist(np.load(path+"/x_test.npy"))
    y_test = np.load(path+"/y_test.npy")

    return x_train_whole[:x_train_whole.shape[0]-100, :], y_train_whole[:y_train_whole.shape[0]-100], x_test, y_test, x_train_whole[x_train_whole.shape[0]-100:, :], y_train_whole[y_train_whole.shape[0]-100:]


def preprocess_mnist(x):
    '''Preprocess the data `x` so that:
    - the maximum possible value in the dataset is 1 (and minimum possible is 0).
    - the shape is in the format: `(N, M)`

    Parameters:
    -----------
    x: ndarray. shape=(N, I_y, I_x). MNIST data samples represented as grayscale images.

    Returns:
    -----------
    ndarray. shape=(N, I_y*I_x). MNIST data samples represented as MLP-compatible feature vectors.
    '''
    x_flat = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    return x_flat/x_flat.max()


def train_val_split(x, y, N_val):
    '''Divide samples into train and validation sets. As usual, each sample should be in EITHER training or validation
    sets, NOT BOTH. Data samples are already shuffled.

    Parameters:
    -----------
    x: ndarray. shape=(N, M). MNIST data samples represented as vector vectors
    y: ndarray. ints. shape=(N,). MNIST class labels.
    N_val: int. Number of data samples to reserve from the training set to form the validation set.

    Returns:
    -----------
    x: ndarray. shape=(N-N_val, M). Training set.
    y: ndarray. shape=(N-N_val,). Training set class labels.
    x_val: ndarray. shape=(N_val, M). Validation set.
    y_val ndarray. shape=(N_val,). Validation set class labels.
    '''
    #Note from Grady: I just did this in the main function
    pass
