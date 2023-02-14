'''neural_decoder.py
Linear and nonlinear decoding neural networks trained with supervised learning to predict class labels
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning

NOTE: Your challenge is to NOT import numpy here!
'''
import random

import tensorflow as tf
import numpy as np
import time


class NeuralDecoder:
    '''Neural network trained to decode the class label from the associated pattern of activation produced by a
    bio-inspired Hebbian learning network.
    '''
    def __init__(self, num_features, num_classes, wt_stdev=0.1):
        '''Constructor to intialize the decoding network weights and bias. The decoder is a single-layer network, so
        there is one set of weights and bias.

        Parameters:
        -----------
        num_features: int. Num input features (M)
        num_classes: int. Num data classes (C)
        wt_stdev: float. Standard deviation of the Gaussian-distributed weights and bias

        NOTE: Remember to wrap your weights and bias as tf.Variables for gradient tracking!
        '''
        # Change/set these
        self.wts = tf.Variable(tf.random.normal(shape=(num_features, num_classes), stddev=wt_stdev))
        self.b = tf.Variable(tf.random.normal(shape=(num_classes,), stddev=wt_stdev))
        self.num_classes = num_classes

    def get_wts(self):
        '''Returns the decoder wts'''
        return self.wts

    def get_b(self):
        '''Returns the decoder bias'''
        return self.b

    def set_wts(self, wts):
        '''Replaces the decoder weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: tf.Variable. shape=(M, C). New decoder network weights.
        '''
        self.wts = wts

    def set_b(self, b):
        '''Replaces the decoder bias with `b` passed in as a parameter.

        Parameters:
        -----------
        b: tf.Variable. shape=(C,). New decoder network bias.
        '''
        self.b = b


    def accuracy(self, y_true, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y_true: tf.constant. shape=(B,). int-coded true classes.
        y_pred: tf.constant. shape=(B,). int-coded predicted classes by the network.

        Returns:
        -----------
        float. accuracy in range [0, 1]

        Hint: tf.where might be helpful.
        '''
        same = tf.where(y_true == y_pred)
        return tf.shape(same)[0]/tf.shape(y_pred)[0]

    def forward(self, x, net_act=False):
        '''Performs the forward pass through the decoder network with data samples `x`

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples

        Returns:
        -----------
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.

        NOTE: Subclasses should implement this (do not implement this method here).
        '''
        pass

    def loss(self, yh, net_act):
        '''Computes the loss on the current mini-batch using the one-hot coded class labels `yh` and `net_act`.

        Parameters:
        -----------
        yh: tf.constant. tf.float32. shape=(B, C). One-hot coded class assignments.
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.

        Returns:
        -----------
        loss: float. Loss computed over the mini-batch.

        NOTE: Subclasses should implement this (do not implement this method here).
        '''
        pass

    def predict(self, x, net_act=None):
        '''Predicts the class of each data sample in `x` using the passed in `net_act`. If `net_act` is not passed in,
        the method should compute it in order to perform the prediction.

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples
        net_act: tf.constant. shape=(B, C) or None. Network activation.

        Returns:
        -----------
        y_preds: tf.constant. shape=(B,). int-coded predicted class for each sample in the mini-batch.
        '''
        if net_act is None:
            net_act = self.forward(x)

        return tf.math.argmax(net_act, axis=1)

    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        '''Helper method used during training to determine whether training should stop before the maximum number of
        training epochs is reached based on the most recent loss values computed on the validation set
        (`recent_val_losses`) the validation loss on the current epoch (`curr_val_loss`) and `patience`.

        - When training begins, the recent history of validation loss values `recent_val_losses` is empty (i.e. `[]`).
        When we have fewer entries in `recent_val_losses` than the `patience`, then we just insert the current val loss.
        - The length of `recent_val_losses` should not exceed `patience` (only the most recent `patience` loss values
        are considered).
        - The recent history of validation loss values (`recent_val_losses`) is assumed to be a "rolling list" or queue.
        Remove the oldest loss value and insert the current validation loss into the list. You may keep track of the
        full history of validation loss values during training, but maintain a separate list in `fit()` for this.

        Conditions that determine whether to stop training early:
        - We never stop early when the number of validation loss values in the recent history list is less than patience
        (training is just starting out).
        - We stop early when the OLDEST validation loss (`curr_val_loss`) is smaller than all recent validation loss
        values. IMPORTANT: Assume that `curr_val_loss` IS one of the recent loss values — so the current loss value
        should be compared with `patience`-1 other loss values.

        Parameters:
        -----------
        recent_val_losses: Python list of floats. len between 0 and `patience` (inclusive).
        curr_val_loss: float. The loss computed on the validation set on the current training epoch.
        patience: int. The patience: how many recent loss values computed on the validation set we should consider when
            deciding whether to stop training early.

        Returns:
        -----------
        recent_val_losses: Python list of floats. len between 1 and `patience` (inclusive).
            The list of recent validation loss values passsed into this method updated to include the current validation
            loss.
        stop. bool. Should we stop training based on the recent validation loss values and the patience value?

        NOTE:
        - This method can be concisely implemented entirely with regular Python (TensorFlow/Numpy not needed).
        - It may be helpful to think of `recent_val_losses` as a queue: the current loss value always gets inserted
        either at the beginning or end. The oldest value is then always on the other end of the list.
        '''
        stop = False
        recent_val_losses.append(curr_val_loss)
        if len(recent_val_losses) > patience:
            recent_val_losses = recent_val_losses[1:patience+1]
            if np.argmin(recent_val_losses) == 0:
                stop = True

        return recent_val_losses, stop

    def extract_at_indices(self, x, indices):
        '''Returns the samples in `x` that have indices `indices` to form a mini-batch.

        Parameters:
        -----------
        x: tf.constant. shape=(N, ...). Data samples or labels
        indices: tf.constant. tf.int32, shape=(B,), Indices of samples to return from `x` to form a mini-batch.
            Indices may be in any order, there may be duplicate indices, and B may be less than N (i.e. a mini-batch).
            For example indices could be [0, 1, 2], [2, 2, 1], or [2, 1].
            In the case of [2, 1] this method would samples with index 2 and index 1 (in that order).

        Returns:
        -----------
        tf.constant. shape=(B, ...). Value extracted from `x` whose sample indices are `indices`.

        Hint: Check out tf.gather. See TF tutorial from last semester (end of final notebook) for example usage.
        '''
        return tf.gather(x, indices)

    def fit(self, x, y, x_val=None, y_val=None, mini_batch_sz=512, lr=1e-4, max_epochs=1000, patience=3, val_every=1,
            verbose=True, print_every=100):
        '''Trains the neural decoder on the training samples `x` (and associated int-coded labels `y`) using early
        stopping and the Adam optimizer.

        Parameters:
        -----------
        x: tf.constant. tf.float32. shape=(N, M). Data samples.
        y: tf.constant. tf.float32. shape=(N,). int-coded class labels
        x_val: tf.constant. tf.float32. shape=(N_val, M). Validation set samples.
        y_val: tf.constant. tf.float32. shape=(N_val,). int-coded validation set class labels.
        mini_batch_sz: int. Number of samples to include in each mini-batch.
        lr: float. Learning rate used with Adam optimizer.
        max_epochs: int. Network should train no more than this many epochs (training could stop early).
        patience: int. Number of most recent computations of the validation set loss to consider when deciding whether
            to stop training early (before `max_epochs` is reached).
        val_every: int. How often (in epoches) to compute validation set accuracy, loss, and check whether to stop training
            early.
        verbose: bool. If set to `False`, there should be no print outs during training. Messages indicating start and
        end of training are fine.


        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.
        val_loss_hist: Python list of floats. len=num_epochs/val_freq.
            Loss computed on the validation set every time it is checked (`val_freq`).
            NOTE: This is the FULL history of validation set loss values, not just the RECENT ones used for early stopping.
        num_epochs: int.
            The number of epochs used to train the network. Must be < max_epochs.

        TODO:
        Go through the usual motions:
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - Compute forward pass and loss for each mini-batch. Have your Adam optimizer apply the gradients to update the
        wts and bias.
        - Record the average training loss values across all mini-batches in each epoch.
        '''
        N, M = x.shape
        one_hot_y = self.one_hot(y, self.num_classes)
        one_hot_y_val = self.one_hot(y_val, self.num_classes)

        # Define loss tracking containers
        train_loss_hist = []
        val_loss_hist = []
        recent_val_loss_hist = []
        stop = False

        for n in range(max_epochs):
            t_0 = time.time()

            with tf.GradientTape() as tape:
                batch_idx = random.sample(range(0, len(x) - 1), mini_batch_sz)
                batch_x = self.extract_at_indices(x, batch_idx)
                batch_y = self.extract_at_indices(one_hot_y, batch_idx)
                net_in = self.forward(batch_x)
                loss = self.loss(batch_y, net_in)
                loss = tf.reduce_mean(loss)
                train_loss_hist.append(loss)

            grads = tape.gradient(loss, [self.wts, self.b])
            opt = tf.keras.optimizers.Adam(lr)
            opt.apply_gradients(zip(grads, [self.wts, self.b]))

            if verbose and n % val_every == 0:
                val_net_in = self.forward(x_val)
                val_loss = self.loss(one_hot_y_val, val_net_in)
                val_loss = tf.reduce_mean(val_loss)
                val_loss_hist.append(val_loss)
                recent_val_loss_hist.append(val_loss)
                recent_val_loss_hist, stop = self.early_stopping(recent_val_loss_hist, val_loss, patience=patience)

            if stop:
                break

            t = time.time() - t_0
            if n == 0:
                print (f'Epoch 0/{max_epochs-1} completed \n 1 epoch took {t:.2f} seconds. Expected runtime is {t*(max_epochs-n):.2f} seconds.')
            elif n % print_every == 0:
                print (f'Epoch {n}/{max_epochs-1} completed, took {t:.2f} seconds. Approx. {int(t*(max_epochs-n))} seconds remain.')

        return train_loss_hist, val_loss_hist, n


# Suggested that SoftmaxDecoder and NonlinearDecoder go below:
class SoftmaxDecoder(NeuralDecoder):
    def forward(self, x, net_act=False):
        '''Performs the forward pass through the decoder network with data samples `x`

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples

        Returns:
        -----------
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.
        '''
        wts = tf.cast(self.wts, tf.double)
        b = tf.cast(self.b, tf.double)
        net_in = tf.matmul(x, wts) + b

        return tf.nn.softmax(net_in) if net_act else net_in

    def loss(self, yh, net_in):
        '''Computes the loss on the current mini-batch using the one-hot coded class labels `yh` and `net_act`.

        Parameters:
        -----------
        yh: tf.constant. tf.float32. shape=(B, C). One-hot coded class assignments.
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.

        Returns:
        -----------
        loss: float. Loss computed over the mini-batch.

        '''
        return tf.nn.softmax_cross_entropy_with_logits(yh, net_in)

    def one_hot(self, y, C):
        '''One-hot codes the vector of class labels `y`

        Parameters:
        -----------
        y: tf.constant. shape=(B,) int-coded class assignments of training mini-batch. 0,...,numClasses-1
        C: int. Number of unique output classes total
        off_value: int. The "off" value that represents all other values in each sample's one-hot vector that is not 1.

        Returns:
        -----------
        y_one_hot: tf.constant. tf.float32. shape=(B, C) One-hot coded class assignments.
            e.g. if off_value=-1, y=[1, 0], and C=3, the one-hot vector would be:
            [[-1., 1., -1.], [-1., 1., -1.]]
        '''
        return tf.one_hot(y, C, off_value=0, dtype='float32')


class NonlinearDecoder(NeuralDecoder):

    def __init__(self, num_features, num_classes, wt_stdev=.1, beta=.005, loss_exp=6):
        super().__init__(num_features, num_classes, wt_stdev)
        self.beta = beta
        self.loss_exp = loss_exp

    def forward(self, x, net_act=False):
        net_in = tf.matmul(x, self.wts) + self.b

        return tf.nn.tanh(tf.math.scalar_mul(self.beta, net_in)) if net_act else net_in

    def one_hot(self, y, C):
        return tf.one_hot(y, C, off_value=-1, dtype='float32')

    def loss(self, yh, net_act):
        diffs = tf.math.subtract(yh, net_act)
        abs_diffs = tf.math.abs(diffs)
        exp_diffs = tf.math.pow(abs_diffs, self.loss_exp)

        return tf.reduce_sum(exp_diffs)
