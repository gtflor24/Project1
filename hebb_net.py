'''hebb_net.py
Bio-inspired neural network that implements the Hebbian learning rule and competition among neurons in the network
Grady and Caleb
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning

NOTE: This network should be implemented in Numpy rather than TensorFlow
'''
import numpy as np
import matplotlib.pyplot as plt
from viz import draw_grid_image
import time


class HebbNet:
    '''Single layer bio-inspired neural network in which neurons compete with
    each other and learning occurs via a variant of Hebbian learning rule
    (Oja's Rule).
    '''
    def __init__(self, num_features, num_neurons, wt_minmax=(0., 1.),
                 kth_place_inhibited=6, inhib_value=0.4,
                 load_wts=False, saved_wts_path='export/wts.npy'):
        '''Hebbian network constructor

        Parameters:
        -----------
        num_features: int. Num input features (M)
        num_neurons: int. Num of neurons in the network (H)
        wt_minmax: tuple of 2 floats. wt_minmax[0] is the min possible wt value when wts initialized. wt_minmax[1] is
            the max possible wt value when wts initialized.
        kth_place_inhibited: int. In the neural competition that occurs when processing each data sample, the neuron
            that achieves the kth highest net_in value ("neuron came in kth place") is inhibited, which means
            the kth place neuron gets netAct value of `-inhib_value`.
        inhib_value: float. Non-negative number (≥0) that represents the netAct value assigned to the inhibited neuron
            (with the kth highest netAct value).
        load_wts: bool. Whether to load weights previously saved off by the network after successful training.
        saved_wts_path: str. Path from the working project directory where the weights previously saved by the net are
            stored. Used if `load_wts` is True.
        '''
        self.num_features = num_features
        self.num_neurons = num_neurons
        self.kth_place_inhibited = kth_place_inhibited
        self.inhib_value = inhib_value
        self.saved_wts_path = saved_wts_path

        if load_wts:
            self.wts = np.load(saved_wts_path)
        else:
            self.wts = np.random.uniform(low=wt_minmax[0], high=wt_minmax[1], size=(num_features, num_neurons))

    def get_wts(self):
        '''Returns the Hebbian network wts'''
        return self.wts

    def set_wts(self, wts):
        '''Replaces the Hebbian network weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: ndarray. shape=(M, H). New Hebbian network weights.
        '''
        self.wts = wts

    def net_in(self, x, verbose=False):
        '''Computes the Hebbian network Dense net_in

        Parameters:
        -----------
        x: ndarray. shape=(B, M)

        Returns:
        -----------
        netIn: ndarray. shape=(B, H)
        '''
        if verbose:
            print(f'{x.shape = }')
            print(f'{self.wts.shape = }')
        return x@self.wts

    def net_act(self, net_in):
        '''Compute the Hebbian network activation, which is a function that reflects competition among the neurons
        based on their net_in values.

        NetAct (also see notebook):
        - 1 for neuron that achieves highest net_in value to sample i
        - -delta for neuron that achieves kth highest net_in value to sample i
        - 0 for all other neurons

        Parameters:
        -----------
        net_in: ndarray. shape=(B, H)

        Returns:
        -----------
        netAct: ndarray. shape=(B, H)

        Tips:
        - It might be helpful to think of competition as an assignment operation.
        - Remember arange indexing? It might be useful depending on your implementation strategy.
        - No loops should be needed.
        '''
        netAct = np.zeros(net_in.shape)

        # I'm sorry for bad variables but I think the code miht be worse. Sorts net_in to find minimum values and
        # then subtracts them to put zeros there. Then I find index of zeros to set equal to inhib_value
        broIDK = net_in-(np.sort(net_in, axis=1)[:, -self.kth_place_inhibited]).reshape((net_in.shape[0], 1))
        netAct[np.arange(net_in.shape[0]), np.where(broIDK == 0)[1]] = -self.inhib_value

        maxes = np.argmax(net_in, axis=1)
        netAct[np.arange(net_in.shape[0]), maxes] = 1
        return netAct

    def update_wts(self, x, net_in, net_act, lr, eps=1e-10, verbose=False):
        '''Update the Hebbian network wts according to a modified Hebbian learning rule (Oja's rule). After each wt
        update, normalize the wts by the largest wt (in absolute value). See notebook for equations.

        Parameters:
        -----------
        net_in: ndarray. shape=(B, H)
        net_act: ndarray. shape=(B, H)
        lr: float. Learning rate hyperparameter
        eps: float. Small non-negative number used in the wt normalization step to prevent possible division by 0.

        Tips:
        - This is definitely a scenario where you should the shapes of everything to guide you through and decide on the
        appropriate operation (elementwise multiplication vs matrix multiplication).
        '''
        # x shape: (B, M)
        # net_act shape: (B, H)
        # x_net_act shape: (M, H)
        x_net_act = x.T @ net_act
        if verbose:
            print(f'{x.shape = }')
            print(f'{net_act.shape = }')
            print(f'{x_net_act.shape = }')

        # net_in shape: (B, H)
        # net_act shape: (B, H)
        # net_scalar shape: (H,)
        combined_net = net_in * net_act
        net_scalar = np.sum(combined_net, axis=0)
        if verbose:
            print(f'{net_in.shape = }')
            print(f'{net_act.shape = }')
            print(f'{combined_net.shape = }')
            print(f'{net_scalar.shape = }')

        # weight_activation shape: (M, H)
        weight_activation = self.wts * net_scalar
        if verbose:
            print(f'{self.wts.shape = }')
            print(f'{net_scalar.shape = }')
            print(f'{weight_activation.shape = }')

        # delta_w shape: (M, H)
        delta_w = x_net_act - weight_activation

        abs_max_weight_change = np.max(np.abs(delta_w))
        delta_w_scalar = lr / (abs_max_weight_change + eps)
        self.wts += delta_w_scalar * delta_w

    def save_wts(self):
        np.save(self.saved_wts_path, self.wts)

    def fit(self, x, n_epochs=1, mini_batch_sz=128, lr=2e-2, plot_wts_live=False, fig_sz=(9, 9), n_wts_plotted=(10, 10),
            print_every=1, save_wts=True):
        '''Trains the Hebbian network on the training samples `x` using unsupervised Hebbian learning (no y classes required!).

        Parameters:
        -----------
        x: ndarray. shape=(N, M). Data samples.
        n_epochs: int. Number of epochs to train the network.
        mini_batch_sz: float. Learning rate used with Adam optimizer.
        lr: float. Learning rate used with Hebbian weight update rule
        plot_wts_live: bool. Whether to plot the weights and update throughout training every `print_every` epochs.
        save_wts: bool. Whether to save the Hebbian network wts (to self.saved_wts_path) after training finishes.

        TODO:
        Very similar workflow to usual:
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - Compute forward pass for each mini-batch then update the weights.
        - If plotting the wts on the current epoch, update the plot (via `draw_grid_image`) to show the current wts.
        - Print out which epoch we are on `print_every` epochs
        - When training is done, save the wts if `save_wts` is True.
        '''
        N, M = x.shape

        if plot_wts_live:
            fig = plt.figure(figsize=fig_sz)

        for epoch_num in range(n_epochs):

            num_batches = int(np.floor(N/mini_batch_sz))
            for j in range(num_batches):
                start = time.time()
                mini_batch = np.random.choice(np.arange(N), size=(mini_batch_sz), replace=True)
                batch_data = x[mini_batch]

                net_in = self.net_in(batch_data)
                net_act = self.net_act(net_in)

                self.update_wts(batch_data, net_in, net_act, lr)

                total = time.time() - start
                if epoch_num % print_every == 0 and j == 0:
                    epochs_remaining = n_epochs - epoch_num
                    est_time = total * int(np.floor(N / mini_batch_sz)) * epochs_remaining
                    hours, remainder = divmod(est_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f'Epoch: {epoch_num}')
                    print(f"Estimated remaining time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

            if plot_wts_live:
                draw_grid_image(self.wts.T, n_wts_plotted[0], n_wts_plotted[1], title=f'Net receptive fields (Epoch {epoch_num})')
                fig.canvas.draw()

        if save_wts:
            self.save_wts()

#Extension Code
class GHA(HebbNet):

    def __init__(self, num_features, num_neurons):
        super().__init__(num_features, num_neurons)
        self.oldWts = None

    def update_wts(self, x, net_in, net_act, lr, eps=1e-10, verbose=False):
        '''Update the Hebbian network wts according to a modified Hebbian learning rule (Oja's rule). After each wt
        update, normalize the wts by the largest wt (in absolute value). See notebook for equations.

        Parameters:
        -----------
        net_in: ndarray. shape=(B, H)
        net_act: ndarray. shape=(B, H)
        lr: float. Learning rate hyperparameter
        eps: float. Small non-negative number used in the wt normalization step to prevent possible division by 0.

        Tips:
        - This is definitely a scenario where you should the shapes of everything to guide you through and decide on the
        appropriate operation (elementwise multiplication vs matrix multiplication).
        '''
        self.oldWts = self.wts
        # x shape: (B, M)
        # net_act shape: (B, H)
        # x_net_act shape: (M, H)
        x_net_act = x.T @ net_act
        if verbose:
            print(f'{x.shape = }')
            print(f'{net_act.shape = }')
            print(f'{x_net_act.shape = }')

        # delta_w shape: (M, H)
        delta_w = lr*(x_net_act - self.wts@np.tril(net_act.T@net_act))
        print(np.sum(self.wts/np.max(self.wts, axis = 1), axis = 1))

        self.wts += delta_w

    def loss(self):
        return np.linalg.norm(self.wts, order = 'fro')-np.linalg.norm(self.oldWts, order = 'fro')

    def fit(self, x, n_epochs=100, mini_batch_sz=1, lr=2e-2, threshold = 0.1, print_every=1, ):
        '''Trains the Hebbian network on the training samples `x` using unsupervised Hebbian learning (no y classes required!).

        Parameters:
        -----------
        x: ndarray. shape=(N, M). Data samples.
        n_epochs: int. Number of epochs to train the network.
        mini_batch_sz: float. Learning rate used with Adam optimizer.
        lr: float. Learning rate used with Hebbian weight update rule
        plot_wts_live: bool. Whether to plot the weights and update throughout training every `print_every` epochs.
        save_wts: bool. Whether to save the Hebbian network wts (to self.saved_wts_path) after training finishes.

        TODO:
        Very similar workflow to usual:
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - Compute forward pass for each mini-batch then update the weights.
        - If plotting the wts on the current epoch, update the plot (via `draw_grid_image`) to show the current wts.
        - Print out which epoch we are on `print_every` epochs
        - When training is done, save the wts if `save_wts` is True.
        '''
        N = x.shape[-2]
        M = x.shape[-1]

        for epoch_num in range(n_epochs):

            num_batches = int(np.floor(N/mini_batch_sz))
            for j in range(num_batches):
                start = time.time()
                mini_batch = np.random.choice(np.arange(N), size=(mini_batch_sz), replace=True)
                batch_data = x[mini_batch]

                net_in = self.net_in(batch_data)
                net_act = self.net_act(net_in)

                self.update_wts(batch_data, net_in, net_act, lr)

                total = time.time() - start
                if epoch_num % print_every == 0 and j == 0:
                    epochs_remaining = n_epochs - epoch_num
                    est_time = total * int(np.floor(N / mini_batch_sz)) * epochs_remaining
                    hours, remainder = divmod(est_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f'Epoch: {epoch_num}')
                    print(f"Estimated remaining time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
