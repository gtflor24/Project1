'''viz.py
Plotting functions
Oliver W. Layton
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import numpy as np
import matplotlib.pylab as plt


def draw_grid_image(x, n_cols, n_rows, sample_dims=(28, 28, 1), title=''):
    '''Renders image data samples or wts (`data`) in a single canvas/image/plot.

    Parameters:
    -----------
    x: ndarray. Data samples or network wts to visualize.
        If passing in data samples: shape=(B, I_y, I_x, n_chans) or (B, I_y*I_x*n_chans) aka (B, M).
        If passing in network wts: shape=(num_neurons, I_y*I_x*n_chans) aka (B, M).
    n_rows: int. Number of samples to include vertically on the image canvas.
    n_cols: int. Number of samples to include horizontally on the image canvas.
    sample_dims: tuple. (I_y, I_x, n_chans). Shape of each data sample (or shape or one neuron's weights).
    title: str. Title to use in plot.

    NOTE: This visualization function is provided to you. No code changes should be needed.
    '''
    # reshape each sample into format: (N, n_rows, n_cols)
    x = x.reshape((len(x), sample_dims[0], sample_dims[1], sample_dims[2]))
    # select only the samples that fit into the grid
    x = x[np.arange(n_rows*n_cols)]

    # make an empty canvas on which we place the individual images
    canvas = np.zeros([sample_dims[0]*n_rows, sample_dims[1]*n_cols, sample_dims[2]])
    # (r,c) becomes the top-left corner
    for r in range(n_rows):
        for c in range(n_cols):
            ind = r*n_cols + c
            canvas[r*sample_dims[0]:(r+1)*sample_dims[0], c*sample_dims[1]:(c+1)*sample_dims[1]] = x[ind]

    # For live updating: clear current plot in figure
    plt.clf()

    max = np.max(np.abs(canvas))
    if sample_dims[2] == 1:
        im = plt.imshow(canvas, cmap='bwr', vmin=-max, vmax=max)
        fig = plt.gcf()
        fig.colorbar(im, ticks=[np.min(canvas), 0, np.max(canvas)])
    else:
        im = plt.imshow(canvas, vmin=-max, vmax=max)

    if title is not None:
        plt.title(title)

    plt.axis('off')
