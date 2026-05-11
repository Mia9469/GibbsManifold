import numpy as np
import matplotlib.pyplot as plt


def plot_rf(rf, out_dim, M, ind=None):
    rf = rf.reshape(out_dim, -1)
    # normalize
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(out_dim, M, M)
    # plotting
    n = int(np.ceil(np.sqrt(rf.shape[0])))
    # fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    # fig.set_size_inches(10, 10)

    # for i in ind:
    #     ax = axes[i // n][i % n]
    #     ax.imshow(rf[i], cmap='gray', vmin=-1, vmax=1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_aspect('equal')
    # n_ind = set(range(rf.shape[0]))-set(ind)
    # for j in n_ind:
    #     ax = axes[j // n][j % n]
    #     ax.imshow(np.ones_like(rf[0]) * -1, cmap='gray', vmin=-1, vmax=1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_aspect('equal')
    # fig.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    # fig.set_size_inches(10, 10)
    # for i in n_ind:
    #     ax = axes[i // n][i % n]
    #     ax.imshow(rf[i], cmap='gray', vmin=-1, vmax=1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_aspect('equal')
    # for j in ind:
    #     ax = axes[j // n][j % n]
    #     ax.imshow(np.ones_like(rf[0]) * -1, cmap='gray', vmin=-1, vmax=1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_aspect('equal')
    # fig.subplots_adjust(wspace=0.0, hspace=0.0)
    
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(10, 10)
    for i in range(rf.shape[0]):
        ax = axes[i // n][i % n]
        ax.imshow(rf[i], cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    
    
    return fig
