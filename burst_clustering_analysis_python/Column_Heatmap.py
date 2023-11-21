import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import scipy.io
import matplotlib.colors as colors
import os
import mat73
from scipy.io import loadmat


def main(path, rawspike_burst):
    # Load data from a MATLAB (.mat) file.
    try:
        data = loadmat(path)
    except:
        data = mat73.loadmat(path)
    backbone_units = np.array(data['scaf_units'])
    backbone_number = backbone_units.shape[0]


    # Create a heatmap of average firing rate for each group
    n = len(rawspike_burst)  # Number of groups.
    fig, ax = plt.subplots(1, n)
    cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', ['black', 'orange', 'white'])
    for i in range(n):
        data = np.mean(rawspike_burst[i], axis=1)
        data = data.reshape(-1, 1)

        heatmap = ax[i].imshow(data, aspect=0.2, cmap=cmap)
        ax.flat[i].axhline(backbone_number - 0.5, linestyle='--', linewidth=3, color='aqua')
        ax[i].set_title("Group "+str(i+1), fontsize=16)
        ax[i].set_xticklabels([])
        ax[0].set_ylabel("Cell Number", fontsize=16)

    # Add colorbar and show plot.
    colorbar = fig.colorbar(heatmap, pad=0.2)
    colorbar.set_label("Firing rate: kHz", fontsize=16)
    plt.tight_layout(w_pad=-10)
    plt.savefig('Avg_FR_columns.png')
    plt.show()
    plt.close(fig)


    # Create a heatmap of CV score between average firing rate of all groups
    fig, ax = plt.subplots(1, 1)
    cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', ['white', 'orange', 'black'])
    std = np.std(np.array([np.mean(rawspike_burst[i], axis=1) for i in range(len(rawspike_burst))]), axis=0).reshape(-1, 1)
    mean = np.mean(np.array([np.mean(rawspike_burst[i], axis=1) for i in range(len(rawspike_burst))]), axis=0).reshape(-1, 1)
    data = std / mean  # CV Score

    # Display heatmap.
    heatmap = ax.imshow(data, aspect=0.2, cmap=cmap)
    ax.axhline(backbone_number - 0.5, linestyle='--', linewidth=3, color='aqua')
    ax.set_title("All bursts", fontsize=16)
    ax.set_xticklabels([])
    ax.set_ylabel("Cell Number", fontsize=16)
    colorbar = fig.colorbar(heatmap, pad=0.2)
    colorbar.set_label("CV(coefficient of variance) score", fontsize=16)
    plt.tight_layout(w_pad=-1)
    plt.savefig('CV_column.png')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main(None, None, None)