import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
import math
from scipy.io import loadmat
import random
import mat73


# Define function for k-means clustering
def kmeans_clustering(data, k, seed):
    # Perform k-means clustering with the specified number of clusters (k)
    print('actual++')
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=seed)
    kmeans.fit(data)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Group the indices based on the cluster labels
    grouped_indices = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        grouped_indices[label].append(i)

    return grouped_indices


# Define function to plot heatmap in a square format
def ax_heatmap_square(data, rows, x_min, x_max, len, burst_displayed, ax):
    # Plot the heatmap
    cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', ['black', 'orange', 'white'])

    # heatmap = ax.imshow(data, cmap=cmap, extent=[-250, 500, rawspike_burst[0].shape[0], 0])
    heatmap = ax.imshow(data, cmap=cmap, vmin=0, vmax=np.amax(data))

    # Adjust aspect ratio to make the heatmap square
    ax.set_aspect((x_max - x_min) / rows / 2 / (len/burst_displayed))  # 1800 is the width, and 52 is the height of the matrix

    return heatmap


def main(path, FileIndex, File_to_plot, k, seeds, rawspike_burst, burst_displayed):
    # Load data from a MATLAB .mat file
    try:
        data = loadmat(path)
        ordering = np.array(data['mean_rate_ordering'])[0]-1
    except:
        data = mat73.loadmat(path)
        ordering = np.array(data['mean_rate_ordering'])-1
    ordering = ordering[::-1]
    ordering = ordering.astype(int)
    backbone_units = np.array(data['scaf_units'])
    backbone_number = backbone_units.shape[0]

    rawspike_burst = [rawspike_burst[i][ordering] for i in range(len(rawspike_burst))]

    # Calculate the pairwise correlation between each cell in each burst
    rawspike_burst_corr = []
    cells = rawspike_burst[0].shape[0]
    I = np.identity(cells)  # identity matrix length of number of cells
    for element in rawspike_burst:  # take the lower triangle of the corr matrix to avoid double counting
        corr_matrix = np.corrcoef(element, rowvar=True)  # treat each row as a datapoint
        corr_matrix = np.nanmax([I, corr_matrix], axis=0)  # ignoring the NaN

        # Extract the lower triangle of the matrix (excluding diagonal)
        mask = np.tri(cells, k=-1, dtype=bool)
        corr_matrix[~mask] = -1
        corr_array = corr_matrix.reshape(corr_matrix.shape[0], -1)
        corr_array = corr_array[corr_array != -1]

        rawspike_burst_corr.append(corr_array)

    # Apply KMeans clustering on the burst data based on their correlation matrix
    seed = seeds[k-2]
    print(seed)
    groups = kmeans_clustering(rawspike_burst_corr, k=k, seed=seed)
    groups = sorted(groups, key=lambda row: row[0] if row else 0)  # sort the groups based on the position of the first burst in each group
    print(groups)


    # Combine the rawspike time windows for bursts that are grouped together
    sep_value = 0
    column_x = np.full((cells, 1), sep_value)  # separate the arrays with a column of value 0
    grouped_combined_rawspike = []  # k elements: each element concatenated rawspike with dim cells*time
    combined_pos = []
    for i in range(k):
        temp_combined = column_x
        temp_pos = []
        for j in range(len(groups[i])):
            temp_pos.append(temp_combined.shape[1])
            temp_combined = np.concatenate((temp_combined, column_x, rawspike_burst[groups[i][j]]), axis=1)
        combined_pos.append(temp_pos)
        grouped_combined_rawspike.append(temp_combined)
    # print(combined_pos)

    # Plot example heatmaps
    if FileIndex == File_to_plot:
        for i in range(k):
            fig, ax = plt.subplots(1, 1)
            num_burst = len(groups[i])
            if num_burst <= burst_displayed:
                data = grouped_combined_rawspike[i]
                ax.set_title(
                    'Group ' + str(i + 1) + '; Displayed bursts: ' + str(num_burst) + '/' + str(num_burst))
            else:
                data = grouped_combined_rawspike[i][:, 0:(combined_pos[i][burst_displayed])]
                ax.set_title(
                    'Group ' + str(i + 1) + '; Displayed bursts: ' + str(burst_displayed) + '/' + str(num_burst))
            num_columns = data.shape[1]
            ax_heatmap_square(data, rows=cells, x_min=0,
                              x_max=num_columns, len=min(burst_displayed, len(groups[i])), burst_displayed=burst_displayed, ax=ax)
            for j in range(min(num_burst, burst_displayed)):
                ax.axvline(combined_pos[i][j], linestyle='--', linewidth=0.8, color='white')
                ax.axhline(backbone_number-0.5, linestyle='--', linewidth=1, color='aqua')
            ax.set_xlabel('Time (ms)')
            plt.savefig('Group'+str(i+1)+'_heatmap.png')
            plt.show()
            plt.close(fig)


        # Plot the PCA data in 2D
        rawspike_burst_corr = np.array(rawspike_burst_corr)
        print('Original Shape: ' + str(rawspike_burst_corr.shape[0]) + ' bursts * ' +
              str(rawspike_burst_corr.shape[1]) + ' correlations')
        pca = PCA(n_components=2)  # reduce dimension to 2D
        reduced_corr = pca.fit_transform(rawspike_burst_corr)
        print('Reduced Shape: ' + str(reduced_corr.shape[0]) + ' bursts * ' +
              str(reduced_corr.shape[1]) + ' correlations')
        variance = pca.explained_variance_ratio_
        for i, ratio in enumerate(variance):
            print(f"Variance explained by PC{i + 1}: {ratio:.4f}")
        letters = np.arange(1, 1+k, 1)
        PCA_groups = []
        for i in range(len(groups)):
            this_group = np.array(reduced_corr[groups[i]])
            PCA_groups.append(this_group)

            plt.scatter(this_group[:, 0], this_group[:, 1], label='Group ' + str(letters[i]))

        plt.xlabel("First PC Varianced Explained: " + str(round(variance[0], 2)), fontsize=16)
        plt.ylabel("Second PC Varianced Explained: " + str(round(variance[1], 2)), fontsize=16)
        plt.legend()
        plt.savefig('PCA.png')
        plt.show()


    # Calculate data for boxplots
    boxplot_data = []
    xlabels = []
    rs = grouped_combined_rawspike
    std = np.std(np.array([np.mean(rs[i], axis=1) for i in range(len(rs))]), axis=0).reshape(-1, 1)
    mean = np.mean(np.array([np.mean(rs[i], axis=1) for i in range(len(rs))]), axis=0).reshape(-1, 1)
    temp = np.divide(std, mean, where=mean != 0, out=np.full_like(std, np.nan))
    boxplot_data.append(temp[:backbone_number, :].reshape(-1))
    boxplot_data.append(temp[backbone_number:, :].reshape(-1))

    return boxplot_data, groups, grouped_combined_rawspike


if __name__ == "__main__":
    main(None, None, None, None, None, None, None)
