import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.io import loadmat
import mat73


# Function to extract columns from the matrix based on given indices
def extract_columns(data, indices):
    extracted_columns = []

    for index_pair in indices:
        start_index, end_index = index_pair
        extracted_columns.append(data[:, int(start_index):int(end_index) + 1])

    return extracted_columns


# Function to perform KMeans clustering and return sum of squared distances
def kmeans_clustering_elbow(data, k, random_state):
    # Perform k-means clustering with the specified number of clusters (k)
    # np.random.seed(42)

    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=random_state)
    kmeans.fit(data)

    # Get the cluster labels for each data point
    # labels = kmeans.labels_
    dist = kmeans.inertia_
    # print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff

    return dist


# Function to perform KMeans clustering and return silhouette coefficient
def kmeans_clustering_sil_coef(data, k, random_state):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=random_state)
    kmeans.fit(data)

    # Get the cluster labels for each data point
    labels = kmeans.labels_
    sil_coeff = silhouette_score(data, labels, metric='euclidean')

    return sil_coeff


def main(path, max_seed, clusters):
    # Load data from a MATLAB .mat file
    try:
        data = loadmat(path)
        peaks = data['tburst'][0]
    except:
        data = mat73.loadmat(path)
        peaks = data['tburst']
    rawspike_smoothed = np.array(data['rate_mat_burst_only']).T

    # Extract burst regions using peaks
    start_end = [[peaks[i]-250, peaks[i]+500] for i in range(peaks.shape[0])]
    start_end = np.round(np.array(start_end))
    print(start_end)
    rawspike_burst = extract_columns(rawspike_smoothed, start_end)

    # Calculate pairwise correlation for each cell in each burst
    rawspike_burst_corr = []
    cells = rawspike_burst[0].shape[0]
    I = np.identity(cells)  # identity matrix length of number of cells
    for element in rawspike_burst:  # take the lower triangle of the corr matrix to avoid double counting
        corr_matrix = np.corrcoef(element, rowvar=True)  # treat each row as a datapoint
        corr_matrix = np.nanmax([I, corr_matrix], axis=0)  # ignoring the NaN

        # Extract lower triangle values
        mask = np.tri(cells, k=-1, dtype=bool)
        corr_matrix[~mask] = -1

        # Reshape the lower triangle to 2D (flatten each row to a 1D vector)
        corr_array = corr_matrix.reshape(corr_matrix.shape[0], -1)
        corr_array = corr_array[corr_array != -1]

        rawspike_burst_corr.append(corr_array)

    # Testing different seeds for KMeans clustering to find the best one
    seeds_elbow = []
    dist_s = []
    seeds_sil = []
    sil_s = []
    max_cluster = min(clusters+1, len(peaks))
    fig, ax = plt.subplots(3, 3)
    selection = np.random.permutation(max_cluster-2) + 2  # randomly order integers from 2 to max_cluster
    selection = selection[:9]  # randomly select k to display sse vs seed
    print("Number of Groups selection: " + str(selection))
    count = 0
    for k in range(2, max_cluster):
        dist = []
        sil = []
        for seed in range(0, max_seed):
            dist.append(kmeans_clustering_elbow(rawspike_burst_corr, k=k, random_state=seed))
            sil.append(kmeans_clustering_sil_coef(rawspike_burst_corr, k=k, random_state=seed))
            print(str(k)+'/'+str(max_cluster), seed)

        seeds_elbow.append(dist.index(min(dist)))
        dist_s.append(min(dist))

        seeds_sil.append(sil.index(max(sil)))
        sil_s.append(max(sil))

        if k in selection:
            ax.flat[count].plot(np.arange(0, max_seed, 1), dist)
            ax.flat[count].set_xlabel("Seed number", fontsize=14)
            ax.flat[count].set_ylabel("Sum of Square Errors", fontsize=14)
            ax.flat[count].set_title("Number of clusters(k) = " + str(k), fontsize=16)
            count += 1

    plt.tight_layout()
    # plt.show()
    plt.close(fig)

    for i in range(2, max_cluster):  # Print best seeds for clustering
        print("k="+str(i), "SSE="+str(round(dist_s[i-2])), "Sil-Coef="+str(round(sil_s[i-2], 2)),
              "Seed_SSE= "+str(seeds_elbow[i-2]), "Seed_Sil-Coef= "+str(seeds_sil[i-2]))

    # Plotting the elbow and silhouette plots
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(2, max_cluster, 1), dist_s)
    ax[0].set_xlabel("Number of cluster", fontsize=16)
    ax[0].set_ylabel("Sum of Square Errors", fontsize=16)
    ax[1].plot(np.arange(2, max_cluster, 1), sil_s)
    ax[1].set_xlabel("Number of cluster", fontsize=16)
    ax[1].set_ylabel("Silhouette Coefficient", fontsize=16)
    plt.tight_layout()
    # plt.savefig(FileIndex + '.png')
    plt.show()

    return seeds_elbow, rawspike_burst


if __name__ == "__main__":
    main(None, None, None)
