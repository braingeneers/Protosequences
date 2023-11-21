import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import mat73


# Find the group that a peak belongs to
def which_group(groups, burst_index):
    for i, array in enumerate(groups):
        if burst_index in array:
            return i  # Found the group index
    return -1  # Burst not found in any group


def main(path, FileIndex, groups, x_min, x_max):
    # Loading .mat data
    try:
        data = loadmat(path)
        peaks = data['tburst'][0]
        smoothed = data['pop_rate'].T[0]
    except:
        data = mat73.loadmat(path)
        peaks = data['tburst']
        smoothed = data['pop_rate'].T
    peaks = np.array([int(peaks[i]) for i in range(len(peaks))])

    # extract peak points to a new array for plotting
    smoothed_peaks = [[], []]
    x_data = np.arange(0, smoothed.shape[0]/1000, 0.001)
    for i in range(peaks.size):
        smoothed_peaks[0].append(x_data[peaks[i]])
        smoothed_peaks[1].append(smoothed[peaks[i]])

    # Find the group that a peak belongs to
    types = []
    for i in range(peaks.shape[0]):
        types.append(which_group(groups, i))
    types = np.array(types)

    # Plotting
    plt.plot(x_data, smoothed)
    plt.scatter(smoothed_peaks[0], smoothed_peaks[1], marker='x', color='orange')  # mark the peak of bursts
    for i in range(types.shape[0]):  # Label the type of each burst
        color = 'red'
        plt.annotate(str(types[i]+1), (smoothed_peaks[0][i], smoothed_peaks[1][i]), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=14, color=color)

    plt.xlabel('Time(s)', fontsize=16)
    plt.ylabel('Population rate (kHz)', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.xlim(x_min, x_max)
    plt.ylim(0, 2.49)
    plt.legend()
    plt.savefig('PopulationRate_Labeled.png')
    plt.show()


if __name__ == "__main__":
    main(None, None, None, None, None)
