import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math


def main(data, labels):
    # Remove NaN values from the loaded data.
    data = [[x for x in lst if not math.isnan(x)] for lst in data]
    # Define colors for the boxes in the boxplot.
    box_colors = ['lightyellow', 'darkgoldenrod'] * round(len(data)/2)

    # Create the boxplot with specific visual configurations.
    bp = plt.boxplot(data, patch_artist=True, medianprops={'color': 'black'})
    for box, color in zip(bp['boxes'], box_colors):
        box.set_facecolor(color)
    new_tick_positions = np.arange(1.5, len(data), 2)  # Set new positions for x-axis ticks.
    plt.xticks(new_tick_positions, labels, fontsize=14)
    plt.title("")
    plt.ylabel("CV score", fontsize=18)
    plt.savefig('Boxplot.png')
    plt.show()


if __name__ == "__main__":
    main(None, None)
