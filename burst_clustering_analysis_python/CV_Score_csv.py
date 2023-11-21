import csv
import os
import pickle as pkl
from scipy.io import loadmat
import mat73
import numpy as np


# # Function to load data arrays from pickle files in a specified folder.
# def load_data_from_folder(folder_path):
#     data = []
#     labels = []
#     filenames = sorted(os.listdir(folder_path))
#     for file in filenames:
#         if file.endswith("boxplot.pkl"):
#             file_path = os.path.join(folder_path, file)
#             with open(file_path, 'rb') as f:
#                 arrays = pkl.load(f)
#                 data.extend(arrays)
#                 label = file[0:3] if file.startswith('L') else file[0:4]
#                 labels.append(label)
#     return data


# Function to generate rows for the CSV file from given data.
def generate_rows(folder_path, label, array1, array2):
    n = len(array1) + len(array2)
    used_label = label[0]+label[2] if label[1] == '0' else label
    sample_labels = [used_label] * n  # Column 1
    unit_numbers = [f"{used_label}_{i+1}" for i in range(n)]  # Column 2

    # Import ordering to retrieve the original unit ordering
    try:
        data = loadmat(folder_path + '/' + label + '_single_recording_metrics.mat')
        ordering = np.array(data['mean_rate_ordering'])[0] - 1
    except:
        data = mat73.loadmat(folder_path + '/' + label + '_single_recording_metrics.mat')
        ordering = np.array(data['mean_rate_ordering'])-1
    scaffold_units = np.array(data['scaf_units'])
    scaffold_number = scaffold_units.shape[0]

    # Reorder the scores to their original positions.
    scores = np.hstack([array1, array2])
    scores_original = [None] * len(scores)
    for idx, value in enumerate(ordering):
        scores_original[idx] = scores[round(value)]
    scaffold_or_not = ['n' if ordering[i] >= scaffold_number else 's' for i in range(len(ordering))]  # Column 3
    shuff_labels = ['o'] * n  # Column 4
    scores_original = [float(scores_original[i]) for i in range(len(scores_original))]  # Column 5
    return zip(sample_labels, unit_numbers, scaffold_or_not, shuff_labels, scores_original)


def main(folder_path, csv_save_path, CV, labels):
    # Prepare datasets for each label.
    datasets = {}
    for i in range(len(labels)):
        label = labels[i]
        datasets[label] = (CV[2*i], CV[2*i+1])
        if i == 0:
            print(datasets[label])

    # Write to CSV
    with open(csv_save_path + 'CV_Score.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_labels", "unit_numbers", 'scaf_labels', 'shuff_labels', 'CV'])  # Writing the headers

        # Write rows for each dataset.
        for label, (array1, array2) in datasets.items():
            rows = generate_rows(folder_path, label, array1, array2)
            writer.writerows(rows)
            print(label)


if __name__ == "__main__":
    main(None, None, None, None)
