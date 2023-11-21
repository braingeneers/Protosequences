import ExtractBurst_ElbowSeeds
import Group_AvgStdColumn_BoxSaved
import PR_LabeledBursts
import Column_Heatmap
import Boxplot
import CV_Score_csv

FileIndices = ['Or1', 'Or2', 'Or3', 'Or4', 'Or5', 'Or6', 'Or7', 'Or8']
input_data_path = ''
csv_save_path = ''

number_of_groups = [3, 2, 4, 5, 4, 3, 10, 4]  # Number of clusters for each dataset
File_to_plot = 'Or5'  # The dataset to plot PCA, population rate, and heatmaps
burst_displayed = 8  # Number of example bursts displayed in each group for heatmaps
max_seed = 10   # Number of seeds to try for each number of groups
max_cluster = 15  # Number of groups to try for each dataset
x_min = 1110  # lower time range of population rate graph
x_max = 1275  # upper time range of population rate graph


boxplot_data = []
for FileIndex, k in zip(FileIndices, number_of_groups):
    path = input_data_path + '/' + FileIndex + '_single_recording_metrics.mat'
    seeds_elbow, rawspike_burst = ExtractBurst_ElbowSeeds.main(path, max_seed, max_cluster)
    boxplot, groups, grouped_combined_rawspike = Group_AvgStdColumn_BoxSaved.main(path, FileIndex, File_to_plot, k,
                                                                                  seeds_elbow, rawspike_burst,
                                                                                  burst_displayed)

    boxplot_data.extend(boxplot)
    if FileIndex == File_to_plot:
        PR_LabeledBursts.main(path, FileIndex, groups, x_min, x_max)
        Column_Heatmap.main(path, grouped_combined_rawspike)

Boxplot.main(boxplot_data, FileIndices)
CV_Score_csv.main(input_data_path, csv_save_path, boxplot_data, FileIndices)
