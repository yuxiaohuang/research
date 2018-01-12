
# Please cite the following paper when using the code


# Modules
import sys
import os
import csv
import numpy as np
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Delimiter type
delimiter_type = ','

# Flag, indicating whether there is a header (1, yes; 0, no)
header = 0

# The row number
row_num = 0

# The column number
col_num = 0

# Global variables
# The column of class
class_col = -1

# The columns of continuous features
# con_feature_col_L = range(13)
con_feature_col_L = range(4)

# The list of number of bins
bins_num_L = []

# The number of bins
bins_num = 3

# The columns of features that should be excluded
exclude_feature_col_L = [0, 1]

# The character for missing values
missing_char = '?'

# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_Dic = {}

# The dictionary of the list of cutoffs
# key: var index
# val: the list of cutoffs
cutoff_L_Dic = {}

# The dictionary of discretized values of continuous features
# key: var
# val: discretized value of continuous features
con_feature_val_L_Dic = {}

# The list of class labels
class_L = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# The index of the two attributes
x1_idx = 2
x2_idx = 3

# Generate figure
def generate_figure():
    # Get con_feature_val_L_Dic
    for col in con_feature_col_L:
        if len(bins_num_L) != 0:
            con_feature_val_L_Dic[col] = range(bins_num_L[col])
        else:
            con_feature_val_L_Dic[col] = range(bins_num)

    # Load the raw file
    with open(raw_file, 'r') as f:
        try:
            spamreader = list(csv.reader(f, delimiter = delimiter_type, skipinitialspace = True))

            # The row number
            global row_num
            row_num = 150

            # The column number
            global col_num
            col_num = len(spamreader[0])

            # Global variables
            # The column of class
            global class_col
            if class_col == -1:
                class_col = col_num - 1

            # The number of rows containing missing values
            missing_row_num = 0

            # Get val_Dic
            for i in range(header, row_num):
                # Initialization
                if not i - header - missing_row_num in val_Dic:
                    val_Dic[i - header - missing_row_num] = {}

                # Get val_Dic
                for j in range(col_num):
                    # Exclude the features suggested by the contributor of the dataset
                    if j in exclude_feature_col_L:
                        continue

                    val_j = spamreader[i][j].strip()

                    # If not missing
                    if val_j != missing_char:
                        if j == class_col:
                            val_Dic[i - header - missing_row_num][j] = class_L.index(val_j)
                        else:
                            val_Dic[i - header - missing_row_num][j] = float(val_j)
                    else:
                        del val_Dic[i - header - missing_row_num]
                        missing_row_num += 1
                        break

            # Get cutoff_L_Dic
            for j in range(col_num):
                # Exclude the features suggested by the contributor of the dataset
                if j in exclude_feature_col_L:
                    continue

                # Get the list of value
                val_L = []
                # If continuous feature
                if j in con_feature_col_L:
                    for i in sorted(val_Dic.keys()):
                        val = float(val_Dic[i][j])
                        val_L.append(val)

                    # Get the list of discretized value
                    bin_num = len(con_feature_val_L_Dic[j])
                    discretize(val_L, bin_num, j)

            # plot decision regions
            plot_decision_regions()

        except UnicodeDecodeError:
            print("UnicodeDecodeError when reading the following file!")
            print(raw_file)


# Discretize val_L into bin_num bins
def discretize(val_L, bin_num, idx):
    split_L = np.array_split(np.sort(val_L), bin_num)
    cutoff_L = [split[-1] for split in split_L]
    cutoff_L = cutoff_L[:-1]
    cutoff_L_Dic[idx] = cutoff_L


def plot_decision_regions():
    # plot the decision surface
    x1_L = [val_Dic[time][x1_idx] for time in sorted(val_Dic.keys())]
    x2_L = [val_Dic[time][x2_idx] for time in sorted(val_Dic.keys())]
    x1_min, x1_max = min(x1_L) - 0.2, max(x1_L) + 0.2
    x2_min, x2_max = min(x2_L) - 0.2, max(x2_L) + 0.2
    # x1_L, x2_L = np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    #
    # # get the discretized value
    # x1_dis_L = np.digitize(x1_L, cutoff_L_Dic[x1_idx], right = True)
    # x2_dis_L = np.digitize(x2_L, cutoff_L_Dic[x2_idx], right = True)
    #
    # # get z_L
    # z_L = []
    # for x1_val in x1_dis_L:
    #     for x2_val in x2_dis_L:
    #         if x1_val == 0:
    #             z_val = 0
    #         elif x1_val == 2 or x2_val == 2:
    #             z_val = 2
    #         else:
    #             z_val = 1
    #         z_L.append(z_val)
    #
    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                        np.arange(x2_min, x2_max, resolution))
    # zz = np.asarray(z_L).reshape(xx1.shape)

    # plt.contourf(xx1, xx2, zz, alpha = 0.4, cmap = cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel('Petal length (cm)', fontsize = 25)
    plt.ylabel('Petal width (cm)', fontsize = 25)
    plt.yticks([0, 0.5, 1.0, 1.6, 2.0, 2.5], fontsize = 25)
    plt.xticks([1, 2, 3, 4, 4.9, 6, 7], fontsize = 25)

    x1_setosa_L = []
    x2_setosa_L = []
    x1_versicolor_L = []
    x2_versicolor_L = []
    x1_virginica_L = []
    x2_virginica_L = []

    for time in sorted(val_Dic.keys()):
        x1_val = val_Dic[time][x1_idx]
        x2_val = val_Dic[time][x2_idx]
        class_val = val_Dic[time][class_col]
        if class_val == 0:
            x1_setosa_L.append(x1_val)
            x2_setosa_L.append(x2_val)
        elif class_val == 1:
            x1_versicolor_L.append(x1_val)
            x2_versicolor_L.append(x2_val)
        elif class_val == 2:
            x1_virginica_L.append(x1_val)
            x2_virginica_L.append(x2_val)

    plt.scatter(x = x1_setosa_L,
                y = x2_setosa_L,
                alpha = 0.6,
                c = 'blue',
                edgecolor = 'blue',
                marker = 's',
                label = 'Setosa',
                s=[200 for i in range(len(x1_setosa_L) + len(x2_setosa_L))])

    plt.scatter(x = x1_versicolor_L,
                y = x2_versicolor_L,
                alpha = 0.6,
                c = 'blue',
                edgecolor = 'blue',
                marker = '*',
                label = 'Versicolor',
                s=[200 for i in range(len(x1_versicolor_L) + len(x2_versicolor_L))])

    plt.scatter(x = x1_virginica_L,
                y = x2_virginica_L,
                alpha = 0.6,
                c = 'green',
                edgecolor = 'green',
                marker = 'o',
                label = 'Virginica',
                s = [200 for i in range(len(x1_virginica_L) + len(x2_virginica_L))])

    print(cutoff_L_Dic[2][1])
    print(cutoff_L_Dic[3][1])

    plt.plot((cutoff_L_Dic[2][1], cutoff_L_Dic[2][1]), (x2_min, cutoff_L_Dic[3][1]), color = 'red', linestyle = '--', linewidth=5.0)
    plt.plot((x1_min, cutoff_L_Dic[2][1]), (cutoff_L_Dic[3][1], cutoff_L_Dic[3][1]), color = 'red', linestyle = '--', linewidth=5.0)
    plt.legend(loc="upper left", fontsize=25)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        top='off')  # ticks along the top edge are off

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        right='off')  # ticks along the top edge are off

    plt.tight_layout()
    plt.savefig(figure_file)


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_file = sys.argv[1]
    figure_file = sys.argv[2]

    # Make directory
    directory = os.path.dirname(figure_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate figure
    generate_figure()

