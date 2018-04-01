
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


# Generate figure
def generate_figure():
    # The list of precision and recall for interactions in 'interaction_data_5_0.txt'
    precision_1_L = [1.0, 1.0, 1.0, 1.0, 1.0]
    recall_1_L = [0.2, 0.4, 0.6, 0.8, 1.0]
    x_1_L = range(1, len(recall_1_L) + 1)

    precision_2_L = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.125, 0.14285714285714285, 0.125, 0.1111111111111111, 0.1, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.125, 0.14285714285714285, 0.125, 0.1111111111111111, 0.125, 0.14285714285714285, 0.125, 0.14285714285714285, 0.125, 0.1111111111111111, 0.125, 0.1111111111111111, 0.125, 0.2222222222222222, 0.25, 0.2222222222222222, 0.25, 0.2857142857142857, 0.25, 0.2857142857142857, 0.375, 0.3333333333333333, 0.375, 0.42857142857142855, 0.5, 0.5555555555555556, 0.5, 0.5555555555555556, 0.625, 0.7142857142857143, 0.8333333333333334, 1.0]
    recall_2_L = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    x_2_L = range(1, len(recall_2_L) + 1)

    # Plot figure 1
    fig_1, ax1 = plt.subplots(1, 1, sharey = True)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    plt.ylim(-0.02, 1.02)
    plt.xlim(1, 5.1)

    ax1.plot(x_1_L, precision_1_L, 'b--', label = 'Precision', linewidth=5.0)
    ax1.plot(x_1_L, recall_1_L, 'r-', label = 'Recall', linewidth=5.0)
    ax1.legend(loc = 'upper left', fontsize = 25)

    plt.sca(ax1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 25)
    plt.xticks(range(1, 6), fontsize = 25)
    plt.xlabel('Number of steps', fontsize = 25)
    plt.ylabel('Precision and Recall', fontsize = 25)

    fig_1.tight_layout()
    plt.savefig(figure_file_1)

    # Plot figure 2
    fig_2, ax2 = plt.subplots(1, 1, sharey = True)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    plt.ylim(-0.02, 1.02)
    plt.xlim(1, 180)

    ax2.plot(x_2_L, precision_2_L, 'b--', label = 'Precision', linewidth=5.0)
    ax2.plot(x_2_L, recall_2_L, 'r-', label = 'Recall', linewidth=5.0)
    ax2.legend(loc = 'upper left', fontsize = 25)

    print(recall_2_L.index(0.2))
    print(recall_2_L.index(0.4))
    print(recall_2_L.index(0.6))
    print(recall_2_L.index(0.8))
    print(recall_2_L.index(1.0))

    plt.sca(ax2)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 25)
    # plt.xticks([recall_2_L.index(0.2) - 1, recall_2_L.index(0.4) - 1, recall_2_L.index(1.0) - 1, 130], fontsize = 25)
    plt.xticks([1, 30, 60, 90, 120, 150, 180], fontsize = 25)
    plt.xlabel('Number of steps', fontsize = 25)
    plt.ylabel('Precision and Recall', fontsize = 25)

    fig_2.tight_layout()
    plt.savefig(figure_file_2)


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    figure_file_1 = sys.argv[1]
    figure_file_2 = sys.argv[2]

    # Make directory
    directory = os.path.dirname(figure_file_1)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(figure_file_2)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate figure
    generate_figure()

