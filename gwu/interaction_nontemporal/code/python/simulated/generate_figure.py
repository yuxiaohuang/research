
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
    precision_1_L = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    recall_1_L = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_1_L = range(len(recall_1_L))
    precision_2_L = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1111111111111111, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1111111111111111, 0.125, 0.125, 0.14285714285714285, 0.125, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1111111111111111, 0.1, 0.1111111111111111, 0.125, 0.1111111111111111, 0.1111111111111111, 0.125, 0.125, 0.14285714285714285, 0.125, 0.125, 0.14285714285714285, 0.125, 0.14285714285714285, 0.125, 0.125, 0.1111111111111111, 0.125, 0.2222222222222222, 0.25, 0.25, 0.2857142857142857, 0.2857142857142857, 0.375, 0.375, 0.42857142857142855, 0.5, 0.5555555555555556, 0.5555555555555556, 0.625, 0.7142857142857143, 0.8333333333333334, 1.0]
    recall_2_L = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    x_2_L = range(len(recall_2_L))

    # Plot two subfigures
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    plt.ylim(0, 1.02)

    ax1.plot(x_1_L, precision_1_L, 'r-', label = 'Precision', linewidth=2.0)
    ax1.plot(x_1_L, recall_1_L, 'b--', label = 'Recall', linewidth=2.0)
    ax1.legend(bbox_to_anchor=(0.4, 0.9),
           bbox_transform=plt.gcf().transFigure)

    plt.sca(ax1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(range(6))
    plt.xlabel('Number of steps')

    plt.sca(ax2)
    plt.xticks([70, 90, 110, 130])
    plt.xlabel('Number of steps')

    ax2.plot(x_2_L[70:], precision_2_L[70:], 'r-', label = 'Precision', linewidth=2.0)
    ax2.plot(x_2_L[70:], recall_2_L[70:], 'b--', label = 'Recall', linewidth=2.0)
    ax2.legend(bbox_to_anchor=(0.85, 0.9),
           bbox_transform=plt.gcf().transFigure)

    fig.tight_layout()
    plt.savefig(figure_file)


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    figure_file = sys.argv[1]

    # Make directory
    directory = os.path.dirname(figure_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate figure
    generate_figure()

