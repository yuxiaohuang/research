# Please cite the following paper when using the code


# Modules
from __future__ import division
import numpy as np
from scipy import stats
import sys
import os
import csv
import math
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import _tree

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list
#_F       : indicates the variable is a flag


# Global variables


# Generate histogram
def generate_histogram():
    # Get the list of ground truth
    ground_truth_L = []
    # Load the interaction_ground_truth file
    with open(interaction_ground_truth_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))
        # Get the target, probability and interaction_ground_truth
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            component_num = (len(spamreader[i]) - 2) // 3

            for j in range(component_num):
                # Name
                name = spamreader[i][j * 3 + 2].strip()
                # If condition
                if '_' in name.replace('src_', ''):
                    # Remove the value
                    name = name[:-2]
                ground_truth_L.append(name)

    # Get the list of attributes and their importance
    attribute_L = []
    importance_L = []

    with open(importance_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        # From the first line to the last
        for i in range(len(spamreader)):
            attribute = spamreader[i][0].strip()
            importance = float(spamreader[i][1].strip())
            attribute_L.append(attribute)
            importance_L.append(importance)

    # Generate histogram
    plt.xlabel('Number of attributes', fontsize = 25)
    plt.ylabel('Importance', fontsize = 25)

    barlist = plt.bar(range(1, len(importance_L) + 1),
            importance_L,
            color='lightblue',
            align='center')
    for i in range(len(attribute_L)):
        if not attribute_L[i] in ground_truth_L:
            barlist[i].set_color('r')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        top='off')  # ticks along the top edge are off

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        right='off')  # ticks along the top edge are off

    # # Rename attributes
    # for i in range(len(attribute_L)):
    #     attribute_L[i] = attribute_L[i].replace('src_', '$x$=')

    plt.xticks([1, 25, 50, len(importance_L)], fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.xlim([0, len(importance_L) + 1])
    plt.tight_layout()
    plt.savefig(histogram_file, dpi=300)
    plt.close()


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    interaction_ground_truth_file = sys.argv[1]
    importance_file = sys.argv[2]
    histogram_file = sys.argv[3]

    directory = os.path.dirname(histogram_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate histogram
    generate_histogram()
