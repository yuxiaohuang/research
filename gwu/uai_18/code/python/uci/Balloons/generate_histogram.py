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
    # Get the list of conditions and their importance
    condition_L = []
    importance_L = []

    with open(importance_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        # From the first line to the last
        for i in range(len(spamreader)):
            condition = spamreader[i][0].strip()
            importance = float(spamreader[i][1].strip())
            condition_L.append(condition)
            importance_L.append(importance)

    # Rename conditions
    for i in range(len(condition_L)):
        if 'src_0' in condition_L[i]:
            condition_L[i] = condition_L[i].replace('src_0_', '$color$=')
            condition_L[i] = condition_L[i].replace('PURPLE', 'purple')
            condition_L[i] = condition_L[i].replace('YELLOW', 'yellow')
        elif 'src_1' in condition_L[i]:
            condition_L[i] = condition_L[i].replace('src_1_', '$size$=')
            condition_L[i] = condition_L[i].replace('LARGE', 'large')
            condition_L[i] = condition_L[i].replace('SMALL', 'small')
        elif 'src_2' in condition_L[i]:
            condition_L[i] = condition_L[i].replace('src_2_', '$act$=')
            condition_L[i] = condition_L[i].replace('DIP', 'dip')
            condition_L[i] = condition_L[i].replace('STRETCH', 'stretch')
        elif 'src_3' in condition_L[i]:
            condition_L[i] = condition_L[i].replace('src_3_', '$age$=')
            condition_L[i] = condition_L[i].replace('ADULT', 'adult')
            condition_L[i] = condition_L[i].replace('CHILD', 'child')

    # Generate histogram
    plt.xlabel('Condition', fontsize = 25)
    plt.ylabel('Importance', fontsize = 25)

    barlist = plt.bar(range(len(importance_L)),
            importance_L,
            color='lightblue',
            align='center')
            #width = [0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.4, 0.4])
    for i in range(3):
        barlist[i].set_color('r')
    barlist[5].set_color('r')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        top='off')  # ticks along the top edge are off

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        right='off')  # ticks along the top edge are off

    plt.xticks(range(len(importance_L)),
               condition_L, rotation=90, fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.xlim([-1, len(importance_L)])
    plt.tight_layout()
    plt.savefig(histogram_file, dpi=300)
    plt.close()


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    importance_file = sys.argv[1]
    histogram_file = sys.argv[2]

    directory = os.path.dirname(histogram_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate histogram
    generate_histogram()
