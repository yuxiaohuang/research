

# Please cite the following paper when using the code


# Modules
import sys
import os
import csv
import numpy as np
import math
import random


# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Generate interaction
def generate_interaction():
    # Read source setting file
    with open(src_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get src_L
        src_L = []
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            src_L.append(spamreader[i][0].strip())

    # Read target setting file
    with open(tar_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get tar_L
        tar_L = []
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            tar_L.append(spamreader[i][0].strip())

    # Write interaction file
    with open(interaction_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(['target', 'probability', 'interaction'])

        for target in tar_L:
            # The interactions
            interaction_LLL = []

            # Get the interaction number
            interaction_num = random.randint(interaction_num_range_L[0], interaction_num_range_L[1])

            while len(interaction_LLL) < interaction_num:
                # Get the probability
                prob = random.uniform(prob_range_L[0], prob_range_L[1])

                # Get the interaction
                interaction_LL = []

                # Get the number of components
                component_num = random.randint(component_num_range_L[0], component_num_range_L[1])

                # Get the name of the components
                var_L = []
                while len(var_L) < component_num:
                    rand_idx = random.randint(0, len(src_L) - 1)
                    if not src_L[rand_idx] in var_L:
                        var_L.append(src_L[rand_idx])

                # Get the window of the components, where win_end > win_start
                win_LL = []
                for j in range(component_num):
                    win_start = win_range_L[0]
                    win_end = win_range_L[1]
                    win_LL.append([win_start, win_end])

                # Add the components to the interaction
                for j in range(component_num):
                    var = var_L[j]
                    win_start = win_LL[j][0]
                    win_end = win_LL[j][1]
                    interaction_LL.append([var, win_start, win_end])

                # Check whether the interaction intersects with the existing ones
                if not check_intersect(interaction_LL, interaction_LLL):
                    interaction_LLL.append(interaction_LL)
                    # Write the target, probability, and the interaction
                    interaction_L = []
                    for [var, win_start, win_end] in interaction_LL:
                        interaction_L.append(var)
                        interaction_L.append(win_start)
                        interaction_L.append(win_end)
                    spamwriter.writerow([target, prob] + interaction_L)


# Check whether i_LL and j_LLL intersect, that is, whether there are two interactions containing sources with the same name
def check_intersect(i_LL, j_LLL):
    for i_L in i_LL:
        for j_LL in j_LLL:
            for j_L in j_LL:
                # Check whether i_L[0] equals j_L[0], i.e., the name of the source is the same
                if j_L[0] == i_L[0]:
                    return True

    return False


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_setting_dir = sys.argv[1]
    tar_setting_dir = sys.argv[2]
    interaction_dir = sys.argv[3]
    interaction_num_range_L = [int(sys.argv[4]), int(sys.argv[5])]
    component_num_range_L = [int(sys.argv[6]), int(sys.argv[7])]
    win_range_L = [int(sys.argv[8]), int(sys.argv[9])]
    prob_range_L = [float(sys.argv[10]), float(sys.argv[11])]

    # Make directory
    directory = os.path.dirname(interaction_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for src_setting_file in os.listdir(src_setting_dir):
        if src_setting_file.endswith(".txt"):
            # Get source setting file number
            num = src_setting_file
            num = num.replace('src_setting_', '')
            num = num.replace('.txt', '')
            # Get source setting file
            src_setting_file = src_setting_dir + 'src_setting_' + num + '.txt'
            # Get target setting file
            tar_setting_file = tar_setting_dir + 'tar_setting_' + num + '.txt'
            # Get interaction file
            interaction_file = interaction_dir + 'interaction_' + num + '.txt'
            # Generate interaction
            generate_interaction()


