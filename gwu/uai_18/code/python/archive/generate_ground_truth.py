

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


# Generate ground_truth
def generate_ground_truth():
    # Read attribute setting file
    with open(attribute_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get x_L
        x_L = []
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            x_L.append(spamreader[i][0].strip())

    # Read class setting file
    with open(class_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get y_L
        y_L = []
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            y_L.append(spamreader[i][0].strip())

    # Write ground_truth file
    with open(ground_truth_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(['class', 'probability', 'ground_truth'])

        for y in y_L:
            # The ground truth
            ground_truth_LLL = []

            # Get the ground_truth number
            ground_truth_num = random.randint(ground_truth_num_range_L[0], ground_truth_num_range_L[1])

            while len(ground_truth_LLL) < ground_truth_num:
                # Get the probability
                prob = random.uniform(prob_range_L[0], prob_range_L[1])

                # Get the ground_truth
                ground_truth_LL = []

                # Get the number of components
                component_num = random.randint(component_num_range_L[0], component_num_range_L[1])

                # Get the name of the components
                var_L = []
                while len(var_L) < component_num:
                    rand_idx = random.randint(0, len(x_L) - 1)
                    if not x_L[rand_idx] in var_L:
                        var_L.append(x_L[rand_idx])

                # Get the window of the components, where win_end > win_start
                win_LL = []
                for j in range(component_num):
                    win_start = win_range_L[0]
                    win_end = win_range_L[1]
                    win_LL.append([win_start, win_end])

                # Add the components to the ground_truth
                for j in range(component_num):
                    var = var_L[j]
                    # Get the negation probability
                    neg_prob = random.uniform(neg_prob_range_L[0], neg_prob_range_L[1])
                    # Generate random number from [0, 1]
                    rand_prob = random.uniform(0, 1)

                    if rand_prob < neg_prob:
                        # Negation, or absence
                        var += '_0'
                    else:
                        var += '_1'

                    win_start = win_LL[j][0]
                    win_end = win_LL[j][1]
                    ground_truth_LL.append([var, win_start, win_end])

                # Check whether the ground_truth intersects with the existing ones
                if check_intersect(ground_truth_LL, ground_truth_LLL) is False:
                    ground_truth_LLL.append(ground_truth_LL)
                    # Write the class, probability, and the ground_truth
                    ground_truth_L = []
                    for [var, win_start, win_end] in ground_truth_LL:
                        ground_truth_L.append(var)
                        ground_truth_L.append(win_start)
                        ground_truth_L.append(win_end)
                    spamwriter.writerow([y, prob] + ground_truth_L)


# Check whether i_LL and j_LLL intersect, that is, whether there are two ground truth containing attributes with the same name
def check_intersect(i_LL, j_LLL):
    for i_L in i_LL:
        for j_LL in j_LLL:
            for j_L in j_LL:
                # Check whether i_L[0] equals j_L[0], i.e., the name of the attribute is the same
                if get_var_name(j_L[0]) == get_var_name(i_L[0]):
                    return True

    return False


# Get the var name (the substring prior to the last '_')
def get_var_name(var_val):
    idx = var_val.rfind('_')
    return var_val[:idx]


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    attribute_setting_dir = sys.argv[1]
    class_setting_dir = sys.argv[2]
    ground_truth_dir = sys.argv[3]
    ground_truth_num_range_L = [int(sys.argv[4]), int(sys.argv[5])]
    component_num_range_L = [int(sys.argv[6]), int(sys.argv[7])]
    win_range_L = [int(sys.argv[8]), int(sys.argv[9])]
    prob_range_L = [float(sys.argv[10]), float(sys.argv[11])]
    neg_prob_range_L = [float(sys.argv[12]), float(sys.argv[13])]

    # Make directory
    directory = os.path.dirname(ground_truth_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for attribute_setting_file in os.listdir(attribute_setting_dir):
        if not attribute_setting_file.startswith('.') and attribute_setting_file.endswith(".txt"):
            # Get attribute setting file number
            num = attribute_setting_file
            num = num.replace('attribute_setting_', '')
            num = num.replace('.txt', '')
            # Get attribute setting file
            attribute_setting_file = attribute_setting_dir + 'attribute_setting_' + num + '.txt'
            # Get class setting file
            class_setting_file = class_setting_dir + 'class_setting_' + num + '.txt'
            # Get ground_truth file
            ground_truth_file = ground_truth_dir + 'ground_truth_' + num + '.txt'
            # Generate ground_truth
            generate_ground_truth()


