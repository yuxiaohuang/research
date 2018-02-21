

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


# Generate x and y data
def generate_data():
    # Load the x setting file
    with open(feature_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get the x list
        x_L = []

        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Get the name of the var, the number of times of its presence, and the probability of its presence
            var = spamreader[i][0].strip()
            num = int(spamreader[i][1].strip())
            num_Dic[var] = num
            prob = float(spamreader[i][2].strip())
            prob_Dic[var] = prob
            # Update the x list
            x_L.append(var)

    random.seed(0)

    # Generate the x value
    for time in range(time_num):
        for x in x_L:
            # Initialization
            if not time in val_Dic:
                val_Dic[time] = {}
            val_Dic[time][x] = 0
            # Generate random number from [0, 1)
            rand_prob = random.random()
            if rand_prob < prob_Dic[x]:
                val_Dic[time][x] = 1

    # Load the ground_truth file
    with open(ground_truth_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get the y, probability and ground_truth
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # y lies in the first column in each row
            y = spamreader[i][0].strip()
            # Probability lies in the second column in each row
            prob = float(spamreader[i][1].strip())
            # ground_truth lies in the remaining columns, with the form component_i, win_start_i, win_end_i
            ground_truth_LL = []
            component_num = (len(spamreader[i]) - 2) // 3
            for j in range(component_num):
                component_L = []
                # Name
                component_L.append(spamreader[i][j * 3 + 2].strip())
                # Window start
                component_L.append(int(spamreader[i][j * 3 + 3].strip()))
                # Window end
                component_L.append(int(spamreader[i][j * 3 + 4].strip()))
                ground_truth_LL.append(component_L)
            if not y in prob_ground_truth_L_Dic:
                prob_ground_truth_L_Dic[y] = []
            prob_ground_truth_L_Dic[y].append([prob, ground_truth_LL])

    # Load the y setting file
    with open(target_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Initialize the y list
        y_L = []

        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Get the name of the var and the probability of its presence
            var = spamreader[i][0].strip()
            prob = float(spamreader[i][1].strip())
            prob_Dic[var] = prob
            # Update the y list
            y_L.append(var)

    # Generate the y value
    for y in y_L:
        # Initialization
        for time in sorted(val_Dic.keys()):
            rand_prob = random.random()
            if rand_prob < prob_Dic[y]:
                # Add noise
                val_Dic[time][y] = 1
            else:
                val_Dic[time][y] = 0

        # Add the impact of the ground truth
        for time in sorted(val_Dic.keys()):
            if val_Dic[time][y] == 1:
                continue

            for [prob, ground_truth_LL] in prob_ground_truth_L_Dic[y]:
                if get_presence(time, ground_truth_LL) is True:
                    # Generate random number from [0, 1]
                    rand_prob = random.uniform(0, 1)
                    if rand_prob < prob:
                        val_Dic[time][y] = 1
                        break

    # Write the raw data file
    # Write the header
    header_L = x_L + y_L
    spamwriter.writerow(header_L)
    # Write the value
    for time in range(time_num):
        val_L = [val_Dic[time][var] for var in header_L]
        spamwriter.writerow(val_L)


# Check the presence of each component in the pie
def get_presence(time, ground_truth_LL):
    # Check the presence of each component in the pie
    for component_L in ground_truth_LL:
        # Get the var_val, window start and window end of the component
        var_val = component_L[0]
        var = var_val[:-2]
        val = var_val[-1:]
        win_start = component_L[1]
        win_end = component_L[2]

        # Check the presence of the component in time window [time - window end, time - window start]
        # Default is absence
        presence_F = False
        for prev_time in range(time - win_end, time - win_start + 1):
            # If the var is measured at the time, and the value at the time equals the one that is necessary to produce the y
            # if prev_time in val_Dic and val_Dic[prev_time][var] == 1:
            if prev_time in val_Dic and val_Dic[prev_time][var] == int(val): # This new implementation allows negation to enter the ground_truth
                presence_F = True

        # If the component is absent in the window above
        if not presence_F:
            return False

    # If all the components in the pie are present in the corresponding windows
    return True


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    feature_setting_dir = sys.argv[1]
    target_setting_dir = sys.argv[2]
    ground_truth_dir = sys.argv[3]
    raw_data_dir = sys.argv[4]
    time_num = int(sys.argv[5])

    # Make directory
    directory = os.path.dirname(raw_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for feature_setting_file in os.listdir(feature_setting_dir):
        if not feature_setting_file.startswith('.') and feature_setting_file.endswith(".txt"):
            # Get src setting file number
            num = feature_setting_file
            num = num.replace('feature_setting_', '')
            num = num.replace('.txt', '')
            # Get src setting file
            feature_setting_file = feature_setting_dir + 'feature_setting_' + num + '.txt'
            # Get tar setting file
            target_setting_file = target_setting_dir + 'target_setting_' + num + '.txt'
            # Get ground_truth file
            ground_truth_file = ground_truth_dir + 'ground_truth_' + num + '.txt'
            # Get raw data
            raw_data_file = raw_data_dir + 'data_' + num + '.txt'

            # The dictionary of value
            # key: time->var
            # val: value of var at the time
            val_Dic = {}

            # The dictionary of probabiity and ground_truth
            # key: y
            # val: list comprised of probability and the ground truth
            prob_ground_truth_L_Dic = {}

            # The dictionary of the number of times of the variable (x and y) being present
            num_Dic = {}

            # The dictionary of the probability of the variable (x and y) being present
            prob_Dic = {}

            # Write the raw data file
            with open(raw_data_file, 'w') as f:
                spamwriter = csv.writer(f, delimiter=',')

                # Generate data
                generate_data()