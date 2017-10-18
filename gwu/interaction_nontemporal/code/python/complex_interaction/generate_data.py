

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


# Generate source and target data
def generate_data():
    # Load the source setting file
    with open(src_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get the source list
        src_L = []

        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Get the name of the var, the number of times of its presence, and the probability of its presence
            var = spamreader[i][0].strip()
            num = int(spamreader[i][1].strip())
            num_Dic[var] = num
            prob = float(spamreader[i][2].strip())
            prob_Dic[var] = prob
            # Update the source list
            src_L.append(var)

    random.seed()

    # Generate the source value
    for time in range(time_num):
        for source in src_L:
            # Initialization
            if not time in val_Dic:
                val_Dic[time] = {}
            val_Dic[time][source] = 0
            # Generate random number from [0, 1)
            rand_prob = random.random()
            if rand_prob < prob_Dic[source]:
                val_Dic[time][source] = 1

    # Write the source file
    with open(src_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(src_L)
        # Write the value
        for time in val_Dic:
            val_L = []
            for source in src_L:
                val_L.append(val_Dic[time][source])
            spamwriter.writerow(val_L)

    # Load the interaction file
    with open(interaction_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get the target, probability and interaction
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Target lies in the first column in each row
            target = spamreader[i][0].strip()
            # Probability lies in the second column in each row
            prob = float(spamreader[i][1].strip())
            # interaction lies in the remaining columns, with the form component_i, win_start_i, win_end_i
            interaction_LL = []
            component_num = (len(spamreader[i]) - 2) // 3
            for j in range(component_num):
                component_L = []
                # Name
                component_L.append(spamreader[i][j * 3 + 2].strip())
                # Window start
                component_L.append(int(spamreader[i][j * 3 + 3].strip()))
                # Window end
                component_L.append(int(spamreader[i][j * 3 + 4].strip()))
                interaction_LL.append(component_L)
            if not target in prob_interaction_L_Dic:
                prob_interaction_L_Dic[target] = []
            prob_interaction_L_Dic[target].append([prob, interaction_LL])

    # Load the target setting file
    with open(tar_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Initialize the target list
        tar_L = []

        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Get the name of the var and the probability of its presence
            var = spamreader[i][0].strip()
            prob = float(spamreader[i][1].strip())
            prob_Dic[var] = prob
            # Update the target list
            tar_L.append(var)

    # Generate the target value
    for target in tar_L:
        # Initialization
        for time in val_Dic:
            rand_prob = random.random()
            if rand_prob < prob_Dic[target]:
                # Add noise
                val_Dic[time][target] = 1
            else:
                val_Dic[time][target] = 0

        # Add the impact of the interactions
        for time in val_Dic:
            if val_Dic[time][target] == 1:
                continue

            for [prob, interaction_LL] in prob_interaction_L_Dic[target]:
                if get_presence(time, interaction_LL) is True:
                    # Generate random number from [0, 1]
                    rand_prob = random.uniform(0, 1)
                    if rand_prob < prob:
                        val_Dic[time][target] = 1
                        break

    # Write the target file
    with open(tar_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')
        # Write the header
        spamwriter.writerow(tar_L)
        # Write the value
        for time in val_Dic:
            val_L = []
            for target in tar_L:
                val_L.append(val_Dic[time][target])
            spamwriter.writerow(val_L)


# Check the presence of each component in the pie
def get_presence(time, interaction_LL):
    # Check the presence of each component in the pie
    for component_L in interaction_LL:
        # Get the name, window start and window end of the component
        var = component_L[0]
        win_start = component_L[1]
        win_end = component_L[2]

        # Check the presence of the component in time window [time - window end, time - window start]
        # Default is absence
        presence_F = False
        for prev_time in range(time - win_end, time - win_start + 1):
            if prev_time in val_Dic and val_Dic[prev_time][var] == 1:
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
    src_setting_dir = sys.argv[1]
    tar_setting_dir = sys.argv[2]
    interaction_dir = sys.argv[3]
    src_data_dir = sys.argv[4]
    tar_data_dir = sys.argv[5]
    time_num = int(sys.argv[6])

    # Make directory
    directory = os.path.dirname(src_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for src_setting_file in os.listdir(src_setting_dir):
        if src_setting_file.endswith(".txt"):
            # Get src setting file number
            num = src_setting_file
            num = num.replace('src_setting_', '')
            num = num.replace('.txt', '')
            # Get src setting file
            src_setting_file = src_setting_dir + 'src_setting_' + num + '.txt'
            # Get tar setting file
            tar_setting_file = tar_setting_dir + 'tar_setting_' + num + '.txt'
            # Get interaction file
            interaction_file = interaction_dir + 'interaction_' + num + '.txt'
            # Get src data file
            src_data_file = src_data_dir + 'src_data_' + num + '.txt'
            # Get tar data file
            tar_data_file = tar_data_dir + 'tar_data_' + num + '.txt'

            # The dictionary of value
            # key: time->var
            # val: value of var at the time
            val_Dic = {}

            # The dictionary of probabiity and interaction
            # key: target
            # val: list comprised of probability and the interactions
            prob_interaction_L_Dic = {}

            # The dictionary of the number of times of the variable (source and target) being present
            num_Dic = {}

            # The dictionary of the probability of the variable (source and target) being present
            prob_Dic = {}

            # Generate data
            generate_data()
