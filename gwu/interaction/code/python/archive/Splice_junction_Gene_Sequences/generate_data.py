

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


# Global variables
# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_Dic = {}
val_rand_Dic = {}

# Maximum time stamp
max_time_stamp = 0


# Generate source and target data
def generate_data():
    # Load the raw file
    with open(raw_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

    global max_time_stamp
    # Get the maximum time stamp
    max_time_stamp = len(spamreader)

    # From the first line to the last (since there is no header)
    for i in range(0, max_time_stamp):
        # Initialization
        if not i in val_Dic:
            val_Dic[i] = {}

        # Get the value of the class
        class_val = spamreader[i][0].strip()

        # Update val_Dic
        for val in ['EI', 'IE']:
            # Get name_val
            name_val = 'class_' + val

            if val == class_val:
                val_Dic[i][name_val] = 1
            else:
                val_Dic[i][name_val] = 0

        # Get the value of the sequence
        seq_val = spamreader[i][2].strip()

        for j in range(60):
            # Get the name of the var
            name = 'pos_' + str(j)

            # Get the value of the var
            val_j = seq_val[j]

            # Update val_Dic
            for val in ['A', 'G', 'T', 'C', 'D', 'N', 'S', 'R']:
                # Get name_val
                name_val = name + '_' + val

                if val == val_j:
                    val_Dic[i][name_val] = 1
                else:
                    val_Dic[i][name_val] = 0

    # Get val_rand_Dic
    time_L = random.sample(list(range(max_time_stamp)), max_time_stamp)
    for i in range(max_time_stamp):
        time = time_L[i]
        if not i in val_rand_Dic:
            val_rand_Dic[i] = {}
        for name_val in sorted(val_Dic[time].keys()):
            val_rand_Dic[i][name_val] = val_Dic[time][name_val]

    write_file(src_data_training_file, 'src', 'training')
    write_file(src_data_testing_file, 'src', 'testing')
    write_file(tar_data_training_file, 'tar', 'training')
    write_file(tar_data_testing_file, 'tar', 'testing')


# Write file
def write_file(file, src_tar_F, training_testing_F):
    # Write file
    with open(file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Get the header
        header_L = []
        for name_val in sorted(val_rand_Dic[0].keys()):
            if ((src_tar_F == 'src' and not 'class' in name_val)
                or (src_tar_F == 'tar' and 'class' in name_val)):
                header_L.append(name_val)

        # Write the header
        spamwriter.writerow(header_L)

        # Write the value
        # Get start and end
        if training_testing_F == 'training':
            start = 0
            end = int(0.8 * max_time_stamp)
        else:
            start = int(0.8 * max_time_stamp)
            end = max_time_stamp

        # Get iteration
        if training_testing_F == 'training':
            iteration = 10
        else:
            iteration = 1

        for i in range(iteration):
            for time in range(start, end):
                # The list of value at the time
                val_L = []

                if time in val_rand_Dic:
                    for name_val in header_L:
                        if name_val in val_rand_Dic[time]:
                            val_L.append(val_rand_Dic[time][name_val])
                        else:
                            val_L.append(0)
                else:
                    for name_val in src_L:
                        val_L.append(0)

                spamwriter.writerow(val_L)


if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_file_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]

    # Make directory
    directory = os.path.dirname(src_data_dir + '/training/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(src_data_dir + '/testing/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/training/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/testing/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    for raw_file in os.listdir(raw_file_dir):
        if raw_file.endswith(".txt"):
            # Get src data training file
            src_data_training_file = src_data_dir + '/training/src_data_' + raw_file
            # Get tar data training file
            tar_data_training_file = tar_data_dir + '/training/tar_data_' + raw_file

            # Get src data testing file
            src_data_testing_file = src_data_dir + '/testing/src_data_' + raw_file
            # Get tar data testing file
            tar_data_testing_file = tar_data_dir + '/testing/tar_data_' + raw_file

            # Update raw_file
            raw_file = raw_file_dir + raw_file

            # Generate data
            generate_data()
