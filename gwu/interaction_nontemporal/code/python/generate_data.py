
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
# The list of name of variables
name_L = ['class', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22']

val_LL = ['1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1', '0, 1']

# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_Dic = {}

# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_raw_Dic = {}

# Maximum time stamp
max_time_stamp = 0

file_type = ".txt"

# Generate source and target data
def generate_data():
    # Load the raw file
    with open(raw_file, 'r') as f:
        try:
            spamreader = list(csv.reader(f, delimiter=','))

            global val_Dic, val_raw_Dic
            # Initialization
            val_Dic = {}
            val_raw_Dic = {}

            global max_time_stamp
            # Get the maximum time stamp
            max_time_stamp = len(spamreader)

            # From the first line to the last (since there is no header)
            for i in range(0, max_time_stamp):
                # Initialization
                if not i in val_Dic:
                    val_Dic[i] = {}
                    val_raw_Dic[i] = {}

                # Get val_Dic and val_raw_Dic
                for j in range(len(name_L)):
                    name = name_L[j]
                    val_j = spamreader[i][j].strip()

                    # Update val_raw_Dic
                    val_raw_Dic[i][name] = val_j

                    for val in val_LL[j].split(','):
                        val = val.strip()
                        # Get name_val
                        name_val = name + '_' + val

                        # Update val_Dic
                        if val == val_j:
                            val_Dic[i][name_val] = 1
                        else:
                            val_Dic[i][name_val] = 0

            if raw_file.endswith("train.txt"):
                write_file(src_data_training_file, 'src', 'training', val_Dic)
                write_file(tar_data_training_file, 'tar', 'training', val_Dic)
                write_file(src_data_training_raw_file, 'src', 'training', val_raw_Dic)
                write_file(tar_data_training_raw_file, 'tar', 'training', val_raw_Dic)


            else:
                write_file(src_data_testing_file, 'src', 'testing', val_Dic)
                write_file(tar_data_testing_file, 'tar', 'testing', val_Dic)
                write_file(src_data_testing_raw_file, 'src', 'testing', val_raw_Dic)
                write_file(tar_data_testing_raw_file, 'tar', 'testing', val_raw_Dic)

        except UnicodeDecodeError:
            print("UnicodeDecodeError when reading the following file!")
            print(raw_file)


# Write file
def write_file(file, src_tar_F, training_testing_F, val_Dic):
    # Write file
    with open(file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Get the header
        header_L = []
        for name_val in sorted(val_Dic[0].keys()):
            if ((src_tar_F == 'src' and not 'class' in name_val)
                or (src_tar_F == 'tar' and 'class' in name_val)):
                header_L.append(name_val)

        # Write the header
        spamwriter.writerow(header_L)

        # Write the value
        # Get start and end
        start = 0
        end = int(max_time_stamp)

        # Get iteration
        if training_testing_F == 'training':
            iteration = 100
        else:
            iteration = 1

        for i in range(iteration):
            for time in range(start, end):
                # The list of value at the time
                val_L = []

                if time in val_Dic:
                    for name_val in header_L:
                        if name_val in val_Dic[time]:
                            val_L.append(val_Dic[time][name_val])
                        else:
                            val_L.append(0)
                else:
                    for name_val in header_L:
                        val_L.append(0)

                spamwriter.writerow(val_L)


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_file_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]

    # Make directory
    directory = os.path.dirname(src_data_dir + '/training/raw/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(src_data_dir + '/testing/raw/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/training/raw/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/testing/raw/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(100):
        for raw_file in os.listdir(raw_file_dir):
            if raw_file.endswith("train.txt"):
                # Get src data training file
                src_data_training_file = src_data_dir + '/training/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data training file
                tar_data_training_file = tar_data_dir + '/training/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data training file
                src_data_training_raw_file = src_data_dir + '/training/raw/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data training file
                tar_data_training_raw_file = tar_data_dir + '/training/raw/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Update raw_file
                raw_file = raw_file_dir + raw_file

                # Generate data
                generate_data()

    for i in range(100):
        for raw_file in os.listdir(raw_file_dir):
            if raw_file.endswith("test.txt"):
                # Get src data testing file
                src_data_testing_file = src_data_dir + '/testing/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data testing file
                tar_data_testing_file = tar_data_dir + '/testing/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data testing file
                src_data_testing_raw_file = src_data_dir + '/testing/raw/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data testing file
                tar_data_testing_raw_file = tar_data_dir + '/testing/raw/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Update raw_file
                raw_file = raw_file_dir + raw_file

                # Generate data
                generate_data()
