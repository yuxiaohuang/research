
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
# The number of names, bins
con_feature_col_L = range(11)
feature_val_LL = [[0, 1, 2]] * 11
class_val_L = [2]

# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_Dic = {}

# The dictionary of discretized value of each var at each time
# key: time->var
# val: discretized value of each var at each time
val_dis_Dic = {}

# Maximum time stamp
max_time_stamp = 0


# Generate source and target data
def generate_data():
    global val_Dic, val_dis_Dic
    val_Dic = {}
    val_dis_Dic = {}

    # Load the raw file
    with open(raw_file, 'r') as f:
        try:
            spamreader = list(csv.reader(f, delimiter=';'))

            global max_time_stamp
            # Get the maximum time stamp
            max_time_stamp = len(spamreader)

            # Get val_Dic
            # From the second line to the last (since there is a header)
            for i in range(1, max_time_stamp):
                # Initialization
                if not i in val_Dic:
                    val_Dic[i - 1] = {}
                # Get val_Dic
                for j in range(len(feature_val_LL) + 1):
                    val_j = spamreader[i][j].strip()
                    val_Dic[i - 1][j] = val_j

            # Get val_dis_Dic
            for j in range(len(feature_val_LL) + 1):
                # Get the list of value
                val_L = []

                for i in sorted(val_Dic.keys()):
                    val = float(val_Dic[i][j])
                    val_L.append(val)

                # Get the list of discretized value
                # If feature
                if j < len(feature_val_LL):
                    bin_num = len(feature_val_LL[j])
                # If class
                else:
                    bin_num = len(class_val_L) + 2
                val_dis_L = discretize(val_L, bin_num)

                # Update val_dis_Dic
                for i in sorted(val_Dic.keys()):
                    # Initialization
                    if not i in val_dis_Dic:
                        val_dis_Dic[i] = {}

                    val_dis = val_dis_L[i]

                    # If feature
                    if j < len(feature_val_LL):
                        for val in feature_val_LL[j]:
                            # Get name_val
                            name_val = 'feature_' + str(j) + '_' + str(val)
                            # Update val_dis_Dic
                            if val == val_dis:
                                val_dis_Dic[i][name_val] = 1
                            else:
                                val_dis_Dic[i][name_val] = 0
                    # If class
                    else:
                        for val in class_val_L:
                            # Get name_val
                            name_val = 'class_' + str(val)
                            # Update val_dis_Dic
                            if val == val_dis:
                                val_dis_Dic[i][name_val] = 1
                            else:
                                val_dis_Dic[i][name_val] = 0

            write_file(src_data_training_file, 'src', 'training')
            write_file(src_data_testing_file, 'src', 'testing')
            write_file(tar_data_training_file, 'tar', 'training')
            write_file(tar_data_testing_file, 'tar', 'testing')

        except UnicodeDecodeError:
            print("UnicodeDecodeError when reading the following file!")
            print(raw_file)


# Discretize val_L into bin_num bins
def discretize(val_L, bin_num):
    split_L = np.array_split(np.sort(val_L), bin_num)
    cutoff_L = [split[-1] for split in split_L]
    cutoff_L = cutoff_L[:-1]
    val_dis_L = np.digitize(val_L, cutoff_L, right = True)
    return val_dis_L


# Write file
def write_file(file, src_tar_F, training_testing_F):
    # Write file
    with open(file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Get the header
        header_L = []
        for name_val in sorted(val_dis_Dic[1].keys()):
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
            end = max_time_stamp - 1

        # Get iteration
        if training_testing_F == 'training':
            iteration = 1
        else:
            iteration = 1

        for i in range(iteration):
            for time in range(start, end):
                # The list of value at the time
                val_L = []

                if time in val_dis_Dic:
                    for name_val in header_L:
                        if name_val in val_dis_Dic[time]:
                            val_L.append(val_dis_Dic[time][name_val])
                        else:
                            val_L.append(0)
                else:
                    for name_val in src_L:
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
        if raw_file.endswith(".csv"):
            # Get src data training file
            src_data_training_file = src_data_dir + '/training/src_data_' + raw_file.replace('csv', 'txt')
            # Get tar data training file
            tar_data_training_file = tar_data_dir + '/training/tar_data_' + raw_file.replace('csv', 'txt')

            # Get src data testing file
            src_data_testing_file = src_data_dir + '/testing/src_data_' + raw_file.replace('csv', 'txt')
            # Get tar data testing file
            tar_data_testing_file = tar_data_dir + '/testing/tar_data_' + raw_file.replace('csv', 'txt')

            # Update raw_file
            raw_file = raw_file_dir + raw_file

            # Generate data
            generate_data()

