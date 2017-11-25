
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


# File type
file_type = ".txt"

# Delimiter type
delimiter_type = ','

# Flag, indicating whether there is a header (1, yes; 0, no)
header = 0

# The row number
row_num = 226

# The column number
col_num = 71

# Global variables
# The column of class
class_col = 70

# The list of class values we are interested in
class_val_L = ['cochlear_age']

# The columns of continuous features
con_feature_col_L = []

# The number of bins
bins_num = 2

# The columns of features that should be excluded
exclude_feature_col_L = [69]

# The character for missing values
missing_char = '*'

# The size of the training set
training_size = 200

# The number of repetition of training set
training_iteration = 100

# The number of repetition of experiments
interation_num = 1

# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_Dic = {}

# The dictionary of discretized value of each var at each time
# key: time->var
# val: discretized value of each var at each time
val_dis_Dic = {}

# The dictionary of continuous value of each var at each time
# key: time->var
# val: continuous value of each var at each time
val_raw_Dic = {}

# The dictionary of discretized values of continuous features
# key: var
# val: discretized value of continuous features
con_feature_val_L_Dic = {}


# Generate source and target data
def generate_data():
    # Get con_feature_val_L_Dic
    global con_feature_val_L_Dic
    con_feature_val_L_Dic = {}
    for col in con_feature_col_L:
        con_feature_val_L_Dic[col] = range(bins_num)

    global val_Dic, val_dis_Dic, val_raw_Dic
    val_Dic = {}
    val_dis_Dic = {}
    val_raw_Dic = {}

    # Load the raw file
    with open(raw_file, 'r') as f:
        try:
            spamreader = list(csv.reader(f, delimiter=delimiter_type, skipinitialspace=True))
            # The number of rows containing missing values
            missing_row_num = 0

            # Get val_Dic
            for i in range(header, row_num):
                # Initialization
                if not i - header - missing_row_num in val_Dic:
                    val_Dic[i - header - missing_row_num] = {}

                # Get val_Dic
                for j in range(col_num):
                    # Exclude the features suggested by the contributor of the dataset
                    if j in exclude_feature_col_L:
                        continue

                    val_j = spamreader[i][j].strip()
                    # If not missing
                    if val_j != missing_char:
                        val_Dic[i - header - missing_row_num][j] = val_j
                    else:
                        del val_Dic[i - header - missing_row_num]
                        missing_row_num += 1
                        break

            # Get val_dis_Dic
            for j in range(col_num):
                # Exclude the features suggested by the contributor of the dataset
                if j in exclude_feature_col_L:
                    continue

                # Get the list of value
                val_L = []
                # If continuous feature
                if j in con_feature_col_L:
                    for i in sorted(val_Dic.keys()):
                        val = float(val_Dic[i][j])
                        val_L.append(val)

                    # Get the list of discretized value
                    bin_num = len(con_feature_val_L_Dic[j])
                    val_dis_L = discretize(val_L, bin_num)
                else:
                    distinct_val_L = []
                    for i in sorted(val_Dic.keys()):
                        val = val_Dic[i][j]
                        val_L.append(val)
                        if not val in distinct_val_L:
                            distinct_val_L.append(val)

                # Update val_raw_Dic and val_dis_Dic
                for i in sorted(val_Dic.keys()):
                    # Initialization
                    if not i in val_raw_Dic:
                        val_raw_Dic[i] = {}
                    if not i in val_dis_Dic:
                        val_dis_Dic[i] = {}

                    # If continuous feature
                    if j in con_feature_col_L:
                        # Get name_val_raw
                        name_val_raw = 'feature_' + str(j)

                        # Update val_raw_Dic
                        val_raw_Dic[i][name_val_raw] = val_Dic[i][j]

                        val_dis = val_dis_L[i]
                        for val in con_feature_val_L_Dic[j]:
                            # Get name_val_dis (one-hot encoding)
                            name_val_dis = 'feature_' + str(j) + '_' + str(val)

                            # Update val_dis_Dic
                            if val == val_dis:
                                val_dis_Dic[i][name_val_dis] = 1
                            else:
                                val_dis_Dic[i][name_val_dis] = 0
                    else:
                        val_dis = val_L[i]

                        # If feature
                        if j != class_col:
                            for k in range(len(distinct_val_L)):
                                val = distinct_val_L[k]
                                # Get name_val_raw
                                name_val_raw = 'feature_' + str(j)

                                # Get name_val_dis (one-hot encoding)
                                name_val_dis = 'feature_' + str(j) + '_' + str(val)

                                # Update val_raw_Dic and val_dis_Dic
                                if val == val_dis:
                                    val_raw_Dic[i][name_val_raw] = k
                                    val_dis_Dic[i][name_val_dis] = 1
                                else:
                                    val_dis_Dic[i][name_val_dis] = 0
                        else:
                            for k in range(len(class_val_L)):
                                val = class_val_L[k]
                                # Get name_val_raw
                                name_val_raw = 'class'

                                # Get name_val_dis (one-hot encoding)
                                name_val_dis = 'class_' + val

                                # Update val_raw_Dic and val_dis_Dic
                                if val == val_dis:
                                    val_raw_Dic[i][name_val_raw] = 1
                                    val_dis_Dic[i][name_val_dis] = 1
                                else:
                                    val_raw_Dic[i][name_val_raw] = 0
                                    val_dis_Dic[i][name_val_dis] = 0

            write_file(src_data_training_file, 'src', 'training', val_dis_Dic)
            write_file(src_data_testing_file, 'src', 'testing', val_dis_Dic)
            write_file(tar_data_training_file, 'tar', 'training', val_dis_Dic)
            write_file(tar_data_testing_file, 'tar', 'testing', val_dis_Dic)

            write_file(src_data_training_raw_file, 'src', 'training', val_raw_Dic)
            write_file(src_data_testing_raw_file, 'src', 'testing', val_raw_Dic)
            write_file(tar_data_training_raw_file, 'tar', 'training', val_raw_Dic)
            write_file(tar_data_testing_raw_file, 'tar', 'testing', val_raw_Dic)

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
        if training_testing_F == 'training':
            start = 0
            end = training_size
        else:
            start = training_size
            end = len(val_Dic.keys())

        # Get iteration
        if training_testing_F == 'training':
            iteration = training_iteration
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
                            print("name_val not in val_Dic[time]!")
                            sys.exit()
                else:
                    print("time not in val_Dic!")
                    sys.exit()

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

    for i in range(interation_num):
        for raw_file in os.listdir(raw_file_dir):
            if raw_file.endswith(file_type):
                # Get src data training file
                src_data_training_file = src_data_dir + '/training/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data training file
                tar_data_training_file = tar_data_dir + '/training/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data testing file
                src_data_testing_file = src_data_dir + '/testing/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data testing file
                tar_data_testing_file = tar_data_dir + '/testing/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data training file
                src_data_training_raw_file = src_data_dir + '/training/raw/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data training file
                tar_data_training_raw_file = tar_data_dir + '/training/raw/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data testing file
                src_data_testing_raw_file = src_data_dir + '/testing/raw/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data testing file
                tar_data_testing_raw_file = tar_data_dir + '/testing/raw/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Update raw_file
                raw_file = raw_file_dir + raw_file

                # Generate data
                generate_data()

