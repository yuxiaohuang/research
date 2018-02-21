
# Please cite the following paper when using the code


# Modules
import sys
import os
import csv
import numpy as np
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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
row_num = 0

# The column number
col_num = 0

# Global variables
# The column of class
class_col = -1

# The list of class values we are interested in
class_val_L = ['T']

# The columns of continuous features
# con_feature_col_L = range(13)
con_feature_col_L = []

# The list of number of bins
bins_num_L = []

# The columns of features that should be excluded
exclude_feature_col_L = []

# The character for missing values
missing_char = '?'

# The percentage of the training set
training_percentage = 0.8

# The number of repetition of training set
# training_iteration = 100
training_iteration = 1

# The number of repetition of experiments
interation_num = 100

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

# The dictionary of continuous value of each var at each time
# key: time->var
# val: standardized value of each var at each time
val_raw_std_Dic = {}

# The dictionary of continuous value of each var at each time
# key: time->var
# val: normalized value of each var at each time
val_raw_mms_Dic = {}

# The dictionary of discretized value of each var at each time
# key: time->var
# val: discretized value of each var at each time
val_dis_rand_Dic = {}

# The dictionary of continuous value of each var at each time
# key: time->var
# val: continuous value of each var at each time
val_raw_rand_Dic = {}

# The dictionary of continuous value of each var at each time
# key: time->var
# val: continuous value of each var at each time
val_raw_std_rand_Dic = {}

# The dictionary of continuous value of each var at each time
# key: time->var
# val: continuous value of each var at each time
val_raw_mms_rand_Dic = {}

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
        con_feature_val_L_Dic[col] = range(bins_num_L[col])

    global val_Dic, val_dis_Dic, val_raw_Dic, val_raw_std_Dic, val_raw_mms_Dic, val_dis_rand_Dic, val_raw_rand_Dic, val_raw_std_rand_Dic, val_raw_mms_rand_Dic
    val_Dic = {}
    val_dis_Dic = {}
    val_raw_Dic = {}
    val_raw_std_Dic = {}
    val_raw_mms_Dic = {}
    val_dis_rand_Dic = {}
    val_raw_rand_Dic = {}
    val_raw_std_rand_Dic = {}
    val_raw_mms_rand_Dic = {}

    # Load the raw file
    with open(raw_file, 'r') as f:
        try:
            spamreader = list(csv.reader(f, delimiter=delimiter_type, skipinitialspace=True))

            # The row number
            global row_num
            row_num = len(spamreader)

            # The column number
            global col_num
            col_num = len(spamreader[0])

            # Global variables
            # The column of class
            global class_col
            if class_col == -1:
                class_col = col_num - 1

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
                        name_val_raw = 'src_' + str(j)

                        # Update val_raw_Dic
                        val_raw_Dic[i][name_val_raw] = val_Dic[i][j]

                        val_dis = val_dis_L[i]
                        for val in con_feature_val_L_Dic[j]:
                            # Get name_val_dis (one-hot encoding)
                            name_val_dis = 'src_' + str(j) + '_' + str(val)

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
                                name_val_raw = 'src_' + str(j)

                                # Get name_val_dis (one-hot encoding)
                                name_val_dis = 'src_' + str(j) + '_' + str(val)

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
                                name_val_raw = 'tar'

                                # Get name_val_dis (one-hot encoding)
                                name_val_dis = 'tar_' + val

                                # Update val_raw_Dic and val_dis_Dic
                                if val == val_dis:
                                    val_raw_Dic[i][name_val_raw] = 1
                                    val_dis_Dic[i][name_val_dis] = 1
                                else:
                                    val_raw_Dic[i][name_val_raw] = 0
                                    val_dis_Dic[i][name_val_dis] = 0

            # Get val_raw_std_Dic and val_raw_mms_Dic
            for name_val_raw in sorted(val_raw_Dic[0].keys()):
                # Initialization
                val_raw_L = []

                for i in sorted(val_raw_Dic.keys()):
                    # Initialization
                    if not i in val_raw_std_Dic:
                        val_raw_std_Dic[i] = {}
                    if not i in val_raw_mms_Dic:
                        val_raw_mms_Dic[i] = {}

                    if not 'tar' in name_val_raw:
                        # Update val_raw_L
                        val_raw_L.append(float(val_raw_Dic[i][name_val_raw]))
                    else:
                        # Update val_raw_L
                        val_raw_L.append(val_raw_Dic[i][name_val_raw])

                if not 'tar' in name_val_raw:
                    # Standardization and min - max normalization
                    stdsc = StandardScaler()
                    mms = MinMaxScaler()
                    val_raw_std_L = stdsc.fit_transform(val_raw_L)
                    val_raw_mms_L = mms.fit_transform(val_raw_L)

                    # Update val_raw_std_Dic and val_raw_mms_Dic
                    for i in sorted(val_raw_Dic.keys()):
                        val_raw_std_Dic[i][name_val_raw] = val_raw_std_L[i]
                        val_raw_mms_Dic[i][name_val_raw] = val_raw_mms_L[i]
                else:
                    # Update val_raw_std_Dic and val_raw_mms_Dic
                    for i in sorted(val_raw_Dic.keys()):
                        val_raw_std_Dic[i][name_val_raw] = val_raw_L[i]
                        val_raw_mms_Dic[i][name_val_raw] = val_raw_L[i]

            # write_file(src_data_training_file, 'src', 'training', val_dis_Dic)
            # write_file(src_data_testing_file, 'src', 'testing', val_dis_Dic)
            # write_file(tar_data_training_file, 'tar', 'training', val_dis_Dic)
            # write_file(tar_data_testing_file, 'tar', 'testing', val_dis_Dic)
            #
            # write_file(src_data_training_raw_file, 'src', 'training', val_raw_Dic)
            # write_file(src_data_testing_raw_file, 'src', 'testing', val_raw_Dic)
            # write_file(tar_data_training_raw_file, 'tar', 'training', val_raw_Dic)
            # write_file(tar_data_testing_raw_file, 'tar', 'testing', val_raw_Dic)
            #
            # write_file(src_data_training_raw_std_file, 'src', 'training', val_raw_std_Dic)
            # write_file(src_data_testing_raw_std_file, 'src', 'testing', val_raw_std_Dic)
            # write_file(tar_data_training_raw_std_file, 'tar', 'training', val_raw_std_Dic)
            # write_file(tar_data_testing_raw_std_file, 'tar', 'testing', val_raw_std_Dic)
            #
            # write_file(src_data_training_raw_mms_file, 'src', 'training', val_raw_mms_Dic)
            # write_file(src_data_testing_raw_mms_file, 'src', 'testing', val_raw_mms_Dic)
            # write_file(tar_data_training_raw_mms_file, 'tar', 'training', val_raw_mms_Dic)
            # write_file(tar_data_testing_raw_mms_file, 'tar', 'testing', val_raw_mms_Dic)

            # Get val_dis_rand_Dic and val_raw_rand_Dic
            time_L = random.sample(list(sorted(val_Dic.keys())), len(val_Dic.keys()))
            for i in sorted(val_Dic.keys()):
                time = time_L[i]
                if not i in val_dis_rand_Dic:
                    val_dis_rand_Dic[i] = {}
                if not i in val_raw_rand_Dic:
                    val_raw_rand_Dic[i] = {}
                if not i in val_raw_std_rand_Dic:
                    val_raw_std_rand_Dic[i] = {}
                if not i in val_raw_mms_rand_Dic:
                    val_raw_mms_rand_Dic[i] = {}
                for name_val in sorted(val_dis_Dic[time].keys()):
                    val_dis_rand_Dic[i][name_val] = val_dis_Dic[time][name_val]
                for name_val in sorted(val_raw_Dic[time].keys()):
                    val_raw_rand_Dic[i][name_val] = val_raw_Dic[time][name_val]
                for name_val in sorted(val_raw_std_Dic[time].keys()):
                    val_raw_std_rand_Dic[i][name_val] = val_raw_std_Dic[time][name_val]
                for name_val in sorted(val_raw_mms_Dic[time].keys()):
                    val_raw_mms_rand_Dic[i][name_val] = val_raw_mms_Dic[time][name_val]

            write_file(src_data_training_file, 'src', 'training', val_dis_rand_Dic)
            write_file(src_data_testing_file, 'src', 'testing', val_dis_rand_Dic)
            write_file(tar_data_training_file, 'tar', 'training', val_dis_rand_Dic)
            write_file(tar_data_testing_file, 'tar', 'testing', val_dis_rand_Dic)

            write_file(src_data_training_raw_file, 'src', 'training', val_raw_rand_Dic)
            write_file(src_data_testing_raw_file, 'src', 'testing', val_raw_rand_Dic)
            write_file(tar_data_training_raw_file, 'tar', 'training', val_raw_rand_Dic)
            write_file(tar_data_testing_raw_file, 'tar', 'testing', val_raw_rand_Dic)

            write_file(src_data_training_raw_std_file, 'src', 'training', val_raw_std_rand_Dic)
            write_file(src_data_testing_raw_std_file, 'src', 'testing', val_raw_std_rand_Dic)
            write_file(tar_data_training_raw_std_file, 'tar', 'training', val_raw_std_rand_Dic)
            write_file(tar_data_testing_raw_std_file, 'tar', 'testing', val_raw_std_rand_Dic)

            write_file(src_data_training_raw_mms_file, 'src', 'training', val_raw_mms_rand_Dic)
            write_file(src_data_testing_raw_mms_file, 'src', 'testing', val_raw_mms_rand_Dic)
            write_file(tar_data_training_raw_mms_file, 'tar', 'training', val_raw_mms_rand_Dic)
            write_file(tar_data_testing_raw_mms_file, 'tar', 'testing', val_raw_mms_rand_Dic)

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
            if ((src_tar_F == 'src' and not 'tar' in name_val)
                or (src_tar_F == 'tar' and 'tar' in name_val)):
                header_L.append(name_val)

        # Write the header
        spamwriter.writerow(header_L)

        # Write the value
        # Get start and end
        if training_testing_F == 'training':
            start = 0
            end = int(training_percentage * len(val_Dic.keys()))
        else:
            start = int(training_percentage * len(val_Dic.keys()))
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
    directory = os.path.dirname(src_data_dir + '/training/raw/std/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(src_data_dir + '/training/raw/mms/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(src_data_dir + '/testing/raw/std/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(src_data_dir + '/testing/raw/mms/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = os.path.dirname(tar_data_dir + '/training/raw/std/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/training/raw/mms/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/testing/raw/std/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/testing/raw/mms/')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for i in range(interation_num):
        for raw_file in os.listdir(raw_file_dir):
            if not raw_file.startswith('.') and raw_file.endswith(".txt"):
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

                # Get src data training file
                src_data_training_raw_std_file = src_data_dir + '/training/raw/std/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data training file
                tar_data_training_raw_std_file = tar_data_dir + '/training/raw/std/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data testing file
                src_data_testing_raw_std_file = src_data_dir + '/testing/raw/std/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data testing file
                tar_data_testing_raw_std_file = tar_data_dir + '/testing/raw/std/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data training file
                src_data_training_raw_mms_file = src_data_dir + '/training/raw/mms/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data training file
                tar_data_training_raw_mms_file = tar_data_dir + '/training/raw/mms/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Get src data testing file
                src_data_testing_raw_mms_file = src_data_dir + '/testing/raw/mms/src_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')
                # Get tar data testing file
                tar_data_testing_raw_mms_file = tar_data_dir + '/testing/raw/mms/tar_data_' + raw_file.replace(file_type, '_' + str(i) + '.txt')

                # Update raw_file
                raw_file = raw_file_dir + raw_file

                # Generate data
                generate_data()

