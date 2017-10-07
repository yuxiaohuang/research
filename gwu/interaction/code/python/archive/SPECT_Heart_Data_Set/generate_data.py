

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
name_L = ['OVERALL_DIAGNOSIS', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22']

# The dictionary of value of each var at each time
# key: time->var
# val: value of each var at each time
val_Dic = {}


# Generate source and target data
def generate_data():
    # Load the raw file
    with open(raw_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

    # Get the maximum time stamp
    max_time_stamp = len(spamreader)

    # From the first line to the last (since there is no header)
    for i in range(0, max_time_stamp):
        # Initialization
        if not i in val_Dic:
            val_Dic[i] = {}
        if not (i + 1) in val_Dic:
            val_Dic[i + 1] = {}

        for j in range(0, len(name_L)):
            # Get the name of the var
            name = name_L[j]

            # Get the value of the var
            val = spamreader[i][j].strip()

            # Update val_Dic
            if 'OVERALL_DIAGNOSIS' == name:
                val_Dic[i + 1][name] = val
            else:
                val_Dic[i][name] = val

    # Write the source file
    with open(src_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Write the header
        spamwriter.writerow(name_L[1:])

        for i in range(10000):
            # Write the value
            for time in range(0, max_time_stamp + 1):
                # The list of value at the time
                val_L = []

                if time in val_Dic:
                    for name in name_L[1:]:
                        if name in val_Dic[time]:
                            val_L.append(val_Dic[time][name])
                        else:
                            val_L.append(0)
                else:
                    for name in name_L[1:]:
                        val_L.append(0)

                spamwriter.writerow(val_L)

    # Write the target file
    with open(tar_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Write the header
        spamwriter.writerow(list(name_L[:1]))

        for i in range(10000):
            # Write the value
            for time in range(0, max_time_stamp + 1):
                # The list of value at the time
                val_L = []

                if time in val_Dic:
                    if name_L[0] in val_Dic[time]:
                        val_L.append(val_Dic[time][name_L[0]])
                    else:
                        val_L.append(0)
                else:
                    val_L.append(0)

                spamwriter.writerow(val_L)


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_file_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]

    # Make directory
    directory = os.path.dirname(src_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for raw_file in os.listdir(raw_file_dir):
        if raw_file.endswith(".txt"):
            # Get src data file
            src_data_file = src_data_dir + 'src_data' + '_' + raw_file
            # Get tar data file
            tar_data_file = tar_data_dir + 'tar_data' + '_' + raw_file

            # Update raw_file
            raw_file = raw_file_dir + raw_file
            
            # Generate data
            generate_data()
