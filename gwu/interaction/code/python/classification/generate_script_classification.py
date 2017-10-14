

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


# Generate the script
def generate_script():
    # Write the script file
    with open(script_file, 'w') as f:
        script = 'python' + ' ' + py_file + ' ' + src_data_training_file + ' ' + tar_data_training_file + ' ' + src_data_testing_file + ' ' + tar_data_testing_file + ' ' + statistics_file
        f.write(script + '\n')

# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    py_file = sys.argv[1]
    script_dir = sys.argv[2]
    src_data_training_dir = sys.argv[3]
    tar_data_training_dir = sys.argv[4]
    src_data_testing_dir = sys.argv[5]
    tar_data_testing_dir = sys.argv[6]
    statistics_file = sys.argv[7]

    # Make directory
    directory = os.path.dirname(script_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for src_data_training_file in os.listdir(src_data_training_dir):
        if src_data_training_file.endswith(".txt"):
            tar_data_training_file = tar_data_training_dir + src_data_training_file.replace('src', 'tar')
            src_data_testing_file = src_data_testing_dir + src_data_training_file.replace('train', 'test')
            tar_data_testing_file = tar_data_testing_dir + src_data_training_file.replace('src', 'tar').replace('train', 'test')
            script_file = script_dir + src_data_training_file.replace('src_data', 'script')
            src_data_training_file = src_data_training_dir + src_data_training_file

            # Generate the script
            generate_script()

