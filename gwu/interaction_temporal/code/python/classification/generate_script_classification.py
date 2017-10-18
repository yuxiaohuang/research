

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
        script = 'python' + ' ' + py_file + ' ' + src_data_training_dir + ' ' + tar_data_training_dir + ' ' + src_data_testing_dir + ' ' + tar_data_testing_dir + ' ' + statistics_file
        f.write(script + '\n')

# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    py_file = sys.argv[1]
    script_file = sys.argv[2]
    src_data_training_dir = sys.argv[3]
    tar_data_training_dir = sys.argv[4]
    src_data_testing_dir = sys.argv[5]
    tar_data_testing_dir = sys.argv[6]
    statistics_file = sys.argv[7]

    # Make directory
    directory = os.path.dirname(script_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate the script
    generate_script()

