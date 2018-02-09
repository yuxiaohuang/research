

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
        script = 'python' + ' ' + py_file + ' ' + attribute_data_dir + attribute_data_file + ' ' + class_data_file + ' ' + knowledge_file + ' ' + log_file + ' ' + max_iteration_cutoff +  ' ' + min_number_of_times_cutoff + ' ' + min_number_of_times_ratio_cutoff + ' ' + p_val_cutoff + ' ' + header
        # Write the file
        f.write(script + '\n')

# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    py_file = sys.argv[1]
    script_dir = sys.argv[2]
    attribute_data_dir = sys.argv[3]
    class_data_dir = sys.argv[4]
    knowledge_dir = sys.argv[5]
    log_dir = sys.argv[6]
    max_iteration_cutoff = sys.argv[7]
    min_number_of_times_cutoff = sys.argv[8]
    min_number_of_times_ratio_cutoff = sys.argv[9]
    p_val_cutoff = sys.argv[10]
    header = sys.argv[11]

    # Make directory
    directory = os.path.dirname(script_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(knowledge_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(log_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for attribute_data_file in os.listdir(attribute_data_dir):
        if not attribute_data_file.startswith('.') and attribute_data_file.endswith(".txt"):
            # Get source data file number
            num = attribute_data_file
            num = num.replace('attribute_data_', '')
            num = num.replace('.txt', '')
            # Get script file
            script_file = script_dir + 'script_' + num + '.txt'
            # Get target data file
            class_data_file = class_data_dir + 'class_data_' + num + '.txt'
            # Get interaction file
            knowledge_file = knowledge_dir + 'knowledge_' + num + '.txt'
            # Get log file
            log_file = log_dir + 'log_' + num + '.txt'

            # Generate the script
            generate_script()

