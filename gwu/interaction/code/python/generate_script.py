

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
        script = 'python3' + ' ' + py_file + ' ' + src_data_dir + src_data_file + ' ' + tar_data_file + ' ' + interaction_file + ' ' + log_file + ' ' + fig_dir_num +  ' ' + p_val_cutoff + ' ' + sample_size_cutoff
        for lag in lag_L:
            script += ' ' + lag
        # Write the file
        f.write(script + '\n')

# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    py_file = sys.argv[1]
    script_dir = sys.argv[2]
    src_data_dir = sys.argv[3]
    tar_data_dir = sys.argv[4]
    interaction_dir = sys.argv[5]
    log_dir = sys.argv[6]
    fig_dir = sys.argv[7]
    p_val_cutoff = sys.argv[8]
    sample_size_cutoff = sys.argv[9]
    lag_L = sys.argv[10:]

    # Make directory
    directory = os.path.dirname(script_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(interaction_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(log_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


    for src_data_file in os.listdir(src_data_dir):
        if src_data_file.endswith(".txt"):
            # Get source data file number
            num = src_data_file
            num = num.replace('src_data_', '')
            num = num.replace('.txt', '')
            # Get script file
            script_file = script_dir + 'script_' + num + '.txt'
            # Get target data file
            tar_data_file = tar_data_dir + 'tar_data_' + num + '.txt'
            # Get interaction file
            interaction_file = interaction_dir + 'interaction_' + num + '.txt'
            # Get log file
            log_file = log_dir + 'log_' + num + '.txt'
            # Get fig_dir
            fig_dir_num = fig_dir + num + '/'
            directory = os.path.dirname(fig_dir_num)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Generate the script
            generate_script()

