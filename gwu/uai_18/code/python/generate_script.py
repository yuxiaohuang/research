

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
        script = 'python' + ' ' + py_file + ' ' + attribute_training_data_file + ' ' + class_training_data_file + ' ' + attribute_testing_data_file + ' ' + class_testing_data_file + ' ' + fit_file + ' ' + predict_file + ' ' + statistics_file + ' ' + log_file + ' ' + max_iteration_cutoff + ' ' + min_number_of_times_cutoff + ' ' + min_number_of_times_ratio_cutoff + ' ' + p_val_cutoff + ' ' + header
        # Write the file
        f.write(script + '\n')


# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    py_file = sys.argv[1]
    script_dir = sys.argv[2]
    attribute_training_data_dir = sys.argv[3]
    class_training_data_dir = sys.argv[4]
    attribute_testing_data_dir = sys.argv[5]
    class_testing_data_dir = sys.argv[6]
    fit_dir = sys.argv[7]
    predict_dir = sys.argv[8]
    statistics_dir = sys.argv[9]
    log_dir = sys.argv[10]
    max_iteration_cutoff = sys.argv[11]
    min_number_of_times_cutoff = sys.argv[12]
    min_number_of_times_ratio_cutoff = sys.argv[13]
    p_val_cutoff = sys.argv[14]
    header = sys.argv[15]

    # Make directory
    directory = os.path.dirname(script_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(fit_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(predict_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(statistics_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(log_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for attribute_training_data_file in os.listdir(attribute_training_data_dir):
        if not attribute_training_data_file.startswith('.') and attribute_training_data_file.endswith(".txt"):
            # Get attribute training data file number
            num = attribute_training_data_file
            num = num.replace('attribute_data_', '')
            num = num.replace('.txt', '')

            # Update attribute_training_data_file
            attribute_training_data_file = attribute_training_data_dir + attribute_training_data_file

            # Get script file
            script_file = script_dir + 'script_' + num + '.txt'

            # Get class training data file
            class_training_data_file = class_training_data_dir + 'class_data_' + num + '.txt'

            # Get attribute testing data file
            attribute_testing_data_file = attribute_testing_data_dir + 'attribute_data_' + num + '.txt'

            # Get class testing data file
            class_testing_data_file = class_testing_data_dir + 'class_data_' + num + '.txt'

            # Get fit file
            fit_file = fit_dir + 'fit_' + num + '.txt'

            # Get predict file
            predict_file = predict_dir + 'predict_' + num + '.txt'

            # Get statistics file
            statistics_file = statistics_dir + 'statistics_' + num + '.txt'

            # Get log file
            log_file = log_dir + 'log_' + num + '.txt'

            # Generate the script
            generate_script()

