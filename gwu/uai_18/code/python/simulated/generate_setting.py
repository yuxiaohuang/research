

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


# Generate the setting
def generate_setting(type):
    # Write the setting file
    with open(setting_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(['var', 'probability'])

        random.seed()

        # Get var number
        var_num = random.randint(var_num_range_L[0], var_num_range_L[1])

        for i in range(var_num):
            if type == 'src':
                # Get the number of times
                num = random.randint(num_range_L[0], num_range_L[1])

            # Get the probability
            prob = random.uniform(prob_range_L[0], prob_range_L[1])

            if type == 'src':
                # Write the var, num and probability
                spamwriter.writerow([type + '_' + str(i), num, prob])
            else:
                # Write the var and probability
                # spamwriter.writerow([type + '_' + str(i), prob])
                spamwriter.writerow([type + '_' + str(1), prob])


# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    setting_dir = sys.argv[1]
    type = sys.argv[2]
    setting_num = int(sys.argv[3])
    var_num_range_L = [int(sys.argv[4]), int(sys.argv[5])]
    prob_range_L = [float(sys.argv[6]), float(sys.argv[7])]
    if type == 'src':
        num_range_L = [int(sys.argv[8]), int(sys.argv[9])]

    # Make directory
    directory = os.path.dirname(setting_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(setting_num):
        # Get setting file
        setting_file = setting_dir + type + '_setting_' + str(i) + '.txt'
        # Generate the setting
        generate_setting(type)
