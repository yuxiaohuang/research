

# Please cite the following paper when using the code


# Modules
import sys
import os
import csv
import numpy as np

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Global variables
# The list of interaction size
interaction_size_L = []
interaction_size_raw_L = []
interaction_size_raw_mms_L = []
interaction_size_raw_std_L = []


# Update interaction_size_L
def generate_average_interaction_size():
    # Load the interaction_file
    with open(interaction_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=' '))
        # From header to the last
        for i in range(len(spamreader)):
            if 'interaction for' in spamreader[i][0]:
                interaction_result = spamreader[i][1].strip()
                interaction_result = interaction_result.replace('[', '')
                interaction_result = interaction_result.replace(']', '')
                interaction_result = interaction_result.replace('\'', '')
                interaction_result = interaction_result.split(',')
                component_num = len(interaction_result) // 3
                if 'raw_std' in interaction_file:
                    interaction_size_raw_std_L.append(component_num)
                elif 'raw_mms' in interaction_file:
                    interaction_size_raw_mms_L.append(component_num)
                elif 'raw' in interaction_file:
                    interaction_size_raw_L.append(component_num)
                else:
                    interaction_size_L.append(component_num)


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    interaction_dir = sys.argv[1]
    average_interaction_size_file = sys.argv[2]

    # Make directory
    directory = os.path.dirname(average_interaction_size_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write statistics file
    with open(average_interaction_size_file, 'w') as f:
        for interaction_file in os.listdir(interaction_dir):
            if not interaction_file.startswith('.') and interaction_file.endswith(".txt"):
                # Get interaction_result file
                interaction_file = interaction_dir + interaction_file

                # Update interaction_size_L
                generate_average_interaction_size()

        # Write average interaction size
        average_interaction_size = None
        if len(interaction_size_L) != 0:
            average_interaction_size = float(sum(interaction_size_L)) / float(len(interaction_size_L))
        f.write('average_interaction_size: ' + str(average_interaction_size) + '\n')

        average_interaction_raw_size = None
        if len(interaction_size_raw_L) != 0:
            average_interaction_raw_size = float(sum(interaction_size_raw_L)) / float(len(interaction_size_raw_L))
        f.write('average_interaction_raw_size: ' + str(average_interaction_raw_size) + '\n')

        average_interaction_raw_mms_size = None
        if len(interaction_size_raw_mms_L) != 0:
            average_interaction_raw_mms_size = float(sum(interaction_size_raw_mms_L)) / float(len(interaction_size_raw_mms_L))
        f.write('average_interaction_raw_mms_size: ' + str(average_interaction_raw_mms_size) + '\n')

        average_interaction_raw_std_size = None
        if len(interaction_size_raw_std_L) != 0:
            average_interaction_raw_std_size = float(sum(interaction_size_raw_std_L)) / float(len(interaction_size_raw_std_L))
        f.write('average_interaction_raw_std_size: ' + str(average_interaction_raw_std_size) + '\n\n')
