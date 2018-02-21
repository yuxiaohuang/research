

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
# The dictionary of value
# key: time->var
# val: value of var at the time
val_Dic = {}

interaction_result_Dic = {}

tar_L = []

interaction_frequency_L = []

component_frequency_Dic = {}

# Initialization
def initialization():
    # Initialization
    global val_Dic, interaction_result_Dic
    val_Dic = {}
    interaction_result_Dic = {}

    # Load source file
    load_data(src_data_file, True)

    # Load target file
    load_data(tar_data_file, False)

    # Load the interaction_result file
    with open(interaction_result_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=' '))
        # Get the target and interaction_ground_truth
        for i in range(len(spamreader)):
            if 'interaction for' in spamreader[i][0]:
                # Target lies in the end of the first column in each row
                target = spamreader[i][0].strip()
                target = target.replace('interaction for ', '')
                target = target.replace(':', '')
                # interaction_result lies in the second column in each row
                interaction_result = spamreader[i][1].strip()
                interaction_result = interaction_result.replace('[', '')
                interaction_result = interaction_result.replace(']', '')
                interaction_result = interaction_result.replace('\'', '')
                interaction_result = interaction_result.split(',')
                component_num = len(interaction_result) // 3
                interaction_result_LL = []

                for j in range(component_num):
                    component_L = []
                    # Name
                    component_L.append(interaction_result[j * 3].strip())
                    # Window start
                    component_L.append(interaction_result[j * 3 + 1].strip())
                    # Window end
                    component_L.append(interaction_result[j * 3 + 2].strip())

                    interaction_result_LL.append(component_L)

                if not target in interaction_result_Dic:
                    interaction_result_Dic[target] = []
                interaction_result_Dic[target].append(interaction_result_LL)


# Load data
def load_data(data_file, x_F):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        # Get tar_L
        if x_F is False:
            # Initialization
            global tar_L
            tar_L = []
            
            for j in range(len(spamreader[0])):
                var = spamreader[0][j].strip()
                tar_L.append(var)

        # Get val_Dic
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            if not i in val_Dic:
                val_Dic[i] = {}
            for j in range(len(spamreader[0])):
                # var's name lies in jth column in the first row
                var = spamreader[0][j].strip()

                # If the value at [i][j] is not missing
                if spamreader[i][j]:
                    # Get the value
                    val = spamreader[i][j].strip()
                    val_Dic[i][var] = int(val)


# Generate frequency
def generate_frequency():
    # Get the frequency of each component and the interaction
    for time in sorted(val_Dic.keys()):
        if target in val_Dic[time]:
            # For each interaction_result
            if target in interaction_result_Dic:
                for interaction_result_LL in interaction_result_Dic[target]:
                    # Ignore the other interactions
                    if not ['src_4_0', '0', '0'] in interaction_result_LL:
                        continue
                    # Flag, indicating whether the interaction is true at the time, 1 by default
                    exist_F = 1

                    for interaction_result_L in interaction_result_LL:
                        # Condition
                        var = interaction_result_L[0]

                        # Initialize component_frequency_Dic
                        if not var in component_frequency_Dic:
                            component_frequency_Dic[var] = []

                        if not var in val_Dic[time] or val_Dic[time][var] == 0:
                            # Update component_frequency_Dic
                            component_frequency_Dic[var].append(0)
                            # Update exist_F
                            exist_F = 0
                        else:
                            # Update component_frequency_Dic
                            component_frequency_Dic[var].append(1)

                    interaction_frequency_L.append(exist_F)


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_dir = sys.argv[1]
    tar_data_dir = sys.argv[2]
    interaction_result_file = sys.argv[3]
    frequency_file = sys.argv[4]

    # Make directory
    directory = os.path.dirname(frequency_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write frequency file
    with open(frequency_file, 'w') as f:
        # Get source and target file
        src_data_file = src_data_dir + os.path.basename(interaction_result_file).replace('interaction', 'src_data')
        tar_data_file = tar_data_dir + os.path.basename(interaction_result_file).replace('interaction', 'tar_data')
        src_data_file = src_data_file.replace('train', 'test')
        tar_data_file = tar_data_file.replace('train', 'test')

        # Initialization
        initialization()

        # Write the name of the dataset
        f.write(interaction_result_file + '\n')

        for target in tar_L:
            # Generate frequency
            generate_frequency()

            # Write frequency file
            # Write the target
            f.write('statistics for target: ' + target + '\n')
            frequency_L  = []
            for var in component_frequency_Dic:
                frequency = np.mean(component_frequency_Dic[var])
                f.write('var: ' + var + '\n')
                f.write('frequency: ' + str(frequency) + '\n\n')
                frequency_L.append(frequency)
            average_var_frequency = np.mean(frequency_L)
            f.write('average var frequency: ' + str(average_var_frequency) + '\n\n')

            theoretical_interaction_frequency = average_var_frequency ** len(frequency_L)
            f.write('theoretical interaction frequency: ' + str(theoretical_interaction_frequency) + '\n\n')

            actual_interaction_frequency = np.mean(interaction_frequency_L)
            f.write('actual interaction frequency: ' + str(actual_interaction_frequency) + '\n\n')