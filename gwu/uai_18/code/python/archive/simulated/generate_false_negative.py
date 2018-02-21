

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


# Helper function
def helper(false_negative_file):
    # Make directory
    directory = os.path.dirname(false_negative_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write false_negative file
    with open(false_negative_file, 'w') as f_false_negative:
        for interaction_ground_truth_file in os.listdir(interaction_ground_truth_dir):
            if not interaction_ground_truth_file.startswith('.') and interaction_ground_truth_file.endswith(".txt"):
                # Get src setting file number
                num = interaction_ground_truth_file
                num = num.replace('interaction_', '')
                num = num.replace('.txt', '')
                # Get interaction_ground_truth_file
                interaction_ground_truth_file = interaction_ground_truth_dir + interaction_ground_truth_file
                # Get interaction_result file
                interaction_result_file = interaction_result_dir + 'interaction_data_' + num + '_0'
                interaction_result_file += os.path.basename(false_negative_file).replace("false_negative_interaction", "")

                # Generate false_negative
                generate_false_negative(interaction_ground_truth_file, interaction_result_file, f_false_negative)


# Generate false_negative
def generate_false_negative(interaction_ground_truth_file, interaction_result_file, f_false_negative):
    # Initialize prob_interaction_ground_truth_L_Dic, interaction_result_Dic, and positive number
    prob_interaction_ground_truth_L_Dic = {}
    interaction_result_Dic = {}

    # Load the interaction_ground_truth file
    with open(interaction_ground_truth_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get the target, probability and interaction_ground_truth
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Target lies in the first column in each row
            target = spamreader[i][0].strip()
            # Probability lies in the second column in each row
            prob = float(spamreader[i][1].strip())
            # interaction_ground_truth lies in the remaining columns, with the form component_i, win_start_i, win_end_i
            interaction_ground_truth_LL = []
            component_num = (len(spamreader[i]) - 2) // 3
            for j in range(component_num):
                component_L = []
                # Name
                component_L.append(spamreader[i][j * 3 + 2].strip())
                # Window start
                component_L.append(int(spamreader[i][j * 3 + 3].strip()))
                # Window end
                component_L.append(int(spamreader[i][j * 3 + 4].strip()))
                interaction_ground_truth_LL.append(component_L)
            if not target in prob_interaction_ground_truth_L_Dic:
                prob_interaction_ground_truth_L_Dic[target] = []
            prob_interaction_ground_truth_L_Dic[target].append([prob, interaction_ground_truth_LL])

    # Load the interaction_result file
    with open(interaction_result_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ' '))
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
                    component_L.append(int(interaction_result[j * 3 + 1].strip()))
                    # Window end
                    component_L.append(int(interaction_result[j * 3 + 2].strip()))
                    interaction_result_LL.append(component_L)
                if not target in interaction_result_Dic:
                    interaction_result_Dic[target] = []
                interaction_result_Dic[target].append(interaction_result_LL)

    # For each target
    for target in prob_interaction_ground_truth_L_Dic:
        # For each interaction_result
        for prob, interaction_ground_truth_LL in prob_interaction_ground_truth_L_Dic[target]:
            # Flag, indicating whether the interaction_ground_truth has been discovered
            equal_F = False
            if target in interaction_result_Dic:
                # For each interaction_result
                for interaction_result_LL in interaction_result_Dic[target]:
                    # If the interaction_ground_truth has been discovered
                    if belong(interaction_ground_truth_LL, interaction_result_LL):
                        equal_F = True
                        break
            # If the interaction_ground_truth has not been discovered
            if equal_F is False:
                f_false_negative.write('interaction_result_file: ' + os.path.basename(interaction_result_file) + '\n')
                f_false_negative.write('false_negative: ' + str(interaction_ground_truth_LL) + '\n\n')


# Check whether the two interaction_results are equal
def equal(interaction_result_i_LL, interaction_result_j_LL):
    # The two interaction_results are equal if one belongs to another, and vice versa
    if belong(interaction_result_i_LL, interaction_result_j_LL) is True and belong(interaction_result_j_LL, interaction_result_i_LL) is True:
        return True
    else:
        return False


# Check whether interaction_result_i belongs to interaction_result_j
def belong(interaction_result_i_LL, interaction_result_j_LL):
    # If interaction_result_i is None or empty
    if interaction_result_i_LL is None or len(interaction_result_i_LL) == 0:
        return True
    # If interaction_result_j is None or empty
    elif interaction_result_j_LL is None or len(interaction_result_j_LL) == 0:
        return False

    # For each variable in interaction_result_i
    for var_i, win_start_i, win_end_i in interaction_result_i_LL:
        # Flag, indicating whether var_i is in interaction_result_j
        belong_F = False

        # For each variable in interaction_result_j
        for var_j, win_start_j, win_end_j in interaction_result_j_LL:
            if var_i == var_j:
                belong_F = True
                break

        # If var_i is not in interaction_result_j
        if belong_F is False:
            return False

    return True


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    interaction_ground_truth_dir = sys.argv[1]
    interaction_result_dir = sys.argv[2]
    false_negative_file = sys.argv[3]
    false_negative_raw_file = sys.argv[4]
    false_negative_raw_std_file = sys.argv[5]
    false_negative_raw_mms_file = sys.argv[6]

    helper(false_negative_file)
    helper(false_negative_raw_file)
    helper(false_negative_raw_std_file)
    helper(false_negative_raw_mms_file)

