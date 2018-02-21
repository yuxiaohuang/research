

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



# Global variable
# The minimum list of components that cover all the ground truth
component_min_LL = []

# The minimum list of component-importance pair that cover all the ground truth
component_importance_min_LL = []



# Generate statistics
def generate_statistics(statistics_file):
    # Initilization
    component_LL = []
    component_importance_LL = []

    # Load the statistics file
    with open(statistics_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        # Get component_LL
        for i in range(len(spamreader)):
            # Name
            name = spamreader[i][0].strip()
            # If condition
            if '_' in name.replace('src_', ''):
                # Remove the value
                name = name[:-2]
            if not [name, '0', '0'] in component_LL:
                component_LL.append([name, '0', '0'])
            # Importance
            importance = spamreader[i][1].strip()
            if not [name, importance] in component_importance_LL:
                component_importance_LL.append([name, importance])

    # Get false positive for the current dataset
    for idx in range(len(component_LL)):
        fp = 0
        # For each target
        for target in prob_interaction_ground_truth_L_Dic:
            # For each interaction_result
            for prob, interaction_ground_truth_LL in prob_interaction_ground_truth_L_Dic[target]:
                # If the interaction_ground_truth has not been discovered
                if belong(interaction_ground_truth_LL, component_LL[:idx]) is False:
                    # Increase true positive
                    fp = 1
                    break
            # If all the interaction_ground_truth have been discovered
            if fp == 1:
                break
        # If all the interaction_ground_truth have been discovered
        if fp == 0:
            break
    global component_min_LL
    global component_importance_min_LL

    # If component_min_LL is empty or is smaller than component_LL[:idx]
    if len(component_min_LL) == 0 or len(component_min_LL) > idx:
        print(statistics_file)
        component_min_LL = list(component_LL[:idx])
        component_importance_min_LL = list(component_importance_LL[:idx])


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
    interaction_ground_truth_file = sys.argv[1]
    importance_file = sys.argv[2]
    importance_raw_file = sys.argv[3]
    importance_raw_std_file = sys.argv[4]
    importance_raw_mms_file = sys.argv[5]
    component_importance_min_LL_file = sys.argv[6]

    directory = os.path.dirname(component_importance_min_LL_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize prob_interaction_ground_truth_L_Dic
    prob_interaction_ground_truth_L_Dic = {}

    # Load the interaction_ground_truth file
    with open(interaction_ground_truth_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))
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
                name = spamreader[i][j * 3 + 2].strip()
                # If condition
                if '_' in name.replace('src_', ''):
                    # Remove the value
                    name = name[:-2]
                component_L.append(name)
                # Window start
                component_L.append(int(spamreader[i][j * 3 + 3].strip()))
                # Window end
                component_L.append(int(spamreader[i][j * 3 + 4].strip()))
                interaction_ground_truth_LL.append(component_L)
            if not target in prob_interaction_ground_truth_L_Dic:
                prob_interaction_ground_truth_L_Dic[target] = []
            prob_interaction_ground_truth_L_Dic[target].append([prob, interaction_ground_truth_LL])

    generate_statistics(importance_file)
    generate_statistics(importance_raw_file)
    generate_statistics(importance_raw_std_file)
    generate_statistics(importance_raw_mms_file)

    # Write statistics
    with open(component_importance_min_LL_file, 'w') as f:
        for component_importance_L in component_importance_min_LL:
            f.write(str(component_importance_L[0]) + ', ' + str(component_importance_L[1]) + '\n')

