

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
def helper(statistics_file):
    # Make directory
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize true positive, false positive, and false negative (across all datasets)
    tp_all = 0
    fp_all = 0
    fn_all = 0

    # Write statistics file
    with open(statistics_file, 'w') as f:
        for interaction_ground_truth_file in os.listdir(interaction_ground_truth_dir):
            if not interaction_ground_truth_file.startswith('.') and interaction_ground_truth_file.endswith(".txt"):
                # Get src setting file number
                num = interaction_ground_truth_file
                num = num.replace('interaction_', '')
                num = num.replace('.txt', '')
                # Get interaction_ground_truth_file
                interaction_ground_truth_file = interaction_ground_truth_dir + interaction_ground_truth_file
                # Get importance_file
                importance_file = importance_dir + 'importance_txt_data_data_' + num + '_0'
                importance_file += os.path.basename(statistics_file).replace("statistics_interaction", "")

                # Generate statistics
                [tp, fp, fn] = generate_statistics(interaction_ground_truth_file, importance_file)

                # Write statistics file
                # Write the name of the dataset
                f.write('dataset_' + num + '\n')
                # Write true positive, false positive, and false negative for the current dataset
                f.write('tp: ' + str(tp) + '\n')
                f.write('fp: ' + str(fp) + '\n')
                f.write('fn: ' + str(fn) + '\n')
                f.write('\n\n')

                # Update true positive, false positive, and false negative across all datasets
                tp_all += tp
                fp_all += fp
                fn_all += fn

        # Write statistics
        # Write true positive, false positive, and false negative across all datasets
        f.write('tp_all: ' + str(tp_all) + '\n')
        f.write('fp_all: ' + str(fp_all) + '\n')
        f.write('fn_all: ' + str(fn_all) + '\n')

        # Write precision and recall across all datasets
        if tp_all + fp_all != 0:
            precision = float(tp_all) / (tp_all + fp_all)
        else:
            precision = 'undefined'
        if tp_all + fn_all != 0:
            recall = float(tp_all) / (tp_all + fn_all)
        else:
            recall = 'undefined'
        if precision != 'undefined' and recall != 'undefined' and (precision != 0 or recall != 0):
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 'undefined'

        f.write('precision: ' + str(precision) + '\n')
        f.write('recall: ' + str(recall) + '\n')
        f.write('f1 score: ' + str(f1_score) + '\n\n')


# Generate statistics
def generate_statistics(interaction_ground_truth_file, importance_file):
    # Initialize prob_interaction_ground_truth_L_Dic and interaction_result_Dic
    prob_interaction_ground_truth_L_Dic = {}
    # The number of ground truth components
    component_ground_truth_num = 0
    # The list of components with number component_ground_truth_num
    component_LL = []
    # The list of components with number component_cutoff_num
    component_cutoff_LL = []

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
                # Update component_ground_truth_num
                component_ground_truth_num += 1
            if not target in prob_interaction_ground_truth_L_Dic:
                prob_interaction_ground_truth_L_Dic[target] = []
            prob_interaction_ground_truth_L_Dic[target].append([prob, interaction_ground_truth_LL])

    # Load the importance file
    with open(importance_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get component_LL
        for i in range(component_ground_truth_num):
            component = spamreader[i][0].strip()
            component_LL.append([component, '0', '0'])

        # Get component_cutoff_LL
        for i in range(min(component_cutoff_num, len(spamreader))):
            component = spamreader[i][0].strip()
            component_cutoff_LL.append([component, '0', '0'])

    # Get true positive and false negative for the current dataset
    tp = 0
    fn = 0

    # For each target
    for target in prob_interaction_ground_truth_L_Dic:
        # For each interaction_result
        for prob, interaction_ground_truth_LL in prob_interaction_ground_truth_L_Dic[target]:
            # If the interaction_ground_truth has been discovered
            if belong(interaction_ground_truth_LL, component_cutoff_LL):
                # Increase true positive
                tp += 1
            else:
                # Increase false negative
                fn += 1

    # Get false positive for the current dataset
    fp = 0
    for component_L in component_LL:
        equal_F = False
        # For each interaction_ground_truth and the probability
        for prob, interaction_ground_truth_LL in prob_interaction_ground_truth_L_Dic[target]:
            # If the interaction_result is a interaction_ground_truth
            if component_L in interaction_ground_truth_LL:
                equal_F = True
                break
        # If the interaction_result is not a interaction_ground_truth
        if equal_F is False:
            # Increase false positive
            fp = 1
            break

    if fp == 1 and fn == 0:
        fp = 0

    return [tp, fp, fn]


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
    importance_dir = sys.argv[2]
    statistics_file = sys.argv[3]
    statistics_raw_file = sys.argv[4]
    statistics_raw_std_file = sys.argv[5]
    statistics_raw_mms_file = sys.argv[6]
    component_cutoff_num = int(sys.argv[7])

    helper(statistics_file)
    helper(statistics_raw_file)
    helper(statistics_raw_std_file)
    helper(statistics_raw_mms_file)
