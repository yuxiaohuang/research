

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


# Generate statistics
def generate_statistics():
    # Initialize prob_causal_interaction_L_Dic and interaction_Dic
    prob_causal_interaction_L_Dic = {}
    interaction_Dic = {}

    # Load the causal interaction file
    with open(causal_interaction_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get the target, probability and causal interaction
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Target lies in the first column in each row
            target = spamreader[i][0].strip()
            # Probability lies in the second column in each row
            prob = float(spamreader[i][1].strip())
            # Causal interaction lies in the remaining columns, with the form interactionce_i, win_start_i, win_end_i
            causal_interaction_LL = []
            interactionce_num = (len(spamreader[i]) - 2) // 3
            for j in range(interactionce_num):
                interactionce_L = []
                # Name
                interactionce_L.append(spamreader[i][j * 3 + 2].strip())
                # Window start
                interactionce_L.append(int(spamreader[i][j * 3 + 3].strip()))
                # Window end
                interactionce_L.append(int(spamreader[i][j * 3 + 4].strip()))
                causal_interaction_LL.append(interactionce_L)
            if not target in prob_causal_interaction_L_Dic:
                prob_causal_interaction_L_Dic[target] = []
            prob_causal_interaction_L_Dic[target].append([prob, causal_interaction_LL])

    # Load the interaction file
    with open(interaction_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ' '))
        # Get the target and causal interaction
        for i in range(len(spamreader)):
            # Target lies in the end of the first column in each row
            target = spamreader[i][0].strip()
            target = target.replace('interaction for ', '')
            target = target.replace(':', '')
            # interaction lies in the second column in each row
            interaction = spamreader[i][1].strip()
            interaction = interaction.replace('[', '')
            interaction = interaction.replace(']', '')
            interaction = interaction.replace('\'', '')
            interaction = interaction.split(',')
            interactionce_num = len(interaction) // 3
            interaction_LL = []
            for j in range(interactionce_num):
                interactionce_L = []
                # Name
                interactionce_L.append(interaction[j * 3].strip())
                # Window start
                interactionce_L.append(interaction[j * 3 + 1].strip())
                # Window end
                interactionce_L.append(interaction[j * 3 + 2].strip())
                interaction_LL.append(interactionce_L)
            if not target in interaction_Dic:
                interaction_Dic[target] = []
            interaction_Dic[target].append(interaction_LL)

    # Get true positive and false positive for the current dataset
    tp = 0
    fp = 0
    # For each target
    for target in interaction_Dic:
        # For each interaction
        for interaction_LL in interaction_Dic[target]:
            # Flag, indicating whether the interaction is a causal interaction
            equal_F = False
            if target in prob_causal_interaction_L_Dic:
                # For each causal interaction and the probability
                for prob, causal_interaction_LL in prob_causal_interaction_L_Dic[target]:
                    # If the interaction is a causal interaction
                    if equal(causal_interaction_LL, interaction_LL):
                        equal_F = True
                        break
            # If the interaction is a causal interaction
            if equal_F is True:
                # Increase true positive
                tp += 1
            else:
                # Increase false positive
                fp += 1

    # Get false negative
    fn = 0
    # For each target
    for target in prob_causal_interaction_L_Dic:
        # For each interaction
        for prob, causal_interaction_LL in prob_causal_interaction_L_Dic[target]:
            # Flag, indicating whether the causal interaction has been discovered
            equal_F = False
            if target in interaction_Dic:
                # For each interaction
                for interaction_LL in interaction_Dic[target]:
                    # If the causal interaction has been discovered
                    if equal(interaction_LL, causal_interaction_LL):
                        equal_F = True
                        break
            # If the causal interaction has not been discovered
            if equal_F is False:
                # Increase false negative
                fn += 1

    return [tp, fp, fn]


# Check whether the two interactions are equal
def equal(interaction_i_LL, interaction_j_LL):
    # The two interactions are equal if one belongs to another, and vice versa
    if belong(interaction_i_LL, interaction_j_LL) is True and belong(interaction_j_LL, interaction_i_LL) is True:
        return True
    else:
        return False


# Check whether interaction_i belongs to interaction_j
def belong(interaction_i_LL, interaction_j_LL):
    # If interaction_i is None or empty
    if interaction_i_LL is None or len(interaction_i_LL) == 0:
        return True
    # If interaction_j is None or empty
    elif interaction_j_LL is None or len(interaction_j_LL) == 0:
        return False

    # For each variable in interaction_i
    for var_i, win_start_i, win_end_i in interaction_i_LL:
        # Flag, indicating whether var_i is in interaction_j
        belong_F = False

        # For each variable in interaction_j
        for var_j, win_start_j, win_end_j in interaction_j_LL:
            if var_i == var_j:
                belong_F = True
                break

        # If var_i is not in interaction_j
        if belong_F is False:
            return False

    return True


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    causal_interaction_dir = sys.argv[1]
    interaction_dir = sys.argv[2]
    statistics_file = sys.argv[3]

    # Make directory
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize true positve, false positive, and false negative (across all datasets)
    tp_all = 0
    fp_all = 0
    fn_all = 0

    # Write statistics file
    with open(statistics_file, 'w') as f:
        for causal_interaction_file in os.listdir(causal_interaction_dir):
            if causal_interaction_file.endswith(".txt"):
                # Get source setting file number
                num = causal_interaction_file
                num = num.replace('interaction_', '')
                num = num.replace('.txt', '')
                # Get causal_interaction_file
                causal_interaction_file = causal_interaction_dir + causal_interaction_file
                # Get interaction file
                interaction_file = interaction_dir + 'interaction_' + num + '.txt'

                # Generate statistics
                [tp, fp, fn] = generate_statistics()

                # Write statistics file
                # Write the name of the dataset
                f.write('dataset_' + num + '\n')
                # Write true positive, false positive and false negative for the current dataset
                f.write('tp: ' + str(tp) + '\n')
                f.write('fp: ' + str(fp) + '\n')
                f.write('fn: ' + str(fn) + '\n')
                f.write('\n\n')

                # Update true positive, false positive and false negative across all datasets
                tp_all += tp
                fp_all += fp
                fn_all += fn

        # Write statistics file
        # Write true positive, false positive and false negative across all datasets
        f.write('tp_all: ' + str(tp_all) + '\n')
        f.write('fp_all: ' + str(fp_all) + '\n')
        f.write('fn_all: ' + str(fn_all) + '\n')
        # Write precision and recall across all datasets
        f.write('precision: ' + str(tp_all / (tp_all + fp_all)) + '\n')
        f.write('recall: ' + str(tp_all / (tp_all + fn_all)) + '\n')