

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
        suffix = os.path.basename(statistics_file).replace('statistics_interaction', '')
        for importance_file in os.listdir(importance_dir):
            if not importance_file.startswith('.') and importance_file.endswith(suffix):
                if not 'raw' in statistics_file and 'raw' in importance_file:
                    continue
                # Get importance_file
                if importance_file.startswith('importance_txt_data_adult-stretch.data'):
                    num = importance_file.replace('importance_txt_data_adult-stretch.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    importance_file = importance_dir + 'importance_txt_data_adult-stretch.data_' + num + suffix
                elif importance_file.startswith('importance_txt_data_adult+stretch.data'):
                    num = importance_file.replace('importance_txt_data_adult+stretch.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    importance_file = importance_dir + 'importance_txt_data_adult+stretch.data_' + num + suffix
                elif importance_file.startswith('importance_txt_data_yellow-small.data'):
                    num = importance_file.replace('importance_txt_data_yellow-small.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    importance_file = importance_dir + 'importance_txt_data_yellow-small.data_' + num + suffix
                elif importance_file.startswith('importance_txt_data_yellow-small+adult-stretch.data'):
                    num = importance_file.replace(
                        'importance_txt_data_yellow-small+adult-stretch.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    importance_file = importance_dir + 'importance_txt_data_yellow-small+adult-stretch.data_' + num + suffix

                # Generate statistics
                [tp, fp, fn] = generate_statistics(importance_file)

                # Write statistics file
                # Write the name of the dataset
                f.write('dataset_' + os.path.basename(importance_file) + '\n')
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
def generate_statistics(importance_file):
    # Initialize prob_interaction_ground_truth_L_Dic and importance_Dic
    prob_interaction_ground_truth_L_Dic = {}
    target_ground_truth = 'tar_T'
    prob_interaction_ground_truth_L_Dic[target_ground_truth] = []
    # The number of ground truth components
    component_ground_truth_num = 0
    # The list of components with number component_ground_truth_num
    component_LL = []
    # The list of components with number component_cutoff_num
    component_cutoff_LL = []

    # # Get prob_interaction_ground_truth_L_Dic
    # if os.path.basename(importance_file).startswith('importance_txt_data_adult-stretch.data'):
    #     prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2', 0, 0]]])
    #     prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_3', 0, 0]]])
    #     component_ground_truth_num = 2
    # elif os.path.basename(importance_file).startswith('importance_txt_data_adult+stretch.data'):
    #     prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2', 0, 0], ['src_3', 0, 0]]])
    #     component_ground_truth_num = 2
    # elif os.path.basename(importance_file).startswith('importance_txt_data_yellow-small.data'):
    #     prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_0', 0, 0], ['src_1', 0, 0]]])
    #     component_ground_truth_num = 2
    # elif os.path.basename(importance_file).startswith('importance_txt_data_yellow-small+adult-stretch.data'):
    #     prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2', 0, 0], ['src_3', 0, 0]]])
    #     prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_0', 0, 0], ['src_1', 0, 0]]])
    #     component_ground_truth_num = 4

    # Get prob_interaction_ground_truth_L_Dic
    if os.path.basename(importance_file).startswith('importance_txt_data_adult-stretch.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2_STRETCH', 0, 0]]])
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_3_ADULT', 0, 0]]])
        # Update positive number
        component_ground_truth_num = 2
    elif os.path.basename(importance_file).startswith('importance_txt_data_adult+stretch.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2_STRETCH', 0, 0], ['src_3_ADULT', 0, 0]]])
        # Update positive number
        component_ground_truth_num = 2
    elif os.path.basename(importance_file).startswith('importance_txt_data_yellow-small.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_0_YELLOW', 0, 0], ['src_1_SMALL', 0, 0]]])
        # Update positive number
        component_ground_truth_num = 2
    elif os.path.basename(importance_file).startswith('importance_txt_data_yellow-small+adult-stretch.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2_STRETCH', 0, 0], ['src_3_ADULT', 0, 0]]])
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_0_YELLOW', 0, 0], ['src_1_SMALL', 0, 0]]])
        # Update positive number
        component_ground_truth_num = 4

    # Load the importance file
    with open(importance_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get component_LL
        for i in range(component_ground_truth_num):
            component = spamreader[i][0].strip()
            # if 'src_0' in component:
            #     component = 'src_0'
            # elif 'src_1' in component:
            #     component = 'src_1'
            # elif 'src_2' in component:
            #     component = 'src_2'
            # elif 'src_3' in component:
            #     component = 'src_3'
            component_LL.append([component, '0', '0'])

        # Get component_cutoff_LL
        for i in range(min(component_cutoff_num, len(spamreader))):
            component = spamreader[i][0].strip()
            # if 'src_0' in component:
            #     component = 'src_0'
            # elif 'src_1' in component:
            #     component = 'src_1'
            # elif 'src_2' in component:
            #     component = 'src_2'
            # elif 'src_3' in component:
            #     component = 'src_3'
            component_cutoff_LL.append([component, '0', '0'])

    # Get true positive and false negative for the current dataset
    tp = 0
    fn = 0

    # For each target
    for target in prob_interaction_ground_truth_L_Dic:
        # For each importance
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
            # If the importance is a interaction_ground_truth
            if component_L in interaction_ground_truth_LL:
                equal_F = True
                break
        # If the importance is not a interaction_ground_truth
        if equal_F is False:
            # Increase false positive
            fp = 1
            break

    if fp == 1 and fn == 0:
        fp = 0

    return [tp, fp, fn]


# Check whether importance_i belongs to importance_j
def belong(importance_i_LL, importance_j_LL):
    # If importance_i is None or empty
    if importance_i_LL is None or len(importance_i_LL) == 0:
        return True
    # If importance_j is None or empty
    elif importance_j_LL is None or len(importance_j_LL) == 0:
        return False

    # For each variable in importance_i
    for var_i, win_start_i, win_end_i in importance_i_LL:
        # Flag, indicating whether var_i is in importance_j
        belong_F = False

        # For each variable in importance_j
        for var_j, win_start_j, win_end_j in importance_j_LL:
            if var_i == var_j:
                belong_F = True
                break

        # If var_i is not in importance_j
        if belong_F is False:
            return False

    return True


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    importance_dir = sys.argv[1]
    statistics_file = sys.argv[2]
    statistics_raw_file = sys.argv[3]
    statistics_raw_std_file = sys.argv[4]
    statistics_raw_mms_file = sys.argv[5]
    component_cutoff_num = int(sys.argv[6])

    helper(statistics_file)
    helper(statistics_raw_file)
    helper(statistics_raw_std_file)
    helper(statistics_raw_mms_file)
