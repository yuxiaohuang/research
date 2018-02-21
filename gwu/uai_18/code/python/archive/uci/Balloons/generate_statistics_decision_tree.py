

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

    # Initialize true positve, false positive, false negative, and run time (across all datasets)
    tp_all = 0
    fp_all = 0
    fn_all = 0
    run_time_all = 0

    # Write statistics file
    with open(statistics_file, 'w') as f:
        suffix = os.path.basename(statistics_file).replace('statistics_interaction', '')
        for interaction_result_file in os.listdir(interaction_result_dir):
            if not interaction_result_file.startswith('.') and interaction_result_file.endswith(suffix):
                if not 'raw' in statistics_file and 'raw' in interaction_result_file:
                    continue
                # Get interaction_result_file
                if interaction_result_file.startswith('interaction_adult-stretch.data'):
                    num = interaction_result_file.replace('interaction_adult-stretch.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    interaction_result_file = interaction_result_dir + 'interaction_adult-stretch.data_' + num + suffix
                elif interaction_result_file.startswith('interaction_adult+stretch.data'):
                    num = interaction_result_file.replace('interaction_adult+stretch.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    interaction_result_file = interaction_result_dir + 'interaction_adult+stretch.data_' + num + suffix
                elif interaction_result_file.startswith('interaction_yellow-small.data'):
                    num = interaction_result_file.replace('interaction_yellow-small.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    interaction_result_file = interaction_result_dir + 'interaction_yellow-small.data_' + num + suffix
                elif interaction_result_file.startswith('interaction_yellow-small+adult-stretch.data'):
                    num = interaction_result_file.replace('interaction_yellow-small+adult-stretch.data_', '')
                    num = num.replace(suffix, '')
                    num = num.replace('.txt', '')
                    interaction_result_file = interaction_result_dir + 'interaction_yellow-small+adult-stretch.data_' + num + suffix

                # Generate statistics
                [tp, fp, fn, run_time] = generate_statistics(interaction_result_file)

                # Write statistics file
                # Write the name of the dataset
                f.write('dataset_' + os.path.basename(interaction_result_file) + '\n')
                # Write true positive, false positive and false negative for the current dataset
                f.write('tp: ' + str(tp) + '\n')
                f.write('fp: ' + str(fp) + '\n')
                f.write('fn: ' + str(fn) + '\n')
                f.write('\n\n')

                # Update true positive, false positive, false negative, and run time across all datasets
                tp_all += tp
                fp_all += fp
                fn_all += fn
                run_time_all += run_time

        # Write statistics
        # Write true positive, false positive and false negative across all datasets
        f.write('run_time_all: ' + str(run_time_all) + '\n')
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
def generate_statistics(interaction_result_file):
    # Initialize prob_interaction_ground_truth_L_Dic and interaction_result_Dic
    prob_interaction_ground_truth_L_Dic = {}
    target_ground_truth = 'tar_T'
    prob_interaction_ground_truth_L_Dic[target_ground_truth] = []
    interaction_result_Dic = {}
    positive_num = 0

    # Get prob_interaction_ground_truth_L_Dic
    if os.path.basename(interaction_result_file).startswith('interaction_adult-stretch.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2_0', 0, 0]]])
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_3_0', 0, 0]]])
        positive_num += 2
    elif os.path.basename(interaction_result_file).startswith('interaction_adult+stretch.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2_0', 0, 0], ['src_3_0', 0, 0]]])
        positive_num += 1
    elif os.path.basename(interaction_result_file).startswith('interaction_yellow-small.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_0_0', 0, 0], ['src_1_0', 0, 0]]])
        positive_num += 1
    elif os.path.basename(interaction_result_file).startswith('interaction_yellow-small+adult-stretch.data'):
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_2_0', 0, 0], ['src_3_0', 0, 0]]])
        prob_interaction_ground_truth_L_Dic[target_ground_truth].append([1.0, [['src_0_0', 0, 0], ['src_1_0', 0, 0]]])
        positive_num += 2

    # Load the interaction_result file
    with open(interaction_result_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ' '))
        # Get the target and interaction_ground_truth
        for i in range(len(spamreader)):
            if 'interaction for' in spamreader[i][0]:
                # Target lies in the end of the first column in each row
                # target = spamreader[i][0].strip()
                # target = target.replace('interaction for ', '')
                # target = target.replace(':', '')
                target = target_ground_truth
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
                    # attribute_val_pair
                    attribute_val_pair = interaction_result[j * 3].strip()
                    if 'YELLOW_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('YELLOW_0', '1')
                    elif 'YELLOW_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('YELLOW_1', '0')
                    elif 'PURPLE_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('PURPLE_0', '0')
                    elif 'PURPLE_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('PURPLE_1', '1')
                    elif 'SMALL_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('SMALL_0', '1')
                    elif 'SMALL_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('SMALL_1', '0')
                    elif 'LARGE_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('LARGE_0', '0')
                    elif 'LARGE_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('LARGE_1', '1')
                    elif 'STRETCH_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('STRETCH_0', '1')
                    elif 'STRETCH_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('STRETCH_1', '0')
                    elif 'DIP_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('DIP_0', '0')
                    elif 'DIP_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('DIP_1', '1')
                    elif 'ADULT_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('ADULT_0', '1')
                    elif 'ADULT_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('ADULT_1', '0')
                    elif 'CHILD_0' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('CHILD_0', '0')
                    elif 'CHILD_1' in attribute_val_pair:
                        attribute_val_pair = attribute_val_pair.replace('CHILD_1', '1')

                    component_L.append(attribute_val_pair)
                    # Window start
                    component_L.append(int(interaction_result[j * 3 + 1].strip()))
                    # Window end
                    component_L.append(int(interaction_result[j * 3 + 2].strip()))
                    interaction_result_LL.append(component_L)
                if not target in interaction_result_Dic:
                    interaction_result_Dic[target] = []
                interaction_result_Dic[target].append(interaction_result_LL)
            elif 'run time' in spamreader[i][0]:
                run_time = float(spamreader[i][0].replace('run time: ', '').strip())

    # Get true positive and false positive for the current dataset
    tp = 0
    fp = 0

    # For each target
    for target in interaction_result_Dic:
        # For each interaction_result
        for interaction_result_LL in interaction_result_Dic[target]:
            # Flag, indicating whether the interaction_result is a interaction_ground_truth
            equal_F = False
            if target in prob_interaction_ground_truth_L_Dic:
                # For each interaction_ground_truth and the probability
                for prob, interaction_ground_truth_LL in prob_interaction_ground_truth_L_Dic[target]:
                    # If the interaction_result is a interaction_ground_truth
                    if equal(interaction_ground_truth_LL, interaction_result_LL):
                        equal_F = True

                        # Sort the two interaction_results based on the component name
                        interaction_ground_truth_sor_LL = sorted(interaction_ground_truth_LL, key = lambda x: x[0])
                        interaction_result_sor_LL = sorted(interaction_result_LL, key = lambda x: x[0])

                        # If the sizes are different
                        if len(interaction_ground_truth_sor_LL) != len(interaction_result_sor_LL):
                            print('interaction_result size different!')
                            exit(1)

                        for i in range(len(interaction_ground_truth_sor_LL)):
                            var_interaction_ground_truth, win_start_interaction_ground_truth, win_end_interaction_ground_truth = interaction_ground_truth_sor_LL[i]
                            var_interaction_result, win_start_interaction_result, win_end_interaction_result = interaction_result_sor_LL[i]

                            # If the component names are different
                            if var_interaction_ground_truth != var_interaction_result:
                                print('component name different!')
                                exit(1)
                        break
            # If the interaction_result is a interaction_ground_truth
            if equal_F is True:
                # Increase true positive
                tp += 1
            elif equal_F is False:
                # Increase false positive
                fp += 1

    # Get false negative
    fn = 0
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
                    if equal(interaction_result_LL, interaction_ground_truth_LL):
                        equal_F = True
                        break
            # If the interaction_ground_truth has not been discovered
            if equal_F is False:
                # Increase false negative
                fn += 1

    return [tp, fp, fn, run_time]


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
    interaction_result_dir = sys.argv[1]
    statistics_file = sys.argv[2]
    statistics_raw_file = sys.argv[3]
    statistics_raw_std_file = sys.argv[4]
    statistics_raw_mms_file = sys.argv[5]

    helper(statistics_file)
    helper(statistics_raw_file)
    helper(statistics_raw_std_file)
    helper(statistics_raw_mms_file)
