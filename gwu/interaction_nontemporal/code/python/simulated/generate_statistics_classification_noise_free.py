

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

# Initialization
def initialization():
    # Initialization
    global val_Dic, interaction_result_Dic, prob_interaction_ground_truth_L_Dic
    val_Dic = {}
    interaction_result_Dic = {}
    prob_interaction_ground_truth_L_Dic = {}

    # Load source file
    load_data(src_data_file, True)

    # Load target file
    load_data(tar_data_file, False)

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
        spamreader = list(csv.reader(f, delimiter=' '))
        # Get the target, interaction_truth, and interaction_ground_truth
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


# Generate statistics
def generate_statistics():
    # Get true positive, false positive, and false negative for the current dataset
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for time in sorted(val_Dic.keys()):
        if target in val_Dic[time]:
            # Ground truth
            val = val_Dic[time][target]
            # Predicited value, 0 by default
            val_hat = 0

            # For each interaction_ground_truth
            if target in prob_interaction_ground_truth_L_Dic:
                for prob, interaction_ground_truth_LL in prob_interaction_ground_truth_L_Dic[target]:
                    exist_ground_truth_F = True

                    for interaction_ground_truth_L in interaction_ground_truth_LL:
                        var = interaction_ground_truth_L[0]
                        if not var in val_Dic[time] or val_Dic[time][var] == 0:
                            exist_ground_truth_F = False
                            break

                    if exist_ground_truth_F is True:
                        break

            # For each interaction_result
            if target in interaction_result_Dic:
                for interaction_result_LL in interaction_result_Dic[target]:
                    exist_F = True

                    for interaction_result_L in interaction_result_LL:
                        var = interaction_result_L[0]
                        if not var in val_Dic[time] or val_Dic[time][var] == 0:
                            exist_F = False
                            break

                    if exist_F is True:
                        # Update predicted value
                        val_hat = 1
                        break

            # Update tp and fp
            if val == 1:
                if exist_ground_truth_F is True:
                    if val_hat == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if val_hat == 1:
                        fp += 1
                    else:
                        tn += 1
            else:
                if val_hat == 0:
                    tn += 1
                else:
                    fp += 1

    return [tp, fp, fn, tn]


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_dir = sys.argv[1]
    tar_data_dir = sys.argv[2]
    interaction_result_dir = sys.argv[3]
    interaction_ground_truth_dir = sys.argv[4]
    statistics_file = sys.argv[5]

    # Make directory
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize true positve, false positive, true negative, and false negative (across all datasets)
    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    # Write statistics file
    with open(statistics_file, 'w') as f:
        for interaction_result_file in os.listdir(interaction_result_dir):
            if not interaction_result_file.startswith('.') and interaction_result_file.endswith(".txt"):
                # Get source, target, and interaction ground truth file
                src_data_file = src_data_dir + interaction_result_file.replace('interaction', 'src_data')
                tar_data_file = tar_data_dir + interaction_result_file.replace('interaction', 'tar_data')
                src_data_file = src_data_file.replace('train', 'test')
                tar_data_file = tar_data_file.replace('train', 'test')
                interaction_ground_truth_file = interaction_ground_truth_dir + interaction_result_file.replace('_data', '')
                interaction_ground_truth_file = interaction_ground_truth_file.replace('_0.txt', '.txt')

                # Update interaction_result file
                interaction_result_file = interaction_result_dir + interaction_result_file

                # Initialization
                initialization()

                # Write the name of the dataset
                f.write(interaction_result_file + '\n')
                for target in tar_L:
                    # Generate statistics
                    [tp, fp, fn, tn] = generate_statistics()

                    # Get precision, recall, f1_score, and accuracy
                    if tp + fp != 0:
                        precision = float(tp) / (tp + fp)
                    else:
                        precision = 'undefined'
                    if tp + fn != 0:
                        recall = float(tp) / (tp + fn)
                    else:
                        recall = 'undefined'
                    if precision != 'undefined' and recall != 'undefined' and (precision != 0 or recall != 0):
                        f1_score = 2 * precision * recall / (precision + recall)
                    else:
                        f1_score = 'undefined'
                    if tp + fp + tn + fn != 0:
                        accuracy = float(tp + tn) / (tp + fp + tn + fn)
                    else:
                        accuracy = 'undefined'

                    # Write statistics file
                    # Write the target
                    f.write('statistics for target: ' + target + '\n')
                    # Write true positive, false positive, true negative, and false negative for the current dataset
                    f.write('tp: ' + str(tp) + '\n')
                    f.write('fp: ' + str(fp) + '\n')
                    f.write('fn: ' + str(fn) + '\n')
                    f.write('tn: ' + str(tn) + '\n')

                    f.write('precision: ' + str(precision) + '\n')
                    f.write('recall: ' + str(recall) + '\n')
                    f.write('f1 score: ' + str(f1_score) + '\n')
                    f.write('accuracy: ' + str(accuracy) + '\n\n')

                    # Update true positive, false positive, true negative, and false negative across all datasets
                    tp_all += tp
                    fp_all += fp
                    fn_all += fn
                    tn_all += tn

        # Write statistics file
        # Write true positive, false positive, true negative, and false negative across all datasets
        f.write('tp_all: ' + str(tp_all) + '\n')
        f.write('fp_all: ' + str(fp_all) + '\n')
        f.write('fn_all: ' + str(fn_all) + '\n')
        f.write('tn_all: ' + str(tn_all) + '\n')

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
        if tp_all + fp_all + tn_all + fn_all != 0:
            accuracy = float(tp_all + tn_all) / (tp_all + fp_all + tn_all + fn_all)
        else:
            accuracy = 'undefined'

        f.write('precision: ' + str(precision) + '\n')
        f.write('recall: ' + str(recall) + '\n')
        f.write('f1 score: ' + str(f1_score) + '\n')
        f.write('accuracy: ' + str(accuracy) + '\n\n')