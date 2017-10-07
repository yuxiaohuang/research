

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


# Generate statistics
def generate_statistics():
    # Initialize prob_interaction_ground_truth_L_Dic and interaction_result_Dic
    prob_interaction_ground_truth_L_Dic = {}
    interaction_result_Dic = {}

    # Load source file
    load_data(src_data_file, True)

    # Load target file
    load_data(tar_data_file, False)

    # Load the interaction_result file
    with open(interaction_result_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ' '))
        # Get the target and interaction_ground_truth
        for i in range(len(spamreader)):
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

    # Get true positive, false positive, and false negative for the current dataset
    tp = 0
    fp = 0
    fn = 0

    # For each target
    for target in interaction_result_Dic:
        for time in sorted(val_Dic.keys()):
            if (time + 1) in val_Dic and target in val_Dic[time + 1]:
                # Ground truth
                val = val_Dic[time + 1][target]
                # Predicited value, 0 by default
                val_hat = 0

                # For each interaction_result
                for interaction_result_LL in interaction_result_Dic[target]:
                    exist_F = True

                    for interaction_result_L in interaction_result_LL:
                        print(interaction_result_L)
                        var = interaction_result_L[0]
                        if not var in val_Dic[time] or val_Dic[time][var] == 0:
                            exist_F = False
                            break

                    if exist_F is True:
                        # Update predicted value
                        val_hat = 1
                        break

                # Update tp and fp
                if val == 1 and val_hat == 1:
                    tp += 1
                elif val == 1 and val_hat == 0:
                    fn += 1
                elif val == 0 and val_hat == 1:
                    fp += 1

    return [tp, fp, fn]


# Load data, get data_type_Dic, val_Dic, x_Dic and y_Dic
# @param        data_file          source / target file
#                                  the data are of the following form
#                                  time, var1    , ..., varn (i.e. header)
#                                  t1  , var1(t1), ..., varn(t1)
#                                                , ...,
#                                  tn  , var1(tn), ..., varn(tn)
# @param        x_F              Flag variable
#                                  True,  if target data
#                                  False, if source data
def load_data(data_file, x_F):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

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


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_dir = sys.argv[1]
    tar_data_dir = sys.argv[2]
    interaction_result_dir = sys.argv[3]
    statistics_file = sys.argv[4]

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
        for interaction_result_file in os.listdir(interaction_result_dir):
            if interaction_result_file.endswith(".txt"):
                # Get source and target file
                src_data_file = src_data_dir + interaction_result_file.replace('interaction', 'src_data')
                tar_data_file = tar_data_dir + interaction_result_file.replace('interaction', 'tar_data')

                # Update interaction_result file
                interaction_result_file = interaction_result_dir + interaction_result_file

                # Generate statistics
                [tp, fp, fn] = generate_statistics()
                precision = float(tp) / (tp + fp)
                recall = float(tp) / (tp + fn)
                f1_score = 2 * precision * recall / (precision + recall)

                # Write statistics file
                # Write the name of the dataset
                f.write(interaction_result_file + '\n')
                # Write true positive, false positive and false negative for the current dataset
                f.write('tp: ' + str(tp) + '\n')
                f.write('fp: ' + str(fp) + '\n')
                f.write('fn: ' + str(fn) + '\n')
                f.write('precision: ' + str(precision) + '\n')
                f.write('recall: ' + str(recall) + '\n\n')
                f.write('F1 score: ' + str(f1_score) + '\n\n')

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
        precision = float(tp_all) / (tp_all + fp_all)
        recall = float(tp_all) / (tp_all + fn_all)
        f1_score = 2 * precision * recall / (precision + recall)
        f.write('precision: ' + str(precision) + '\n')
        f.write('recall: ' + str(recall) + '\n\n')
        f.write('F1 score: ' + str(f1_score) + '\n\n')