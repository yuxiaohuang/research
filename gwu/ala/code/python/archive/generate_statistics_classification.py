

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
# key: class_label pair->time
# val: the value of the class_label pair at the time
y_time_val_Dic = {}

# The dictionary of value
# key: class_label pair->time
# val: the predicted value of the class_label pair at the time
y_time_val_predicted_Dic = {}


# Load data
def load_data(data_file, time_val_Dic):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        # Get time_val_Dic
        # For each time
        for time in range(header, len(spamreader)):
            # For each column
            for j in range(len(spamreader[0])):
                # If there is a header
                if header == 1:
                    # var's name lies in jth column in the first row
                    var = spamreader[0][j].strip()
                else:
                    # var's name is j
                    var = str(j)

                # If the value at [time][j] is numeric
                if spamreader[time][j].isnumeric() is True:
                    # Get the value and convert it to integer
                    val = int(spamreader[time][j].strip())

                    # Initialize time_val_Dic
                    if not var in time_val_Dic:
                        time_val_Dic[var] = {}

                    # Update time_val_Dic
                    time_val_Dic[var][time] = val


# Generate statistics
def generate_statistics():
    # Write the name of the dataset
    f.write(os.path.basename(class_testing_data_file) + '\n\n')

    # Get true positive (tp), true negative (tn), false positive (fp), and false negative (fn) for the current dataset
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # For each class_label pair
    for y in sorted(y_time_val_Dic.keys()):
        # Get true positive (tp_y), true negative (tn_y), false positive (fp_y), and false negative (fn_y) for the current class_label pair
        tp_y = 0
        tn_y = 0
        fp_y = 0
        fn_y = 0

        # For each time
        for time in sorted(y_time_val_Dic[y].keys()):
            # Ground truth
            val = y_time_val_Dic[y][time]
            # Predicited value
            val_predicted = y_time_val_predicted_Dic[y][time]

            # Update true positive (tp_y), true negative (tn_y), false positive (fp_y), and false negative (fn_y) for the current class_label pair
            if val == 1 and val_predicted == 1:
                # Update true positive
                tp_y += 1
            elif val == 1 and val_predicted == 0:
                # Update false negative
                fn_y += 1
            elif val == 0 and val_predicted == 1:
                # Update false positive
                fp_y += 1
            elif val == 0 and val_predicted == 0:
                # Update true negative
                tn_y += 1

        # Write statistics file
        write_statistics_file(tp_y, tn_y, fp_y, fn_y, y)

        # Update true positive (tp), true negative (tn), false positive (fp), and false negative (fn) for the current dataset
        tp += tp_y
        tn += tn_y
        fp += fp_y
        fn += fn_y

    # Write statistics file
    write_statistics_file(tp, tn, fp, fn, 'the current dataset')

    # Update the overall true positive (tp_all), true negative (tn_all), false positive (fp_all), and false negative (fn_all)
    global tp_all, tn_all, fp_all, fn_all
    tp_all += tp
    tn_all += tn
    fp_all += fp
    fn_all += fn


# Write statistics file
def write_statistics_file(tp, tn, fp, fn, y):
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

    f.write('statistics for: ' + y + '\n')

    f.write('tp: ' + str(tp) + '\n')
    f.write('tn: ' + str(tn) + '\n')
    f.write('fp: ' + str(fp) + '\n')
    f.write('fn: ' + str(fn) + '\n')

    f.write('precision: ' + str(precision) + '\n')
    f.write('recall: ' + str(recall) + '\n')
    f.write('f1_score: ' + str(f1_score) + '\n')
    f.write('accuracy: ' + str(accuracy) + '\n\n')


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    class_testing_data_dir = sys.argv[1]
    predict_dir = sys.argv[2]
    statistics_file = sys.argv[3]
    header = int(sys.argv[4])

    # Make directory
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize the overall true positive (tp_all), true negative (tn_all), false positive (fp_all), and false negative (fn_all)
    tp_all = 0
    tn_all = 0
    fp_all = 0
    fn_all = 0

    # Write statistics file
    with open(statistics_file, 'w') as f:
        for class_testing_data_file in os.listdir(class_testing_data_dir):
            if not class_testing_data_file.startswith('.') and class_testing_data_file.endswith(".txt"):
                # Get class testing data file number
                num = class_testing_data_file
                num = num.replace('class_data_', '')
                num = num.replace('.txt', '')

                # Update class_testing_data_file
                class_testing_data_file = class_testing_data_dir + class_testing_data_file

                # Get predict file
                predict_file = predict_dir + 'predict_' + num + '.txt'

                # Load class testing data file
                load_data(class_testing_data_file, y_time_val_Dic)

                # Load predict file
                load_data(predict_file, y_time_val_predicted_Dic)

                # If not same number of timepoints
                if len(y_time_val_Dic.keys()) != len(y_time_val_predicted_Dic.keys()):
                    print("Not same number of timepoints!")
                    exit(1)

                generate_statistics()

        # Write statistics file
        write_statistics_file(tp_all, tn_all, fp_all, fn_all, 'all datasets')