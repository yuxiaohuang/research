# Please cite the following paper when using the code


# Modules
from __future__ import division
import numpy as np
from scipy import stats
import sys
import os
import csv
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import time
from multiprocessing import Pool

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list
#_F       : indicates the variable is a flag


# Global variables

# The dictionary of value
# key: var->time
# val: value of var at the time
val_Dic = {}

# Class and feature vector
X_training_L = []
y_training_L = []
X_testing_L = []
y_testing_L = []

# The list of name+val
name_val_L = []

# Call classification in a parallel fashion
def parallel(src_data_training_file):
    if src_data_training_file.endswith(".txt"):
        tar_data_training_file = tar_data_training_dir + src_data_training_file.replace('src', 'tar')
        src_data_testing_file = src_data_testing_dir + src_data_training_file.replace('train', 'test')
        tar_data_testing_file = tar_data_testing_dir + src_data_training_file.replace('src', 'tar').replace('train',
                                                                                                            'test')
        src_data_training_file = src_data_training_dir + src_data_training_file

        # Write the name of the dataset
        f.write(src_data_training_file.replace('src_data_', '') + '\n')

        # Classification
        classification(src_data_training_file, tar_data_training_file, src_data_testing_file, tar_data_testing_file)


# Classification
def classification(src_data_training_file, tar_data_training_file, src_data_testing_file, tar_data_testing_file):
    # Get X_training_L and y_training_L
    global X_training_L, y_training_L
    [X_training_L, y_training_L] = get_feature_and_class_vectors(src_data_training_file, tar_data_training_file)

    # Get X_testing_L and y_testing_L
    global X_testing_L, y_testing_L
    [X_testing_L, y_testing_L] = get_feature_and_class_vectors(src_data_testing_file, tar_data_testing_file)

    # Get name_val_L
    global name_val_L
    name_val_L = []
    for var in sorted(val_Dic[1].keys()):
        if 'class' in var or 'tar' in var:
            name_val_L.append(var)

    global model, run_time

    # # Random forest
    # random_forest()

    # # SVM
    # svm()

    # KNN
    knn()


# Get feature and class vectors
def get_feature_and_class_vectors(src_data_file, tar_data_file):
    # Initialization
    global val_Dic
    val_Dic = {}

    # Load src file
    load_data(src_data_file)

    # Load tar file
    load_data(tar_data_file)

    # Get feature vectors, X_LL and y_LL
    X_LL = get_feature_vector('X')
    y_LL = get_feature_vector('y')

    return [X_LL, y_LL]


# Load data
def load_data(data_file):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

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

                    # Get val_Dic
                    if val == '1':
                        val_Dic[i][var] = 1
                    else:
                        val_Dic[i][var] = 0


# Get feature vector
def get_feature_vector(X_y_F):
    # Get val_LL
    val_LL = []
    for time in sorted(val_Dic.keys()):
        val_L = []

        for var in sorted(val_Dic[time].keys()):
            if ((X_y_F == 'X' and ('class' in var or 'tar' in var))
                or (X_y_F == 'y' and (not 'class' in var and not 'tar' in var))):
                continue

            val = val_Dic[time][var]
            val_L.append(val)
        val_LL.append(val_L)

    return val_LL


# Random forest
def random_forest():
    f.write('random forest' + '\n')

    col = -1
    for y in name_val_L:
        f.write('statistics for class: ' + y + '\n')

        # Get the corresponding class vector
        col += 1
        y_training_col_L = []
        for y_L in y_training_L:
            y_training_col_L.append(y_L[col])

        start_time = time.clock()

        # Training
        model = RandomForestClassifier()
        model.fit(X_training_L, y_training_col_L)

        end_time = time.clock()
        run_time = end_time - start_time

        # Testing
        y_testing_col_L = []
        for y_L in y_testing_L:
            y_testing_col_L.append(y_L[col])

        y_testing_hat_col_L = model.predict(X_testing_L)

        get_statistics(y_testing_col_L, y_testing_hat_col_L)

        # Write run time
        f.write('run time: ' + str(run_time) + '\n\n')
        f.flush()


# SVM
def svm():
    f.write('svm' + '\n')

    col = -1
    for y in name_val_L:
        f.write('statistics for class: ' + y + '\n')

        # Get the corresponding class vector
        col += 1
        y_training_col_L = []
        for y_L in y_training_L:
            y_training_col_L.append(y_L[col])

        start_time = time.clock()

        # Training
        model = SVC()
        model.fit(X_training_L, y_training_col_L)

        end_time = time.clock()
        run_time = end_time - start_time

        # Testing
        y_testing_col_L = []
        for y_L in y_testing_L:
            y_testing_col_L.append(y_L[col])

        y_testing_hat_col_L = model.predict(X_testing_L)

        get_statistics(y_testing_col_L, y_testing_hat_col_L)

        # Write run time
        f.write('run time: ' + str(run_time) + '\n\n')
        f.flush()


# KNN
def knn():
    f.write('knn' + '\n')

    col = -1
    for y in name_val_L:
        f.write('statistics for class: ' + y + '\n')

        # Get the corresponding class vector
        col += 1
        y_training_col_L = []
        for y_L in y_training_L:
            y_training_col_L.append(y_L[col])

        start_time = time.clock()

        # Training
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(X_training_L, y_training_col_L)

        end_time = time.clock()
        run_time = end_time - start_time

        # Testing
        y_testing_col_L = []
        for y_L in y_testing_L:
            y_testing_col_L.append(y_L[col])

        y_testing_hat_col_L = model.predict(X_testing_L)

        get_statistics(y_testing_col_L, y_testing_hat_col_L)

        # Write run time
        f.write('run time: ' + str(run_time) + '\n\n')
        f.flush()


# Get statistics
def get_statistics(y_L, y_hat_L):
    # Get true positive, false positive, true negative, and false negative for the current dataset
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Check whether equal length
    if len(y_L) != len(y_hat_L):
        print("Not equal length!")
        return

    for i in range(len(y_L)):
        y = y_L[i]
        y_hat = y_hat_L[i]

        # Update tp, fp, tn, and fn
        if y == 1 and y_hat == 1:
            tp += 1
        elif y == 1 and y_hat == 0:
            fn += 1
        elif y == 0 and y_hat == 1:
            fp += 1
        else:
            tn += 1

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

    # Write true positive, false positive, true negative, and false negative for the current dataset
    f.write('tp: ' + str(tp) + '\n')
    f.write('fp: ' + str(fp) + '\n')
    f.write('fn: ' + str(fn) + '\n')
    f.write('tn: ' + str(tn) + '\n')

    f.write('precision: ' + str(precision) + '\n')
    f.write('recall: ' + str(recall) + '\n')
    f.write('f1 score: ' + str(f1_score) + '\n')
    f.write('accuracy: ' + str(accuracy) + '\n')


# Get statistics across all datasets
def get_statistics_all():
    # Initialization
    tp_all = 0
    fp_all = 0
    tn_all = 0
    fn_all = 0
    run_time_all = 0

    with open(statistics_file, 'r') as f:
        # Update tp_all, fp_all, tn_all, fn_all, and run_time_all
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line)  # Non-blank lines in a list

    for line in lines:
        if 'tp: ' in line:
            tp = int(line.replace('tp: ', '').strip())
            tp_all += tp
        elif 'fp: ' in line:
            fp = int(line.replace('fp: ', '').strip())
            fp_all += fp
        elif 'tn: ' in line:
            tn = int(line.replace('tn: ', '').strip())
            tn_all += tn
        elif 'fn: ' in line:
            fn = int(line.replace('fn: ', '').strip())
            fn_all += fn
        elif 'run time: ' in line:
            run_time = float(line.replace('run time: ', '').strip())
            run_time_all += run_time

    # Write true positive, false positive, true negative, and false negative across all datasets
    f_all.write('tp_all: ' + str(tp_all) + '\n')
    f_all.write('fp_all: ' + str(fp_all) + '\n')
    f_all.write('fn_all: ' + str(fn_all) + '\n')
    f_all.write('tn_all: ' + str(tn_all) + '\n')

    # Get precision, recall, f1 score, and accuracy
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

    # Write precision, recall, f1 score, accuracy, and run_time_all
    f_all.write('precision_all: ' + str(precision) + '\n')
    f_all.write('recall_all: ' + str(recall) + '\n')
    f_all.write('f1 score_all: ' + str(f1_score) + '\n')
    f_all.write('accuracy_all: ' + str(accuracy) + '\n')
    f_all.write('run_time_all: ' + str(run_time_all) + '\n\n')
    f_all.flush()


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_training_dir = sys.argv[1]
    tar_data_training_dir = sys.argv[2]
    src_data_testing_dir = sys.argv[3]
    tar_data_testing_dir = sys.argv[4]
    statistics_file = sys.argv[5]
    statistics_all_file = sys.argv[6]

    with open(statistics_file, 'w') as f:
        num_cores = 10
        p = Pool(num_cores)
        p.map(parallel, os.listdir(src_data_training_dir))

    with open(statistics_all_file, 'w') as f_all:
        # Get statistics across all datasets
        get_statistics_all()