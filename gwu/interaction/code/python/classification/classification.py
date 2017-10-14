# Please cite the following paper when using the code


# Modules
from __future__ import division
import numpy as np
from scipy import stats
import sys
import csv
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

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

# The maximum time stamp
max_time_stamp = 0

# Class and feature vector
X_training_L = []
y_training_L = []
X_testing_L = []
y_testing_L = []

name_val_L = []

# Initialize true positve, false positive, and false negative (across all datasets)
tp_all = 0
fp_all = 0
fn_all = 0
tn_all = 0

# Classification
def classification():
    # Get X_training_L and y_training_L
    global X_training_L, y_training_L
    [X_training_L, y_training_L] = get_feature_and_class_vectors(src_data_training_file, tar_data_training_file)

    # Get X_testing_L and y_testing_L
    global X_testing_L, y_testing_L
    [X_testing_L, y_testing_L] = get_feature_and_class_vectors(src_data_testing_file, tar_data_testing_file)

    # Get name_val_L
    global name_val_L
    for var in sorted(val_Dic[1].keys()):
        if 'class' in var:
            name_val_L.append(var)

    # Random forest
    random_forest()

    # SVM
    svm()

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
            if ((X_y_F == 'X' and 'class' in var)
                or (X_y_F == 'y' and not 'class' in var)):
                continue

            val = val_Dic[time][var]
            val_L.append(val)
        val_LL.append(val_L)

    return val_LL


# Random forest
def random_forest():
    f.write('random forest' + '\n')

    # Initialize true positve, false positive, and false negative (across all datasets)
    global tp_all, fp_all, fn_all, tn_all
    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    col = -1
    for y in name_val_L:
        f.write('statistics for class: ' + y + '\n')

        # Get the corresponding class vector
        col += 1
        y_training_col_L = []
        for y_L in y_training_L:
            y_training_col_L.append(y_L[col])

        # Training
        model = RandomForestClassifier()
        model.fit(X_training_L, y_training_col_L)

        # Testing
        y_testing_col_L = []
        for y_L in y_testing_L:
            y_testing_col_L.append(y_L[col])

        y_testing_hat_col_L = model.predict(X_testing_L)

        get_statistics(y_testing_col_L, y_testing_hat_col_L)
    get_statistics_all()


# SVM
def svm():
    f.write('svm' + '\n')

    # Initialize true positve, false positive, and false negative (across all datasets)
    global tp_all, fp_all, fn_all, tn_all
    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    col = -1
    for y in name_val_L:
        f.write('statistics for class: ' + y + '\n')

        # Get the corresponding class vector
        col += 1
        y_training_col_L = []
        for y_L in y_training_L:
            y_training_col_L.append(y_L[col])

        # Training
        model = SVC()
        model.fit(X_training_L, y_training_col_L)

        # Testing
        y_testing_col_L = []
        for y_L in y_testing_L:
            y_testing_col_L.append(y_L[col])

        y_testing_hat_col_L = model.predict(X_testing_L)

        get_statistics(y_testing_col_L, y_testing_hat_col_L)
    get_statistics_all()


# KNN
def knn():
    f.write('knn' + '\n')

    # Initialize true positve, false positive, and false negative (across all datasets)
    global tp_all, fp_all, fn_all, tn_all
    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    col = -1
    for y in name_val_L:
        f.write('statistics for class: ' + y + '\n')

        # Get the corresponding class vector
        col += 1
        y_training_col_L = []
        for y_L in y_training_L:
            y_training_col_L.append(y_L[col])

        # Training
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(X_training_L, y_training_col_L)

        # Testing
        y_testing_col_L = []
        for y_L in y_testing_L:
            y_testing_col_L.append(y_L[col])

        y_testing_hat_col_L = model.predict(X_testing_L)

        get_statistics(y_testing_col_L, y_testing_hat_col_L)
    get_statistics_all()


# Get statistics
def get_statistics(y_L, y_hat_L):
    # Get precision, recall, f1_score, and accuracy
    # Get true positive, false positive, true negative, and false negative  for the current dataset
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

    if tp + fp != 0:
        precision = float(tp) / (tp + fp)
    else:
        precision = 'undefined'
    if tp + fn != 0:
        recall = float(tp) / (tp + fn)
    else:
        recall = 'undefined'
    if tp + fp != 0 and tp + fn != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 'undefined'
    if tp + fp + tn + fn != 0:
        accuracy = float(tp + tn) / (tp + fp + tn + fn)
    else:
        accuracy = 'undefined'

    # Write statistics file
    # Write true positive, false positive and false negative for the current dataset
    f.write('tp: ' + str(tp) + '\n')
    f.write('fp: ' + str(fp) + '\n')
    f.write('fn: ' + str(fn) + '\n')
    f.write('tn: ' + str(tn) + '\n')

    f.write('precision: ' + str(precision) + '\n')
    f.write('recall: ' + str(recall) + '\n')
    f.write('f1 score: ' + str(f1_score) + '\n')
    f.write('accuracy: ' + str(accuracy) + '\n\n')

    # Update true positive, false positive, true negative, and false negative
    # Initialize true positve, false positive, and false negative (across all datasets)
    global tp_all, fp_all, fn_all, tn_all
    tp_all += tp
    fp_all += fp
    fn_all += fn
    tn_all += tn


# Get statistics across all tars
def get_statistics_all():
    # Write statistics file
    # Write true positive, false positive, true negative, and false negative across all tars
    f.write('tp_all: ' + str(tp_all) + '\n')
    f.write('fp_all: ' + str(fp_all) + '\n')
    f.write('fn_all: ' + str(fn_all) + '\n')
    f.write('tn_all: ' + str(tn_all) + '\n')

    # Write precision and recall across all tars
    if tp_all + fp_all != 0:
        precision = float(tp_all) / (tp_all + fp_all)
    else:
        precision = 'undefined'
    if tp_all + fn_all != 0:
        recall = float(tp_all) / (tp_all + fn_all)
    else:
        recall = 'undefined'
    if tp_all + fp_all != 0 and tp_all + fn_all != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 'undefined'
    if tp_all + fp_all + tn_all + fn_all != 0:
        accuracy = float(tp_all + tn_all) / (tp_all + fp_all + tn_all + fn_all)
    else:
        accuracy = 'undefined'

    f.write('precision_all: ' + str(precision) + '\n')
    f.write('recall_all: ' + str(recall) + '\n')
    f.write('f1 score_all: ' + str(f1_score) + '\n')
    f.write('accuracy_all: ' + str(accuracy) + '\n\n')


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_training_file = sys.argv[1]
    tar_data_training_file = sys.argv[2]
    src_data_testing_file = sys.argv[3]
    tar_data_testing_file = sys.argv[4]
    statistics_file = sys.argv[5]

    with open(statistics_file, 'a') as f:
        # Write the name of the dataset
        f.write(src_data_training_file.replace('src_data', '') + '\n')
        # Classification
        classification()
