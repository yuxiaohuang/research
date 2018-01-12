# Please cite the following paper when using the code


# Modules
from __future__ import division
import numpy as np
from scipy import stats
import sys
import os
import csv
import math
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import _tree

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

# The list of feature+val
feature_val_L = []

# The list of interactions
interaction_LL = []


# Call classification in a parallel fashion
def parallel(src_data_training_file):
    if not src_data_training_file.startswith('.') and src_data_training_file.endswith(".txt"):
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
        if 'target' in var or 'tar' in var or 'class' in var:
            name_val_L.append(var)

    # Get feature_val_L
    global feature_val_L
    feature_val_L = []
    for var in sorted(val_Dic[1].keys()):
        if not 'target' in var and not 'tar' in var and not 'class' in var:
            feature_val_L.append(var)

    global model, run_time

    # Decision tree
    decision_tree(src_data_training_file)

    # # Random forest
    # random_forest(src_data_training_file)

    # # SVM
    # svm()

    # # Multi-layer Perceptron
    # mlp()

    # # KNN
    # knn()

    # # Gaussian Naive Bayes
    # gnb()


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

                # Get the value
                val = spamreader[i][j].strip()

                # Get val_Dic
                val_Dic[i][var] = float(val)


# Get feature vector
def get_feature_vector(X_y_F):
    # Get val_LL
    val_LL = []
    for time in sorted(val_Dic.keys()):
        val_L = []

        for var in sorted(val_Dic[time].keys()):
            if ((X_y_F == 'X' and ('target' in var or 'tar' in var or 'class' in var))
                or (X_y_F == 'y' and (not 'target' in var and not 'tar' in var and not 'class' in var))):
                continue

            val = val_Dic[time][var]
            val_L.append(val)
        val_LL.append(val_L)

    return val_LL


# Decision tree
def decision_tree(src_data_training_file):
    f.write('decision tree' + '\n')

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
        model = tree.DecisionTreeClassifier(min_samples_leaf = 30)
        # model = tree.DecisionTreeClassifier(max_leaf_nodes = 5)
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

        # Decision tree file
        decision_tree_file = os.path.dirname(statistics_file) + '/tree/' + os.path.basename(
            src_data_training_file).replace("src_data", "tree")
        decision_tree_file = decision_tree_file.replace(".txt", "")
        decision_tree_file += os.path.basename(statistics_file).replace("statistics_classification", "")
        decision_tree_file = decision_tree_file.replace(".txt", ".dot")

        # Create file
        directory = os.path.dirname(decision_tree_file)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except FileExistsError:
            print('FileExistsError!')

        # Write decision tree file
        decision_tree_file = open(decision_tree_file, 'w')
        tree.export_graphviz(model, out_file = decision_tree_file, feature_names = feature_val_L, class_names = ['0', '1'])
        decision_tree_file.close()

        # Get interaction_LL
        # Initialization
        global interaction_LL
        interaction_LL = []
        get_interaction_LL(model, y)

        # Interaction file
        Interaction_file = os.path.dirname(statistics_file) + '/interaction/' + os.path.basename(
            src_data_training_file).replace("src_data", "interaction")
        Interaction_file = Interaction_file.replace(".txt", "")
        Interaction_file += os.path.basename(statistics_file).replace("statistics_classification", "")

        # Create file
        directory = os.path.dirname(Interaction_file)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except FileExistsError:
            print('FileExistsError!')

        # Write interaction file
        with open(Interaction_file, 'w') as f_interaction:
            spamwriter_interaction = csv.writer(f_interaction, delimiter=' ')
            for interaction_L in interaction_LL:
                # If not one-hot encoding
                if y[-2:] != '_1':
                    spamwriter_interaction.writerow(['interaction for ' + y + '_1: ', interaction_L])
                else:
                    spamwriter_interaction.writerow(['interaction for ' + y + ': ', interaction_L])
                f_interaction.flush()

            # Write run time
            spamwriter_interaction.writerow(['run time: ' + str(run_time)])


# Get interaction_LL
def get_interaction_LL(tree, y):
    tree_ = tree.tree_
    helper(tree_, 0, [], y)


# Helper function
def helper(tree_, node, interaction_L, y):
    # If leaf node
    if tree_.feature[node] == _tree.TREE_UNDEFINED:
        value = tree_.value[node]
        idx = np.argmax(value)
        if idx == 1:
            if not interaction_L in interaction_LL:
                temp = list(interaction_L)
                interaction_LL.append(temp)
                return
    # Otherwise
    else:
        # Left subtree
        temp_left = list(interaction_L)
        feature_val = feature_val_L[tree_.feature[node]]
        # If one-hot encoding
        if '_' in feature_val.replace('src_', ''):
            if feature_val[-1:] == '0':
                feature_val_left = feature_val[:-1] + '1'
            elif feature_val[-1:] == '1':
                feature_val_left = feature_val[:-1] + '0'
            else:
                feature_val_left = feature_val + '_0'
        else:
            feature_val_left = feature_val + '_0'
        temp_left.append([feature_val_left, 0, 0])
        helper(tree_, tree_.children_left[node], temp_left, y)

        # Right subtree
        temp_right = list(interaction_L)
        feature_val = feature_val_L[tree_.feature[node]]
        # If one-hot encoding
        if '_' in feature_val.replace('src_', ''):
            if feature_val[-1:] == '0' or feature_val[-1:] == '1':
                feature_val_right = feature_val
            else:
                feature_val_right = feature_val + '_1'
        else:
            feature_val_right = feature_val + '_1'
        temp_right.append([feature_val_right, 0, 0])
        helper(tree_, tree_.children_right[node], temp_right, y)


# Random forest
def random_forest(src_data_training_file):
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

        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Feature importance text file
        importance_txt_file = os.path.dirname(statistics_file) + '/importance_txt/' + os.path.basename(src_data_training_file).replace("src", "importance_txt")
        importance_txt_file = importance_txt_file.replace(".txt", "")
        importance_txt_file += os.path.basename(statistics_file).replace("statistics_classification", "")

        # Create file
        directory = os.path.dirname(importance_txt_file)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except FileExistsError:
            print('FileExistsError!')

        # Write file
        with open(importance_txt_file, 'w') as f_importance_txt:
            for idx in indices:
                f_importance_txt.write(str(feature_val_L[idx]) + ', ' + str(importances[idx]) + '\n' )

        # Feature importance histogram figure
        importance_hist_fig = os.path.dirname(statistics_file) + '/importance_hist/' + os.path.basename(
            src_data_training_file).replace("src", "importance_hist")
        importance_hist_fig = importance_hist_fig.replace(".txt", "")
        importance_hist_fig += os.path.basename(statistics_file).replace("statistics_classification", "")
        importance_hist_fig = importance_hist_fig.replace(".txt", ".pdf")

        # Create figure
        directory = os.path.dirname(importance_hist_fig)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except FileExistsError:
            print('FileExistsError!')

        # Draw figure
        plt.title('Feature Importances')
        # plt.xlabel('Condition')
        # plt.ylabel('Importance')
        bar_L = []
        for i in range(min(len(indices), 25)):
            idx = indices[i]
            bar_L.append(importances[idx])

        plt.bar(range(min(len(indices), 25)),
                bar_L,
                color='lightblue',
                align='center')

        xticks_L = []
        for i in range(min(len(indices), 25)):
            idx = indices[i]
            xticks_L.append(feature_val_L[idx])

        plt.xticks(range(min(len(indices), 25)),
                   xticks_L, rotation=90)
        plt.xlim([-1, min(len(indices), 25)])
        plt.tight_layout()
        plt.savefig(importance_hist_fig, dpi=300)
        plt.close()


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


# Multi-layer Perceptron
def mlp():
    f.write('Multi-layer Perceptron' + '\n')

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
        model = MLPClassifier()
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
        model = KNeighborsClassifier()
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


# Gaussian Naive Bayes
def gnb():
    f.write('Gaussian Naive Bayes' + '\n')

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
        model = GaussianNB()
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
    f1_score_max = None
    accuracy_max = None


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
        elif 'f1 score: ' in line:
            f1_score = line.replace('f1 score: ', '').strip()
            if not "undefined" in f1_score:
                f1_score = float(f1_score)
                if f1_score_max is None or f1_score_max < f1_score:
                    f1_score_max = f1_score
        elif 'accuracy: ' in line:
            accuracy = line.replace('accuracy: ', '').strip()
            if not "undefined" in accuracy:
                accuracy = float(accuracy)
                if accuracy_max is None or accuracy_max < accuracy:
                    accuracy_max = accuracy

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
    f_all.write('f1_score_all: ' + str(f1_score) + '\n')
    f_all.write('accuracy_all: ' + str(accuracy) + '\n')
    f_all.write('run_time_all: ' + str(run_time_all) + '\n\n')
    f_all.write('f1_score_max: ' + str(f1_score_max) + '\n')
    f_all.write('accuracy_max: ' + str(accuracy_max) + '\n\n')
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
