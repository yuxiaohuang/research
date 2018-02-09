# Please cite the following paper when using the code


# Modules
from __future__ import division
import numpy as np
from scipy import stats
import sys
import csv
import math
import time
from operator import itemgetter
import random

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list
# _F       : indicates the variable is a flag


# Global variables

# The dictionary of value
# key: time->var (attribute_value pair or class_label pair)
# val: the value of the var at the time
time_var_val_Dic = {}

# The dictionary of value
# key: attribute_value pair->time
# val: the value of the attribute_value pair at the time
x_time_val_Dic = {}

# The dictionary of value
# key: class_label pair->time
# val: the value of the class_label pair at the time
y_time_val_Dic = {}

# The dictionary of value
# key: class_label pair->time
# val: the predicted value of the class_label pair at the time
y_time_val_predicted_Dic = {}

# The dictionary of list of time where class_label pair is measured and attribute_value pair is true
# key: class_label pair->attribute_value pair
# val: the list of time where class_label pair is measured and attribute_value pair is true
y_cond_x_time_L_Dic = {}

# The dictionary of list of (x, importance) pairs sorted in descending order of the importance
# key: class_label pair
# val: the list
sorted_x_importance_pair_LL_Dic = {}

# The dictionary of list of (x, importance) pairs sorted in descending order of the importance
# Here, only the pairs whose importance is significant are included in the list
# key: class_label pair
# val: the list
significant_sorted_x_importance_pair_LL_Dic = {}


# Initialization
def initialization(attribute_data_file, class_data_file):
    # Initialize time_var_val_Dic, x_time_val_Dic, and y_time_val_Dic
    global time_var_val_Dic, x_time_val_Dic, y_time_val_Dic
    time_var_val_Dic, x_time_val_Dic, y_time_val_Dic = {}, {}, {}

    # Load attribute data file
    load_data(attribute_data_file, True)

    # Load class data file
    load_data(class_data_file, False)

    # Update min_number_of_times_cutoff
    global min_number_of_times_cutoff
    if min_number_of_times_cutoff == 0:
        min_number_of_times_cutoff = int(min_number_of_times_ratio_cutoff * len(time_var_val_Dic.keys()))


# Load data, get time_var_val_Dic, x_time_val_Dic, and y_time_val_Dic
def load_data(data_file, x_F):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        # Get time_var_val_Dic, x_time_val_Dic, and y_time_val_Dic
        # For each time
        for time in range(header, len(spamreader)):
            # Initialize time_var_val_Dic
            if not time in time_var_val_Dic:
                time_var_val_Dic[time] = {}

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

                    # Update time_var_val_Dic
                    time_var_val_Dic[time][var] = val

                    # Initialize and update x_time_val_Dic and y_time_val_Dic
                    # If attribute data file
                    if x_F is True:
                        if not var in x_time_val_Dic:
                            x_time_val_Dic[var] = {}
                        x_time_val_Dic[var][time] = val
                    # If class data file
                    else:
                        if not var in y_time_val_Dic:
                            y_time_val_Dic[var] = {}
                        y_time_val_Dic[var][time] = val


# The fit function
def fit():
    # For each class_label pair
    for y in sorted(y_time_val_Dic.keys()):
        # Write log file
        spamwriter_log.writerow(["fitting for class_label pair: ", y])
        spamwriter_log.writerow([])
        f_log.flush()

        # Initialize iteration number and old_sorted_x_importance_pair_LL
        iteration = 0
        old_sorted_x_importance_pair_LL = []

        # Termination condition: when the iteration number exceeds the max iteration number cutoff
        while iteration <= max_iteration_cutoff:
            # Get the new list of (x, importance) pairs sorted in descending order of the importance
            new_sorted_x_importance_pair_LL = get_sorted_x_importance_pair_LL(old_sorted_x_importance_pair_LL, y)

            # Write log file
            spamwriter_log.writerow(["old_sorted_x_importance_pair_LL: ", old_sorted_x_importance_pair_LL])
            spamwriter_log.writerow([])
            spamwriter_log.writerow(["new_sorted_x_importance_pair_LL: ", new_sorted_x_importance_pair_LL])
            spamwriter_log.writerow([])
            f_log.flush()

            # Get the list of (x, importance) pairs sorted in descending order of the importance
            # Here, only the pairs whose importance is significant are included
            old_significant_sorted_x_importance_pair_LL = get_significant_sorted_x_importance_pair_LL(
                old_sorted_x_importance_pair_LL)
            new_significant_sorted_x_importance_pair_LL = get_significant_sorted_x_importance_pair_LL(
                new_sorted_x_importance_pair_LL)

            # Write log file
            spamwriter_log.writerow(
                ["old_significant_sorted_x_importance_pair_LL: ", old_significant_sorted_x_importance_pair_LL])
            spamwriter_log.writerow([])
            spamwriter_log.writerow(
                ["new_significant_sorted_x_importance_pair_LL: ", new_significant_sorted_x_importance_pair_LL])
            spamwriter_log.writerow([])
            f_log.flush()

            # Update old_sorted_x_importance_pair_LL
            old_sorted_x_importance_pair_LL = list(new_sorted_x_importance_pair_LL)

            # Update iteration
            iteration += 1

        # Update sorted_x_importance_pair_LL_Dic
        sorted_x_importance_pair_LL_Dic[y] = list(new_sorted_x_importance_pair_LL)

        # Update significant_sorted_x_importance_pair_LL_Dic
        significant_sorted_x_importance_pair_LL_Dic[y] = list(new_significant_sorted_x_importance_pair_LL)

        # Write fit file
        spamwriter_fit.writerow(["fitting for class_label pair: ", y])
        spamwriter_fit.writerow([])

        spamwriter_fit.writerow(
            ["sorted_x_importance_pair_LL_Dic[y]: ", sorted_x_importance_pair_LL_Dic[y]])
        spamwriter_fit.writerow([])

        spamwriter_fit.writerow(
            ["significant_sorted_x_importance_pair_LL_Dic[y]: ", significant_sorted_x_importance_pair_LL_Dic[y]])
        spamwriter_fit.writerow([])

        f_fit.flush()


# Get the list of (x, importance) pairs sorted in descending order of the importance
def get_sorted_x_importance_pair_LL(old_sorted_x_importance_pair_LL, y):
    # If old_sorted_x_importance_pair_LL is None or empty
    if old_sorted_x_importance_pair_LL is None or len(old_sorted_x_importance_pair_LL) == 0:
        # Initialize old_sorted_x_importance_pair_LL
        for x in sorted(x_time_val_Dic.keys()):
            # Get p(y | x)
            p_y_cond_x = get_p_y_cond_x(y, x)

            # If p(y | x) is not None
            if not p_y_cond_x is None:
                # Update old_sorted_x_importance_pair_LL
                old_sorted_x_importance_pair_LL.append([x, p_y_cond_x])

        # Sort old_sorted_x_importance_pair_LL in descending order of the importance (i.e., p(y | x))
        old_sorted_x_importance_pair_LL = sorted(old_sorted_x_importance_pair_LL, key=itemgetter(1), reverse=True)

    # Initialize new_sorted_x_importance_pair_LL
    new_sorted_x_importance_pair_LL = []

    # For each index in old_sorted_x_importance_pair_LL
    for i in range(len(old_sorted_x_importance_pair_LL)):
        # Get p(y | xi and not xjs)
        p_y_cond_xi_and_not_xjs = get_p_y_cond_xi_and_not_xjs(old_sorted_x_importance_pair_LL, y, i)

        # Get xi
        xi = old_sorted_x_importance_pair_LL[i][0]

        # Update new_sorted_x_importance_pair_LL
        new_sorted_x_importance_pair_LL.append([xi, p_y_cond_xi_and_not_xjs])

    # Sort new_sorted_x_importance_pair_LL in descending order of the importance (i.e, p(y | xi and not xjs))
    new_sorted_x_importance_pair_LL = sorted(new_sorted_x_importance_pair_LL, key=itemgetter(1), reverse=True)

    return new_sorted_x_importance_pair_LL


# Get p(y | x)
def get_p_y_cond_x(y, x):
    # Get the list of timepoints where y is measured and x is true
    y_cond_x_time_L = get_y_cond_x_time_L(y, x)

    # If y_cond_x_time_L is None or empty
    if y_cond_x_time_L is None or len(y_cond_x_time_L) == 0:
        # p(y | x) does not exist
        return None

    # Get p(y | x)
    p_y_cond_x = np.mean([y_time_val_Dic[y][time] for time in y_cond_x_time_L])

    return p_y_cond_x


# Get the list of timepoints where y is measured and x is true
def get_y_cond_x_time_L(y, x):
    # If y_cond_x_time_L has been obtained
    if y in y_cond_x_time_L_Dic and x in y_cond_x_time_L_Dic[y]:
        return y_cond_x_time_L_Dic[y][x]

    # Initialize y_cond_x_time_L_Dic
    if not y in y_cond_x_time_L_Dic:
        y_cond_x_time_L_Dic[y] = {}

    # Initilize y_cond_x_time_L
    y_cond_x_time_L = []

    # Get y_cond_x_time_L
    # For each timepoint where y is measured
    for time in sorted(y_time_val_Dic[y].keys()):
        # If x is also true at the time
        if x in time_var_val_Dic[time] and time_var_val_Dic[time][x] == 1:
            # Update y_cond_x_time_L
            y_cond_x_time_L.append(time)

    # Get y_cond_x_time_L_Dic[y][x]
    # If y_cond_x_time_L is None or the number of times in y_cond_x_time_L is smaller than min_number_of_times_cutoff
    if y_cond_x_time_L is None or len(y_cond_x_time_L) < min_number_of_times_cutoff:
        # Ignore x, since p(y | x) is not reliable
        y_cond_x_time_L_Dic[y][x] = []
    else:
        y_cond_x_time_L_Dic[y][x] = y_cond_x_time_L

    return y_cond_x_time_L


# Get p(y | xi and not xjs)
def get_p_y_cond_xi_and_not_xjs(sorted_x_importance_pair_LL, y, i):
    # Get xi
    xi = sorted_x_importance_pair_LL[i][0]

    # Initialize y_cond_xi_and_not_xjs_time_L
    y_cond_xi_and_not_xjs_time_L = get_y_cond_x_time_L(y, xi)

    # If y_cond_xi_and_not_xjs_time_L is None or the number of times in y_cond_xi_and_not_xjs_time_L is smaller than min_number_of_times_cutoff
    if y_cond_xi_and_not_xjs_time_L is None or len(y_cond_xi_and_not_xjs_time_L) < min_number_of_times_cutoff:
        # Ignore xi, since p(y | xi and not xjs) is not reliable
        return None

    # For each index in the sorted list of (x, importance) pairs (in descending order of importance)
    for j in range(len(sorted_x_importance_pair_LL)):
        # Ignore the same index
        if j == i:
            continue

        # Get xj
        xj = sorted_x_importance_pair_LL[j][0]

        # Update y_cond_xi_and_not_xjs_time_L based on xj
        y_cond_xi_and_not_xjs_time_L = get_y_cond_xi_and_not_xjs_time_L(y_cond_xi_and_not_xjs_time_L, xj)

    # Get p(y | xi and not xjs)
    # If y_cond_xi_and_not_xjs_time_L is None or the number of times in y_cond_xi_and_not_xjs_time_L is smaller than min_number_of_times_cutoff
    if y_cond_xi_and_not_xjs_time_L is None or len(y_cond_xi_and_not_xjs_time_L) < min_number_of_times_cutoff:
        # Ignore xi, since p(y | xi and not xjs) is not reliable
        p_y_cond_xi_and_not_xjs = None
    else:
        p_y_cond_xi_and_not_xjs = np.mean([y_time_val_Dic[y][time] for time in y_cond_xi_and_not_xjs_time_L])

    return p_y_cond_xi_and_not_xjs


# Get y_cond_xi_and_not_xjs_time_L based on xj
def get_y_cond_xi_and_not_xjs_time_L(time_L, xj):
    # If time_L is None or the number of times in time_L is not larger than min_number_of_times_cutoff
    if time_L is None or len(time_L) <= min_number_of_times_cutoff:
        # Ignore xj
        return time_L

    # Initialize y_cond_xi_and_not_xjs_time_L
    y_cond_xi_and_not_xjs_time_L = []

    # Get y_cond_xi_and_not_xjs_time_L
    for time in time_L:
        # If xj is not measured or not true at the time
        if not xj in time_var_val_Dic[time] or time_var_val_Dic[time][xj] != 1:
            # Update y_cond_xi_and_not_xjs_time_L
            y_cond_xi_and_not_xjs_time_L.append(time)

    # If y_cond_xi_and_not_xjs_time_L is None or the number of times in y_cond_xi_and_not_xjs_time_L is smaller than min_number_of_times_cutoff
    if y_cond_xi_and_not_xjs_time_L is None or len(y_cond_xi_and_not_xjs_time_L) < min_number_of_times_cutoff:
        # Ignore xj
        return time_L
    else:
        return y_cond_xi_and_not_xjs_time_L


# Get the list of (x, importance) pairs sorted in descending order of the importance
# Here, only the pairs whose importance is significant are included
def get_significant_sorted_x_importance_pair_LL(sorted_x_importance_pair_LL):
    # If sorted_x_importance_pair_LL is None or empty
    if sorted_x_importance_pair_LL is None or len(sorted_x_importance_pair_LL) == 0:
        return sorted_x_importance_pair_LL

    # Initialize significant_sorted_x_importance_pair_LL
    significant_sorted_x_importance_pair_LL = []

    # Get the z-value
    z_val_L = stats.zscore([importance for x, importance in sorted_x_importance_pair_LL])

    # Get the p-value
    p_val_L = stats.norm.sf(z_val_L)

    # Get significant_sorted_x_importance_pair_LL
    # For each index in p_val_L
    for idx in range(len(p_val_L)):
        # Get p-value
        p_val = p_val_L[idx]

        # If p-value is no larger than the p-value cutoff
        if p_val <= p_val_cutoff:
            # Update significant_sorted_x_importance_pair_LL by including (x, importance) pairs whose importance is significant
            significant_sorted_x_importance_pair_LL.append(sorted_x_importance_pair_LL[idx])

    return significant_sorted_x_importance_pair_LL


# Check whether two lists of pairs are equal
def are_lists_equal(pair_i_LL, pair_j_LL):
    # The two lists of pairs are equal if one belongs to the other, and vice versa
    if belong(pair_i_LL, pair_j_LL) is True and belong(pair_j_LL, pair_i_LL) is True:
        return True
    else:
        return False


# Check whether pair_i_LL (one list of pairs) belongs to pair_j_LL (the other)
def belong(pair_i_LL, pair_j_LL):
    # If pair_i_LL is None or empty
    if pair_i_LL is None or len(pair_i_LL) == 0:
        return True
    # If pair_j_LL is None or empty
    elif pair_j_LL is None or len(pair_j_LL) == 0:
        return False

    # For each pair in pair_i_LL
    for pair in pair_i_LL:
        # If the pair is not in pair_j_LL
        if not pair in pair_j_LL:
            return False

    return True


# The predict function
def predict():
    # For each class_label pair
    for y in sorted(y_time_val_Dic.keys()):
        # Initialize y_time_val_predicted_Dic
        if not y in y_time_val_predicted_Dic:
            y_time_val_predicted_Dic[y] = {}

        # For each time
        for time in sorted(time_var_val_Dic.keys()):
            # Initialize p_y_cond_xis_and_not_xjs_L
            p_y_cond_xis_and_not_xjs_L = [1]

            # For each (x, importance) pair in sorted_x_importance_pair_LL_Dic[y] sorted in descending order of the importance
            for x, importance in sorted_x_importance_pair_LL_Dic[y]:
                # If x is true at the time
                if x in time_var_val_Dic[time] and time_var_val_Dic[time][x] == 1:
                    # Get the last probability in p_y_cond_xis_and_not_xjs_L
                    last_prob = p_y_cond_xis_and_not_xjs_L[-1]

                    # Update the last probability to last_prob * p(y | x)
                    p_y_cond_xis_and_not_xjs_L[-1] = last_prob * importance

                    # Add last_prob * (1 - p(y | x)) to p_y_cond_xis_and_not_xjs_L
                    p_y_cond_xis_and_not_xjs_L.append(last_prob * (1 - importance))

            # Get p(y | xis and not xjs)
            p_y_cond_xis_and_not_xjs = sum(p_y_cond_xis_and_not_xjs_L[:-1])

            # Predict value
            if random.uniform(0, 1) <= p_y_cond_xis_and_not_xjs:
                val_predicted = 1
            else:
                val_predicted = 0

            # Update y_time_val_predicted_Dic
            y_time_val_predicted_Dic[y][time] = val_predicted

    # Write predict file

    # Get the header
    header_L = [y for y in sorted(y_time_val_predicted_Dic.keys())]
    # Write the header
    spamwriter_predict.writerow(header_L)

    # For each time
    for time in sorted(time_var_val_Dic.keys()):
        # Get the list of value at the time
        val_L = [y_time_val_predicted_Dic[y][time] for y in sorted(y_time_val_predicted_Dic.keys())]
        # Write the list of value at the time
        spamwriter_predict.writerow(val_L)


# Main function
if __name__ == "__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    attribute_training_data_file = sys.argv[1]
    class_training_data_file = sys.argv[2]
    attribute_testing_data_file = sys.argv[3]
    class_testing_data_file = sys.argv[4]
    fit_file = sys.argv[5]
    predict_file = sys.argv[6]
    log_file = sys.argv[7]
    max_iteration_cutoff = int(sys.argv[8])
    min_number_of_times_cutoff = int(sys.argv[9])
    min_number_of_times_ratio_cutoff = float(sys.argv[10])
    p_val_cutoff = float(sys.argv[11])
    header = int(sys.argv[12])

    # Set random seed, so that the results are reproducible
    random.seed(0)

    # Initialization
    initialization(attribute_training_data_file, class_training_data_file)

    # Initialize spamwriter_fit
    with open(fit_file, 'w') as f_fit:
        # Write the fit file
        spamwriter_fit = csv.writer(f_fit, delimiter=' ')

        # Initialize spamwriter_log
        with open(log_file, 'w') as f_log:
            # Write the log file
            spamwriter_log = csv.writer(f_log, delimiter=' ')

            # The fit function
            fit()

    # Initialization
    initialization(attribute_testing_data_file, class_testing_data_file)

    # Initialize spamwriter_predict
    with open(predict_file, 'w') as f_predict:
        # Write the predict file
        spamwriter_predict = csv.writer(f_predict, delimiter=' ')

        # The predict function
        predict()