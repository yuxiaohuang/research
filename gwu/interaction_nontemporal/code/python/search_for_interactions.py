# Please cite the following paper when using the code


# Modules
from __future__ import division
import numpy as np
from scipy import stats
import sys
import csv
import math
import time

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list
#_F       : indicates the variable is a flag


# Global variables
# The list of time windows, where each window, win, is a list, [win_start, win_end]
win_LL = []

# The list of timepoints
time_series_L = []

# The list of conditions
x_LL = []

# The dictionary of sources
# key: var
# val: 1
x_Dic = {}

# The dictionary of targets
# key: var
# val: 1
y_Dic = {}

# The dictionary of time series
# key: time
# val: 1
time_series_Dic = {}

# The dictionary of value
# key: var->time
# val: value of var at the time
val_Dic = {}

# The dictionary of var
# key: time
# val: the vars occur at the time
var_Dic = {}

# The dictionary of P(target)
# key: target
# val: P(target)
pro_y_Dic = {}

# The dictionary of 1 - P(target)
# key: target
# val: 1 - P(target)
not_pro_y_Dic = {}

# The dictionary of #(target), that is the number of timepoints where the target is measured
# key: target
# val: #(target)
num_y_Dic = {}

# The dictionary of #(target = 1), that is the number of timepoints where the target is 1
# key: target
# val: #(target = 1)
num_y_1_Dic = {}

# The dictionary of P(target | not condition)
# key: target->condition
# val: P(target | not condition)
pro_y_cond_not_x_Dic = {}

# The dictionary of 1 - P(target | not condition)
# key: target->condition
# val: 1 - P(target | not condition)
not_pro_y_cond_not_x_Dic = {}

# The dictionary of #(target and not condition), that is the number of timepoints where the target is measured but cannot be changed by the condition
# key: target->condition
# val: #(target and not condition)
num_y_cond_not_x_Dic = {}

# The dictionary of #(target = 1 and not condition), that is the number of timepoints where the target is 1 but cannot be changed by the condition
# key: target->condition
# val: #(target = 1 and not condition)
num_y_1_cond_not_x_Dic = {}

# The dictionary of the timepoints where the target can be changed by the condition
# key: target->condition
# val: The timepoints where the target can be changed by the condition
y_cond_x_time_LL_Dic = {}

# The dictionary records the conditions of discovered interactions
# key: condition
# val: 1
discovered_Dic = {}

# The dictionary records the replaced condition
# key: condition
# val: 1
replaced_Dic = {}

# The dictionary records the list of conjunctions for which the condition was conditioned to check the sufficient condition
# key: condition
# val: list of conjunctions
conditioned_Dic = {}

# The dictionary of conjunctions
# key: target
# val: list of conjunctions
interaction_Dic = {}

# The maximum time stamp
max_time_stamp = 0

# The minimum size of the samples
sample_size_cutoff = 30

# The condition when checking the necessary condition
condition_check_necessary_cond = 0


# Initialization
# @param        source_data_file           source data file, which includes variables that can be the causes, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        target_data_file           target data file, which includes variables that can be the effects, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
def initialization(source_data_file, target_data_file):
    # Load source file
    load_data(source_data_file, True)

    # Load target file
    load_data(target_data_file, False)

    # Get windows
    get_win_LL(lag_L)

    # Get conditions
    get_x_LL()

    # Get time series
    get_time_series()

    # Get the statistics (pro_y_Dic, not_pro_y_Dic, num_y_Dic, and num_y_1_Dic) of the targets
    get_y_statistics(y_Dic)

    # Get max time stamp
    global max_time_stamp
    max_time_stamp = time_series_L[len(time_series_L) - 1]


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

        # Get data_type_Dic, val_Dic, x_Dic and y_Dic
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            if not i in var_Dic:
                var_Dic[i] = []
            for j in range(len(spamreader[0])):
                # var's name lies in jth column in the first row
                var = spamreader[0][j].strip()

                # If the value at [i][j] is not missing
                if spamreader[i][j]:
                    # Get the value
                    val = spamreader[i][j].strip()

                    # Get val_Dic
                    if not var in val_Dic:
                        val_Dic[var] = {}
                    if val == '1':
                        val_Dic[var][i] = 1
                        var_Dic[i].append(var)
                    else:
                        val_Dic[var][i] = 0

                    # If source file
                    if x_F:
                        # Get x_Dic
                        if not var in x_Dic:
                            x_Dic[var] = 1
                    # If target file
                    else:
                        # Get y_Dic
                        if not var in y_Dic:
                            y_Dic[var] = 1


# Get windows
def get_win_LL(lag_L):
    for i in range(0, len(lag_L), 2):
        win_L = [int(lag_L[i]), int(lag_L[i + 1])]
        win_LL.append(win_L)


# Get conditions
def get_x_LL():
    for x in sorted(x_Dic.keys()):
        for win_L in win_LL:
            condition_L = [x, win_L[0], win_L[1]]
            x_LL.append(condition_L)


# Get the time series
def get_time_series():
    # Get time_series_Dic
    for var in sorted(val_Dic.keys()):
        for time in sorted(val_Dic[var].keys()):
            if not time in time_series_Dic:
                time_series_Dic[time] = 1

    # Get time_series_L
    for time in sorted(time_series_Dic.keys()):
        time_series_L.append(time)

    # Sort time_series_L
    time_series_L.sort()


# Get the statistics of the target
def get_y_statistics(y_L):
    for y in y_L:
        val_L = []
        for time in sorted(val_Dic[y].keys()):
            # Remove the impact of the conjunction from the data
            if val_Dic[y][time] != -1:
                val_L.append(val_Dic[y][time])

        # Update
        pro_y_Dic[y] = np.mean(val_L)
        not_pro_y_Dic[y] = 1 - pro_y_Dic[y]
        num_y_Dic[y] = len(val_L)
        num_y_1_Dic[y] = sum(val_L)

        # Initialization
        pro_y_cond_not_x_Dic[y] = {}
        not_pro_y_cond_not_x_Dic[y] = {}
        num_y_cond_not_x_Dic[y] = {}
        num_y_1_cond_not_x_Dic[y] = {}
        y_cond_x_time_LL_Dic[y] = {}

        y_cond_X_time_LL = get_y_cond_X_time_LL(y, [])

        for index in range(len(x_LL)):
            get_y_cond_x_statistics(y, index, y_cond_X_time_LL)


# Get the statistics of the target conditioned on the condition
def get_y_cond_x_statistics(y, index, y_cond_X_time_LL):
    # Get the timepoints where the target can be changed by the condition
    y_cond_x_time_LL = get_y_cond_X_time_LL(y, [index])

    # Update
    y_cond_x_time_LL_Dic[y][index] = y_cond_x_time_LL

    # If y_cond_X_time_LL is None
    if y_cond_X_time_LL is None:
        y_cond_X_time_LL = get_y_cond_X_time_LL(y, [])

    # Get the timepoints where the target can be changed by the conjunction but not the condition
    y_cond_X_min_x_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL)

    val_L = []
    for [time] in y_cond_X_min_x_and_not_x_time_LL:
        # Remove the impact of the conjunction from the data
        if val_Dic[y][time] != -1:
            val_L.append(val_Dic[y][time])

    # If the condition is not always present
    if len(val_L) > 0:
        # Update
        pro_y_cond_not_x_Dic[y][index] = np.mean(val_L)
        not_pro_y_cond_not_x_Dic[y][index] = 1 - pro_y_cond_not_x_Dic[y][index]
        num_y_cond_not_x_Dic[y][index] = len(val_L)
        num_y_1_cond_not_x_Dic[y][index] = sum(val_L)


# Search for interactions
def search_for_interactions():
    # For each target
    for y in sorted(y_Dic.keys()):
        # Write target to the log file
        spamwriter_log.writerow(['search target: ', y])
        spamwriter_log.writerow('')
        f_log.flush()

        # Initialize interaction_Dic[y]
        interaction_Dic[y] = []

        # The dictionary records the conditions in a discovered interaction
        # key: condition
        # val: 1
        global discovered_Dic
        discovered_Dic = {}

        # Helper: clear data structure
        [X_L, y_cond_X_time_LL] = helper_clear_data_structure(y)

        # Do, while the conjunction is not empty after shrinking
        while True:
            # Check the sufficient condition (to produce the target)
            # Flag sample_size_cutoff_met_F, indicating whether there is enough sample
            # Flag sufficient_F, indicating whether the conjunction is sufficient
            # Flag add_F, indicating whether a condition has been added to the conjunction when checking the sufficient condition
            X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F = check_sufficient_cond(y, X_L, y_cond_X_time_LL, p_val_cutoff_X, p_val_cutoff_X, False)

            # If 1) enough sample and 2) the conjunction is sufficient
            if sample_size_cutoff_met_F is False and sufficient_F is True:
                # Check the necessary condition and remove unnecessary conditions
                X_L = check_necessary_cond(y, X_L)

                # The timepoints where the target can be changed by the conjunction
                # This step was missed in the previous version
                y_cond_X_time_LL = get_y_cond_X_time_LL(y, X_L)

                # Helper for interaction: update and output
                [X_L, y_cond_X_time_LL] = helper_for_interaction(y, X_L, y_cond_X_time_LL)
                continue

            # Flag expand_F, indicating whether the conjunction can be expanded, False by default
            expand_F = False
            # If 1) enough sample and 2) no condition has been added
            if sample_size_cutoff_met_F is False and add_F is False:
                # Expand the conjunction
                [X_L, y_cond_X_time_LL, expand_F] = expand(y, X_L, y_cond_X_time_LL)

            # If 1) not enough sample or 2.1) no condition has been added and 2.2) the conjunction cannot be expanded
            if sample_size_cutoff_met_F is True or (add_F is False and expand_F is False):
                # Shrink the conjunction
                [X_L, y_cond_X_time_LL] = shrink(y, X_L, False)

                # Termination condition
                # If the conjunction is empty
                if len(X_L) == 0:
                    break


# Helper: clear data structure
def helper_clear_data_structure(y):
    # The dictionary records the replaced conditions
    # key: condition
    # val: 1
    global replaced_Dic
    replaced_Dic = {}

    # The dictionary records the list of conjunctions for which the condition was conditioned to check the sufficient condition
    # key: condition
    # val: list of conjunctions
    global conditioned_Dic
    conditioned_Dic = {}

    # The conjunction, empty by default
    X_L = []

    # The timepoints where the target can be changed by the conjunction
    y_cond_X_time_LL = get_y_cond_X_time_LL(y, X_L)

    return [X_L, y_cond_X_time_LL]


# Get the timepoints where the target can be changed by the conjunction
def get_y_cond_X_time_LL(y, X_L):
    # Initialization
    y_cond_X_time_LL = []

    # If the conjunction is None or empty, return the timepoints where the target is measured
    if X_L is None or len(X_L) == 0:
        for time in sorted(val_Dic[y].keys()):
            y_cond_X_time_LL.append([time])
        return y_cond_X_time_LL

    # Get the minimum window length of conditions in the conjunction
    min_win_len = get_min_win_len(X_L)

    # Get the dictionary of start and end
    [start_Dic, end_Dic] = get_start_end_Dic(X_L)

    # Get X_time_Dic
    # key: var
    # val: number of times the var occurs
    X_time_Dic = {}

    # Flag, indicating whether we have started recording the timepoints where all the conditions in the conjunction are present
    # Default is False
    recorded_F = False

    # Get y_cond_X_time_LL
    for time in time_series_L:
        for index in X_L:
            if index in start_Dic and time in start_Dic[index]:
                if index in X_time_Dic:
                    X_time_Dic[index] += 1
                else:
                    X_time_Dic[index] = 1
            if index in end_Dic and time in end_Dic[index]:
                if index in X_time_Dic:
                    X_time_Dic[index] -= 1
                if X_time_Dic[index] == 0:
                    del X_time_Dic[index]
        # If all the conditions in the conjunction are present
        if len(X_time_Dic) == len(X_L):
            if recorded_F is False:
                time_L = []
                recorded_F = True
            # If the target is measured at this time
            if time in val_Dic[y]:
                time_L.append(time)
            # If the last timepoint or the length of the intersection equals the minimum window length
            if time == max_time_stamp or len(time_L) == min_win_len:
                y_cond_X_time_LL.append(time_L)
                recorded_F = False
        # If some conditions are absent and we have been recording time
        elif recorded_F:
            if len(time_L) > 0:
                y_cond_X_time_LL.append(time_L)
            recorded_F = False

    return y_cond_X_time_LL


# Get the timepoints where the target cannot be changed by the conjunction
def get_y_cond_not_X_time_LL(y, X_L):
    # Initialization
    y_cond_not_X_time_LL = []

    # If the conjunction is None or empty, return the timepoints where the target is measured
    if X_L is None or len(X_L) == 0:
        for time in sorted(val_Dic[y].keys()):
            y_cond_not_X_time_LL.append([time])
        return y_cond_not_X_time_LL

    # Get the timepoints where the target is measured
    y_time_LL = get_y_cond_X_time_LL(y, [])
    # Get the timepoints where the target can be changed by the conjunction
    y_cond_X_time_LL = get_y_cond_X_time_LL(y, X_L)
    # Get the timepoints where the target cannot be changed by the conjunction
    y_cond_not_X_time_LL = get_y_cond_X_and_not_x_time_LL(y_time_LL, y_cond_X_time_LL)

    return y_cond_not_X_time_LL


# Get the dictionary of window start and window end
def get_start_end_Dic(X_L):
    # Initialization
    start_Dic = {}
    end_Dic = {}

    for index in X_L:
        [var, win_start, win_end] = x_LL[index]
        # Initialization
        if not index in start_Dic:
            start_Dic[index] = {}
        if not index in end_Dic:
            end_Dic[index] = {}

        for time in sorted(val_Dic[var].keys()):
            # If var occurs at this time
            if val_Dic[var][time] == 1:
                start_time = time + win_start
                end_time = time + win_end + 1
                # Update the two dictionaries
                start_Dic[index][start_time] = 1
                end_Dic[index][end_time] = 1

    return [start_Dic, end_Dic]


# Check sufficient condition
def check_sufficient_cond(y, X_L, y_cond_X_time_LL, p_val_cutoff_X, p_val_cutoff_X_min_x_and_not_x, check_necessary_cond_F):
    # Write the target and conjunction to the log file
    spamwriter_log.writerow(["check_sufficient_cond target: ", y])
    spamwriter_log.writerow(["check_sufficient_cond X_L: ", decode(X_L)])
    spamwriter_log.writerow(["check_suf_p_val_cutoff_X: ", p_val_cutoff_X])
    spamwriter_log.writerow(["check_sufficient_cond p_val_cutoff_X_min_x_and_not_x: ", p_val_cutoff_X_min_x_and_not_x])
    f_log.flush()

    # Flag, indicating whether there is enough sample, False by default
    sample_size_cutoff_met_F = False
    # Flag, indicating whether the conjunction is sufficient, False by default
    sufficient_F = False
    # Flag, indicating whether a condition has been added to the conjunction when checking the sufficient condition, False by default
    add_F = False

    # If the conjunction is None or empty
    if X_L is None or len(X_L) == 0:
        # Write empty line to the log file
        spamwriter_log.writerow('')
        f_log.flush()
        return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]

    # Get P(target | conjunction)
    pro_y_cond_X, num_y_cond_X, num_y_1_cond_X = get_pro_num_y_cond_X(y, y_cond_X_time_LL)
    # Write to the log file
    spamwriter_log.writerow(["check_sufficient_cond pro_y_cond_X: ", pro_y_cond_X])
    spamwriter_log.writerow(["check_sufficient_cond num_y_cond_X: ", num_y_cond_X])
    spamwriter_log.writerow(["check_sufficient_cond num_y_1_cond_X: ", num_y_1_cond_X])
    f_log.flush()

    # If not enough sample
    if num_y_cond_X <= sample_size_cutoff:
        # Update sample_size_cutoff_met_F, since there is no enough sample
        sample_size_cutoff_met_F = True

    # If P(target | conjunction) is None
    if pro_y_cond_X is None:
        # Write empty line to the log file
        spamwriter_log.writerow('')
        f_log.flush()
        return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]

    # Get numerator
    pro_y = pro_y_Dic[y]
    numerator = pro_y_cond_X - pro_y
    # Write to the log file
    spamwriter_log.writerow(["check_sufficient_cond numerator: ", numerator])
    f_log.flush()

    # Get denominator
    num_y = num_y_Dic[y]
    num_y_1 = num_y_1_Dic[y]
    pro = (num_y_1_cond_X + num_y_1) / (num_y_cond_X + num_y)
    denominator = math.sqrt(pro * (1 - pro) * (1 / num_y_cond_X + 1 / num_y))

    # If denominator is zero
    if denominator == 0:
        # Write empty line to the log file
        spamwriter_log.writerow('')
        f_log.flush()
        return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]

    # Get z value
    z_val = numerator / denominator
    # Get p value
    p_val = stats.norm.sf(z_val)
    # Write z value and p value to the log file
    spamwriter_log.writerow(["check_sufficient_cond z_val: ", z_val])
    spamwriter_log.writerow(["check_sufficient_cond p_val: ", p_val])
    spamwriter_log.writerow('')
    f_log.flush()

    # If the conjunction does not significantly increase the occurrence of the target
    if p_val >= p_val_cutoff_X:
        return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]

    # Check the sufficient conditioned on conjunction \ condition
    for index in range(len(x_LL)):
        # Write the condition to the log file
        spamwriter_log.writerow(["check_sufficient_cond x_LL[index]: ", x_LL[index]])
        f_log.flush()

        # If the condition:
        #     1) is always present
        # or  2) is always absent
        # or  3) is in the conjunction
        # or  4) the condition is the negation of a condition in the interaction
        if (not index in pro_y_cond_not_x_Dic[y]
            or y_cond_x_time_LL_Dic[y][index] is None
            or are_same_var(index, X_L) is True
            or (index in discovered_Dic and discovered_Dic[index] == 0)):

            # Write empty line to the log file
            spamwriter_log.writerow('')
            f_log.flush()
            continue

        # Get X_vote_F_Lm
        X_vote_F_L = get_X_vote_F_L(X_L, index)

        # If the condition has been used
        if X_vote_F_L is not None:
            # Get the vote of the condition
            vote_F = X_vote_F_L[1]
            # Write the vote to the log file
            spamwriter_log.writerow(["vote_F: ", vote_F])

            if vote_F is not None and vote_F >= p_val_cutoff_X_min_x_and_not_x:
                # Write empty line to the log file
                spamwriter_log.writerow('')
                f_log.flush()
                return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]
            continue

        # Get the timepoints where the target can be changed by the condition
        y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]
        # Get the timepoints where the target can be changed by the conjunction but not the condition
        y_cond_X_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL)
        # Get P(target | conjunction and not condition)
        pro_y_cond_X_and_not_x, num_y_cond_X_and_not_x, num_y_1_cond_X_and_not_x = get_pro_num_y_cond_X(y, y_cond_X_and_not_x_time_LL)
        spamwriter_log.writerow(["check_sufficient_cond pro_y_cond_X_and_not_x: ", pro_y_cond_X_and_not_x])
        spamwriter_log.writerow(["check_sufficient_cond num_y_cond_X_and_not_x: ", num_y_cond_X_and_not_x])
        spamwriter_log.writerow(["check_sufficient_cond num_y_1_cond_X_and_not_x: ", num_y_1_cond_X_and_not_x])
        f_log.flush()

        # Initialize conditioned_Dic
        if not index in conditioned_Dic:
            conditioned_Dic[index] = []

        # # If not enough sample or P(target | conjunction and not condition) is None
        # if num_y_cond_X_and_not_x <= sample_size_cutoff or pro_y_cond_X_and_not_x is None:
        # If P(target | conjunction and not condition) is None
        if pro_y_cond_X_and_not_x is None:
            # The condition cannot vote
            vote_F = None

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(X_L), vote_F])

            # Write empty line to the log file
            spamwriter_log.writerow('')
            f_log.flush()
            continue

        # # Get numerator
        # pro_y_cond_not_x = pro_y_cond_not_x_Dic[y][index]
        # numerator = pro_y_cond_X_and_not_x - pro_y_cond_not_x
        # spamwriter_log.writerow(["check_sufficient_cond numerator: ", numerator])
        # f_log.flush()
        #
        # # Get denominator
        # num_y_cond_not_x = num_y_cond_not_x_Dic[y][index]
        # num_y_1_cond_not_x = num_y_1_cond_not_x_Dic[y][index]
        # pro = (num_y_1_cond_X_and_not_x + num_y_1_cond_not_x) / (num_y_cond_X_and_not_x + num_y_cond_not_x)
        # denominator = math.sqrt(pro * (1 - pro) * (1 / num_y_cond_X_and_not_x + 1 / num_y_cond_not_x))

        # # Get numerator
        numerator = pro_y_cond_X_and_not_x - pro_y
        # Write to the log file
        spamwriter_log.writerow(["check_sufficient_cond numerator: ", numerator])
        f_log.flush()

        # Get denominator
        pro = (num_y_1_cond_X_and_not_x + num_y_1) / (num_y_cond_X_and_not_x + num_y)
        denominator = math.sqrt(pro * (1 - pro) * (1 / num_y_cond_X_and_not_x + 1 / num_y))

        # Update based on Balloons dataset
        if pro == 0:
            # Get z value
            z_val = None
            # Get p value
            p_val = 1.0
        elif pro == 1:
            # The condition cannot vote
            vote_F = None

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(X_L), vote_F])

            # Write empty line to the log file
            spamwriter_log.writerow('')
            f_log.flush()
            continue
        else:
            # Get z value
            z_val = numerator / denominator
            # Get p value
            p_val = stats.norm.sf(z_val)

        # Write z value and p value to the log file
        spamwriter_log.writerow(["check_sufficient_cond z_val: ", z_val])
        spamwriter_log.writerow(["check_sufficient_cond p_val: ", p_val])
        spamwriter_log.writerow('')
        f_log.flush()

        # The condition can vote
        vote_F = p_val

        # Update conditioned_Dic
        conditioned_Dic[index].append([list(X_L), vote_F])

        # If the conjunction \ condition does not significantly increase the occurrence of the target
        if p_val >= p_val_cutoff_X_min_x_and_not_x:
            # If the condition has not been discovered in an interaction
            if not index in discovered_Dic or discovered_Dic[index] == 0:
                # If  1) the function is called when checking the sufficient condition
                # and 2) not enough sample (so that expand will not be called)
                if (check_necessary_cond_F is False
                    and sample_size_cutoff_met_F is True):
                    # Add the condition to the conjunction
                    add(y, X_L, index)
                    # Update add_F
                    add_F = True

                    # Update y_cond_X_time_LL
                    y_cond_X_time_LL = get_y_cond_X_time_LL(y, X_L)

                    # Write the conjunction to the log file
                    spamwriter_log.writerow(["add X_L: ", decode(X_L)])
                    spamwriter_log.writerow('')
                    f_log.flush()
                    # Print the conjunction
                    print(["add X_L: ", decode(X_L)])

            return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]

    # Update sufficient_F
    sufficient_F = True

    return [X_L, y_cond_X_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F]


# Get X_vote_F_L
def get_X_vote_F_L(X_L, index):
    if not index in conditioned_Dic:
        return None
    else:
        X_vote_F_LL = conditioned_Dic[index]
        for X_vote_F_L in X_vote_F_LL:
            if X_equal(X_L, X_vote_F_L[0]):
                return X_vote_F_L

    return None


# Check whether the two conjunctions are the same
def X_equal(X_i_L, X_j_L):
    for index in X_i_L:
        if not index in X_j_L:
            return False

    for index in X_j_L:
        if not index in X_i_L:
            return False

    return True


# Check whether the condition is duplicate
def duplicate(X_L, index):
    # For each condition in the conjunction
    for index_X in X_L:
        # If the name of the two conditions are the same
        if x_LL[index_X][0] == x_LL[index][0]:
            return True

    return False


# Get the actual conditions in the conjunction
def decode(X_L):
    temp_L = []

    # If the conjunction is None or empty
    if X_L is None or len(X_L) == 0:
        return temp_L

    for index in X_L:
        temp_L.append(x_LL[index])

    return temp_L


# Get P(target | conjunction), #(target | conjunction), and #(target = 1 | conjunction)
def get_pro_num_y_cond_X(y, time_LL):
    # Initialization
    pro_y_cond_X = None
    num_y_cond_X = 0
    num_y_1_cond_X = 0

    # If time_LL is None or empty
    if time_LL is None or len(time_LL) == 0:
        return [pro_y_cond_X, num_y_cond_X, num_y_1_cond_X]

    # Get pro_y_cond_X, num_y_cond_X, and num_y_1_cond_X
    # For each time_L
    for time_L in time_LL:
        # Get temp_L
        # Initialization
        temp_L = []
        for time in time_L:
            if time in val_Dic[y]:
                temp_L.append(val_Dic[y][time])

        if len(temp_L) == 0:
            continue

        # If temp_L does not contain removed value of the target
        if min(temp_L) != -1:
            # Update num_y_cond_X, num_y_1_cond_X, and denominator
            num_y_cond_X += 1
            num_y_1_cond_X += max(temp_L)

    if num_y_cond_X != 0:
        pro_y_cond_X = num_y_1_cond_X / num_y_cond_X

    return [pro_y_cond_X, num_y_cond_X, num_y_1_cond_X]


# # Get P(target | combination), #(target | combination), and #(target = 1 | combination)
# def get_pro_num_y_cond_X(y, time_LL):
#     # Initialization
#     pro_y_cond_X = None
#     num_y_cond_X = 0
#     num_y_1_cond_X = 0
#
#     # If time_LL is None or empty
#     if time_LL is None or len(time_LL) == 0:
#         return [pro_y_cond_X, num_y_cond_X, num_y_1_cond_X]
#
#     # Get pro_y_cond_X, num_y_cond_X, and num_y_1_cond_X
#     denominator = 0
#
#     # For each time_L
#     for time_L in time_LL:
#         # Get temp_L
#         # Initialization
#         temp_L = []
#         for time in time_L:
#             if time in val_Dic[y]:
#                 temp_L.append(val_Dic[y][time])
#
#         if len(temp_L) == 0:
#             continue
#
#         # If temp_L does not contain removed value of the target
#         if min(temp_L) != -1:
#             # Update num_y_cond_X, num_y_1_cond_X, and denominator
#             num_y_cond_X += 1
#             num_y_1_cond_X += max(temp_L)
#             denominator += math.pow(not_pro_y_Dic[y], len(temp_L))
#
#     if denominator != 0:
#         numerator = num_y_cond_X - num_y_1_cond_X
#         pro_y_cond_X = 1 - numerator / denominator
#
#     return [pro_y_cond_X, num_y_cond_X, num_y_1_cond_X]


# Get the minimum window length of conditions in the conjunction
def get_min_win_len(X_L):
    # Initialization
    min_win_len = None

    # If the conjunction is None or empty, return the minimum window length, 1
    if X_L is None or len(X_L) == 0:
        return 1

    # For each condition in the conjunction
    for index in X_L:
        # Get window start, window end, and the length
        win_start = x_LL[index][1]
        win_end = x_LL[index][2]
        win_len = win_end - win_start + 1
        # Update the minimum length
        if min_win_len is None or min_win_len > win_len:
            min_win_len = win_len

    return min_win_len


# Get the timepoints where the target can be changed by the conjunction but not the condition
def get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL):
    if y_cond_x_time_LL is None or len(y_cond_x_time_LL) == 0:
        return y_cond_X_time_LL

    # Get y_cond_x_time_Dic
    y_cond_x_time_Dic = {}
    for y_cond_x_time_L in y_cond_x_time_LL:
        for time in y_cond_x_time_L:
            y_cond_x_time_Dic[time] = 1

    # Initialization
    y_cond_X_min_x_and_not_x_time_LL = []
    for y_cond_X_time_L in y_cond_X_time_LL:
        y_cond_X_min_x_and_not_x_time_L = []
        for time in y_cond_X_time_L:
            if not time in y_cond_x_time_Dic:
                y_cond_X_min_x_and_not_x_time_L.append(time)
        if len(y_cond_X_min_x_and_not_x_time_L) > 0:
            y_cond_X_min_x_and_not_x_time_LL.append(y_cond_X_min_x_and_not_x_time_L)

    return y_cond_X_min_x_and_not_x_time_LL


# Helper for interaction: update and output
def helper_for_interaction(y, X_L, y_cond_X_time_LL):
    # Update interaction_Dic
    # Get the interaction where the time window of each condition is the intersection of time windows of conditions with the same name
    X_int_L = get_X_int_L(X_L)
    # Add the interaction to interaction_Dic
    interaction_Dic[y].append(X_int_L)

    # Update discovered_Dic
    # Mark each condition in the interaction and the duplicated ones as discovered (i.e, adding the key to the dict)
    for index in range(len(x_LL)):
        if duplicate(X_L, index) is True:
            discovered_Dic[index] = 1

    # Update discovered_Dic
    for index in range(len(x_LL)):
        # If the condition under consideration and a condition in the interaction are the same variable and index is not in discovered_Dic
        if are_same_var(index, X_L) is True and not index in discovered_Dic:
            discovered_Dic[index] = 0

    # Remove the impact of the conjunction from the data
    remove_impact(y, y_cond_X_time_LL)

    # Update the statistics (pro_y_Dic, not_pro_y_Dic, num_y_Dic, and num_y_1_Dic) of the targets
    get_y_statistics([y])

    # Write the conjunction to spamwriter_interaction
    spamwriter_interaction.writerow(['interaction for ' + y + ': ', X_int_L])
    f_interaction.flush()

    # Print the conjunction
    print(['interaction for ' + y + ': ', X_int_L])

    # Helper: clear data structure
    [X_L, y_cond_X_time_LL] = helper_clear_data_structure(y)

    return [X_L, y_cond_X_time_LL]


# # Expand the conjunction by adding the condition that yields the minimum P(target | conjunction and not condition)
# def expand(y, X_L, y_cond_X_time_LL):
#     # Write the target and conjunction to the log file
#     spamwriter_log.writerow(["expand target: ", y])
#     spamwriter_log.writerow(["expand X_L: ", decode(X_L)])
#     f_log.flush()
#
#     # Flag, indicating whether the conjunction can be expanded, False by default
#     expand_F = False
#
#     # This is the condition that yields the minimum P(target | conjunction and not condition)
#     min_condition = None
#     # This is the minimum P(target | conjunction and not condition)
#     min_pro = None
#
#     # For each condition in x_LL
#     for index in range(len(x_LL)):
#         # If the condition:
#         #     1) is not in the conjunction
#         # and 2) has not been discovered in an interaction
#         # and 3) has not been replaced when shrinking the conjunction
#         # and 4) is not always present
#         # and 5) the condition is not always absent
#         # and 6) enough sample when the condition is present
#         # and 7) the condition under consideration and any condition in the interaction are not the same variable
#         if (not index in X_L
#             and (not index in discovered_Dic or discovered_Dic[index] == 0)
#             and not index in replaced_Dic
#             and index in pro_y_cond_not_x_Dic[y]
#             and y_cond_x_time_LL_Dic[y][index] is not None
#             and len(y_cond_x_time_LL_Dic[y][index]) > sample_size_cutoff
#             and are_same_var(index, X_L) is False):
#             spamwriter_log.writerow(["expand x_LL[index]: ", x_LL[index]])
#             f_log.flush()
#
#             # Get the target's value that can be changed by the condition
#             y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]
#             # Get the timepoints where the target can be changed by the conjunction but not the condition
#             y_cond_X_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL)
#             # Get P(target | conjunction and not condition)
#             pro_y_cond_X_and_not_x, num_y_cond_X_and_not_x, num_y_1_cond_X_and_not_x = get_pro_num_y_cond_X(y, y_cond_X_and_not_x_time_LL)
#
#             # Write the log file
#             spamwriter_log.writerow(["expand pro_y_cond_X_and_not_x: ", pro_y_cond_X_and_not_x])
#             spamwriter_log.writerow(["expand num_y_cond_X_and_not_x: ", num_y_cond_X_and_not_x])
#             spamwriter_log.writerow(["expand num_y_1_cond_X_and_not_x: ", num_y_1_cond_X_and_not_x])
#             f_log.flush()
#
#             # If:
#             #    1) P(target | conjunction and not condition) is None
#             # or 2) the condition is the absence of the conditions in the discovered interactions
#             # or 3) not enough sample
#             if (pro_y_cond_X_and_not_x is None
#                 or (pro_y_cond_X_and_not_x == 0 and index in discovered_Dic and discovered_Dic[index] == 0)
#                 or num_y_cond_X_and_not_x <= sample_size_cutoff):
#                 continue
#
#             # Update min_condition and min_pro
#             if min_pro is None or min_pro > pro_y_cond_X_and_not_x:
#                 min_condition = index
#                 min_pro = pro_y_cond_X_and_not_x
#
#     # If the conjunction cannot be expanded anymore
#     if min_condition is None:
#         return [X_L, y_cond_X_time_LL, expand_F]
#
#     # Update expand_F
#     expand_F = True
#
#     # Update y_cond_X_time_LL
#     y_cond_X_time_LL = get_y_cond_X_x_time_LL(y, X_L, y_cond_X_time_LL, min_condition)
#
#     # Add min_condition to the conjunction
#     add(y, X_L, min_condition)
#
#     # Write X_L to the log file
#     spamwriter_log.writerow(['expand X_L' + ': ', decode(X_L)])
#     f_log.flush()
#
#     # Print the conjunction
#     print(['expand X_L: ', decode(X_L)])
#
#     return [X_L, y_cond_X_time_LL, expand_F]


# Expand the conjunction by adding the condition that yields the minimum value of P(target | conjunction and not condition) - P(target | not condition)
def expand(y, X_L, y_cond_X_time_LL):
    # Write the target and conjunction to the log file
    spamwriter_log.writerow(["expand target: ", y])
    spamwriter_log.writerow(["expand X_L: ", decode(X_L)])
    f_log.flush()

    # Flag, indicating whether the conjunction can be expanded, False by default
    expand_F = False

    # This is the condition that yields the minimum z value
    min_condition = None
    # This is the minimum probability
    min_pro = None

    # For each condition in x_LL
    for index in range(len(x_LL)):
        # If the condition:
        #     1) has not been discovered in an interaction
        # and 2) has not been replaced when shrinking the conjunction
        # and 3, 4)  is not always present
        # and 5, 6) the condition is not always absent
        # and 7) the condition under consideration and any condition in the interaction are not the same variable
        if ((not index in discovered_Dic or discovered_Dic[index] == 0)
            and not index in replaced_Dic
            and index in num_y_cond_not_x_Dic[y]
            and num_y_cond_not_x_Dic[y][index] > sample_size_cutoff
            and y_cond_x_time_LL_Dic[y][index] is not None
            and len(y_cond_x_time_LL_Dic[y][index]) > sample_size_cutoff
            and are_same_var(index, X_L) is False):

            spamwriter_log.writerow(["expand x_LL[index]: ", x_LL[index]])
            f_log.flush()

            # Get the target's value that can be changed by the condition
            y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]
            # Get the timepoints where the target can be changed by the conjunction but not the condition
            y_cond_X_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL)
            # Get P(target | conjunction and not condition)
            pro_y_cond_X_and_not_x, num_y_cond_X_and_not_x, num_y_1_cond_X_and_not_x = get_pro_num_y_cond_X(y, y_cond_X_and_not_x_time_LL)
            # Write the log file
            spamwriter_log.writerow(["expand pro_y_cond_X_and_not_x: ", pro_y_cond_X_and_not_x])
            spamwriter_log.writerow(["expand num_y_cond_X_and_not_x: ", num_y_cond_X_and_not_x])
            spamwriter_log.writerow(["expand num_y_1_cond_X_and_not_x: ", num_y_1_cond_X_and_not_x])
            f_log.flush()

            # If:
            #    1) P(target | conjunction and not condition) is None
            # or 2) not enough sample
            if (pro_y_cond_X_and_not_x is None
                or num_y_cond_X_and_not_x <= sample_size_cutoff):
                continue

            # Update min_condition and min_pro
            if min_pro is None or min_pro > pro_y_cond_X_and_not_x:
                min_condition = index
                min_pro = pro_y_cond_X_and_not_x

    # If the conjunction cannot be expanded anymore
    if min_condition is None:
        return [X_L, y_cond_X_time_LL, expand_F]

    # Update expand_F
    expand_F = True

    # Update y_cond_X_time_LL
    y_cond_X_time_LL = get_y_cond_X_x_time_LL(y, X_L, y_cond_X_time_LL, min_condition)

    # Add min_condition to the conjunction
    add(y, X_L, min_condition)

    # Write X_L to the log file
    spamwriter_log.writerow(['expand X_L' + ': ', decode(X_L)])
    f_log.flush()
    # Print the conjunction
    print(['expand X_L: ', decode(X_L)])
    return [X_L, y_cond_X_time_LL, expand_F]


# # Expand the combination by adding the component that yields the minimum z value of P(target | combination and not component) - P(target | not component)
# def expand(y, X_L, y_cond_X_time_LL):
#     # Write the target and combination to the log file
#     spamwriter_log.writerow(["expand target: ", y])
#     spamwriter_log.writerow(["expand X_L: ", decode(X_L)])
#     f_log.flush()
#
#     # Flag, indicating whether the combination can be expanded, False by default
#     expand_F = False
#
#     # This is the component that yields the minimum z value
#     min_component = None
#     # This is the minimum z value
#     min_z_val = None
#
#     # For each condition in x_LL
#     for index in range(len(x_LL)):
#         # If the condition:
#         #     1) has not been discovered in an interaction
#         # and 2) has not been replaced when shrinking the conjunction
#         # and 3, 4)  is not always present
#         # and 5, 6) the condition is not always absent
#         # and 7) the condition under consideration and any condition in the interaction are not the same variable
#         if (not index in discovered_Dic
#             # not index in discovered_Dic or discovered_Dic[index] == 0
#             and not index in replaced_Dic
#             and index in num_y_cond_not_x_Dic[y]
#             and num_y_cond_not_x_Dic[y][index] > sample_size_cutoff
#             and y_cond_x_time_LL_Dic[y][index] is not None
#             and len(y_cond_x_time_LL_Dic[y][index]) > sample_size_cutoff
#             and are_same_var(index, X_L) is False):
#
#             spamwriter_log.writerow(["expand x_LL[index]: ", x_LL[index]])
#             f_log.flush()
#
#             # Get the target's value that can be changed by the component
#             y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]
#
#             # Get the timepoints where the target can be changed by the combination but not the component
#             y_cond_X_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL)
#             # Get P(target | combination and not component)
#             pro_y_cond_X_and_not_x, num_y_cond_X_and_not_x, num_y_1_cond_X_and_not_x = get_pro_num_y_cond_X(y, y_cond_X_and_not_x_time_LL)
#             # Get P(target | not component)
#             pro_y_cond_not_x = pro_y_cond_not_x_Dic[y][index]
#
#             # Write the log file
#             spamwriter_log.writerow(["expand pro_y_cond_X_and_not_x: ", pro_y_cond_X_and_not_x])
#             spamwriter_log.writerow(["expand num_y_cond_X_and_not_x: ", num_y_cond_X_and_not_x])
#             spamwriter_log.writerow(["expand num_y_1_cond_X_and_not_x: ", num_y_1_cond_X_and_not_x])
#             spamwriter_log.writerow(["expand pro_y_cond_not_x: ", pro_y_cond_not_x])
#             f_log.flush()
#
#             # If:
#             #    1) P(target | combination and not component) is None
#             # or 2) not enough sample
#             if (pro_y_cond_X_and_not_x is None
#                 or num_y_cond_X_and_not_x <= sample_size_cutoff):
#                 continue
#
#             # Get numerator
#             numerator = pro_y_cond_X_and_not_x - pro_y_cond_not_x
#             spamwriter_log.writerow(["check_sufficient_cond numerator: ", numerator])
#             f_log.flush()
#
#             # Get denominator
#             num_y_cond_not_x = num_y_cond_not_x_Dic[y][index]
#             num_y_1_cond_not_x = num_y_1_cond_not_x_Dic[y][index]
#             spamwriter_log.writerow(["expand num_y_1_cond_not_x: ", num_y_1_cond_not_x])
#             pro = (num_y_1_cond_X_and_not_x + num_y_1_cond_not_x) / (num_y_cond_X_and_not_x + num_y_cond_not_x)
#             denominator = math.sqrt(pro * (1 - pro) * (1 / num_y_cond_X_and_not_x + 1 / num_y_cond_not_x))
#
#             # # If denominator is zero
#             # if denominator == 0:
#             #     continue
#
#             # Update based on Ballons dataset
#             if pro == 0:
#                 min_component = index
#                 break
#             elif pro == 1:
#                 continue
#
#             # Get z value
#             z_val = numerator / denominator
#
#             # Write z value to the log file
#             spamwriter_log.writerow(["expand z_val: ", z_val])
#             spamwriter_log.writerow('')
#             f_log.flush()
#
#             if min_z_val is None or min_z_val > z_val:
#                 min_component = index
#                 min_z_val = z_val
#
#     # If the combination cannot be expanded anymore
#     if min_component is None:
#         return [X_L, y_cond_X_time_LL, expand_F]
#
#     # Update expand_F
#     expand_F = True
#
#     # Update y_cond_X_time_LL
#     y_cond_X_time_LL = get_y_cond_X_x_time_LL(y, X_L, y_cond_X_time_LL, min_component)
#
#     # Add min_component to the combination
#     add(y, X_L, min_component)
#
#     # Write X_L to the log file
#     spamwriter_log.writerow(['expand X_L' + ': ', decode(X_L)])
#     f_log.flush()
#
#     # Print the combination
#     print(['expand X_L: ', decode(X_L)])
#
#     return [X_L, y_cond_X_time_LL, expand_F]


# Get the timepoints where the target can be changed by the conjunction and the condition
def get_y_cond_X_x_time_LL(y, X_L, y_cond_X_time_LL, index):
    # Initialization
    y_cond_X_x_time_LL = []

    # If the conjunction is None or empty
    if X_L is None or len(X_L) == 0:
        # Get the timepoints where the target can be changed by the condition
        y_cond_X_x_time_LL = get_y_cond_X_time_LL(y, [index])
    else:
        # Get the timepoints where the target can be changed by the condition
        y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]

        # Get x_time_Dic
        # key: var
        # Val: time
        x_time_Dic = {}
        for y_cond_x_time_L in y_cond_x_time_LL:
            for time in y_cond_x_time_L:
                x_time_Dic[time] = 1

        # Get y_cond_X_x_time_LL
        for y_cond_X_time_L in y_cond_X_time_LL:
            y_cond_X_x_time_L = []
            for time in y_cond_X_time_L:
                if time in x_time_Dic:
                    y_cond_X_x_time_L.append(time)
            if len(y_cond_X_x_time_L) > 0:
                y_cond_X_x_time_LL.append(y_cond_X_x_time_L)

    return y_cond_X_x_time_LL


# Check the necessary condition and remove unnecessary conditions
def check_necessary_cond(y, X_L):
    # Write the target and conjunction to the log file
    spamwriter_log.writerow(["check_necessary_cond target: ", y])
    spamwriter_log.writerow(["check_necessary_cond X_L: ", decode(X_L)])
    spamwriter_log.writerow('')
    f_log.flush()

    while True:
        # Clear replaced_Dic
        global replaced_Dic
        replaced_Dic = {}

        # Flag, indicating the existence of unnecessary condition, False by default
        unnecessary_F = False

        for i in range(len(X_L)):
            # Initialize
            temp_L = list(X_L)

            # Get conjunction \ condition by shrinking
            temp_L, y_cond_temp_time_LL = shrink(y, temp_L, True)

            global condition_check_necessary_cond
            condition_check_necessary_cond = list(set(X_L) - set(temp_L))[0]

            # Check the sufficient condition (to produce the target)
            # Flag sample_size_cutoff_met_F, indicating whether there is enough sample
            # Flag sufficient_F, indicating whether the conjunction is sufficient
            # Flag add_F, indicating whether a condition has been added to the conjunction when checking the sufficient condition
            temp_L, y_cond_temp_time_LL, sample_size_cutoff_met_F, sufficient_F, add_F = check_sufficient_cond(y, temp_L, y_cond_temp_time_LL, p_val_cutoff_X, p_val_cutoff_X_min_x_and_not_x, True)

            # If the conjunction \ condition still significantly increases the occurrence of the target
            if sufficient_F is True:
                # Update unnecessary_F
                unnecessary_F = True
                break

        # If there is unnecessary condition
        if unnecessary_F is True:
            # Remove the condition (since it is not necessary)
            X_L = list(temp_L)
        else:
            break

    return X_L


# Add index to the conjunction
def add(y, X_L, index):
    # Add the index to the conjunction
    X_L.append(index)

    # Get the conjunction where the time window of each condition is the intersection of time windows of conditions with the same name
    X_int_L = get_X_int_L(X_L)

    # get X_L
    # For each condition in X_int_L
    for condition_L in X_int_L:
        # If the condition is not in the conjunction
        if not condition_L in decode(X_L):
            # Get the index of the condition in x_LL, None by default
            index = None
            for i in range(len(x_LL)):
                if x_LL[i] == condition_L:
                    index = i
                    break

            if index is None:
                # Add the condition to x_LL
                x_LL.append(condition_L)

                # Get the index
                index = len(x_LL) - 1

                # Get the statistics of the target conditioned on the condition
                get_y_cond_x_statistics(y, index, None)

            # Add the idnex to X_L
            X_L.append(index)


# Remove the impact of the conjunction from the data
def remove_impact(y, y_cond_X_time_LL):
    # Remove the impact of the conjunction from the data
    for y_cond_X_time_L in y_cond_X_time_LL:
        for time in y_cond_X_time_L:
            # If the target was changed by the conjunction at the current time
            # if time in val_Dic[y] and val_Dic[y][time] == 1:
            if time in val_Dic[y]:
                val_Dic[y][time] = -1


# Shrink the conjunction by removing the condition that yields,
#    1) the maximum P(target | conjunction \ condition and not condition)
# or 2) the maximum P(target | conjunction \ condition)
def shrink(y, X_L, check_necessary_cond_F):
    # Write the target and conjunction to the log file
    spamwriter_log.writerow(["shrink target: ", y])
    spamwriter_log.writerow(["before shrink X_L: ", decode(X_L)])
    f_log.flush()

    # If the conjunction is None or empty
    if X_L is None or len(X_L) == 0:
        return [X_L, []]

    # This is the condition that yields the maximum probability
    max_condition = None
    # This is the maximum probability
    max_pro = None
    # This is the timepoints where the target can be changed by the remaining conjunction but not max_condition
    max_y_cond_X_min_x_time_LL = []

    # For each condition in the conjunction
    for index in X_L:
        # If:
        #     1) the condition is always present
        # or  2.1) the function is called when checking the necessary condition and 2.2) the condition has been replaced and put back when checking the condition
        if (not index in pro_y_cond_not_x_Dic[y]
            or (check_necessary_cond_F is True and index in replaced_Dic)):
            continue

        # Write the condition to the log file
        spamwriter_log.writerow(["shrink x_LL[index]: ", x_LL[index]])
        f_log.flush()

        # Get conjunction \ condition
        temp_L = list(X_L)
        temp_L.remove(index)

        # Get the timepoints where the target can be changed by temp_L (i.e., conjunction \ condition)
        y_cond_X_min_x_time_LL = get_y_cond_X_time_LL(y, temp_L)
        # Get the target's value that can be changed by the condition
        y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]
        # Get the timepoints where the target can be changed by the conjunction but not the condition
        y_cond_X_min_x_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_min_x_time_LL, y_cond_x_time_LL)
        # Get P(target | conjunction \ condition and not condition)
        pro_y_cond_X_min_x_and_not_x, num_y_cond_X_min_x_and_not_x, num_y_1_cond_X_min_x_and_not_x = get_pro_num_y_cond_X(y, y_cond_X_min_x_and_not_x_time_LL)
        # Write the log file
        spamwriter_log.writerow(["shrink pro_y_cond_X_min_x_and_not_x: ", pro_y_cond_X_min_x_and_not_x])
        spamwriter_log.writerow(["shrink num_y_cond_X_min_x_and_not_x: ", num_y_cond_X_min_x_and_not_x])
        spamwriter_log.writerow(["shrink num_y_1_cond_X_min_x_and_not_x: ", num_y_1_cond_X_min_x_and_not_x])
        f_log.flush()

        # If P(target | conjunction \ condition and not condition) is None
        if pro_y_cond_X_min_x_and_not_x is None:
            max_condition = None
            # Write empty line to the log file
            spamwriter_log.writerow('')
            f_log.flush()
            break

        # Update max_condition and max_pro
        if max_pro is None or max_pro < pro_y_cond_X_min_x_and_not_x:
            max_condition = index
            max_pro = pro_y_cond_X_min_x_and_not_x
            max_y_cond_X_min_x_time_LL = y_cond_X_min_x_time_LL

    # If  P(target | conjunction \ condition and not condition) is not None for any condition
    if max_condition is not None:
        # Remove max_condition from the conjunction
        X_L.remove(max_condition)
        # Write X_L to the log file
        spamwriter_log.writerow(['after shrink X_L' + ': ', decode(X_L)])
        spamwriter_log.writerow('')
        f_log.flush()
        # Print the conjunction
        print(['shrink X_L: ', decode(X_L)])

        # Update replaced_Dic
        replaced_Dic[max_condition] = 1
        return [X_L, max_y_cond_X_min_x_time_LL]

    # This is the condition that yields the maximum probability
    max_condition = None
    # This is the maximum probability
    max_pro = None
    # This is the timepoints where the target can be changed by the remaining conjunction but not max_condition
    max_y_cond_X_time_LL = []

    # For each condition in the conjunction
    for index in X_L:
        # If 1) the function is called when checking the necessary condition and 2) the condition has been replaced and put back when checking the necessary condition
        if check_necessary_cond_F is True and index in replaced_Dic:
            continue

        spamwriter_log.writerow(["shrink x_LL[index]: ", x_LL[index]])
        f_log.flush()

        # Get conjunction \ condition
        temp_L = list(X_L)
        temp_L.remove(index)

        # Get the timepoints where the target can be changed by temp_L (i.e., conjunction \ condition)
        y_cond_X_min_x_time_LL = get_y_cond_X_time_LL(y, temp_L)
        # Get P(target | conjunction \ condition)
        pro_y_cond_X_min_x, num_y_cond_X_min_x, num_y_1_cond_X_min_x = get_pro_num_y_cond_X(y, y_cond_X_min_x_time_LL)
        # Write the log file
        spamwriter_log.writerow(["shrink pro_y_cond_X_min_x: ", pro_y_cond_X_min_x])
        spamwriter_log.writerow(["shrink num_y_cond_X_min_x: ", num_y_cond_X_min_x])
        spamwriter_log.writerow(["shrink num_y_1_cond_X_min_x: ", num_y_1_cond_X_min_x])
        f_log.flush()

        # If P(target | conjunction \ condition) is None
        if pro_y_cond_X_min_x is None:
            continue

        # Update max_condition and max_pro
        if max_pro is None or max_pro < pro_y_cond_X_min_x:
            max_condition = index
            max_pro = pro_y_cond_X_min_x
            max_y_cond_X_time_LL = y_cond_X_min_x_time_LL

    # If P(target | conjunction \ condition) is None for some condition
    if max_condition is None:
        # Use the last condition in the conjunction (which was added the most recently) as max_condition
        max_condition = X_L[len(X_L) - 1]

    # Remove max_condition from the conjunction
    X_L.remove(max_condition)
    # Write X_L to the log file
    spamwriter_log.writerow(['shrink X_L' + ': ', decode(X_L)])
    f_log.flush()
    # Print the conjunction
    print(['shrink X_L: ', decode(X_L)])

    # Update replaced_Dic
    replaced_Dic[max_condition] = 1
    return [X_L, max_y_cond_X_time_LL]


# # Shrink the conjunction by removing the condition that yields,
# #    1) the maximum z value of P(target | conjunction \ condition and not condition) - P(target | not condition)
# # or 2) the maximum P(target | conjunction \ condition)
# def shrink(y, X_L, check_necessary_cond_F):
#     # Write the target and conjunction to the log file
#     spamwriter_log.writerow(["shrink target: ", y])
#     spamwriter_log.writerow(["before shrink X_L: ", decode(X_L)])
#     f_log.flush()
#
#     # If the conjunction is None or empty
#     if X_L is None or len(X_L) == 0:
#         return [X_L, []]
#
#     # This is the condition that yields the maximum z value
#     max_condition = None
#     # This is the maximum z value
#     max_z_val = None
#     # This is the timepoints where the target can be changed by the remaining conjunction but not max_condition
#     max_y_cond_X_time_LL = []
#
#     # For each condition in the conjunction
#     for index in X_L:
#         # If:
#         #     1) the condition is always present
#         # or  2.1) the function is called when checking the necessary condition and 2.2) the condition has been replaced and put back when checking the condition
#         if (not index in pro_y_cond_not_x_Dic[y]
#             or (check_necessary_cond_F is True and index in replaced_Dic)):
#             continue
#
#         # Write the condition to the log file
#         spamwriter_log.writerow(["shrink x_LL[index]: ", x_LL[index]])
#         f_log.flush()
#
#         # Get conjunction \ condition
#         temp_L = list(X_L)
#         temp_L.remove(index)
#
#         # Get the timepoints where the target can be changed by temp_L (i.e., conjunction \ condition)
#         y_cond_X_time_LL = get_y_cond_X_time_LL(y, temp_L)
#         # Get the target's value that can be changed by the condition
#         y_cond_x_time_LL = y_cond_x_time_LL_Dic[y][index]
#         # Get the timepoints where the target can be changed by the conjunction but not the condition
#         y_cond_X_min_x_and_not_x_time_LL = get_y_cond_X_and_not_x_time_LL(y_cond_X_time_LL, y_cond_x_time_LL)
#         # Get P(target | conjunction \ condition and not condition)
#         pro_y_cond_X_min_x_and_not_x, num_y_cond_X_min_x_and_not_x, num_y_1_cond_X_min_x_and_not_x = get_pro_num_y_cond_X(y, y_cond_X_min_x_and_not_x_time_LL)
#         # Get P(target | not condition)
#         pro_y_cond_not_x = pro_y_cond_not_x_Dic[y][index]
#         # Write the log file
#         spamwriter_log.writerow(["shrink pro_y_cond_X_min_x_and_not_x: ", pro_y_cond_X_min_x_and_not_x])
#         spamwriter_log.writerow(["shrink num_y_cond_X_min_x_and_not_x: ", num_y_cond_X_min_x_and_not_x])
#         spamwriter_log.writerow(["shrink num_y_1_cond_X_min_x_and_not_x: ", num_y_1_cond_X_min_x_and_not_x])
#         spamwriter_log.writerow(["shrink pro_y_cond_not_x: ", pro_y_cond_not_x])
#         f_log.flush()
#
#         # If P(target | conjunction \ condition and not condition) is None or P(target | not condition) is None
#         if pro_y_cond_X_min_x_and_not_x is None or pro_y_cond_not_x is None:
#             max_condition = None
#             # Write empty line to the log file
#             spamwriter_log.writerow('')
#             f_log.flush()
#             break
#
#         # Get numerator
#         numerator = pro_y_cond_X_min_x_and_not_x - pro_y_cond_not_x
#         spamwriter_log.writerow(["shrink numerator: ", numerator])
#         f_log.flush()
#
#         # Get denominator
#         num_y_cond_not_x = num_y_cond_not_x_Dic[y][index]
#         num_y_1_cond_not_x = num_y_1_cond_not_x_Dic[y][index]
#         pro = (num_y_1_cond_X_min_x_and_not_x + num_y_1_cond_not_x) / (num_y_cond_X_min_x_and_not_x + num_y_cond_not_x)
#         denominator = math.sqrt(pro * (1 - pro) * (1 / num_y_cond_X_min_x_and_not_x + 1 / num_y_cond_not_x))
#
#         # # If denominator is zero
#         # if denominator == 0:
#         #     max_condition = None
#         #
#         #     # Write empty line to the log file
#         #     spamwriter_log.writerow('')
#         #     f_log.flush()
#         #
#         #     break
#
#         # Update based on Balloons dataset
#         if pro == 0:
#             # Write empty line to the log file
#             spamwriter_log.writerow('')
#             f_log.flush()
#
#             continue
#         elif pro == 1:
#             max_condition = None
#             # Write empty line to the log file
#             spamwriter_log.writerow('')
#             f_log.flush()
#             break
#
#         # Get z value
#         z_val = numerator / denominator
#         # Write z value to the log file
#         spamwriter_log.writerow(["shrink z_val: ", z_val])
#         spamwriter_log.writerow('')
#         f_log.flush()
#
#         # Update max_condition and max_z_val
#         if max_z_val is None or max_z_val < z_val:
#             max_condition = index
#             max_z_val = z_val
#             max_y_cond_X_time_LL = y_cond_X_time_LL
#
#     # If neither P(target | conjunction \ condition and not condition) nor P(target | not condition) is None for any condition
#     if max_condition is not None:
#         # Remove max_condition from the conjunction
#         X_L.remove(max_condition)
#         # Write X_L to the log file
#         spamwriter_log.writerow(['after shrink X_L' + ': ', decode(X_L)])
#         spamwriter_log.writerow('')
#         f_log.flush()
#         # Print the conjunction
#         print(['shrink X_L: ', decode(X_L)])
#
#         # Update replaced_Dic
#         replaced_Dic[max_condition] = 1
#         return [X_L, max_y_cond_X_time_LL]
#
#     # This is the condition that yields the maximum probability
#     max_condition = None
#     # This is the maximum probability
#     max_pro = None
#     # This is the timepoints where the target can be changed by the remaining conjunction but not max_condition
#     max_y_cond_X_time_LL = []
#
#     # For each condition in the conjunction
#     for index in X_L:
#         # If 1) the function is called when checking the necessary condition and 2) the condition has been replaced and put back when checking the necessary condition
#         if check_necessary_cond_F is True and index in replaced_Dic:
#             continue
#
#         spamwriter_log.writerow(["shrink x_LL[index]: ", x_LL[index]])
#         f_log.flush()
#
#         # Get conjunction \ condition
#         temp_L = list(X_L)
#         temp_L.remove(index)
#
#         # Get the timepoints where the target can be changed by temp_L (i.e., conjunction \ condition)
#         y_cond_X_not_x_time_LL = get_y_cond_X_time_LL(y, temp_L)
#         # Get P(target | conjunction \ condition)
#         pro_y_cond_X_not_x, num_y_cond_X_not_x, num_y_1_cond_X_not_x = get_pro_num_y_cond_X(y, y_cond_X_not_x_time_LL)
#         # Write the log file
#         spamwriter_log.writerow(["shrink pro_y_cond_X_not_x: ", pro_y_cond_X_not_x])
#         spamwriter_log.writerow(["shrink num_y_cond_X_not_x: ", num_y_cond_X_not_x])
#         spamwriter_log.writerow(["shrink num_y_1_cond_X_not_x: ", num_y_1_cond_X_not_x])
#         f_log.flush()
#
#         # If P(target | conjunction \ condition) is None
#         if pro_y_cond_X_not_x is None:
#             continue
#
#         # Update max_condition and max_pro
#         if max_pro is None or max_pro < pro_y_cond_X_not_x:
#             max_condition = index
#             max_pro = pro_y_cond_X_not_x
#             max_y_cond_X_time_LL = y_cond_X_not_x_time_LL
#
#     # If P(target | conjunction \ condition) is None for some condition
#     if max_condition is None:
#         # Use the last condition in the conjunction (which was added the most recently) as max_condition
#         max_condition = X_L[len(X_L) - 1]
#
#     # Remove max_condition from the conjunction
#     X_L.remove(max_condition)
#     # Write X_L to the log file
#     spamwriter_log.writerow(['shrink X_L' + ': ', decode(X_L)])
#     f_log.flush()
#
#     # Print the conjunction
#     print(['shrink X_L: ', decode(X_L)])
#
#     # Update replaced_Dic
#     replaced_Dic[max_condition] = 1
#
#     return [X_L, max_y_cond_X_time_LL]


# Check if idx_sup is a superset of some condition in the conjunction
def is_sup_set(idx_sup, X_L):
    var_sup, win_start_sup, win_end_sup = x_LL[idx_sup]

    for idx_sub in X_L:
        # If the two conditions are the same
        if idx_sub == idx_sup:
            continue

        var_sub, win_start_sub, win_end_sub = x_LL[idx_sub]

        if (var_sub == var_sup
            and win_start_sub >= win_start_sup
            and win_end_sub <= win_end_sup):
            return True

    return False

# Are the condition under consideration and a condition in the interaction the same variable
def are_same_var(idx_i, X_L):
    var_i, win_start_i, win_end_i = x_LL[idx_i]

    for idx_j in X_L:
        # If the two conditions are the same
        if idx_j == idx_i:
            continue

        var_j, win_start_j, win_end_j = x_LL[idx_j]

        if get_var_name(var_i) == get_var_name(var_j):
            return True

    return False


# Get the var name (the substring prior to the last '_')
def get_var_name(var_val):
    idx = var_val.rfind('_')
    return var_val[:idx]


# Get the conjunction where the time window of each condition is the intersection of time windows of conditions with the same name
def get_X_int_L(X_L):
    # The dictionary of the intersection of time windows
    int_win_Dic = {}

    # Get the name of the conditions
    for index in X_L:
        var, win_start, win_end = x_LL[index]
        if not var in int_win_Dic:
            int_win_Dic[var] = []

    # Get the intersection of time windows
    for var in sorted(int_win_Dic.keys()):
        win_LL = []

        # For each condition in the conjunction
        for index in X_L:
            var_ind, win_start_ind, win_end_ind = x_LL[index]

            # If the two conditions have the same name
            if var_ind == var:
                # Flag, indicating whehter the current window intersects with a window in win_L, False by default
                int_F = False

                # For each time window
                for i in range(len(win_LL)):
                    win_start = win_LL[i][0]
                    win_end = win_LL[i][1]

                    # Get the intersection
                    if win_end < win_end_ind:
                        if win_end >= win_start_ind:
                            win_LL[i][0] = max(win_start, win_start_ind)
                            win_LL[i][1] = win_end
                            int_F = True
                            break
                    elif win_end_ind >= win_start:
                        win_LL[i][0] = max(win_start, win_start_ind)
                        win_LL[i][1] = win_end_ind
                        int_F = True
                        break

                # If there is no window that intersects with the current one
                if int_F is False:
                    # Add the current one
                    win_LL.append([win_start_ind, win_end_ind])

        # Update int_win_Dic
        int_win_Dic[var] = list(win_LL)

    # Initialize
    X_int_L = []

    # For each var
    for var in sorted(int_win_Dic.keys()):
        # Add each condition
        for [win_start, win_end] in int_win_Dic[var]:
            X_int_L.append([var, win_start, win_end])

    return X_int_L


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    source_data_file = sys.argv[1]
    target_data_file = sys.argv[2]
    interaction_file = sys.argv[3]
    log_file = sys.argv[4]
    fig_dir = sys.argv[5]
    p_val_cutoff_X = float(sys.argv[6])
    p_val_cutoff_X_min_x_and_not_x = float(sys.argv[7])
    sample_size_cutoff = int(sys.argv[8])
    lag_L = sys.argv[9:]

    start_time = time.clock()

    # Initialization
    initialization(source_data_file, target_data_file)

    with open(log_file, 'w') as f_log:
        # Write the log file
        spamwriter_log = csv.writer(f_log, delimiter=' ')
        with open(interaction_file, 'w') as f_interaction:
            # Write the causal conjunction file
            spamwriter_interaction = csv.writer(f_interaction, delimiter=' ')

            # Search for the interactions
            search_for_interactions()

            end_time = time.clock()
            run_time = end_time - start_time

            # Write run time
            spamwriter_interaction.writerow(['run time: ' + str(run_time)])