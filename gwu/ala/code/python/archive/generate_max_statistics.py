

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


# Generate the max statistics
def generate_max_statistics():
    # Make directory
    directory = os.path.dirname(max_statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize the max precision, recall, f1 score, and accuracy
    precision_all_max = None
    recall_all_max = None
    f1_score_all_max = None
    accuracy_all_max = None

    # Write max_statistics_file
    with open(max_statistics_file, 'w') as f_max:
        for statistics_all_file in os.listdir(statistics_dir):
            if not statistics_all_file.startswith('.') and (statistics_all_file.endswith("all_classification.txt") or statistics_all_file.endswith("_all.txt")):
                # Update statistics_all_file
                statistics_all_file = statistics_dir + statistics_all_file

                # Load the statistics_all_file
                with open(statistics_all_file, 'r') as f:
                    spamreader = list(csv.reader(f, delimiter = ','))
                    # Get the value
                    # From the second line to the last (since the first line is the header)
                    for i in range(len(spamreader)):
                        # Omit empty line
                        if len(spamreader[i]) == 0:
                            continue

                        # Omit undefined statistic
                        if 'undefined' in spamreader[i][0]:
                            continue

                        if 'precision_all: ' in spamreader[i][0]:
                            precision_all = float(spamreader[i][0].strip().replace('precision_all: ', ''))
                            if precision_all_max is None or precision_all_max < precision_all:
                                precision_all_max = precision_all
                        elif 'recall_all: ' in spamreader[i][0]:
                            recall_all = float(spamreader[i][0].strip().replace('recall_all: ', ''))
                            if recall_all_max is None or recall_all_max < recall_all:
                                recall_all_max = recall_all
                        elif 'f1_score_all: ' in spamreader[i][0]:
                            f1_score_all = float(spamreader[i][0].strip().replace('f1_score_all: ', ''))
                            if f1_score_all_max is None or f1_score_all_max < f1_score_all:
                                f1_score_all_max = f1_score_all
                        elif 'accuracy_all: ' in spamreader[i][0]:
                            accuracy_all = float(spamreader[i][0].strip().replace('accuracy_all: ', ''))
                            if accuracy_all_max is None or accuracy_all_max < accuracy_all:
                                accuracy_all_max = accuracy_all

        # Write the max statistics
        f_max.write('precision_all_max: ' + str(precision_all_max) + '\n')
        f_max.write('recall_all_max: ' + str(recall_all_max) + '\n')
        f_max.write('f1_score_all_max: ' + str(f1_score_all_max) + '\n')
        f_max.write('accuracy_all_max: ' + str(accuracy_all_max) + '\n')


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    statistics_dir = sys.argv[1]
    max_statistics_file = sys.argv[2]

    # Generate the max statistics
    generate_max_statistics()

