

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
def generate_max_statistics_interaction():
    # Make directory
    directory = os.path.dirname(max_statistics_interaction_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize the max precision, recall, and f1 score
    precision_max = None
    recall_max = None
    f1_score_max = None

    # Write max_statistics_interaction_file
    with open(max_statistics_interaction_file, 'w') as f_max:
        for statistics_file in os.listdir(statistics_dir):
            if not statistics_file.startswith('.') and statistics_file.endswith('.txt'):
                # Update statistics_file
                statistics_file = statistics_dir + statistics_file

                # Load the statistics_file
                with open(statistics_file, 'r') as f:
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

                        if 'precision: ' in spamreader[i][0]:
                            precision = float(spamreader[i][0].strip().replace('precision: ', ''))
                            if precision_max is None or precision_max < precision:
                                precision_max = precision
                        elif 'recall: ' in spamreader[i][0]:
                            recall = float(spamreader[i][0].strip().replace('recall: ', ''))
                            if recall_max is None or recall_max < recall:
                                recall_max = recall
                        elif 'f1 score: ' in spamreader[i][0]:
                            f1_score = float(spamreader[i][0].strip().replace('f1 score: ', ''))
                            if f1_score_max is None or f1_score_max < f1_score:
                                f1_score_max = f1_score

        # Write the max statistics
        f_max.write('precision_max: ' + str(precision_max) + '\n')
        f_max.write('recall_max: ' + str(recall_max) + '\n')
        f_max.write('f1_score_max: ' + str(f1_score_max) + '\n')


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    statistics_dir = sys.argv[1]
    max_statistics_interaction_file = sys.argv[2]

    # Generate the max statistics
    generate_max_statistics_interaction()

