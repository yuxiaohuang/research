import os
import sys
import csv
import glob

from scipy.stats import ttest_ind_from_stats


def print_table(result_dir, p_val, table_file):
    """Print table"""

    files = glob.glob(result_dir + '**/*.txt', recursive=True)
    # Get the score file of others
    files_others = [file for file in files if
                    '/others/' in os.path.abspath(file) and '/score_file/' in os.path.abspath(file)]
    # Get the score file of ours
    files_ours = [file_others.replace('others', 'ours') for file_others in files_others]

    # The dictionary of datasets
    #     datasets = {'audiology' : 'audiology',
    #                 'balance-scale' : 'balance-scale',
    #                 'adult-stretch' : 'balloon (adult-stretch)',
    #                 'adult+stretch' : 'balloon (adult+stretch)',
    #                 'yellow-small' : 'balloon (yellow-small)',
    #                 'yellow-small+adult-stretch' : 'balloon (yellow-small+adult-stretch)',
    #                 'breast-cancer-wisconsin' : 'breast-cancer-wisconsin',
    #                 'car' : 'car',
    #                 'connect-4' : 'connect-4',
    #                 'hayes-roth' : 'hayes-roth',
    #                 'king-rook-vs-king-pawn' : 'king-rook-vs-king-pawn',
    #                 'lenses' : 'lenses',
    #                 'lymphography' : 'lymphography',
    #                 'monks-1' : 'monks-problems (monks-1)',
    #                 'monks-2' : 'monks-problems (monks-2)',
    #                 'monks-3' : 'monks-problems (monks-3)',
    #                 'mushroom' : 'mushroom',
    #                 'nursery' : 'nursery',
    #                 'poker' : 'poker',
    #                 'primary-tumor' : 'primary-tumor',
    #                 'soybean-large' : 'soybean (large)',
    #                 'soybean-small' : 'soybean (small)',
    #                 'SPECT' : 'SPECT',
    #                 'tic-tac-toe' : 'tic-tac-toe',
    #                 'voting-records' : 'voting-records'}

    datasets = {'audiology': '1',
                'balance-scale': '2',
                'adult-stretch': '3',
                'adult+stretch': '4',
                'yellow-small': '5',
                'yellow-small+adult-stretch': '6',
                'breast-cancer-wisconsin': '7',
                'car': '8',
                'connect-4': '9',
                'hayes-roth': '10',
                'king-rook-vs-king-pawn': '11',
                'lenses': '12',
                'lymphography': '13',
                'monks-1': '14',
                'monks-2': '15',
                'monks-3': '16',
                'mushroom': '17',
                'nursery': '18',
                'poker': '19',
                'primary-tumor': '20',
                'soybean-large': '21',
                'soybean-small': '22',
                'SPECT': '23',
                'tic-tac-toe': '24',
                'voting-records': '25'}

    # The dictionary of methods
    #     methods = {'AdaBoostClassifier' : 'AdaBoost',
    #               'DecisionTreeClassifier' : 'Decision Tree',
    #               'GaussianNB' : 'Gaussian Naive Bayes',
    #               'GaussianProcessClassifier' : 'Gaussian Process',
    #               'KNeighborsClassifier' : 'KNN',
    #               'LogisticRegression' : 'Logistic Regression',
    #               'MLPClassifier' : 'Multi-layer Perceptron',
    #               'RandomForestClassifier' : 'Random Forest',
    #               'SVC' : 'Support Vector Machine'}

    methods = {'AdaBoostClassifier': 'AB',
               'DecisionTreeClassifier': 'DT',
               'GaussianNB': 'GNB',
               'GaussianProcessClassifier': 'GP',
               'KNeighborsClassifier': 'KNN',
               'LogisticRegression': 'LR',
               'MLPClassifier': 'MLP',
               'RandomForestClassifier': 'RF',
               'SVC': 'SVC'}

    # Make directory
    directory = os.path.dirname(table_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the table file
    with open(table_file, 'w') as f:
        # Table content
        # Write the methods
        content = '\t' + '&' + '\t&'.join([methods[method] for method in sorted(methods.keys())]) + '\\\\'
        f.write(content + '\n')

        for dataset in datasets.keys():
            content = datasets[dataset] + '\t' + '&'
            scores = []

            for method in sorted(methods.keys()):
                dataset_method_score_file_others = get_score_file(files_others, dataset, method)
                dataset_method_score_file_ours = get_score_file(files_ours, dataset, method)

                if dataset_method_score_file_others is not None:
                    mean_others, std_others = get_mean_std(dataset_method_score_file_others)
                    nobs_others = get_nobs(dataset_method_score_file_others)
                else:
                    mean_others, std_others, nobs_others = None, None, None

                if dataset_method_score_file_ours is not None:
                    mean_ours, std_ours = get_mean_std(dataset_method_score_file_ours)
                    nobs_ours = get_nobs(dataset_method_score_file_ours)
                else:
                    mean_ours, std_ours, njobs_ours = None, None, None

                if (mean_others is not None
                    and std_others is not None
                    and nobs_others is not None
                    and mean_ours is not None
                    and std_ours is not None
                    and nobs_ours is not None):

                    dif = round(float(mean_others) - float(mean_ours), 2)

                    if dif < 0:
                        score = format(float(mean_others), '.2f') + ' + ' + str(
                            format(abs(dif), '.2f')) + ' $\\triangle$'
                    elif dif > 0:
                        score = format(float(mean_others), '.2f') + ' - ' + str(
                            format(abs(dif), '.2f')) + ' $\\triangledown$'
                    else:
                        score = format(float(mean_others), '.2f') + ' + 0.00'

                    if (float(std_others) + float(std_ours)) != 0:
                        # significance test
                        statistic, pvalue = ttest_ind_from_stats(mean1=float(mean_others),
                                                                 std1=float(std_others),
                                                                 nobs1=nobs_others,
                                                                 mean2=float(mean_ours),
                                                                 std2=float(std_ours),
                                                                 nobs2=nobs_ours,
                                                                 equal_var=False)

                        if statistic < 0 and pvalue < p_val:
                            score = score.replace('triangle', 'blacktriangle')
                        elif statistic > 0 and pvalue < p_val:
                            score = score.replace('triangledown', 'blacktriangledown')
                elif (mean_others is not None
                      and std_others is not None
                      and nobs_others is not None):
                    score = mean_others + ' $\pm$ NA'
                else:
                    score = 'NA $\pm$ NA'

                scores.append(score)

            content += '\t&'.join(scores) + '\\\\'
            f.write(content + '\n')


def get_score_file(files, dataset, method):
    """Get the score file containing keywords dataset and method"""

    for file in files:
        if os.path.exists(file) and dataset in file and method in file:
            return file

    return None


def get_mean_std(file):
    """Get mean and std"""

    with open(file, 'r') as f:
        # Read the file
        spamreader = list(csv.reader(f, delimiter=':'))

    # Initialize mean and std
    mean, std = None, None

    for i in range(len(spamreader)):
        # If spamreader[i] is not empty
        if spamreader[i] is not None and len(spamreader[i]) > 0:
            # Get the identifier on the left-hand side of ':'
            identifier = spamreader[i][0]

            if 'The mean of the cross validation scores' in identifier:
                # Get the mean on the right-hand side of ':'
                mean = spamreader[i][1].strip()

            if 'The std of the cross validation scores' in identifier:
                # Get the std on the right-hand side of ':'
                std = spamreader[i][1].strip()

    return [mean, std]


def get_nobs(file):
    """Get nobs"""

    with open(file, 'r') as f:
        # Read the file
        spamreader = list(csv.reader(f, delimiter=','))

    # Initialize nobs
    nobs = None

    for i in range(len(spamreader)):
        # If spamreader[i] is not empty
        if spamreader[i] is not None and len(spamreader[i]) > 0:
            # Get the identifier on the left-hand side of ','
            identifier = spamreader[i][0]

            if identifier.isdigit() is True:
                identifier = int(identifier)
                if nobs is None or nobs < identifier:
                    nobs = identifier

    return nobs + 1


if __name__ == "__main__":
    # Get the pathname of the result directory from command line
    result_dir = sys.argv[1]

    # Get the p-value from command line
    p_val = float(sys.argv[2])

    # Get the pathname of the table file
    table_file = sys.argv[3]

    # Print table
    print_table(result_dir, p_val, table_file)