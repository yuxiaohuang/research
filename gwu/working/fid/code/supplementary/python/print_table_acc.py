import os
import sys
import csv
import glob
import pandas as pd
import numpy as np

from scipy.stats import ttest_ind_from_stats


def print_table_acc():
    """
    Print table (accuracy)
    :return:
    """

    datasets = {'breast-cancer-wisconsin': '1',
                'iris': '2',
                'parkinsons': '3',
                'drug_consumption': '4'}

    methods = {'GaussianNB': 'GNB',
               'LogisticRegression': 'LR'}

    files = glob.glob(result_dir + '**/*.csv', recursive=True)

    # Get the score file of others
    files_others = [file for file in files if
                    ('/GaussianNB/' in os.path.abspath(file)
                     or '/LogisticRegression/' in os.path.abspath(file))
                    and '/cv_results_file/' in os.path.abspath(file)]

    # Get the score file of blr
    files_blr = [file for file in files if
                    '/blr/' in os.path.abspath(file) and '/cv_results_file/' in os.path.abspath(file)]

    # Make directory
    directory = os.path.dirname(table_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the table file
    with open(table_file, 'w') as f:
        # Table content
        # Write the methods
        content = 'No.' + '\t' + '&' + '\t&'.join(['\phantom{0.0}' + methods[method] for method in sorted(methods.keys())]) + '\\\\'
        f.write(content + '\n')

        for dataset in datasets.keys():
            content = dataset + '\t' + '&'
            scores = []

            for method in sorted(methods.keys()):
                dataset_method_score_files_others = get_score_files(files_others, dataset, method)
                dataset_method_score_files_blr = get_score_files(files_blr, dataset, '')

                if len(dataset_method_score_files_others) > 0:
                    nobs_others, mean_others, std_others = get_nobs_mean_std(dataset_method_score_files_others)
                else:
                    nobs_others, mean_others, std_others = None, None, None

                if len(dataset_method_score_files_blr) > 0:
                    nobs_blr, mean_blr, std_blr = get_nobs_mean_std(dataset_method_score_files_blr)
                else:
                    nobs_blr, mean_blr, std_blr = None, None, None

                if mean_others is not None and mean_blr is not None:
                    dif = mean_others - mean_blr

                    if dif < 0:
                        score = str(format(mean_others, '.' + str(num_decimal_places) + 'f')) + ' $+$ ' + str(format(abs(dif), '.' + str(num_decimal_places) + 'f')) + ' $\\vartriangle$'
                    elif dif > 0:
                        score = str(format(mean_others, '.' + str(num_decimal_places) + 'f')) + ' $-$ ' + str(format(abs(dif), '.' + str(num_decimal_places) + 'f')) + ' $\\triangledown$'
                    else:
                        score = str(format(mean_others, '.' + str(num_decimal_places) + 'f')) + ' $+$ 0'

                    if (std_others is not None
                            and std_blr is not None
                            and std_others + std_blr != 0):
                        # significance test
                        statistic, pvalue = ttest_ind_from_stats(mean1=mean_others,
                                                                 std1=std_others,
                                                                 nobs1=nobs_others,
                                                                 mean2=mean_blr,
                                                                 std2=std_blr,
                                                                 nobs2=nobs_blr,
                                                                 equal_var=False)

                        if statistic < 0 and pvalue < p_val:
                            score = score.replace('vartriangle', 'blacktriangle')
                        elif statistic > 0 and pvalue < p_val:
                            score = score.replace('triangledown', 'blacktriangledown')
                else:
                    score = '\phantom{0.00} $+$'

                scores.append(score)

            content += '\t&'.join(scores) + '\\\\'
            f.write(content + '\n')


def get_score_files(files, dataset, method):
    """
    Get the score files containing keywords dataset and method
    :param files: the score files
    :param dataset: the name of the dataset
    :param method: the name of the method
    :return: the score files containing keywords dataset and method
    """

    score_files = []

    for file in files:
        if os.path.exists(file) is True:
            for str_ in os.path.abspath(file).split('/'):
                if str_ == dataset:
                    for str_ in os.path.abspath(file).split('/'):
                        if str_ == method:
                            score_files.append(file)

    return score_files


def get_nobs_mean_std(score_files):
    """
    Get the number of observations, mean and std
    :param score_files: the score files
    :return: the number of observations, mean and std
    """

    scores = []

    for score_file in score_files:
        # Get the cv results
        df = pd.read_csv(score_file, header=0)

        # Add the test score from each fold to scores
        for col in df.columns:
            if 'split' in col and 'test_score' in col:
                scores.append(df[col][0])

    return [len(scores), round(np.mean(scores), num_decimal_places), round(np.std(scores), num_decimal_places)]


if __name__ == "__main__":
    # Get the pathname of the result directory from command line
    result_dir = sys.argv[1]

    # Get the number of decimal places
    num_decimal_places = int(sys.argv[2])

    # Get the p-value from command line
    p_val = float(sys.argv[3])

    # Get the pathname of the table file
    table_file = sys.argv[4]

    # Print table (accuracy)
    print_table_acc()