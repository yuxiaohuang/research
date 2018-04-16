# Please cite the following paper when using the code

import sys
import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed


class Setting:
    """The Setting class"""

    def __init__(self, data_file, result_dir):

        # The label encoder
        self.encoder = LabelEncoder()

        # The percentage of the testing set
        self.test_size = 0.3

        # The scaler
        self.scaler = StandardScaler()

        # The random state
        self.random_state = 0

        # The maximum number of iterations
        self.max_iter = 100

        # The minimum number of samples in each bin
        self.min_samples_bin = 1

        # The value of C
        self.C = 1

        # The average for precision_recall_fscore_support
        self.average = 'micro'

        # The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        self.n_jobs = -1

        # The pathname of the parameter file directory
        self.parameter_file_dir = result_dir + 'parameter_file/'

        # The name of the parameter file
        self.parameter_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the parameter_file
        self.parameter_file_type = '.txt'

        # The pathname of the MSE figure directory
        self.mse_fig_dir = result_dir + 'mse_fig/'

        # The name of the MSE figure
        self.mse_fig_name = os.path.basename(data_file).split('.')[0]

        # The type of the MSE figure
        self.mse_fig_type = '.pdf'

        # The pathname of the probability distribution figure directory
        self.prob_dist_fig_dir = result_dir + 'prob_dist_fig/'

        # The name of the probability distribution figure
        self.prob_dist_fig_name = os.path.basename(data_file).split('.')[0]

        # The type of the probability distribution figure
        self.prob_dist_fig_type = '.pdf'

        # The pathname of the probability distribution file directory
        self.prob_dist_file_dir = result_dir + 'prob_dist_file/'

        # The name of the probability distribution file
        self.prob_dist_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the probability distribution file
        self.prob_dist_file_type = '.csv'

        # The pathname of the score file directory
        self.score_file_dir = result_dir + 'score_file/'

        # The name of the score file
        self.score_file_name = os.path.basename(data_file).split('.')[0]

        # The type of the score file
        self.score_file_type = '.txt'


class Names:
    """The Names class"""

    def __init__(self):

        # The header
        self.header = None

        # The place holder for missing values
        self.place_holder_for_missing_vals = '?'

        # The (name of the) columns
        self.columns = None

        # The (name of the) target
        self.target = None

        # The (name of the) features that should be excluded
        self.exclude_features = []

        # The (name of the) categorical features
        self.categorical_features = []

        # The (name of the) features
        # This is not a parameter in the names file,
        # since it can either be inferred based on self.columns and self.target
        self.features = None

        # The parameter names
        self.para_names = ['header',
                           'place_holder_for_missing_vals',
                           'columns',
                           'target',
                           'exclude_features',
                           'categorical_features']


class Data:
    """The Data class"""

    def __init__(self,
                 X,
                 X_train,
                 X_test,
                 y,
                 y_train,
                 y_test):

        # The feature data
        self.X = X

        # The feature data for training
        self.X_train = X_train

        # The feature data for testing
        self.X_test = X_test

        # The target data
        self.y = y

        # The target data for training
        self.y_train = y_train

        # The target data for testing
        self.y_test = y_test


def match_data_names(files):
    """
    Match data file with names file
    :param files: the pathname of the data and names files
    :return: matched [data_file, names_file] lists
    """

    # Initialization
    data_names = []

    for data_file in files:
        # If data file
        if data_file.endswith('data.txt') or data_file.endswith('data.csv'):
            for names_file in files:
                # If names file
                if names_file.endswith('names.txt'):
                    # If data and names file match
                    if (data_file.replace('data.txt', '') == names_file.replace('names.txt', '')
                        or data_file.replace('data.csv', '') == names_file.replace('names.txt', '')):
                        # Update data_names
                        data_names.append([data_file, names_file])

    return data_names


def pipeline(data_file, names_file, result_dir):
    """
    The pipeline for data preprocessing, train, test, and evaluate the ALA classifier
    :param data_file: the pathname of the data file
    :param names_file: the pathname of the names file
    :param result_dir: the pathname of the result directory
    :return:
    """

    # Data preprocessing
    setting, names, data = data_preprocessing(data_file, names_file, result_dir)

    # Train, test, and evaluate the ALA classifier
    train_test_eval(setting, names, data)


def data_preprocessing(data_file, names_file, result_dir):
    """
    Data preprocessing
    :param data_file: the pathname of the data file
    :param names_file: the pathname of the names file
    :param result_dir: the pathname of the result directory
    :return: the setting, names, and data object
    """

    # Get the Setting object
    result_dir += os.path.basename(os.path.dirname(data_file)) + '/'
    setting = Setting(data_file, result_dir)

    # Get the Names object
    names = get_names(names_file)

    # Get the Data object
    data = get_data(data_file, setting, names)

    if setting.parameter_file_dir is not None:
        # Write the parameter file
        write_parameter_file(data_file, names_file, setting, names)

    return [setting, names, data]


def get_names(names_file):
    """
    Get the Names object
    :param names_file: the pathname of the names file
    :return: the Names object
    """

    with open(names_file, 'r') as f:
        # Read the names file
        spamreader = list(csv.reader(f, delimiter='='))

    # Declare the Names object
    names = Names()

    # For each parameter
    for para_name in names.para_names:
        # For each row in the names file
        for i in range(len(spamreader)):
            # If spamreader[i] is not empty
            if spamreader[i] is not None and len(spamreader[i]) > 0:
                # Get the string on the left-hand side of '='
                str_left = spamreader[i][0]

                # Ignore comments
                if str_left.startswith('#'):
                    continue

                if para_name in str_left:
                    # If there are values for the parameter
                    if len(spamreader[i]) > 1:
                        # Get the string on the right-hand side of '='
                        str_right = spamreader[i][1]

                        # Split the string into strings
                        strs = str_right.split(",")

                        # Get the (non-empty) values
                        vals = [str.strip() for str in strs if len(str.strip()) > 0]

                        # If vals is not empty
                        if len(vals) > 0:
                            vals = [float(val) if val.isdigit() is True else val for val in vals]
                            get_para_vals(names, para_name, vals)

    # Get the features
    names.features = [feature for feature in names.columns if (feature != names.target
                                                            and feature not in names.exclude_features)]

    return names


def get_para_vals(names, para_name, vals):
    """
    Get parameter values
    :param names: the Names object
    :param para_name: the parameter name
    :param vals: the values
    :return:
    """

    vals = vals[0] if len(vals) == 1 else vals

    if para_name == 'header':
        names.header = int(vals)
    elif para_name == 'place_holder_for_missing_vals':
        names.place_holder_for_missing_vals = vals
    elif para_name == 'columns':
        names.columns = vals
    elif para_name == 'target':
        names.target = vals
    elif para_name == 'exclude_features':
        names.exclude_features = vals
    elif para_name == 'categorical_features':
        names.categorical_features = vals


def get_data(data_file, setting, names):
    """
    Get the Data object
    :param data_file: the pathname of the data file
    :param setting: the Setting object
    :param names: the Names object
    :return: the Data object
    """

    # Load data
    df = pd.read_csv(data_file, header=names.header)

    # Replace missing_representation with NaN
    df = df.replace(names.place_holder_for_missing_vals, np.NaN)
    # Remove rows that contain missing values
    df = df.dropna(axis=0)

    # Get df.columns
    df.columns = list(names.columns)

    if len(names.exclude_features) > 0:
        # Remove features that should be excluded
        df = df.drop([names.exclude_features], axis=1)

    # Get the feature vector
    X = df[names.features]

    # One-hot encoding on categorical features
    if len(names.categorical_features) > 0:
        X = pd.get_dummies(X, columns=names.categorical_features)
        names.features = X.columns

    # Cast X to float
    X = X.astype(float)

    # Get the target vector
    y = df[names.target]

    # Encode the target
    y = setting.encoder.fit_transform(y)

    # Randomly choose setting.test_size% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=setting.test_size,
                                                        random_state=setting.random_state,
                                                        stratify=y)

    # Standardize the features
    X_train = setting.scaler.fit_transform(X_train.astype(float))
    X_test = setting.scaler.transform(X_test.astype(float))

    # Declare the Data object
    data = Data(X, X_train, X_test, y, y_train, y_test)

    return data


def write_parameter_file(data_file, names_file, setting, names):
    """
    Write the parameter file
    :param data_file: the pathname of the data file
    :param names_file: the pathname of the names file
    :param setting: the Setting object
    :param names: the Names object
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.parameter_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the parameters
    parameters = """
    ###--------------------------------------------------------------------------------------------------------
    ### Parameter file for AnotherLogisticAlgorithm (ALA) classifier
    ###--------------------------------------------------------------------------------------------------------
    
    ###--------------------------------------------------------------------------------------------------------
    ### The pathname of the data file
    ###--------------------------------------------------------------------------------------------------------
    
    data_file = """ + data_file + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The pathname of the names file
    ###--------------------------------------------------------------------------------------------------------
    
    names_file = """ + names_file + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The header
    ###--------------------------------------------------------------------------------------------------------
    
    header = """ + str(names.header) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The place holder for missing values
    ###--------------------------------------------------------------------------------------------------------
    
    place_holder_for_missing_vals = """ + str(names.place_holder_for_missing_vals) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The (name of the) columns
    ###--------------------------------------------------------------------------------------------------------
    
    columns = """ + ', '.join(names.columns) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The (name of the) target
    ###--------------------------------------------------------------------------------------------------------
    
    target = """ + names.target + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The (name of the) features
    ###--------------------------------------------------------------------------------------------------------
    
    features = """ + ', '.join(names.features) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The (name of the) features that should be excluded
    ###--------------------------------------------------------------------------------------------------------
    
    exclude_features = """ + ', '.join(names.exclude_features) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The (name of the) categorical features
    ###--------------------------------------------------------------------------------------------------------
    
    categorical_features = """ + ', '.join(names.categorical_features) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The label encoder
    ###--------------------------------------------------------------------------------------------------------
    
    encoder = """ + str(type(setting.encoder)) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The percentage of the testing set
    ###--------------------------------------------------------------------------------------------------------
    
    test_size = """ + str(setting.test_size) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The scaler
    ###--------------------------------------------------------------------------------------------------------
    
    scaler = """ + str(type(setting.scaler)) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The random state
    ###--------------------------------------------------------------------------------------------------------
    
    random_state = """ + str(setting.random_state) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The maximum number of iterations
    ###--------------------------------------------------------------------------------------------------------
    
    max_iter = """ + str(setting.max_iter) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The minimum number of samples in each bin
    ###--------------------------------------------------------------------------------------------------------
    
    min_samples_bin = """ + str(setting.min_samples_bin) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The value of C
    ###--------------------------------------------------------------------------------------------------------
    
    C = """ + str(setting.C) + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The average for precision_recall_fscore_support
    ###--------------------------------------------------------------------------------------------------------
    
    average = """ + setting.average + """
    
    ###--------------------------------------------------------------------------------------------------------
    ### The number of jobs to run in parallel, -1 indicates (all CPUs are used)
    ###--------------------------------------------------------------------------------------------------------
    
    n_jobs = """ + str(setting.n_jobs) + """
    """

    parameter_file = setting.parameter_file_dir + setting.parameter_file_name + setting.parameter_file_type
    # Write the parameter file
    with open(parameter_file, 'w') as f:
        f.write(parameters + '\n')


def train_test_eval(setting, names, data):
    """
    Train, test, and evaluate the ALA classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :return:
    """

    # Declare the ALA classifier
    ala = ALA.ALA(setting.max_iter, setting.min_samples_bin, setting.C)

    # Train ala
    ala.fit(data.X_train, data.y_train)

    # Test ala
    y_pred = ala.predict(data.X_test)

    # Evaluate ala
    eval(setting, names, data, ala, y_pred)


def eval(setting, names, data, ala, y_pred):
    """
    Evaluate the ALA classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param ala: the ALA classifier
    :param y_pred: the predicted values of the target
    :return:
    """

    set_plt()

    if setting.mse_fig_dir is not None:
        # Plot the mean square error figure
        plot_mse_fig(setting, ala)

    if setting.prob_dist_fig_dir is not None:
        # Plot the probability distribution figures
        plot_prob_dist_fig(setting, names, data.X, ala)

    if setting.prob_dist_file_dir is not None:
        # Write the probability distribution file
        write_prob_dist_file(setting, names, data.X, ala)

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, data.y_test, y_pred)


def set_plt():
    """
    Set plt
    :return:
    """
    size = 30

    # Set text size
    plt.rc('font', size=size)

    # Set axes title size
    plt.rc('axes', titlesize=size)

    # Set x and y labels size
    plt.rc('axes', labelsize=size)

    # Set xtick and ytick size
    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)

    # Set legend size
    plt.rc('legend', fontsize=size)

    # Set figure title size
    plt.rc('figure', titlesize=size)

    plt.switch_backend('agg')


def plot_mse_fig(setting, ala):
    """
    Plot the mean square error figure
    :param setting: the Setting object
    :param ala: the ALA classifier
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.mse_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.plot(range(1, len(ala.mses_) + 1), ala.mses_)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.tight_layout()
    mse_fig = setting.mse_fig_dir + setting.mse_fig_name + setting.mse_fig_type
    plt.savefig(mse_fig, dpi=300)


def plot_prob_dist_fig(setting, names, X, ala):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature vector
    :param ala: the ALA classifier
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.prob_dist_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    ala.get_prob_dist_dict(setting.scaler.transform(X))

    # For each unique value of the target
    for yu in sorted(ala.prob_dist_dict_.keys()):
        # Get the original value of yu
        yu_orig = str(setting.encoder.inverse_transform(yu))

        # For each xj
        for j in sorted(ala.prob_dist_dict_[yu].keys()):
            xijs = sorted(ala.prob_dist_dict_[yu][j].keys())
            pijs = [ala.prob_dist_dict_[yu][j][xij] for xij in xijs]
            xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))
            xijs_orig = [round(xij_orig, 2) for xij_orig in xijs_orig]

            # Get the pandas series
            df = pd.DataFrame(list(zip(xijs_orig, pijs)), columns=['Feature value', 'Probability'])

            xj = 'x0' if j == 0 else names.features[j - 1]

            # Plot the histogram of the series
            df.plot(x='Feature value',
                    y='Probability',
                    kind='bar',
                    figsize=(20, 10),
                    title=('P(' + yu_orig + ' | ' + xj + ')'),
                    legend=False,
                    color='b')

            # Set the x-axis label
            plt.xlabel("Feature value")
            # Set the y-axis label
            plt.ylabel("Probability")

            plt.tight_layout()
            prob_dist_fig = (setting.prob_dist_fig_dir + setting.prob_dist_fig_name + '_' + yu_orig + '_' + xj
                             + setting.prob_dist_fig_type)
            plt.savefig(prob_dist_fig)


def write_prob_dist_file(setting, names, X, ala):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature vector
    :param ala: the ALA object
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.prob_dist_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    ala.get_prob_dist_dict(setting.scaler.transform(X))

    prob_dist_file = setting.prob_dist_file_dir + setting.prob_dist_file_name + setting.prob_dist_file_type

    with open(prob_dist_file, 'w') as f:
        # Write header
        f.write("yu, xj, xij, pij" + '\n')

        # For each unique value of the target
        for yu in sorted(ala.prob_dist_dict_.keys()):
            # Get the original value of yu
            yu_orig = str(setting.encoder.inverse_transform(yu))

            # For each xj
            for j in sorted(ala.prob_dist_dict_[yu].keys()):
                xj = 'x0' if j == 0 else names.features[j - 1]
                xijs = sorted(ala.prob_dist_dict_[yu][j].keys())
                pijs = [ala.prob_dist_dict_[yu][j][xij] for xij in xijs]
                xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))

                for idx in range(len(pijs)):
                    pij = pijs[idx]
                    xij_orig = xijs_orig[idx]
                    f.write(yu_orig + ', ' + xj + ', ' + str(xij_orig) + ', ' + str(pij) + '\n')


def write_score_file(setting, y_test, y_pred):
    """
    Write the score file
    :param setting: the Setting object
    :param y_test: the testing set
    :param y_pred: the predicted values of the target
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting.score_file_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    score_file = setting.score_file_dir + setting.score_file_name + setting.score_file_type

    with open(score_file, 'w') as f:
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=setting.average)

        # Write header
        f.write("precision, recall, fscore" + '\n')

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n')


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get all files in data_dir
    files = glob.glob(data_dir + '**/*.txt', recursive=True) + glob.glob(data_dir + '**/*.csv', recursive=True)

    # Match data file with names file
    data_names = match_data_names(files)

    # The parallel pipelines for data preprocessing, train, test, and evaluate the ALA classifier
    # n_jobs = -1 indicates (all CPUs are used)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=-1)(delayed(pipeline)(data_file, names_file, result_dir)
                        for data_file, names_file in data_names)