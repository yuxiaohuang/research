import os
import sys
import csv
import glob
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

def print_table_data():
    """
    Print table (data)
    :return:
    """

    # Add setting_dir folder
    sys.path.append(setting_dir)
    # Import the Setting module
    import Setting

    # Match data file withm names file
    data_names = match_data_names()

    # Make directory
    directory = os.path.dirname(table_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the table file
    with open(table_file, 'w') as f:
        # Write the header
        content = 'No.' + '\t' + '&' + 'Name' + '\t' + '&' + '\#Sample' + '\t' + '&' + '\#Feature' + '\t' + '&' + '\#Condition' + '\\\\'
        f.write(content + '\n')

        for i in range(len(datasets)):
            # Initialize the content
            content = str(i + 1)

            # Get the short name of the i'th dataset
            dataset_short = datasets[i][0]
            # Get the long name of the i'th dataset
            dataset_long = datasets[i][1]

            for data_files, names_file in data_names:
                if dataset_short in names_file:
                    # Data preprocessing: get the Setting, Names, and Data object
                    get_setting_names_data(data_files, names_file, '', Setting)

                    content += '\t' + '&' + dataset_long + '\t' + '&' + str(n_sample) + '\t' + '&' + str(n_feature) + '\t' + '&' + str(n_condition) + '\\\\'
                    f.write(content + '\n')

def match_data_names():
    """
    Match data file with names file
    :return: matched [data_file, names_file] lists
    """

    files = glob.glob(data_dir + '**/*.txt', recursive=True) + glob.glob(data_dir + '**/*.csv', recursive=True)

    # Initialization
    data_names = []

    for names_file in files:
        # If names file
        if names_file.endswith('names.txt'):
            names_file_name = os.path.basename(names_file)
            names_file_name = names_file_name.replace('names.txt', '')
            data_files = []
            for data_file in files:
                # If data file
                if data_file.endswith('data.txt') or data_file.endswith('data.csv'):
                    data_file_name = os.path.basename(data_file)
                    data_file_name = data_file_name.replace('.train', '')
                    data_file_name = data_file_name.replace('.test', '')
                    data_file_name = data_file_name.replace('data.txt', '')
                    data_file_name = data_file_name.replace('data.csv', '')
                    # If data and names file match
                    if data_file_name == names_file_name:
                        data_files.append(data_file)

            # Update data_names
            data_names.append([data_files, names_file])

    return data_names

def get_setting_names_data(data_files, names_file, result_dir, Setting):
    """
    Data preprocessing
    :param data_files: the pathname of the data files
    :param names_file: the pathname of the names file
    :param result_dir: the pathname of the result directory
    :param result_dir: the Setting module
    :return:
    """

    data_file = data_files[0].replace('.train', '')
    data_file = data_file.replace('.test', '')

    # Get the Setting object
    # If data_file and names_file are in ./data/current/
    if os.path.basename(os.path.dirname(os.path.dirname(data_file))) == 'data':
        result_dir += os.path.basename(os.path.dirname(data_file)) + '/'
    # If data_file and names_file are in ./data/parent/current/
    else:
        result_dir += os.path.basename(os.path.dirname(os.path.dirname(data_file))) + '/' + os.path.basename(
            os.path.dirname(data_file)) + '/'
    setting = Setting.Setting(names_file, result_dir)

    # Get the Names object
    names = get_names(names_file)

    # Get the Data object
    get_data(data_files, setting, names)

def get_names(names_file):
    """
    Get the Names object
    :param names_file: the pathname of the names file
    :return: the Names object
    """

    with open(names_file, 'r') as f:
        # Read the names file
        spamreader = list(csv.reader(f, delimiter='='))

    # Add code_dir folder
    sys.path.append(code_dir)
    # Import the Names module
    import Names

    # Declare the Names object
    names = Names.Names()

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

    if para_name == 'header':
        names.header = int(vals[0])
    elif para_name == 'delim_whitespace':
        names.delim_whitespace = str(vals[0])
    elif para_name == 'sep':
        names.sep = str(vals[0])
    elif para_name == 'place_holder_for_missing_vals':
        names.place_holder_for_missing_vals = str(vals[0])
    elif para_name == 'columns':
        names.columns = [str(val) for val in vals]
    elif para_name == 'target':
        names.target = str(vals[0])
    elif para_name == 'exclude_features':
        names.exclude_features = [str(val) for val in vals]
    elif para_name == 'categorical_features':
        names.categorical_features = [str(val) for val in vals]

def get_data(data_files, setting, names):
    """
    Get the Data object
    :param data_files: the pathname of the data files
    :param setting: the Setting object
    :param names: the Names object
    :return: the Data object
    """

    # If one data file
    if len(data_files) == 1:
        data_file = data_files[0]

        # Get X and y
        X, y = get_X_y(data_file, names)
    elif len(data_files) == 2:
        training_data_file = data_files[0] if 'train' in data_files[0] else data_files[1]
        testing_data_file = data_files[0] if 'test' in data_files[0] else data_files[1]

        # Get X_train and y_train
        X_train, y_train = get_X_y(training_data_file, names)

        # Get X_test and y_test
        X_test, y_test = get_X_y(testing_data_file, names)

        # Combine training and testing data
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
    else:
        print("Wrong number of data files!")
        exit(1)

    # Update n_feature
    global n_feature
    n_feature = X.shape[1]

    # Encode X and y
    X, y = encode_X_y(X, y, setting, names)

    # Update n_condition
    global n_condition
    n_condition = X.shape[1]

    # Update the name of features
    names.features = X.columns
    # Transform X from dataframe into numpy array
    X = X.values

    # Oversampling when the minimum number of class label is 1
    if min(np.bincount(y)) == 1:
        ros = RandomOverSampler(random_state=setting.random_state)
        X, y = ros.fit_sample(X, y)

    # Update n_sample
    global n_sample
    n_sample = X.shape[0]

    # Cross validation using StratifiedKFold or LeaveOneOut
    if X.shape[0] > max(setting.min_samples_importance, setting.min_samples_interaction):
        cv = StratifiedKFold(n_splits=min(min(np.bincount(y)), setting.n_splits), random_state=setting.random_state)
    else:
        cv = LeaveOneOut()

    # Get the train and test indices
    train_test_indices = [[train_index, test_index] for train_index, test_index in cv.split(X, y)]

    # Update the number of splits
    setting.n_splits = len(train_test_indices)

    # Add code_dir folder
    sys.path.append(code_dir)
    # Import the Data module
    import Data

    # Declare the Data object
    data = Data.Data(X, y, train_test_indices)

    return data

def get_X_y(data_file, names):
    """
    Get X and y
    :param data_file: the pathname of the data file
    :param names: the Names object
    :return: the feature and target vector
    """

    # Load data
    if names.delim_whitespace is True:
        df = pd.read_csv(data_file, header=names.header, delim_whitespace=names.delim_whitespace)
    else:
        df = pd.read_csv(data_file, header=names.header, sep=names.sep)

    # Replace '/' with '_'
    df = df.replace('/', '_')

    # Replace missing_representation with NaN
    df = df.replace(names.place_holder_for_missing_vals, np.NaN)
    # Remove rows that contain missing values
    df = df.dropna(axis=0)

    # Get df.columns
    df.columns = list(names.columns)

    if len(names.exclude_features) > 0:
        # Remove features that should be excluded
        df = df.drop(names.exclude_features, axis=1)

    # Get the feature vector
    X = df[names.features]

    # Get the target vector
    y = df[names.target]

    return [X, y]

def encode_X_y(X, y, setting, names):
    """
    Encode X and y
    :param X: the feature vector
    :param y: the target vector
    :param setting: the Setting object
    :param names: the Names object
    :return: the encoded feature and target vector
    """

    # One-hot encoding on categorical features
    if len(names.categorical_features) > 0:
        X = pd.get_dummies(X, columns=names.categorical_features)

    # Cast X to float
    X = X.astype(float)

    # Encode the target
    y = setting.encoder.fit_transform(y)

    return [X, y]

if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the code directory from command line
    code_dir = sys.argv[2]

    # Get the pathname of the Setting module from command line
    setting_dir = sys.argv[3]

    # Get the pathname of the table file
    table_file = sys.argv[4]

    # GLobal variable
    n_sample = 0
    n_feature = 0
    n_condition = 0

    # The list of datasets
    datasets = [['audiology', 'audiology'],
                ['balance-scale', 'balance-scale'],
                ['adult-stretch', 'balloon (adult-stretch)'],
                ['adult+stretch', 'balloon (adult+stretch)'],
                ['yellow-small', 'balloon (yellow-small)'],
                ['yellow-small+adult-stretch', 'balloon (yellow-small+adult-stretch)'],
                ['breast-cancer-wisconsin', 'breast-cancer-wisconsin'],
                ['car', 'car'],
                ['connect-4', 'connect-4'],
                ['hayes-roth', 'hayes-roth'],
                ['king-rook-vs-king-pawn', 'king-rook-vs-king-pawn'],
                ['lenses', 'lenses'],
                ['lymphography', 'lymphography'],
                ['monks-1', 'monks-problems (monks-1)'],
                ['monks-2', 'monks-problems (monks-2)'],
                ['monks-3', 'monks-problems (monks-3)'],
                ['mushroom', 'mushroom'],
                ['nursery', 'nursery'],
                ['poker', 'poker'],
                ['primary-tumor', 'primary-tumor'],
                ['soybean-large', 'soybean (large)'],
                ['soybean-small', 'soybean (small)'],
                ['SPECT', 'SPECT'],
                ['tic-tac-toe', 'tic-tac-toe'],
                ['voting-records', 'voting-records']]

    # Print table (data)
    print_table_data()