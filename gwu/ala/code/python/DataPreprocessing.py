# Please cite the following paper when using the code

import os
import glob
import csv
import pandas as pd
import numpy as np
import Names
import Data

from sklearn.model_selection import train_test_split


class DataPreprocessing():
    """The Data Processing class"""

    def __init__(self, data_dir):
        """
        Get all files in data_dir
        :param data_dir: the pathname of the data directory
        """

        self.files = glob.glob(data_dir + '**/*.txt', recursive=True) + glob.glob(data_dir + '**/*.csv', recursive=True)

    def match_data_names(self):
        """
        Match data file with names file
        :param files: the pathname of the data and names files
        :return: matched [data_file, names_file] lists
        """

        # Initialization
        data_names = []

        for names_file in self.files:
            # If names file
            if names_file.endswith('names.txt'):
                names_file_name = os.path.basename(names_file)
                names_file_name = names_file_name.replace('names.txt', '')
                data_files = []
                for data_file in self.files:
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

    def get_setting_names_data(self, data_files, names_file, result_dir, Setting):
        """
        Data preprocessing
        :param data_files: the pathname of the data files
        :param names_file: the pathname of the names file
        :param result_dir: the pathname of the result directory
        :param result_dir: the Setting module
        :return: the setting, names, and data object
        """

        data_file = data_files[0].replace('.train', '')
        data_file = data_file.replace('.test', '')

        # Get the Setting object
        # If data_file and names_file are in the same directory
        if os.path.dirname(data_file) == os.path.dirname(names_file):
            result_dir += os.path.basename(os.path.dirname(names_file)) + '/'
        else:
            result_dir += os.path.basename(os.path.dirname(data_file)) + '/' + os.path.basename(os.path.dirname(names_file)) + '/'
        setting = Setting.Setting(names_file, result_dir)

        # Get the Names object
        names = self.get_names(names_file)

        # Get the Data object
        data = self.get_data(data_files, setting, names)

        if setting.parameter_file_dir is not None:
            # Write the parameter file
            self.write_parameter_file(data_files, names_file, setting, names)

        return [setting, names, data]

    def get_names(self, names_file):
        """
        Get the Names object
        :param names_file: the pathname of the names file
        :return: the Names object
        """

        with open(names_file, 'r') as f:
            # Read the names file
            spamreader = list(csv.reader(f, delimiter='='))

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
                                self.get_para_vals(names, para_name, vals)

        # Get the features
        names.features = [feature for feature in names.columns if (feature != names.target
                                                                and feature not in names.exclude_features)]

        return names

    def get_para_vals(self, names, para_name, vals):
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

    def get_data(self, data_files, setting, names):
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
            X, y = self.get_X_y(data_file, names)

            # Encode X and y
            X, y = self.encode_X_y(X, y, setting, names)

            # Randomly choose setting.test_size% of the data for testing
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=setting.test_size,
                                                                random_state=setting.random_state,
                                                                stratify=y)
        elif len(data_files) == 2:
            training_data_file = data_files[0] if 'train' in data_files[0] else data_files[1]
            testing_data_file = data_files[0] if 'test' in data_files[0] else data_files[1]

            # Get X_train and y_train
            X_train, y_train = self.get_X_y(training_data_file, names)

            # Get X_test and y_test
            X_test, y_test = self.get_X_y(testing_data_file, names)

            # Combine training and testing data
            X = pd.concat([X_train, X_test])
            y = pd.concat([y_train, y_test])

            # Encode X and y
            X, y = self.encode_X_y(X, y, setting, names)

            X_train = X.iloc[:X_train.shape[0], :]
            X_test = X.iloc[X_train.shape[0]:, :]

            y_train = y[:y_train.shape[0]]
            y_test = y[y_train.shape[0]:]
        else:
            print("Wrong number of data files!")
            exit(1)

        # Standardize the features
        X_train = setting.scaler.fit_transform(X_train.astype(float))
        X_test = setting.scaler.transform(X_test.astype(float))

        # Declare the Data object
        data = Data.Data(X, X_train, X_test, y, y_train, y_test)

        names.features = X.columns

        return data

    def get_X_y(self, data_file, names):
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

    def encode_X_y(self, X, y, setting, names):
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

    def write_parameter_file(self, data_files, names_file, setting, names):
        """
        Write the parameter file
        :param data_file: the pathname of the data files
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
        
        data_files = """ + ', '.join(data_files) + """
        
        ###--------------------------------------------------------------------------------------------------------
        ### The pathname of the names file
        ###--------------------------------------------------------------------------------------------------------
        
        names_file = """ + names_file + """
        
        ###--------------------------------------------------------------------------------------------------------
        ### The header
        ###--------------------------------------------------------------------------------------------------------
        
        header = """ + str(names.header) + """
        
        ###--------------------------------------------------------------------------------------------------------
        ### The delimiter
        ###--------------------------------------------------------------------------------------------------------
        
        delim_whitespace = """ + str(names.delim_whitespace) + """
        
        ###--------------------------------------------------------------------------------------------------------
        ### The separator
        ###--------------------------------------------------------------------------------------------------------
        
        sep = """ + str(names.sep) + """
        
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
        
        average = """ + ', '.join(setting.average) + """
        
        ###--------------------------------------------------------------------------------------------------------
        ### The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        ###--------------------------------------------------------------------------------------------------------
        
        n_jobs = """ + str(setting.n_jobs) + """
        """

        parameter_file = setting.parameter_file_dir + setting.parameter_file_name + setting.parameter_file_type
        # Write the parameter file
        with open(parameter_file, 'w') as f:
            f.write(parameters + '\n')