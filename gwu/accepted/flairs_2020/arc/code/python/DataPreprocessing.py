# Please cite the following paper when using the code

import os
import glob
import csv
import pandas as pd
import numpy as np
import Names
import Data

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut


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
        # If data_file and names_file are in ./data/current/
        if os.path.basename(os.path.dirname(os.path.dirname(data_file))) == 'data':
            result_dir += os.path.basename(os.path.dirname(data_file)) + '/'
        # If data_file and names_file are in ./data/parent/current/
        else:
            result_dir += os.path.basename(os.path.dirname(os.path.dirname(data_file))) + '/' + os.path.basename(
                os.path.dirname(data_file)) + '/'
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
                        # If there is a value for the parameter
                        if len(spamreader[i]) > 1:
                            # Get the string on the right-hand side of '='
                            str_right = spamreader[i][1]

                            if para_name == 'combine_classes':
                                # Split the string into groups of new-old classes
                                groups = [group.strip() for group in str_right.split(';')
                                          if len(group.strip()) > 0]

                                # If groups is not empty
                                if len(groups) > 0:
                                    # Get the combine_classes dictionary
                                    combine_classes = {}

                                    for group in groups:
                                        # Split the group into new and old classes
                                        new_old_classes = [new_old_class.strip() for new_old_class in group.split(':')
                                                           if len(new_old_class.strip()) > 0]

                                        # If there are both new and old classes
                                        if len(new_old_classes) == 2:
                                            # Get the new class
                                            new_class = new_old_classes[0]

                                            # Get the old classes
                                            old_classes = [old_class.strip() for old_class in new_old_classes[1].split(',')
                                                           if len(old_class.strip()) > 0]

                                            # Update the combine_classes dictionary
                                            combine_classes[new_class] = old_classes

                                    self.get_para_vals(names, para_name, combine_classes)
                            else:
                                # Split the string into values
                                vals = [str_.strip() for str_ in str_right.split(',')
                                        if len(str_.strip()) > 0]

                                # If vals is not empty
                                if len(vals) > 0:
                                    vals = [float(val) if val.isdigit() is True else val for val in vals]
                                    self.get_para_vals(names, para_name, vals)


        # Get the features
        names.features = [feature for feature in names.columns
                          if (feature != names.target and feature not in names.exclude_features)]

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
        elif para_name == 'combine_classes':
            names.combine_classes = vals
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

            # Get data frame
            df = self.get_df(data_file, names)
        elif len(data_files) == 2:
            training_data_file = data_files[0] if 'train' in data_files[0] else data_files[1]
            testing_data_file = data_files[0] if 'test' in data_files[0] else data_files[1]

            # Get data frame for training
            df_train = self.get_df(training_data_file, names)

            # Get data frame for testing
            df_test = self.get_df(testing_data_file, names)

            # Combine training and testing data frame
            df = pd.concat([df_train, df_test])
        else:
            print("Wrong number of data files!")
            exit(1)

        if len(names.exclude_features) > 0:
            # Remove features that should be excluded
            df = df.drop(names.exclude_features, axis=1)

        # Replace missing_representation with NaN
        df = df.replace(names.place_holder_for_missing_vals, np.NaN)
        # Impute missing values using the mode
        for column in df.columns:
            df[column].fillna(df[column].mode()[0], inplace=True)

        # Get the feature vector
        X = df[names.features]

        # Get the target vector
        y = df[names.target]

        # If there are classes that should be combined
        if len(names.combine_classes.keys()) > 0:
            for new_class in names.combine_classes.keys():
                old_classes = names.combine_classes[new_class]
                for old_class in old_classes:
                    y = y.replace(to_replace=old_class, value=new_class)

        # Encode X and y
        X, y = self.encode_X_y(X, y, setting, names)

        # Update the name of features
        names.features = X.columns
        # Transform X from dataframe into numpy array
        X = X.values

        # Oversampling when y is imbalanced
        if len(np.unique(np.unique(y, return_counts=True)[1])) != 1:
            ros = RandomOverSampler(random_state=setting.random_state)
            X, y = ros.fit_sample(X, y)

        # Cross validation using StratifiedKFold or LeaveOneOut
        if X.shape[0] > setting.min_samples_importance:
            cv = StratifiedKFold(n_splits=min(min(np.bincount(y)), setting.n_splits), random_state=setting.random_state)
        else:
            cv = LeaveOneOut()

        # Get the train and test indices
        train_test_indices = [[train_index, test_index] for train_index, test_index in cv.split(X, y)]

        # Update the number of splits
        setting.n_splits = len(train_test_indices)

        # Declare the Data object
        data = Data.Data(X, y, train_test_indices)

        return data

    def get_df(self, data_file, names):
        """
        Get data frame
        :param data_file: the pathname of the data file
        :param names: the Names object
        :return: the data frame
        """

        # Load data
        if names.delim_whitespace is True:
            df = pd.read_csv(data_file, header=names.header, delim_whitespace=names.delim_whitespace)
        else:
            df = pd.read_csv(data_file, header=names.header, sep=names.sep)

        # Get df.columns
        df.columns = list(names.columns)

        # Replace '/' with '_'
        df = df.replace('/', '_')

        return df

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
        ### The k-fold cross validation
        ###--------------------------------------------------------------------------------------------------------
        
        n_splits = """ + str(setting.n_splits) + """
        
        ###--------------------------------------------------------------------------------------------------------
        ### The scaler
        ###--------------------------------------------------------------------------------------------------------
        
        scaler = """ + str(type(setting.scaler)) + """

        ###--------------------------------------------------------------------------------------------------------
        ### The random state
        ###--------------------------------------------------------------------------------------------------------

        random_state = """ + str(setting.random_state) + """

        ###--------------------------------------------------------------------------------------------------------
        ### The minimum number of samples required for calculating importance
        ###--------------------------------------------------------------------------------------------------------

        min_samples_importance = """ + str(setting.min_samples_importance) + """

        ###--------------------------------------------------------------------------------------------------------
        ### The number of jobs to run in parallel, -1 indicates (all CPUs are used)
        ###--------------------------------------------------------------------------------------------------------

        n_jobs = """ + str(setting.n_jobs) + """
        """

        parameter_file = setting.parameter_file_dir + setting.parameter_file_name + setting.parameter_file_type
        # Write the parameter file
        with open(parameter_file, 'w') as f:
            f.write(parameters + '\n')

