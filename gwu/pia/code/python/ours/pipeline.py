# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

import Setting
import PIA


def get_result_from_data(data_dir, result_dir, dp_dir):
    """
    Get result from data
            
    Parameters
    ----------
    data_dir : the pathname of the data directory
    result_dir : the pathname of the result directory
    dp_dir : the pathname of the DataPreprocessing module directory
    """

    # Add code_dir folder
    sys.path.append(dp_dir)
    
    # Import the DataPreprocessing module
    import DataPreprocessing
    # Get the DataPreprocessing object
    dp = DataPreprocessing.DataPreprocessing(data_dir)

    # Match data file with names file
    data_names = dp.match_data_names()

    # The parallel pipelines for data preprocessing, train, test, and evaluate the ALA classifier
    # n_jobs = -1 indicates (all CPUs are used)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=-1)(delayed(pipeline)(dp, data_file, names_file, result_dir)
                        for data_file, names_file in data_names)


def pipeline(dp, data_files, names_file, result_dir):
    """
    The pipeline for data preprocessing, principle interaction analysis (PIA), train, test, and evaluate the classifiers
    
    Parameters
    ----------
    dp : the DataPreprocessing module
    data_files : the pathname of the data files
    names_file : the pathname of the names file
    result_dir : the pathname of the result directory
    """

    # Data preprocessing: get the Setting, Names, and Data object
    setting, names, data = dp.get_setting_names_data(data_files, names_file, result_dir, Setting)
    
    # Get the PIA object
    pia = PIA.PIA(setting.min_samples_importance, setting.min_samples_interaction, setting.random_state)

    # The fit function
    pia.fit(data.X, data.y)

    # Write the interaction file
    write_interaction_file(setting, names, pia)


def write_interaction_file(setting, names, pia):
    """
    Write the interaction file

    Parameters
    ----------
    setting: the Setting object
    names : the Names object
    pia : the PIA object
    """

    # Get the pathname of the interaction file
    interaction_file = setting.interaction_file_dir + setting.interaction_file_name + setting.interaction_file_type

    # Make directory
    directory = os.path.dirname(interaction_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(interaction_file, 'w') as f:
        # Write header
        f.write("class, interaction, probability" + '\n')

        # For each class of the target
        for class_ in sorted(pia.D.keys()):
            for I, prob in pia.D[class_]:
                f.write(str(setting.encoder.inverse_transform([class_])[0]) + ',' + ' & '.join([names.features[c] for c in I]) + ', ' + str(prob) + '\n')


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)

