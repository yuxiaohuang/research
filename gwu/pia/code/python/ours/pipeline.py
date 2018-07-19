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
    pia = PIA.PIA(setting.min_samples_importance, setting.min_samples_interaction, setting.p_val, setting.random_state)

    # The fit-transform
    data.X_train_I = pia.fit_transform(data.X_train, data.y_train)
    # The transform
    data.X_test_I = pia.transform(data.X_test)
    # Update names.features_I
    for class_ in pia.D.keys():
        for I, prob in pia.D[class_]:
            # Here, we only add interaction with multiple conditions
            if len(I) > 1:
                names.features_I.append([names.features[c] for c in I])

    # Write the interaction file
    write_interaction_file(setting, names, pia)
        
    # Train, test, and evaluate the classifier
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(train_test_eval)(setting, names, data, clf_name, pia)
                                    for clf_name in setting.classifiers.keys())


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


def train_test_eval(setting, names, data, clf_name, pia):
    """
    Train, test, and evaluate the classifier
    
    Parameters
    ----------
    setting : the Setting object
    names : the Names object
    data : the Data object
    clf_name : the name of the classifier
    pia : the PIA object
    """

    classifier = setting.classifiers[clf_name]

    if clf_name == 'RandomForestClassifier':
        clf = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
        clf_I = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
    elif clf_name == 'AdaBoostClassifier':
        clf = classifier(random_state=setting.random_state)
        clf_I = classifier(random_state=setting.random_state)
    elif clf_name == 'MLPClassifier':
        clf = classifier(random_state=setting.random_state)
        clf_I = classifier(random_state=setting.random_state)
    elif clf_name == 'KNeighborsClassifier':
        clf = classifier(n_jobs=setting.n_jobs)
        clf_I = classifier(n_jobs=setting.n_jobs)
    elif clf_name == 'GaussianNB':
        clf = classifier()
        clf_I = classifier()
    elif clf_name == 'DecisionTreeClassifier':
        clf = classifier(random_state=setting.random_state)
        clf_I = classifier(random_state=setting.random_state)
    elif clf_name == 'LogisticRegression':
        clf = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
        clf_I = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
    elif clf_name == 'GaussianProcessClassifier':
        clf = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
        clf_I = classifier(random_state=setting.random_state, n_jobs=setting.n_jobs)
    elif clf_name == 'SVC':
        clf = classifier(random_state=setting.random_state)
        clf_I = classifier(random_state=setting.random_state)

    # Train clf
    clf.fit(data.X_train, data.y_train)

    # Test clf
    y_pred = clf.predict(data.X_test)
    
    # Train clf_I, with interaction
    clf_I.fit(data.X_train_I, data.y_train)

    # Test clf, with interaction
    y_pred_I = clf_I.predict(data.X_test_I)

    # Evaluate clf
    eval(setting, names, data, clf, clf_I, y_pred, y_pred_I, clf_name, pia)    


def eval(setting, names, data, clf, clf_I, y_pred, y_pred_I, clf_name, pia):
    """
    Evaluate the classifier
    
    Parameters
    ----------
    setting: the Setting object
    names: the Names object
    data: the Data object
    clf: the classifier
    clf_I: the classifier, with interaction
    y_pred: the predicted values of the target
    y_pred_I: the predicted values of the target, with interaction
    clf_name: the name of the classifier
    pia : the PIA object
    """

    setting.set_plt()

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, data.y_test, y_pred, y_pred_I, clf_name)

    if (setting.feature_importance_fig_dir is not None
        and (isinstance(clf, setting.classifiers['RandomForestClassifier']) is True)):
        # Plot the feature importance figures
        plot_feature_importance_fig(setting, names, clf_I, clf_name)


def write_score_file(setting, y_test, y_pred, y_pred_I, clf_name):
    """
    Write the score file
    
    Parameters
    ----------
    setting: the Setting object
    y_test: the actual values of the target
    y_pred: the predicted values of the target
    y_pred_I: the predicted values of the target, with interaction
    clf_name: the name of the classifier
    """

    # Get the directory of the score file
    score_file_dir = setting.score_file_dir + clf_name + '/'
    # Get the pathname of the score file
    score_file = score_file_dir + setting.score_file_name + setting.score_file_type

    # Make directory
    directory = os.path.dirname(score_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(score_file, 'w') as f:
        for average in setting.average:      
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=average)
            # Write the precision, recall, and fscore
            f.write("precision, recall, fscore using " + average + ':' + '\n')
            f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n\n')
         
            precision_I, recall_I, fscore_I, support_I = precision_recall_fscore_support(y_test, y_pred_I, average=average)
            # Write the precision, recall, and fscore, with interaction
            f.write("precision, recall, fscore using " + average + ', with interaction:' + '\n')
            f.write(str(precision_I) + ', ' + str(recall_I) + ', ' + str(fscore_I) + '\n\n')
            
            dif_precision, dif_recall, dif_fscore = precision_I - precision, recall_I - recall, fscore_I - fscore
            # Write dif_precision, dif_recall, and dif_fscore
            f.write("precision (with interaction) - precison, recall (with interaction) - recall, fscore (with interaction) - fscore:" + '\n')
            f.write(str(dif_precision) + ', ' + str(dif_recall) + ', ' + str(dif_fscore) + '\n\n')
            
        accuracy = accuracy_score(y_test, y_pred)
        # Write the accuracy
        f.write("accuracy:" + '\n')
        f.write(str(accuracy) + '\n')
        
        accuracy_I = accuracy_score(y_test, y_pred_I)
        # Write the accuracy, with interaction
        f.write("accuracy, with interaction:" + '\n')
        f.write(str(accuracy_I) + '\n')

        dif_accuracy = accuracy_I - accuracy
        # Write dif_accuracy
        f.write("accuracy (with interaction) - accuracy:" + '\n')
        f.write(str(dif_accuracy) + '\n\n')


def plot_feature_importance_fig(setting, names, clf, clf_name):
    """
    Plot the feature importance figures
    
    Parameters
    ----------
    setting : the Setting object
    names : the Names object
    clf : the classifier
    clf_name: the name of the classifier
    """

    # Get the directory of the feature importance file
    feature_importance_fig_dir = setting.feature_importance_fig_dir + clf_name + '/'
    # Get the pathname of the feature importance figure
    feature_importance_fig = (feature_importance_fig_dir + setting.feature_importance_fig_name + setting.feature_importance_fig_type)

    # Make directory
    directory = os.path.dirname(feature_importance_fig)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the feature importances
    importances = clf.feature_importances_

    # Get the number of conditions
    nums = [len(feature) if isinstance(feature, list) is True else 1 for feature in names.features_I]

    # Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
    f_importances = pd.Series(importances, nums)

    # Sort the array in descending order of the importances
    f_importances.sort_values(ascending=False, inplace=True)

    # Get the top k feature-importance pairs
    f_importances_top_k = f_importances[:min(len(f_importances), 10)]

    # Get the colors
    colors = ['b' if num == 1 else 'r' for num, importance in f_importances_top_k.iteritems()]

    # Make the bar plot from f_importances_top_k
    f_importances_top_k.plot(kind='bar', figsize=(20,10), rot=0, fontsize=30, color=colors)

    plt.xlabel('Number of conditions', fontsize=30)
    plt.ylabel('Importance', fontsize=30)
    plt.tight_layout()
    plt.savefig(feature_importance_fig)


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)

