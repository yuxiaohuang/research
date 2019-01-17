# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


def pipeline_all_datasets():
    """
    The pipeline for all data sets
    :return:
    """

    # Add code_dir folder
    sys.path.append(dp_dir)

    # Import DataPreprocessing module
    import DataPreprocessing
    dp = DataPreprocessing.DataPreprocessing(data_dir)

    # Match data files with names file
    data_names = dp.match_data_names()

    # The pipeline for each data set (in parallel)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=-1)(delayed(pipeline_one_dataset)(dp, data_files, names_file)
                        for data_files, names_file in data_names)


def pipeline_one_dataset(dp, data_files, names_file):
    """
    The pipeline for one data set
    :param dp: the DataPreprocessing module
    :param data_files: the pathname of the data files
    :param names_file: the pathname of the names file
    :return:
    """

    # Data preprocessing: get the Setting, Names, and Data object
    setting, names, data = dp.get_setting_names_data(data_files, names_file, result_dir, Setting)

    # The pipeline for each data set and each classifier (in parallel)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(pipeline_one_dataset_one_classifier)(setting, names, data, clf_name)
                                    for clf_name in setting.classifiers.keys())

def pipeline_one_dataset_one_classifier(setting, names, data, clf_name):
    """
    The pipeline for one data set and one classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the sklearn pipeline
    pipe_clf = Pipeline([('scaler', setting.scaler),
                         (clf_name, setting.classifiers[clf_name])])

    # Hyperparameter tuning using GridSearchCV
    gs = GridSearchCV(estimator=pipe_clf,
                      param_grid=setting.param_grids[clf_name],
                      scoring=setting.scoring,
                      n_jobs=setting.n_jobs,
                      cv=StratifiedKFold(n_splits=setting.n_splits,
                                         random_state=setting.random_state))
    gs.fit(data.X, data.y)

    # Get the results
    get_results(setting, names, data, gs, clf_name)


def get_results(setting, names, data, gs, clf_name):
    """
    Get the results
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param gs: the GridSearchCV object
    :param clf_name: the name of the classifier
    :return:
    """

    setting.set_plt()

    if (setting.prob_dists_fig_dir is not None
        and clf_name == 'LogisticRegression'):
        # Plot the probability distribution figures
        plot_prob_dists_fig(setting, names, data.X, data.y, gs.best_estimator_, clf_name)

    if (setting.prob_dists_file_dir is not None
        and clf_name == 'LogisticRegression'):
        # Write the probability distribution file
        write_prob_dists_file(setting, names, data.X, data.y, gs.best_estimator_, clf_name)

    if (setting.feature_importances_fig_dir is not None
        and clf_name == 'RandomForestClassifier'):
        # Plot the feature importances figure
        plot_feature_importances_fig(setting, names, gs.best_estimator_.named_steps[clf_name], clf_name)

    if (setting.feature_importances_file_dir is not None
        and clf_name == 'RandomForestClassifier'):
        # Write the feature importances file
        write_feature_importances_file(setting, names, gs.best_estimator_.named_steps[clf_name], clf_name)

    if setting.cv_results_file_dir is not None:
        # Write the cv results file
        write_cv_results_file(setting, gs.cv_results_, clf_name)

    if setting.best_params_file_dir is not None:
        # Write the best hyperparameters file
        write_best_params_file(setting, gs.best_params_, clf_name)


def plot_prob_dists_fig(setting, names, X, y, clf, clf_name):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param y: the target vector
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the probability distribution figure
    prob_dists_fig_dir = setting.prob_dists_fig_dir + clf_name + '/'

    # Make directory
    directory = os.path.dirname(prob_dists_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    prob_dists = get_prob_dists(X, clf, clf_name)

    for class_ in range(len(np.unique(y))):
        # Get the original value of class_ before the encoding
        class_ori = str(setting.encoder.inverse_transform(class_))

        for j in sorted(prob_dists[class_].keys()):
            # Get the name of the jth feature
            xj_name = names.features[j]

            # Get the original value of the jth feature before the scaling
            xijs_ori = [round(xij, 2) for xij in sorted(prob_dists[class_][j].keys())]

            # Get the probabilities
            pijs = [round(prob_dists[class_][j][xij], 5) for xij in
                    sorted(prob_dists[class_][j].keys())]

            # Get the dataframe
            df = pd.DataFrame(list(zip(xijs_ori, pijs)), columns=[xj_name, 'Probability'])

            # Plot the histogram
            df.plot(x=xj_name,
                    y='Probability',
                    kind='bar',
                    yticks=[0, 0.25, 0.5, 0.75, 1],
                    ylim=(0, 1),
                    figsize=(20, 10),
                    title=class_ori,
                    legend=False,
                    color='b')

            # Set the x-axis label
            plt.xlabel(xj_name)
            # Set the y-axis label
            plt.ylabel('Probability')

            if len(xijs_ori) > 50:
                plt.tick_params(labelbottom='off')

            plt.tight_layout()
            prob_dists_fig = (prob_dists_fig_dir + setting.prob_dists_fig_name + '_' + class_ori + '_' + xj_name
                             + setting.prob_dists_fig_type)
            plt.savefig(prob_dists_fig)


def get_prob_dists(X, clf, clf_name):
    """
    Get the dictionary of probability distribution
    :param X: the feature matrix
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return: the dictionary of probability distribution
    """

    prob_dists = {}

    for j in range(X.shape[1]):
        # Get the jth feature
        Xj = np.zeros(X.shape)
        # Standardize the data
        Xj[:, j] = clf.named_steps['scaler'].fit_transform(X)[:, j]

        # Get the unique value and the corresponding index of the jth feature
        xijs, idxs = np.unique(X[:, j], return_index=True)

        for i in idxs:
            # Get the probability of each class
            probs = clf.named_steps[clf_name].predict_proba(Xj[i, :].reshape(1, -1)).ravel()

            for class_ in range(len(probs)):
                # Get the probability
                prob = probs[class_]

                if class_ not in prob_dists:
                    prob_dists[class_] = {}
                if j not in prob_dists[class_]:
                    prob_dists[class_][j] = {}

                # Update prob_dists
                prob_dists[class_][j][X[i, j]] = prob

    return prob_dists


def write_prob_dists_file(setting, names, X, y, clf, clf_name):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature matrix
    :param y: the target vector
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the probability distribution file
    prob_dists_file_dir = setting.prob_dists_file_dir + clf_name + '/'
    # Get the pathname of the probability distribution file
    prob_dists_file = prob_dists_file_dir + setting.prob_dists_file_name + setting.prob_dists_file_type

    # Make directory
    directory = os.path.dirname(prob_dists_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    prob_dists = get_prob_dists(X, clf, clf_name)

    with open(prob_dists_file, 'w') as f:
        # Write header
        f.write("Class, Feature, Value, Probability" + '\n')

        for class_ in range(len(np.unique(y))):
            # Get the original value of class_ before the encoding
            class_ori = str(setting.encoder.inverse_transform(class_))

            for j in sorted(prob_dists[class_].keys()):
                # Get the name of the jth feature
                xj_name = names.features[j]

                # Get the original value of the jth feature before the scaling
                xijs_ori = [round(xij_ori, 2) for xij_ori in np.unique(sorted(X[:, j]))]

                # Get the probabilities
                pijs = [round(prob_dists[class_][j][xij], 5) for xij in
                        np.unique(sorted(prob_dists[class_][j].keys()))]

                for idx in range(len(pijs)):
                    pij = pijs[idx]
                    xij_ori = xijs_ori[idx]
                    f.write(class_ori + ', ' + xj_name + ', ' + str(xij_ori) + ', ' + str(pij) + '\n')


def plot_feature_importances_fig(setting, names, clf, clf_name):
    """
    Plot the feature importances figure
    :param setting: the Setting object
    :param names: the Names object
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the feature importances figure
    feature_importances_fig_dir = setting.feature_importances_fig_dir + clf_name + '/'
    # Get the pathname of the feature importances file
    feature_importances_fig = feature_importances_fig_dir + setting.feature_importances_fig_name + setting.feature_importances_fig_type

    # Make directory
    directory = os.path.dirname(feature_importances_fig)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the feature importances dataframe
    df = pd.DataFrame(list(zip(names.features, clf.feature_importances_)), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

    # Plot the histogram
    df.plot(x='Feature',
            y='Importance',
            kind='bar',
            yticks=[0, 0.25, 0.5, 0.75, 1],
            ylim=(0, 1),
            figsize=(20, 10),
            legend=False,
            color='b')

    # Set the x-axis label
    plt.xlabel('Feature')
    # Set the y-axis label
    plt.ylabel('Importance')

    # Hide X label
    plt.axes().xaxis.label.set_visible(False)

    if len(names.features) > 50:
        plt.tick_params(labelbottom='off')

    plt.tight_layout()
    plt.savefig(feature_importances_fig)


def write_feature_importances_file(setting, names, clf, clf_name):
    """
    Write the feature importances file
    :param setting: the Setting object
    :param names: the Names object
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the feature importances file
    feature_importances_file_dir = setting.feature_importances_file_dir + clf_name + '/'
    # Get the pathname of the feature importances file
    feature_importances_file = feature_importances_file_dir + setting.feature_importances_file_name + setting.feature_importances_file_type

    # Make directory
    directory = os.path.dirname(feature_importances_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the feature importances dataframe
    df = pd.DataFrame(list(zip(names.features, clf.feature_importances_)), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

    df.to_csv(path_or_buf=feature_importances_file)


def write_cv_results_file(setting, cv_results, clf_name):
    """
    Write the cv results file
    :param setting: the Setting object
    :param cv_results: the cv results
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the cv results file
    cv_results_file_dir = setting.cv_results_file_dir + clf_name + '/'
    # Get the pathname of the cv results
    cv_results_file = cv_results_file_dir + setting.cv_results_file_name + setting.cv_results_file_type

    # Make directory
    directory = os.path.dirname(cv_results_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(cv_results).sort_values(by=['rank_test_score', 'std_test_score'])

    cv_results.to_csv(path_or_buf=cv_results_file)


def write_best_params_file(setting, best_params, clf_name):
    """
    Write the best hyperparameters file
    :param setting: the Setting object
    :param best_params: the best hyperparameters
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the best hyperparameters file
    best_params_file_dir = setting.best_params_file_dir + clf_name + '/'
    # Get the pathname of the best hyperparameters file
    best_params_file = best_params_file_dir + setting.best_params_file_name + setting.best_params_file_type

    # Make directory
    directory = os.path.dirname(best_params_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    pd.DataFrame.from_dict(best_params).to_csv(path_or_buf=best_params_file)


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # The pipeline for all data sets
    pipeline_all_datasets()