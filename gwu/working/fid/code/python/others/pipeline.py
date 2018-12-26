# Please cite the following paper when using the code

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Setting

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


def get_result_from_data(data_dir, result_dir, dp_dir):
    """
    Get result from data
    :param data_dir: the pathname of the data directory
    :param result_dir: the pathname of the result directory
    :param dp_dir: the pathname of the DataPreprocessing module directory
    :return:
    """

    # Add code_dir folder
    sys.path.append(dp_dir)

    # Import the DataPreprocessing module
    import DataPreprocessing
    # Get the DataPreprocessing object
    dp = DataPreprocessing.DataPreprocessing(data_dir)

    # Match data file withm names file
    data_names = dp.match_data_names()

    # The parallel pipelines for data preprocessing, train, test, and evaluate the ALA classifier
    # n_jobs = -1 indicates (all CPUs are used)
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=-1)(delayed(pipeline)(dp, data_file, names_file, result_dir)
                        for data_file, names_file in data_names)


def pipeline(dp, data_files, names_file, result_dir):
    """
    The pipeline for data preprocessing, train, test, and evaluate the ALA classifier
    :param dp: the DataPreprocessing module
    :param data_files: the pathname of the data files
    :param names_file: the pathname of the names file
    :param result_dir: the pathname of the result directory
    :return:
    """

    # Data preprocessing: get the Setting, Names, and Data object
    setting, names, data = dp.get_setting_names_data(data_files, names_file, result_dir, Setting)

    # Train, test, and evaluate the classifier
    # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
    Parallel(n_jobs=setting.n_jobs)(delayed(train_test_eval)(setting, names, data, clf_name)
                                    for clf_name in setting.classifiers.keys())


def train_test_eval(setting, names, data, clf_name):
    """
    Train, test, and evaluate the classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param clf_name: the name of the classifier
    :return:
    """

    classifier = setting.classifiers[clf_name]

    if clf_name == 'RandomForestClassifier':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'AdaBoostClassifier':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state))])
    elif clf_name == 'MLPClassifier':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state))])
    elif clf_name == 'KNeighborsClassifier':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(n_jobs=setting.n_jobs))])
    elif clf_name == 'GaussianNB':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier())])
    elif clf_name == 'DecisionTreeClassifier':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state))])
    elif clf_name == 'LogisticRegression_ovr':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'LogisticRegression_multinomial_lbfgs':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(multi_class='multinomial', solver='lbfgs', random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'LogisticRegression_multinomial_sag':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(multi_class='multinomial', solver='sag', random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'LogisticRegression_multinomial_newton-cg':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(multi_class='multinomial', solver='newton-cg', random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'GaussianProcessClassifier':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state, n_jobs=setting.n_jobs))])
    elif clf_name == 'SVC':
        pipe_clf = Pipeline([('scaler', setting.scaler),
                             ('clf',  classifier(random_state=setting.random_state))])

    # Get the cross validation scores
    scores = cross_val_score(estimator=pipe_clf,
                             X=data.X,
                             y=data.y,
                             cv=StratifiedKFold(n_splits=setting.n_splits, random_state=setting.random_state),
                             n_jobs=setting.n_jobs)

    # Refit clf on the whole data
    pipe_clf.fit(data.X, data.y)

    # Evaluate clf
    eval(setting, names, data, pipe_clf.named_steps['clf'], scores, clf_name)


def eval(setting, names, data, clf, scores, clf_name):
    """
    Evaluate the classifier
    :param setting: the Setting object
    :param names: the Names object
    :param data: the Data object
    :param clf: the classifier
    :param scores: the cross validation scores
    :param clf_name: the name of the classifier
    :return:
    """

    setting.set_plt()

    if setting.score_file_dir is not None:
        # Write the score file
        write_score_file(setting, scores, clf_name)

    if (setting.prob_dist_fig_dir is not None
        and (isinstance(clf, setting.classifiers['LogisticRegression_ovr']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_lbfgs']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_sag']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_newton-cg']) is True)):
        # Plot the probability distribution figures
        plot_prob_dist_fig(setting, names, data.X, clf, clf_name)

    if (setting.prob_dist_file_dir is not None
        and (isinstance(clf, setting.classifiers['LogisticRegression_ovr']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_lbfgs']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_sag']) is True
             or isinstance(clf, setting.classifiers['LogisticRegression_multinomial_newton-cg']) is True)):
        # Write the probability distribution file
        write_prob_dist_file(setting, names, data.X, clf, clf_name)


def write_score_file(setting, scores, clf_name):
    """
    Write the score file
    :param setting: the Setting object
    :param scores: the cross validation scores
    :param clf_name: the name of the classifier
    :return:
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
        # Write the mean of the cross validation scores
        f.write("The mean of the cross validation scores: " + str(round(np.mean(scores), 2)) + '\n')

        # Write the std of the cross validation scores
        f.write("The std of the cross validation scores: " + str(round(np.std(scores), 2)) + '\n')

        # Write the min of the cross validation scores
        f.write("The min of the cross validation scores: " + str(round(min(scores), 2)) + '\n')

        # Write the max of the cross validation scores
        f.write("The max of the cross validation scores: " + str(round(max(scores), 2)) + '\n')


def plot_prob_dist_fig(setting, names, X, clf, clf_name):
    """
    Plot the probability distribution figures.
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature vector
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the probability distribution figure
    prob_dist_fig_dir = setting.prob_dist_fig_dir + clf_name + '/'

    # Make directory
    directory = os.path.dirname(prob_dist_fig_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    prob_dist_dict = get_prob_dist_dict(setting.scaler.transform(X), clf)

    # For each unique value of the target
    for yu in sorted(prob_dist_dict.keys()):
        # Get the original value of yu
        yu_orig = str(setting.encoder.inverse_transform(yu))

        # For each xj
        for j in sorted(prob_dist_dict[yu].keys()):
            xijs = sorted(prob_dist_dict[yu][j].keys())
            pijs = [round(prob_dist_dict[yu][j][xij], 2) for xij in xijs]
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

            if len(xijs_orig) > 50:
                plt.tick_params(labelbottom='off')

            plt.tight_layout()
            # Get the pathname of the probability distribution figure
            prob_dist_fig = (prob_dist_fig_dir + setting.prob_dist_fig_name + '_' + yu_orig + '_' + xj
                             + setting.prob_dist_fig_type)
            plt.savefig(prob_dist_fig)


def get_prob_dist_dict(X, clf):
    """
    Get the dictionary of probability distribution
    :param X: the feature vector
    :param clf: the classifier
    :return: the dictionary of probability distribution
    """

    # Initialization
    prob_dist_dict = {}

    # For each xj
    for j in range(X.shape[1] + 1):
        # Initialize X_sparse
        X_sparse = np.zeros((X.shape[0], X.shape[1]))

        if j > 0:
            # Update xj in X_sparse
            X_sparse[:, j - 1] = X[:, j - 1]

        # Get the unique value and the corresponding index in xj
        if j == 0:
            xus, idxs = np.unique([1], return_index=True)
        else:
            xus, idxs = np.unique(X_sparse[:, j - 1], return_index=True)

        # For each unique index
        for i in idxs:
            # Get xij
            xij = 1 if j == 0 else X_sparse[i, j - 1]
            # Get the probability of each label
            probs = clf.predict_proba(X_sparse[i, :].reshape(1, -1)).ravel()

            # For each unique value of the target
            for yu in range(len(probs)):
                # Get the probability
                prob = probs[yu]

                # Initialization
                if yu not in prob_dist_dict:
                    prob_dist_dict[yu] = {}
                if j not in prob_dist_dict[yu]:
                    prob_dist_dict[yu][j] = {}

                # Update prob_dist_dict
                prob_dist_dict[yu][j][xij] = prob

    return prob_dist_dict


def write_prob_dist_file(setting, names, X, clf, clf_name):
    """
    Write the probability distribution file
    :param setting: the Setting object
    :param names: the Names object
    :param X: the feature vector
    :param clf: the classifier
    :param clf_name: the name of the classifier
    :return:
    """

    # Get the directory of the probability distribution file
    prob_dist_file_dir = setting.prob_dist_file_dir + clf_name + '/'
    # Get the pathname of the probability distribution file
    prob_dist_file = prob_dist_file_dir + setting.prob_dist_file_name + setting.prob_dist_file_type

    # Make directory
    directory = os.path.dirname(prob_dist_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the dictionary of probability distribution
    prob_dist_dict = get_prob_dist_dict(setting.scaler.transform(X), clf)

    with open(prob_dist_file, 'w') as f:
        # Write header
        f.write("yu, xj, xij, pij" + '\n')

        # For each unique value of the target
        for yu in sorted(prob_dist_dict.keys()):
            # Get the original value of yu
            yu_orig = str(setting.encoder.inverse_transform(yu))

            # For each xj
            for j in sorted(prob_dist_dict[yu].keys()):
                xj = 'x0' if j == 0 else names.features[j - 1]
                xijs = sorted(prob_dist_dict[yu][j].keys())
                pijs = [prob_dist_dict[yu][j][xij] for xij in xijs]
                xijs_orig = [1] if j == 0 else np.unique(sorted(X.iloc[:, j - 1]))

                for idx in range(len(pijs)):
                    pij = pijs[idx]
                    xij_orig = xijs_orig[idx]
                    f.write(yu_orig + ', ' + xj + ', ' + str(xij_orig) + ', ' + str(pij) + '\n')


if __name__ == "__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[2]

    # Get the pathname of the DataPreprocessing module directory
    dp_dir = sys.argv[3]

    # Get result from data
    get_result_from_data(data_dir, result_dir, dp_dir)