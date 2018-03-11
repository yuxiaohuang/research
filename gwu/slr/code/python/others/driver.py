# Please cite the following paper when using the code


import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression

# Global variables
# Flag variable, indicating whether there is a header
header = None

# The representation (other than NaN) for missing values
missing_representation = '?'

# The features that should be excluded
exclude_features = []

# The target, the last column of the data frame by default
target = -1

# The categorical features, empty by default
categorical_features = []

# The features
features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

# The percentage of the testing set, 0.3 by default
test_size = 0.3

# The scaler, StandardScaler by default
scaler = StandardScaler()

# The label encoder for the target
target_le = LabelEncoder()

# The random state
random_state = 0


# Data preprocessing
def data_preprocessing():
    # Load data
    df = pd.read_csv(data_file, header=header)

    # Replace missing_representation with NaN
    df = df.replace(missing_representation, np.NaN)

    # Remove rows that contain missing values
    df = df.dropna(axis=0)

    # Remove columns that should be excluded
    df = df.drop(df.columns[exclude_features], axis=1)

    # Get the features
    if target == -1:
        X = df.iloc[:, :target].values
    else:
        X = np.hstack((df.iloc[:, :target], df.iloc[:, target + 1:]))

    # One-hot encoding on categorical features
    if len(categorical_features) > 0:
        X = pd.get_dummies(X, columns=features[categorical_features]).values

    # Get the target
    # If the target is the last column
    if target == -1:
        y = np.ravel(df.iloc[:, target:].values)
    else:
        y = np.ravel(df.iloc[:, target:target + 1].values)

    y = target_le.fit_transform(y)

    # Randomly choose test_size% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Standardization on the features
    X_train = scaler.fit_transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))

    return [X, y, X_train, X_test, y_train, y_test]


# Train, test, and evaluate classifers
def train_test_evaluate(classifier):
    # If logistic regression
    if classifier is LogisticRegression:
        clf = classifier(max_iter=100, multi_class='multinomial', solver='newton-cg', random_state=random_state)

    # The fit function
    clf.fit(X_train, y_train)

    # The predict function
    y_pred = clf.predict(X_test)

    # Get the statistics
    get_statistics(y_pred)

    # If logistic regression
    if classifier is LogisticRegression:
        # Get the probabilities
        get_probabilities(clf)

        
# Get the statistics
def get_statistics(y_pred):
    with open(statistics_file, 'w') as f:
        # Get the precision, recall, fscore, and support
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')

        # Write header
        f.write("precision, recall, fscore" + '\n')
        f.flush()

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n')
        f.flush()


# Get the probabilities
def get_probabilities(clf):
    # If the scaler is not None
    if scaler is not None:
        # Scale the data
        X_scaled = scaler.fit_transform(X.astype(float))
    else:
        X_scaled = X

    # Initialize pijs
    pijs = {}

    with open(probabilities_file, 'w') as f:
        # Write header
        f.write("yu, xj, xij, pij" + '\n')
        f.flush()

        # For each xj
        for j in range(X.shape[1]):
            # Initialize X_sparse
            X_sparse = np.zeros((X.shape[0], X.shape[1]))

            # Update xj in X_sparse
            X_sparse[:, j] = X_scaled[:, j]

            # Get the unique value and the corresponding index in xj
            xus, idxus = np.unique(X[:, j], return_index=True)

            # For each unique index
            for idxu in idxus:
                # Get the probability of each label
                probs = clf.predict_proba(X_sparse[idxu, :].reshape(1, -1)).ravel()

                # For each class label
                for idx in range(len(probs)):
                    # Get the probability
                    prob = probs[idx]

                    # Transform labels back to original encoding
                    yu = target_le.inverse_transform(idx)

                    # Initialization
                    if yu not in pijs:
                        pijs[yu] = {}
                    if j not in pijs[yu]:
                        pijs[yu][j] = []

                    # Update pijs
                    pijs[yu][j].append(prob)

            # For each unique value of the target
            for yu in pijs:
                for idx in range(len(pijs[yu][j])):
                    prob = pijs[yu][j][idx]
                    xu = xus[idx]

                    f.write(yu + ', ' + features[j] + ', ' + str(xu) + ', ' + str(prob) + '\n')
                    f.flush()

            # For each unique value of the target
            for yu in pijs:
                # Get the pandas series
                df = pd.DataFrame(pijs[yu][j])

                # Plot the histogram of the series
                df.plot(kind='bar', figsize=(16, 9), fontsize=30, legend=False)
                plt.xticks(range(len(xus)), xus)
                plt.xlabel('Feature value', fontsize=30)
                plt.ylabel('Probability', fontsize=30)
                plt.title('P(' + yu + ' | ' + features[j] + ')', fontsize=30, loc='center')

                probabilities_fig = probabilities_file.replace('.txt', '_' + str(yu) + '_' + str(features[j]) + '.pdf')
                plt.tight_layout()
                plt.savefig(probabilities_fig)


# Main function
if __name__ == "__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    data_file = sys.argv[1]
    statistics_file = sys.argv[2]
    probabilities_file = sys.argv[3]

    # Data preprocessing
    X, y, X_train, X_test, y_train, y_test = data_preprocessing()

    # The classifiers
    classifiers = [LogisticRegression]

    for classifier in classifiers:
        train_test_evaluate(classifier)