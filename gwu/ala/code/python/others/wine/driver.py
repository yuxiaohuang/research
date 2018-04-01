# Please cite the following paper when using the code


import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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

# The features that should be excluded
exclude_features = []

# The target, the last column of the data frame by default
target = 0

# Flag variable, indicating whether the target is categorical, False by default
categorical_target = True

# The categorical features, empty by default
categorical_features = []

# The percentage of the testing set, 0.3 by default
test_size = 0.3


# Data preprocessing
def data_preprocessing():
    # Load data
    df = pd.read_csv(data_file, header=header)

    # Replace ? with NaN
    df = df.replace('?', np.NaN)

    # Remove rows that contain missing values
    df = df.dropna(axis=0)

    # Remove columns that should be excluded
    df = df.drop(df.columns[exclude_features], axis=1)

    # Get the features
    if target == -1:
        X = df.iloc[:, :target]
    else:
        X = np.hstack((df.iloc[:, :target], df.iloc[:, target + 1:]))

    # One-hot encoding on categorical features
    # ohe = OneHotEncoder(categorical_features=categorical_features, sparse=False)
    # X = ohe.fit_transform(X)
    if len(categorical_features) > 0:
        X = pd.get_dummies(X)

    # Get the target
    # If the target is the last column
    if target == -1:
        y = df.iloc[:, target:]
    else:
        y = df.iloc[:, target:target + 1]

    # If the target is categorical
    if categorical_target is True:
        # Label encoding the target
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)

    # Randomly choose test_size% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)

    # Standardization on the features
    std_sc = StandardScaler()
    X_train_std = std_sc.fit_transform(X_train)
    X_test_std = std_sc.transform(X_test)

    return [X_train_std, X_test_std, y_train, y_test]


# Get the statistics file
def get_statistics_file():
    # Get the precision, recall, fscore, and support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_hat, average='micro')

    with open(statistics_file, 'w') as f:
        # Write header
        f.write("precision, recall, fscore" + '\n')

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n')


# Main function
if __name__ == "__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    data_file = sys.argv[1]
    statistics_file = sys.argv[2]

    # Data preprocessing
    X_train_std, X_test_std, y_train, y_test = data_preprocessing()

    # The classifier
    # model = RandomForestClassifier()
    model = LogisticRegression(max_iter=100, multi_class='multinomial', solver='newton-cg')

    # The fit function
    model.fit(X_train_std, y_train)

    # The predict function
    y_hat = model.predict(X_test_std)

    print(y_hat)

    print(y_test)

    # Get the statistics file
    get_statistics_file()