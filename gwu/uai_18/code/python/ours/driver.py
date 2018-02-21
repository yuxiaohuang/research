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
import EnsembleLogisticRegression


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


# Get the weight file
def get_weight_file():
    with open(weight_file, 'w') as f:
        # Write header
        f.write("yu, xj, wj0, wj1" + '\n')

        # For each unique value of the target
        for yu in elg.ws_:
            # For each xj
            for j in elg.ws_[yu]:
                wj0 = elg.ws_[yu][j][0]
                wj1 = elg.ws_[yu][j][1]
                f.write(str(yu) + ', ' + str(j) + ', ' + str(wj0) + ', ' + str(wj1) + '\n')
                f.flush()


# Get the mse figure
def get_mse_fig():
    plt.plot(range(1, len(elg.mses_) + 1), elg.mses_)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(mse_fig, dpi=300)


# Get the prediction file
def get_prediction_file():
    with open(prediction_file, 'w') as f:
        # Write header
        f.write("y_hat" + '\n')

        # For each value
        for val in y_hat:
            # Write the value
            f.write(str(val) + '\n')


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
    weight_file = sys.argv[2]
    mse_fig = sys.argv[3]
    prediction_file = sys.argv[4]
    statistics_file = sys.argv[5]
    max_iter = sys.argv[6]
    eta = sys.argv[7]

    # Data preprocessing
    X_train_std, X_test_std, y_train, y_test = data_preprocessing()

    # The EnsembleLogisticRegression classifier
    if max_iter.isdigit() is False and eta.isdigit() is False:
        elg = EnsembleLogisticRegression.EnsembleLogisticRegression()
    elif max_iter.isdigit() is True and eta.isdigit() is False:
        elg = EnsembleLogisticRegression.EnsembleLogisticRegression(max_iter=int(max_iter))
    elif max_iter.isdigit() is False and eta.isdigit() is True:
        elg = EnsembleLogisticRegression.EnsembleLogisticRegression(eta=float(eta))
    else:
        elg = EnsembleLogisticRegression.EnsembleLogisticRegression(max_iter=int(max_iter), eta=float(eta))

    # The fit function
    elg.fit(X_train_std, y_train)

    # Get the weight file
    get_weight_file()

    # Get the mse figure
    get_mse_fig()

    # The predict function
    y_hat = elg.predict(X_test_std)

    print(y_test)

    # Get the prediction file
    get_prediction_file()

    # Get the statistics file
    get_statistics_file()