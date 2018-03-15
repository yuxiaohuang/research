# Please cite the following paper when using the code


import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import SimpleLogisticRegression


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

# The features for Iris
features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

# The features for breast-cancer-wisconsin
# features = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class']

# The features for Wine
# features = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280_OD315 of diluted wines', 'Proline']

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


# Get the SimpleLogisticRegression classifier
def get_slg(max_iter, C, min_bin_size):
    # Use default value
    slg = SimpleLogisticRegression.SimpleLogisticRegression()

    return slg


# Get the weights
def get_weights():
    with open(weight_file, 'w') as f:
        # Write header
        f.write("yu, xj, wj0, wj1" + '\n')
        f.flush()

        # For each unique value of the target
        for yu in slg.ws_:
            # Transform labels back to original encoding
            yu_str = str(target_le.inverse_transform(yu))

            # For each xj
            for j in slg.ws_[yu]:
                # For each bin
                for bin in slg.ws_[yu][j]:
                    wj0 = slg.ws_[yu][j][bin][0]
                    wj1 = slg.ws_[yu][j][bin][1]

                    if j == 0:
                        f.write(str(yu_str) + ', latent, ' + str(slg.bins_[j]) + ', ' + str(wj0) + ', ' + str(wj1) + '\n')
                    else:
                        f.write(str(yu_str) + ', ' + features[j - 1] + ', ' + str([slg.bins_[j][bin], slg.bins_[j][bin + 1]]) + ', ' + str(wj0) + ', ' + str(wj1) + '\n')
                    f.flush()


# Get the mse figure
def get_mse():
    plt.plot(range(1, len(slg.mses_) + 1), slg.mses_)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(mse_fig, dpi=300)


# Get the prediction file
def get_prediction():
    with open(prediction_file, 'w') as f:
        # Write header
        f.write("y_pred" + '\n')
        f.flush()

        # For each predicted value
        for val in y_pred:
            # Write the value
            f.write(str(val) + '\n')
            f.flush()


# Get the statistics file
def get_statistics():
    with open(statistics_file, 'w') as f:
        # Get the precision, recall, fscore, and support
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')

        # Write header
        f.write("precision, recall, fscore" + '\n')
        f.flush()

        # Write the precision, recall, and fscore
        f.write(str(precision) + ', ' + str(recall) + ', ' + str(fscore) + '\n')
        f.flush()

        # Write empty lines
        f.write('\n\n')
        f.flush()

        # Write header
        f.write("i, y_testi, prob_y_testi, y_predi, prob_y_predi" + '\n')
        f.flush()

        # For each row
        for i in range(len(y_pred)):
            if y_pred[i] != y_test[i]:
                # Get p(y_test[i]) and p(y_pred[i])
                for yu, prob in yu_probs_log[i]:
                    if yu == y_test[i]:
                        prob_y_testi = prob
                    elif yu == y_pred[i]:
                        prob_y_predi = prob

                # Write the wrong(yu, prob) pairs
                f.write(str(i) + ': [' + str(y_test[i]) + ', ' + str(prob_y_testi) + ']' + ', ' + '[' + str(y_pred[i]) + ', ' + str(prob_y_predi) + ']' + '\n')
                f.flush()


# Get the probabilities
def get_probabilities():
    with open(probabilities_file, 'w') as f:
        # Write header
        f.write("yu, xj, xij, pij" + '\n')
        f.flush()

        # If the scaler is not None
        if scaler is not None:
            # Scale the data
            X_scaled = scaler.fit_transform(X.astype(float))
        else:
            X_scaled = X

        # For each unique value of the target
        for yu in slg.ws_:
            # Transform labels back to original encoding
            yu_str = str(target_le.inverse_transform(yu))

            # For each xj
            for j in range(X.shape[1] + 1):
                if j == 0:
                    # Get the unique value and the corresponding index in xj
                    xus, idxus = np.unique([1], return_index=True)
                else:
                    # Get the unique value and the corresponding index in xj
                    xus, idxus = np.unique(X[:, j - 1], return_index=True)

                # Initialize xijs and pijs
                xijs, pijs = [], []

                # For each unique index
                for idxu in idxus:
                    # Get pij
                    pij = slg.get_pij(X_scaled, yu, idxu, j)

                    # Update pijs
                    pijs.append(pij)

                    # Get xij
                    if j == 0:
                        xij = 1
                    else:
                        xij = X[idxu][j - 1]

                    # Update xijs
                    xijs.append(xij)

                    if j == 0:
                        f.write(yu_str + ', latent, ' + str(xij) + ', ' + str(pij) + '\n')
                    else:
                        f.write(yu_str + ', ' + features[j - 1] + ', ' + str(xij) + ', ' + str(pij) + '\n')
                    f.flush()

                # Get the pandas series
                df = pd.DataFrame(pijs)

                # Plot the histogram of the series
                df.plot(kind='bar', figsize=(16, 9), fontsize=30, legend=False)
                plt.xticks(range(len(xijs)), xijs)
                plt.xlabel('Feature value', fontsize=30)
                plt.ylabel('Probability', fontsize=30)

                if j == 0:
                    plt.title('P(' + yu_str + ' | latent)', fontsize=30, loc='center')
                else:
                    plt.title('P(' + yu_str + ' | ' + features[j - 1] + ')', fontsize=30, loc='center')

                if j == 0:
                    prob_fig = probabilities_file.replace('.txt', '_' + yu_str + '_latent.pdf')
                else:
                    prob_fig = probabilities_file.replace('.txt', '_' + yu_str + '_' + features[j - 1] + '.pdf')

                plt.tight_layout()
                plt.savefig(prob_fig)


# Main function
if __name__ == "__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    data_file = sys.argv[1]
    weight_file = sys.argv[2]
    mse_fig = sys.argv[3]
    prediction_file = sys.argv[4]
    statistics_file = sys.argv[5]
    probabilities_file = sys.argv[6]
    max_iter = sys.argv[7]
    C = sys.argv[8]
    min_bin_size = sys.argv[9]

    # Data preprocessing
    X, y, X_train, X_test, y_train, y_test = data_preprocessing()

    # Get the SimpleLogisticRegression classifier
    slg = get_slg(max_iter, C, min_bin_size)

    # The fit function
    slg.fit(X_train, y_train)

    # Get the weights
    get_weights()

    # Get the mse
    get_mse()

    # The predict function
    y_pred, yu_probs_log = slg.predict(X_test)

    # Get the prediction
    get_prediction()

    # Get the statistics
    get_statistics()

    # Get the probabilities
    get_probabilities()