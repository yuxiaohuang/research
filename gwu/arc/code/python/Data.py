# Please cite the following paper when using the code

import copy

class Data:
    """The Data class"""

    def __init__(self, X, y, X_trains, X_tests, y_trains, y_tests):

        # The feature vector
        self.X = copy.deepcopy(X)

        # The target vector
        self.y = copy.deepcopy(y)

        # The dictionary of training feature vectors
        self.X_trains = copy.deepcopy(X_trains)

        # The dictionary of testing feature vectors
        self.X_tests = copy.deepcopy(X_tests)

        # The dictionary of training target vector
        self.y_trains = copy.deepcopy(y_trains)

        # The dictionary of testing target vector
        self.y_tests = copy.deepcopy(y_tests)
