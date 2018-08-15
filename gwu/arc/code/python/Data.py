# Please cite the following paper when using the code

import copy

class Data:
    """The Data class"""

    def __init__(self, X, y, train_test_indices):

        # The feature vector
        self.X = copy.deepcopy(X)

        # The target vector
        self.y = copy.deepcopy(y)

        # The train and test indices
        self.train_test_indices = copy.deepcopy(train_test_indices)

