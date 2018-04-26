# Please cite the following paper when using the code

class Data:
    """The Data class"""

    def __init__(self,
                 X,
                 X_train,
                 X_test,
                 y,
                 y_train,
                 y_test):

        # The feature data
        self.X = X

        # The feature data for training
        self.X_train = X_train

        # The feature data for testing
        self.X_test = X_test

        # The target data
        self.y = y

        # The target data for training
        self.y_train = y_train

        # The target data for testing
        self.y_test = y_test