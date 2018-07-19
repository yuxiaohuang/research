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

        # The feature vector
        self.X = X

        # The feature vector for training
        self.X_train = X_train

        # The feature vector for testing
        self.X_test = X_test

        # The target vector
        self.y = y

        # The target vector for training
        self.y_train = y_train

        # The target vector for testing
        self.y_test = y_test
        
        # The feature vector for training, with interaction
        self.X_train_I = None

        # The feature vector for testing, with interaction
        self.X_test_I = None

