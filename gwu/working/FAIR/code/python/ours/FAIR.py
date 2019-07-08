# Please cite the following paper when using the code

import numpy as np

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class FAIR(BaseEstimator, ClassifierMixin):
    """
    Fast and Accurate Interaction Recognizer
    """
    
    def __init__(self,
                 n_iter=10,
                 min_support=0.1,
                 min_confidence=1,
                 max_conds=10,
                 random_state=0,
                 n_jobs=10):
        # The number of iterations for searching for the rules
        self.n_iter = n_iter

        # The maximum number of conditions to consider when searching for the rules
        self.max_conds = max_conds

        # The minimum support required by the rules
        self.min_support = min_support

        # The minimum confidence required by the rules
        self.min_confidence = min_confidence
        
        # The random state
        self.random_state = random_state

        # The number of jobs to run in parallel, -1 by default (all CPUs are used)
        self.n_jobs = n_jobs

        # The number of samples
        self.m = None

        # The number of conditions
        self.n = None

        # The class labels
        self.classes = None

        # The rule
        self.rule = None

        # The detected significant rules
        self.sig_rules = None

        # The random number generator
        self.rng = None

        # The available samples
        self.avail_samples = None

        # The available conditions
        self.avail_conds = None

        # The removed conditions
        self.removed_conds = None
        
    def fit(self, X, y):
        """
        The fit function for all classes
        :param X: the condition matrix
        :param y: the target vector
        :return:
        """

        # Initialization for all classes
        self.init(X, y)

        # The fit function for one class
        # Set backend="threading" to share memory between parent and threads
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.fit_one_class)(X, y, class_)
                                                          for class_ in sorted(self.classes))

    def init(self, X, y):
        """
        Initialization for all classes
        :param X: the condition matrix
        :param y: the target vector
        :return:
        """

        # The maximum number of conditions to consider when searching for the rules
        self.max_conds = min(self.max_conds, X.shape[1])

        # The number of samples
        self.m = X.shape[0]

        # The number of conditions
        self.n = X.shape[1]

        # The class labels
        self.classes = np.unique(y)

        # The rule
        self.rule = {}

        # The detected significant rules
        self.sig_rules = {}

        # The random number generator
        self.rng = {}

        # The available samples
        self.avail_samples = {}

        # The available conditions
        self.avail_conds = {}

        # The removed conditions
        self.removed_conds = {}
              
    def fit_one_class(self, X, y, class_):
        """
        The fit function for one class
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :return:
        """

        # Initialization for one class
        self.init_one_class(class_)

        # Greedy search
        # Set backend="threading" to share memory between parent and threads
        Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.greedy_search)(X, y, class_, iter)
                                                          for iter in range(self.n_iter))

    def init_one_class(self, class_):
        """
        Initialization for one class
        :param class_: a class of the target
        :return:
        """

        # The rule
        self.rule[class_] = {}

        # The detected significant rules
        self.sig_rules[class_] = {}

        # The random number generator
        self.rng[class_] = {}

        # The available samples
        self.avail_samples[class_] = {}

        # The available conditions
        self.avail_conds[class_] = {}

        # The removed conditions
        self.removed_conds[class_] = {}

    def greedy_search(self, X, y, class_, iter):
        """
        Greedy search
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        """

        # Initialization for one iteration
        self.init_one_iter(y, class_, iter)

        # Repeat, until no significant rules (satisfying both min_support and min_confidence) can be found
        while True:
            # If the rule is significant
            if self.significant(class_, iter) is True:
                # Prune the rule by removing the unnecessary conditions
                self.prune(X, y, class_, iter)

                # Update for one iteration
                self.update_one_iter(y, class_, iter)
            else:
                # If no condition can be added
                if self.add(X, y, class_, iter) is False:
                    # If the rule is empty after removing a condition
                    if self.remove(X, y, class_, iter) is False:
                        break

    def init_one_iter(self, y, class_, iter):
        """
        Initialization for one iteration
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        """

        # The detected significant rules
        self.sig_rules[class_][iter] = []

        # The random number generator
        self.rng[class_][iter] = np.random.RandomState(seed=self.random_state + iter)

        # The available samples
        self.avail_samples[class_][iter] = np.array(range(self.m))

        # The available conditions
        self.avail_conds[class_][iter] = self.rng[class_][iter].choice(self.n, size=self.max_conds, replace=False).astype(int)

        # The removed conditions
        self.removed_conds[class_][iter] = {}

        # The rule
        self.init_rule(y, class_, iter)

    def init_rule(self, y, class_, iter):
        """
        Initialize a rule
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        """

        # Get the conjunction
        C = np.array([]).astype(int)

        # Get the samples where C is true
        C_samples = self.avail_samples[class_][iter]

        # Get the samples where both class_ and C are true
        class_C_samples = C_samples[np.where(y[C_samples] == class_)]

        # Get the rule
        self.rule[class_][iter] = np.array([C, C_samples, class_C_samples])
    
    def significant(self, class_, iter):
        """
        Check whether the rule is significant
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        True: if the rule is significant
        False: otherwise
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        # If C is empty
        if len(C) == 0:
            return False

        # If the rule meets min_support and min_confidence
        if (len(C_samples) > 0
            and float(len(C_samples)) / self.m >= self.min_support
            and float(len(class_and_C_samples)) / len(C_samples) >= self.min_confidence):
            return True

        return False

    def prune(self, X, y, class_, iter):
        """
        Prune the rule by removing the unnecessary conditions
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        """

        # Repeat, until all the conditions in the rule are necessary
        while True:
            # Unpack the rule
            C, C_samples, class_and_C_samples = self.rule[class_][iter]

            if len(C) <= 1:
                break

            # Flag variable, indicating whether a unnecessary condition exists, None by default
            unnecessary_c = None

            # Get the condition-importance pairs
            c_importance = self.get_c_importance(X, y, class_, iter, C)

            # Sort the condition-importance pairs in ascending order of importance
            c_importance_sorted = sorted(c_importance, key=lambda x: x[1], reverse=False)

            for c, importance in c_importance_sorted:
                # Remove c from c_importance_sorted
                C_setminus_c = np.delete(C, np.where(C == c)[0])

                # Get the available samples
                avail_samples = self.avail_samples[class_][iter]

                # Get the samples where C_setminus_c is true
                C_setminus_c_samples = avail_samples[np.where(np.prod(X[avail_samples, C_setminus_c]) == 1)]

                # Get the samples where both class_ and C_setminus_c are true
                class_C_setminus_c_samples = C_setminus_c_samples[np.where(y[C_setminus_c_samples] == class_)]

                # Update the rule
                self.rule[class_][iter] = [C_setminus_c, C_setminus_c_samples, class_C_setminus_c_samples]

                # If the rule is still significant (i.e., c is not necessary)
                if self.significant(class_, iter) is True:
                    unnecessary_c = c
                    break

            # If all the conditions are necessary
            if unnecessary_c is None:
                # Reverse the rule
                self.rule[class_][iter] = [C, C_samples, class_and_C_samples]

                break

    def get_c_importance(self, X, y, class_, iter, C):
        """
        Get the condition-importance pairs
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :param C: a conjunction of conditions
        :return: the condition-importance pairs
        """

        # Get the importance of each condition
        importances = [self.get_importance(X, y, class_, iter, c) for c in C]

        # Get the condition-importance pairs
        c_importance = [[C[i], importances[i]] for i in range(len(C)) if importances[i] is not None]

        return c_importance

    def get_importance(self, X, y, class_, iter, c):
        """
        Get the importance of condition c with respect to C
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :param c: a condition
        :return: the importance of condition c with respect to C
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        if c in C:
            # Get the samples where C is true
            C_and_c_samples = C_samples

            # Get the samples where both class_ and C are true
            class_C_and_c_samples = class_and_C_samples

            # Get C_setminus_c
            C_setminus_c = np.delete(C, np.where(C == c)[0])

            # Get the available samples
            avail_samples = self.avail_samples[class_][iter]

            # Get the samples where C_setminus_c is true and c is false
            if len(C_setminus_c) == 0:
                C_setminus_c_and_not_c_samples = avail_samples[np.where(X[avail_samples, c] == 0)]
            else:
                C_setminus_c_and_not_c_samples = avail_samples[np.where(np.prod(X[np.ix_(avail_samples, C_setminus_c)]) == 1
                                                                        and X[avail_samples, c] == 0)]

            # Get the samples where both class_ and C_setminus_c are true but c is false
            class_C_setminus_c_and_not_c_samples = C_setminus_c_and_not_c_samples[np.where(y[C_setminus_c_and_not_c_samples] == class_)]
        else:
            # Get the samples where both C and c are true
            C_and_c_samples = C_samples[np.where(X[C_samples, c] == 1)]

            # Get the samples where class_, C, and c are true
            class_C_and_c_samples = C_and_c_samples[np.where(y[C_and_c_samples] == class_)]

            # Get the samples where C is true and c is false
            C_setminus_c_and_not_c_samples = C_samples[np.where(X[C_samples, c] == 0)]

            # Get the samples where both class_ and C_setminus_c are true but c is false
            class_C_setminus_c_and_not_c_samples = C_setminus_c_and_not_c_samples[np.where(y[C_setminus_c_and_not_c_samples] == class_)]

        if (len(C_and_c_samples) > 0
            and len(C_setminus_c_and_not_c_samples) > 0
            and float(len(C_and_c_samples)) / self.m >= self.min_support
            and float(len(C_setminus_c_and_not_c_samples)) / self.m >= self.min_support):
            ratio_and_c = float(len(class_C_and_c_samples)) / len(C_and_c_samples)
            ratio_and_not_c = float(len(class_C_setminus_c_and_not_c_samples)) / len(C_setminus_c_and_not_c_samples)
            importance = ratio_and_c - ratio_and_not_c
        else:
            importance = None

        return importance

    def update_one_iter(self, y, class_, iter):
        """
        Update for one iteration
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        # Get the support
        support = float(len(C_samples)) / self.m

        # Get the confidence
        confidence = float(len(class_and_C_samples)) / len(C_samples)

        # Add the rule
        idx = 0
        while idx < len(self.sig_rules[class_][iter]):
            # If the rule has the same conditions with a detected rule
            if sorted(self.sig_rules[class_][iter][idx][0]) == sorted(C):
                # Update the supports and confidences
                self.sig_rules[class_][iter][idx][1] = np.append(self.sig_rules[class_][iter][idx][1], support)
                self.sig_rules[class_][iter][idx][2] = np.append(self.sig_rules[class_][iter][idx][2], confidence)
                break
            idx += 1
        # If the rule does not have the same conditions with a detected rule
        if idx == len(self.sig_rules[class_][iter]):
            self.sig_rules[class_][iter].append([C, np.array([support]), np.array([confidence])])

        # Update the available samples
        self.avail_samples[class_][iter] = np.setdiff1d(self.avail_samples[class_][iter], C_samples)

        # Update the available conditions
        self.avail_conds[class_][iter] = np.setdiff1d(self.avail_conds[class_][iter], C)

        # Clear the removed conditions
        self.removed_conds[class_][iter] = {}

        # Update the rule
        self.init_rule(y, class_, iter)
    
    def add(self, X, y, class_, iter):
        """
        Across the available conditions that meet all of the following three requirements
        Add the one, c, with the highest importance
        1. c is not in C
        2. c has not been removed
        3. the importance of c with respect to C is not None
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        True : if c exists
        False : otherwise
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]
        
        # The list of available conditions meeting requirements 1-3
        cs = np.array([]).astype(int)
        
        for c in self.avail_conds[class_][iter]:
            # If requirement 1 is not met
            if c in C:
                continue

            # If requirement 2 is not met
            if c in self.removed_conds[class_][iter].keys():
                continue

            # Add c to cs
            cs = np.append(cs, c)

        # Get the condition-importance pairs
        c_importance = self.get_c_importance(X, y, class_, iter, cs)

        # If requirement 3 is not met
        if len(c_importance) == 0:
            return False

        # Sort the condition-importance pairs in descending order of importance
        c_importance_sorted = sorted(c_importance, key=lambda x: x[1], reverse=True)

        # Get the best condition (the one with the highest importance)
        c = c_importance_sorted[0][0]

        # Update C by adding the best to C
        C = np.append(C, c)

        # Update C_samples
        C_samples = C_samples[np.where(X[C_samples, c] == 1)]

        # Update class_and_C_samples
        class_and_C_samples = C_samples[np.where(y[C_samples] == class_)]

        # Update the rule
        self.rule[class_][iter] = [C, C_samples, class_and_C_samples]
                
        return True
            
    def remove(self, X, y, class_, iter):
        """
        Across the available conditions that meet both of the following two requirements
        Remove the one, c, with the lowest importance
        1. c is in C
        2. the importance of c with respect to C_set_minus_c is not None
        If c does not exist, remove a random condition from C
        :param X: the condition matrix
        :param y: the target vector
        :param class_: a class of the target
        :param iter: the current iteration
        :return:
        True : if the rule is not empty after removing a condition
        False : otherwise
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        if len(C) <= 1:
            return False

        # Get the condition-importance pairs
        c_importance = self.get_c_importance(X, y, class_, iter, C)

        # If both requirements 1 and 2 are met
        if len(c_importance) != 0:
            # Sort the condition-importance pairs in ascending order of importance
            c_importance_sorted = sorted(c_importance, key=lambda x: x[1], reverse=False)

            # Get the worst condition (the one with the lowest importance)
            c = c_importance_sorted[0][0]
        else:
            # Get the random condition
            c = self.rng[class_][iter].choice(C, size=1, replace=False)[0]

        # Update C by removing c from C
        C = np.delete(C, np.where(C == c)[0])

        # Get the available samples
        avail_samples = self.avail_samples[class_][iter]

        # Update C_samples
        C_samples = avail_samples[np.where(np.prod(X[np.ix_(avail_samples, C)]) == 1)]

        # Update class_and_C_samples
        class_and_C_samples = C_samples[np.where(y[C_samples] == class_)]

        # Update the rule
        self.rule[class_][iter] = [C, C_samples, class_and_C_samples]

        # Update the removed conditions
        self.removed_conds[class_][iter][c] = 1
        
        return True

    def predict(self, X):
        """
        Predict the class of each sample in X
        :param X: the condition matrix
        :return: the predicted class of each sample in X
        """

        # Predict the probability of each class (of each sample in X) using the best base classifier
        probabilities = self.gs_base.best_estimator_.predict_proba(X)

        for class_ in sorted(self.sig_rules.keys()):
            for rule in self.sig_rules[class_][iter]:
                # Unpack the rule
                C, C_samples, class_and_C_samples = rule

                # Get the samples where the rule fires
                samples = np.where(np.prod(X[:, C] == 1))[0]

                if len(samples) > 0:
                    # Get the probability of the rule
                    probability = float(len(class_and_C_samples)) / len(C_samples)

                    # Update the probabilities
                    probabilities[samples][class_] = max(probabilities[samples][class_], probability)

        return np.argmax(probabilities, axis=1)
