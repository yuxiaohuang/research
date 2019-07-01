# Please cite the following paper when using the code

import numpy as np

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomPARC(BaseEstimator, ClassifierMixin):
    """
    Random Probabilistic Add-on Rule-based Classifier
    """
    
    def __init__(self,
                 base=None,
                 n_iter=2,
                 min_support=0,
                 min_confidence=1,
                 max_conds=10,
                 random_state=0,
                 n_jobs=1):
        # The base classifier
        self.base = base

        # The number of iterations for searching for the rules
        self.n_iter = n_iter

        # The minimum support required by the rules
        self.min_support = min_support

        # The minimum confidence required by the rules
        self.min_confidence = min_confidence

        # The maximum number of conditions to consider when searching for the rules
        self.max_conds = max_conds
        
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

        # The available samples
        self.avail_samples = None

        # The available conditions
        self.avail_conds = None

        # The removed conditions
        self.removed_conds = None

        # The random number generator
        self.rng = None
        
    def fit(self, X, y):
        """
        The fit function for all classes
        :param X:
        :param y:
        :return:
        """
        """
        
        
        Parameters
        ----------
        X : the condition matrix
        y : the target vector
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

        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        """

        # The maximum number of conditions to consider when searching for the rules
        if self.max_conds > int(X.shape[1] ** 0.5):
            self.max_conds = int(X.shape[1] ** 0.5)

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

        # The available samples
        self.avail_samples = {}

        # The available conditions
        self.avail_conds = {}

        # The removed conditions
        self.removed_conds = {}

        # The random number generator
        self.rng = np.random.RandomState(seed=self.random_state)
              
    def fit_one_class(self, X, y, class_):
        """
        The fit function for one class
        
        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
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

        Parameters
        ----------
        class_ : a class of the target
        """

        # The rule
        self.rule[class_] = {}

        # The detected significant rules
        self.sig_rules[class_] = []

        # The available samples
        self.avail_samples[class_] = {}

        # The available conditions
        self.avail_conds[class_] = {}

        # The removed conditions
        self.removed_conds[class_] = {}

    def greedy_search(self, X, y, class_, iter):
        """
        Greedy search

        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
        iter: the current iteration
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

                print('sig', class_, iter, self.sig_rules[class_])
            else:
                # If no condition can be added
                if self.add(X, y, class_, iter) is False:
                    # If the rule is empty after removing a condition
                    if self.remove(X, y, class_, iter) is False:
                        break

    def init_one_iter(self, y, class_, iter):
        """
        Initialization for one iteration

        Parameters
        ----------
        y : the target vector
        class_ : a class of the target
        iter: the current iteration
        """

        # The available samples
        self.avail_samples[class_][iter] = np.array(range(self.m))

        # The available conditions
        self.avail_conds[class_][iter] = self.rng.choice(self.n, size=self.max_conds, replace=False).astype(int)

        # The removed conditions
        self.removed_conds[class_][iter] = {}

        # The rule
        self.init_rule(y, class_, iter)

    def init_rule(self, y, class_, iter):
        """
        Initialize a rule

        Parameters
        ----------
        y : the target vector
        class_ : a class of the target
        iter: the current iteration
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

        Parameters
        ----------
        class_ : a class of the target
        iter: the current iteration
        
        Returns
        ----------    
        True : if the rule is significant
        False : otherwise
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        # If C is empty
        if len(C) == 0:
            return False

        # print(C_samples, class_and_C_samples, self.min_support, self.min_confidence)

        # If the rule meets min_support and min_confidence
        if (float(len(C_samples)) / self.m >= self.min_support
                and float(len(class_and_C_samples)) / len(C_samples) >= self.min_confidence):
            return True

        return False

    def prune(self, X, y, class_, iter):
        """
        Prune the rule by removing the unnecessary conditions

        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
        iter: the current iteration
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

        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
        iter: the current iteration
        C : a conjunction of conditions

        Returns
        ----------
        The condition-importance pairs
        """

        # Get the importance of each condition
        importances = [self.get_importance(X, y, class_, iter, c) for c in C]

        # Get the condition-importance pairs
        c_importance = [[C[i], importances[i]] for i in range(len(C)) if importances[i] is not None]

        return c_importance

    def get_importance(self, X, y, class_, iter, c):
        """
        Get the importance of condition c with respect to C

        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
        iter: the current iteration
        c : a condition

        Returns
        ----------
        The importance of condition c with respect to C
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
                C_setminus_c_and_not_c_samples = avail_samples[np.where(np.prod(X[avail_samples, C_setminus_c]) == 1
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

        if (float(len(C_and_c_samples)) / self.m >= self.min_support
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

        Parameters
        ----------
        class_ : a class of the target
        iter: the current iteration
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        # Add the rule
        self.sig_rules[class_].append(self.rule[class_][iter])

        # Update the rule
        self.init_rule(y, class_, iter)

        # Update the available samples
        self.avail_samples[class_][iter] = np.setdiff1d(self.avail_samples[class_][iter], C_samples)

        # Update the available conditions
        self.avail_conds[class_][iter] = np.setdiff1d(self.avail_conds[class_][iter], C)

        # Clear the removed conditions
        self.removed_conds[class_][iter] = {}
    
    def add(self, X, y, class_, iter):
        """
        Across the available conditions that meet all of the following three requirements
        Add the one, c, with the highest importance
        1. c is not in C
        2. c has not been removed
        3. the importance of c with respect to C is not None
        
        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
        iter : the current iteration
        
        Returns
        ----------    
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
            if c in self.removed_conds.keys():
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

        print(C_samples, class_and_C_samples)

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
        
        Parameters
        ----------
        X : the condition matrix
        y : the target vector
        class_ : a class of the target
        iter : the current iteration
        
        Returns
        ----------    
        True : if the rule is not empty after removing a condition
        False : otherwise
        """

        # Unpack the rule
        C, C_samples, class_and_C_samples = self.rule[class_][iter]

        if len(C) <= 1:
            return False

        # Get the condition-importance pairs
        c_importance = self.get_c_importance(X, y, C)

        # If both requirements 1 and 2 are met
        if len(c_importance) != 0:
            # Sort the condition-importance pairs in ascending order of importance
            c_importance_sorted = sorted(c_importance, key=lambda x: x[1], reverse=False)

            # Get the worst condition (the one with the lowest importance)
            c = c_importance_sorted[0][0]
        else:
            # Get the random condition
            c = self.rng.choice(C, size=1, replace=False)

        # Update C by removing c from C
        C = np.delete(C, np.where(C == c)[0])

        # Get the available samples
        avail_samples = self.avail_samples[class_][iter]

        # Update C_samples
        C_samples = avail_samples[np.where(np.prod(X[avail_samples, C]) == 1)]

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

        Parameters
        ----------
        X : the condition matrix

        Returns
        ----------
        The predicted class of each sample in X
        """

        # Predict the probability of each class (of each sample in X) using the baseline classifier

        probabilities = self.base.predict_proba(X)

        for class_ in sorted(self.sig_rules.keys()):
            for rule in self.sig_rules[class_]:
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
