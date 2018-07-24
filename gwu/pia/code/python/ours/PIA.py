# Please cite the following paper when using the code

import numpy as np
from numpy import prod
from scipy import stats
import random

class PIA:
    """
    Principle Interaction Analysis
    """
    
    def __init__(self, min_samples_importance=1, min_samples_interaction=1, random_state=0):
        # The minimum number of samples required for calculating importance
        self.min_samples_importance = min_samples_importance
        
        # The minimum number of samples required for an interaction
        self.min_samples_interaction = min_samples_interaction
        
        # The random_state
        self.random_state = random_state
        
        # Set random seed
        np.random.RandomState(seed=self.random_state)
        
    def fit(self, X, y):
        """
        Detect the interactions for each class label
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        """

        # The distribution of each class
        self.dist = {}
        
        # The disjunction of detected interactions
        self.D = {}
        
        # For each class of the target
        for class_ in sorted(np.unique(y)):
            # Get the distribution of class_
            self.dist[class_] = [1 if class_ == y[i] else 0 for i in range(X.shape[0])]
        
            self.D[class_] = []
            
            # The conditions that have been removed
            self.removed = {}
        
            # The conditions that have been deleted
            self.deleted = {}
        
            # The conjunction
            C = []
            
            # Greedy search
            self.greedy_search(X, y, class_, C)
              
    def greedy_search(self, X, y, class_, C):
        """
        Greedy search
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        """
            
        # Do, while one of the following three requirements is met
        # 1. C is sufficient
        # 2. add_best returns [C, True]
        # 3. remove_worst returns [C, True]
        while True:
            # If C is sufficient
            if self.sufficient(X, y, class_, C) is True:
                # Remove the unnecessary conditions from C
                C = self.remove_unnecessary(X, y, class_, C)
            
                # Get the samples where C is true
                C_samples = self.get_samples(X, C)
                # Get the distribution of class_ where C is true
                dist_C = [1 if class_ == y[i] else 0 for i in C_samples]
                # Get the probability, P(class_ | C)
                prob = np.mean(dist_C)

                # Add [C, prob] to D
                self.D[class_].append([C, prob])
                
                # Clear removed
                self.removed = {}
                
                for c in C:
                    # Get a subset of C by excluding c
                    C_setminus_c = [x for x in C if x != c]   
                    
                    # Recursively call greedy_search using the above subset of C
                    self.greedy_search(X, y, class_, C_setminus_c)
                
                # Delete the conditions
                for c in C:
                    self.deleted[c] = 1
                    
                # Initialize the conjunction
                C = []
                    
            # If C is not sufficient
            else:
                C, success = self.add_best(X, y, class_, C)
                
                if success is False:
                    C, success = self.remove_worst(X, y, class_, C)
                    
                    if success is False:
                        C = self.remove_random(X, y, class_, C, 0)
                        
                        if len(C) == 0:
                            break
                
    def sufficient(self, X, y, class_, C):
        """
        The sufficient condition
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        
        Returns
        ----------    
        True : if the sufficient condition is met
        False : otherwise
        """
        
        # Check if P(class_ | C) >> P(class_)
        if self.sig(X, y, class_, C, []) is False:
            return False
        
        # For each condition that is not in C
        for x in range(X.shape[1]):
            if x in C:
                continue
            # Check if P(class_ | C and not x) >> P(class_)
            if self.sig(X, y, class_, C, [x]) is False:
                return False

        return True
    
    def sig(self, X, y, class_, C, xs):
        """
        Check if P(class_ | C and not xs) >> P(class_)
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        xs : a conjunction of conditions
        
        Returns
        ----------    
        True : if P(class_ | C and not xs) >> P(class_)
        False : otherwise
        """

        # Get the samples where C is true
        C_samples = self.get_samples(X, C)
        # Get the samples where xs is false
        not_xs_samples = self.get_not_samples(X, xs)
        # Get the samples where C is true and xs is false
        C_and_not_xs_samples = list(set(C_samples) & set(not_xs_samples))
        # Get the distribution of class_ where C is true and xs is false
        dist_C_and_not_xs = [1 if class_ == y[i] else 0 for i in C_and_not_xs_samples]

        # If there are no sufficient samples, xs cannot be used, return True
        if len(dist_C_and_not_xs) < self.min_samples_importance:
            return True

        return True if np.mean(dist_C_and_not_xs) == 1 else False
    
    def get_samples(self, X, C):
        """
        Get the samples where every condition in C is true
        
        Parameters
        ----------
        X : the feature vector
        C : a conjunction of conditions
        
        Returns
        ----------    
        The samples where every condition in C is true
        """      
        
        if len(C) > 0:  
            return [i for i in range(X.shape[0]) if prod(X[i, C]) == 1]
        else:
            return [i for i in range(X.shape[0])]
    
    def get_not_samples(self, X, C):
        """
        Get the samples where at least one condition in C is not true
        
        Parameters
        ----------
        X : the feature vector
        C : a conjunction of conditions
        
        Returns
        ----------    
        The samples where at least one condition in C is not true
        """
        
        if len(C) > 0:  
            return [i for i in range(X.shape[0]) if prod(X[i, C]) == 0]
        else:
            return [i for i in range(X.shape[0])]     
                
    def remove_unnecessary(self, X, y, class_, C):
        """
        Remove the unnecessary conditions from C
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        
        Returns
        ----------    
        A subset of C where each condition is necessary        
        """
        
        while True:
            # Record the size of C
            size = len(C)
            
            # Get the condition-importance pairs
            c_importances = self.get_c_importances(X, y, class_, C, C)
            
            # Sort the condition-importance pairs in ascending order of importance
            c_importances_sorted = sorted(c_importances, key=lambda x : x[1], reverse=False)
            
            for c, importance in c_importances_sorted:
                # Remove c from c_importances_sorted
                C_setminus_c = [c_importance[0] for c_importance in c_importances_sorted if c_importance[0] != c]
                
                # If c is not necessary
                if self.sufficient(X, y, class_, C_setminus_c) is True:
                    # Remove c from C
                    C.remove(c)
                    break
                    
            # If all conditions are necessary
            if size == len(C):
                break
                
        return C
    
    def get_c_importances(self, X, y, class_, C, cs):
        """
        Get the condition-importance pairs, where:
        the condition, c, in a pair is a condition in cs
        the importance is the one of c with respect to C
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        cs : a conjunction of conditions
        
        Returns
        ----------    
        The condition-importance pairs        
        """        
        
        # Get the importance of each condition
        importances = [self.get_importance(X, y, class_, C, c) for c in cs]
                
        # Get the condition-importance pairs
        c_importances = [[cs[i], importances[i]] for i in range(len(cs)) if importances[i] is not None]
        
        return c_importances
    
    def get_importance(self, X, y, class_, C, c):
        """
        Get the importance of condition c with respect to conjunction C
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        c : a condition
        
        Returns
        ----------    
        The importance of condition c with respect to conjunction C    
        """ 
        
        # Get C_and_c
        C_and_c = list(C)
        if c not in C_and_c:
            C_and_c.append(c)
        # Get the samples where C_and_c is true
        C_and_c_samples = self.get_samples(X, C_and_c)
        # Get the distribution of class_ where C_and_c is true
        dist_C_and_c = [1 if class_ == y[i] else 0 for i in C_and_c_samples]
        
        # Get C \setminus c
        C_setminus_c = list(C)
        if c in C_setminus_c:
            C_setminus_c.remove(c)
        # Get the samples where C_setminus_c is true
        C_setminus_c_samples = self.get_samples(X, C_setminus_c)     
        # Get the samples where c is false
        not_c_samples = self.get_not_samples(X, [c])   
        # Get the samples where C_setminus_c is true and c is false
        C_setminus_c_and_not_c_samples = list(set(C_setminus_c_samples) & set(not_c_samples))
        # Get the distribution where C_setminus_c is true and c is false
        dist_C_setminus_c_and_not_c = [1 if class_ == y[i] else 0 for i in C_setminus_c_and_not_c_samples]
                
        if len(dist_C_and_c) < self.min_samples_importance or len(dist_C_setminus_c_and_not_c) < self.min_samples_importance:
            importance = None
        else:
            importance = np.mean(dist_C_and_c) - np.mean(dist_C_setminus_c_and_not_c)

        return importance
    
    def add_best(self, X, y, class_, C):
        """
        Across the conditions that meet all of the following four requirements
        Add the one, c, with the highest importance
        1. c is not in C
        2. c has not been removed or deleted
        3. C_and_c is not a superset of any detected interaction
        4. the importance of c with respect to C is not None
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        
        Returns
        ----------    
        [C_and_c, True] : if the above condition, c, exists 
        [C, False] : otherwise 
        """
        
        # The list of c meeting requirements 1-3
        cs = []
        
        for c in range(X.shape[1]):
            # If requirement 1 is not met
            if c in C:
                continue

            # If requirement 2 is not met
            if c in self.removed.keys() or c in self.deleted.keys():
                continue
                
            # Get C_and_c
            C_and_c = list(C)
            C_and_c.append(c)
            
            # If requirement 3 is not met
            if self.super_set(class_, C_and_c) is True:
                continue
                
            cs.append(c)  
                            
        # Get the condition-importance pairs
        c_importances = self.get_c_importances(X, y, class_, C, cs)        
        
        if len(c_importances) == 0:
            return [C, False]

        # Sort the condition-importance pairs in descending order of importance
        c_importances_sorted = sorted(c_importances, key=lambda x : x[1], reverse=True)  
        # Get the best feature (the one with the highest importance)
        best = c_importances_sorted[0][0]
        # Add best to C
        C.append(best)
                
        return [C, True]
    
    def super_set(self, class_, C):
        """
        If C is a superset of interactions of class_
        
        Parameters
        ----------
        class_ : a class of the target
        C : a conjunction of conditions
        
        Returns
        ----------    
        True : if C is a superset of interactions of class_
        False : otherwise 
        """
        for I, prob in self.D[class_]:
            if set(I) <= set(C):
                return True
            
        return False
            
    def remove_worst(self, X, y, class_, C):
        """
        Across the conditions that meet all of the following two requirements
        Remove the one, c, with the lowest importance
        1. c is in C
        2. the importance of c with respect to C_set_minus_c is not None
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        
        Returns
        ----------    
        [C_set_minus_c, True] : if the above condition, c, exists 
        [C, False] : otherwise 
        """
        
        c_importances = self.get_c_importances(X, y, class_, C, C)
            
        if len(c_importances) == 0:
            return [C, False]
        
        # Sort the condition-importance pairs in ascending order of importance
        c_importances_sorted = sorted(c_importances, key=lambda x : x[1], reverse=False) 
        # Get the worst feature (the one with the lowest importance)
        worst = c_importances_sorted[0][0]
        # Remove worst from C
        C.remove(worst)
        # Update self.removed
        self.removed[worst] = 1
        
        return [C, True]
    
    def remove_random(self, X, y, class_, C, random_state):
        """
        Remove a random condition, c, from C
        
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        class_ : a class of the target
        C : a conjunction of conditions
        random_state : seed from random number generator
        
        Returns
        ----------    
        C_set_minus_c
        """
        
        if len(C) == 0:
            return C
        
        # Get random_c
        random_idx = np.random.randint(low=0, high=len(C))
        random_c = C[random_idx]
        # Remove random_c from C
        C.remove(random_c)
        # Update self.removed
        self.removed[random_c] = 1
        
        return C