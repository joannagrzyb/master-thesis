import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from math import sqrt
import strlearn as sl
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class HDDT(BaseEnsemble, ClassifierMixin):
    """
    References
    ----------
    .. [1] https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ CART 
    .. [2] Cieslak, David A., et al. "Hellinger distance decision trees are robust and skew-insensitive." Data Mining and Knowledge Discovery 24.1 (2012): 136-158.
    """

    def __init__(self, base_clf=GaussianNB()):
        self.base_clf = base_clf
        self.Hellinger = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, _ = np.unique(y, return_inverse=True)
        self._clf = clone(self.base_clf).fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
                
        self._X, self._y = X, y
        

        self._clf = clone(self.base_clf).fit(self._X, self._y)
        
        # self.Hellinger = [self.hellinger_distance(self._clf, self._X, self._y)]
        # print(self.Hellinger)
        # dataset = sl.streams.StreamGenerator().get_chunk()
        # print(dataset)
        # split = self.get_split(self._X) # argumentem powinien być cały dataset X+y
        # print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
        tree = self.build_tree(self._X, 1, 1) # argumentem powinien być cały dataset X+y
        # print(tree)
        
        
        return self

    def predict(self, X):
        """
        Predict classes for X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict(X)

    def predict_proba(self, X):
        """
        Predict classes for X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict_proba(X)
    
    
    # Make a prediction with a decision tree
    def prediction(self, node, row):
    	if row[node['index']] < node['value']:
    		if isinstance(node['left'], dict):
    			return self.predict(node['left'], row)
    		else:
    			return node['left']
    	else:
    		if isinstance(node['right'], dict):
    			return self.predict(node['right'], row)
    		else:
    			return node['right']
    
    # Calculate Hellinger distance based on ref. [2] 
    def hellinger_distance(self, clf, X, y):
        # TPR - True Positive Rate (or sensitivity, recall, hit rate)
        tprate = sl.metrics.recall(y, clf.predict(X))
        # TNR - True Negative Rate (or specificity, selectivity)
        tnrate = sl.metrics.specificity(y, clf.predict(X))
        # FPR - False Positive Rate (or fall-out)
        fprate = 1 - tnrate 
        # FNR - False Negative Rate (or miss rate)
        fnrate = 1 - tprate
        # Calculate Hellinger distance
        if tprate > fnrate:
            hd = sqrt((sqrt(tprate)-sqrt(fprate))**2 + (sqrt(1-tprate)-sqrt(1-fprate))**2)
        else:
            hd = 0
        return hd
    
    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
    	left, right = list(), list()
    	for row in dataset:
    		if row[index] < value:
    			left.append(row)
    		else:
    			right.append(row)
    	return left, right
    
    # Select the best split point for a dataset
    def get_split(self, dataset):
    	# class_values = list(set(row[-1] for row in dataset))
    	b_index, b_value, b_score, b_groups = 999, 999, 999, None
    	hellinger = self.hellinger_distance(self._clf, self._X, self._y)
    	for index in range(len(dataset[0])-1):
    		for row in dataset:
    			groups = self.test_split(index, row[index], dataset)
    			# print('X%d < %.3f Hellinger=%.3f' % ((index+1), row[index], hellinger))
    			if hellinger < b_score:
    				b_index, b_value, b_score, b_groups = index, row[index], hellinger, groups
    	return {'index':b_index, 'value':b_value, 'groups':b_groups}
    
    # Create a terminal node value
    def to_terminal(self, group):
    	outcomes = [row[-1] for row in group]
    	return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
    	left, right = node['groups']
    	del(node['groups'])
    	# check for a no split
    	if not left or not right:
    		node['left'] = node['right'] = self.to_terminal(left + right)
    		return
    	# check for max depth
    	if depth >= max_depth:
    		node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
    		return
    	# process left child
    	if len(left) <= min_size:
    		node['left'] = self.to_terminal(left)
    	else:
    		node['left'] = self.get_split(left)
    		self.split(node['left'], max_depth, min_size, depth+1)
    	# process right child
    	if len(right) <= min_size:
    		node['right'] = self.to_terminal(right)
    	else:
    		node['right'] = self.get_split(right)
    		self.split(node['right'], max_depth, min_size, depth+1)

    # Build a decision tree
    def build_tree(self, train, max_depth, min_size):
    	root = self.get_split(train)
    	self.split(root, max_depth, min_size, 1)
    	return root