from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold
import numpy as np
from math import sqrt
import strlearn as sl
import statistics 

class HDWE(ClassifierMixin, BaseEnsemble):
    
    """
    References
    ----------
    .. [1] Wang, Haixun, et al. "Mining concept-drifting data streams using ensemble classifiers." Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining. 2003.
    .. [2] Cieslak, David A., et al. "Hellinger distance decision trees are robust and skew-insensitive." Data Mining and Knowledge Discovery 24.1 (2012): 136-158.
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, n_splits=5):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self.candidate_scores = []
        self.weights_ = []
        self.hd_weights = []

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)
            
        # Calculate its scores
        scores = np.zeros(self.n_splits)
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            scores[fold] = self.hellinger_distance(fold_candidate, self.X_[test], self.y_[test])
        
        # Save scores
        candidate_score = np.mean(scores)
        self.candidate_scores.append(candidate_score)
        # Normalize scores
        if len(self.candidate_scores) > 0:
            # !!! normalizacja z użyciem std i mean, ale chyba źle
            # std = statistics.stdev(self.candidate_scores)
            # mean = statistics.mean(self.candidate_scores)
            # candidate_weight = (candidate_score-mean)/std
            
            # !!! normalizacja z wartoscia min i max, co daje zakres (0, 1), ale włącznie z 0 i 1 - co trochę psuje wykres
            # normalized = [(candidate_s-min(self.candidate_scores))/(max(self.candidate_scores)-min(self.candidate_scores)) for candidate_s in self.candidate_scores]
            # candidate_weight = normalized[-1]
            
            # !!! normalizacja, w której wartości sumują się do "1"
            if sum(self.candidate_scores) == 0:
                normalized = [0]
            else:
                normalized = [float(candidate_s)/sum(self.candidate_scores) for candidate_s in self.candidate_scores]
            candidate_weight = normalized[-1]
        else:
            candidate_weight = 0
        
        # Calculate weights of current ensemble
        self.hd_weights = [self.hellinger_distance(clf, self.X_, self.y_) for clf in self.ensemble_]
        # Normalize weights
        if len(self.hd_weights) > 0:
            # !!! normalizacja, w której mamy wartości od 0 do 1 (włącznie)
            # normalized_weights = [(hd_w-min(self.hd_weights))/(max(self.hd_weights)-min(self.hd_weights)) for hd_w in self.hd_weights] 
            
            # !!! normalizacja, w której wartości sumują się do "1"
            if sum(self.hd_weights) == 0:
                normalized_weights = [0]
            else:
                normalized_weights = [float(hd_w)/sum(self.hd_weights) for hd_w in self.hd_weights]  
            self.weights_ = normalized_weights

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]

        return self

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

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    # Prediction without calculated weights
    # def predict(self, X):
    #     """
    #     Predict classes for X.

    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         The training input samples.

    #     Returns
    #     -------
    #     y : array-like, shape (n_samples, )
    #         The predicted classes.
    #     """

    #     # Check is fit had been called
    #     check_is_fitted(self, "classes_")
    #     X = check_array(X)
    #     if X.shape[1] != self.X_.shape[1]:
    #         raise ValueError("number of features does not match")

    #     esm = self.ensemble_support_matrix(X)
    #     average_support = np.mean(esm, axis=0)
    #     prediction = np.argmax(average_support, axis=1)

    #     # Return prediction
    #     return self.classes_[prediction]

    # Prediction (making decision) use Hellinger distance weights 
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
    
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")
    
        # Weight support before acumulation
        weighted_support = (
            self.ensemble_support_matrix(
                X) * np.array(self.weights_)[:, np.newaxis, np.newaxis]
        )
    
        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        prediction = np.argmax(acumulated_weighted_support, axis=1)
    
        # Return prediction
        return self.classes_[prediction]