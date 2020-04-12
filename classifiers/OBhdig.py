"""Online Bagging with HDIG"""
# Probably it will not work

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import entropy, binned_statistic
from math import sqrt
# from utils import minority_majority_name, minority_majority_split
# from utils.imbalancedStreams import minority_majority_split, minority_majority_name
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize


# !!! to jest niedokończone i może źle działać

class OnlineBaggingHDIG(BaseEnsemble, ClassifierMixin):
    """
    Online Bagging.
    """

    def __init__(self, base_estimator=None, n_estimators=10, balance_ratio=0.5, n_bins=10):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.balance_ratio = balance_ratio
        self.n_bins = n_bins
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.label_encoder = None
        self.bin_size = None
        self.n_features = None
        self.previous_X = None
        self.previous_y = None
        # self.ens_weights = None

    def fit(self, X, y):
        """Fitting."""
        # Number of features
        # self.n_features = len(X[0])
        self.n_features = X[0].size
        
        self.partial_fit(X, y)
        return self
    
    def distance_hdig(self, Xp, yp, X, y):
        """
        Distance based on Hellinger Distance and Information Gain.
        
        References
        ----------
        .. [1] Lichtenwalter, Ryan N., and Nitesh V. Chawla. "Adaptive methods for classification in arbitrarily imbalanced and drifting data streams." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2009.
        
        Parameters
        ----------
        Xp : previous chunk t - samples
        yp : previous chunk t - classes
        X : current chunk t+1 - samples
        y : current chunk t+1 - classes
        self.n_bins : number of bins for numerical feature value 
        
        Returns
        -------
          : distance/weight
        """
        
        self.n_features = X[0].size
        
        # Number of elements/samples in the array y 
        len_y = len(y)
        
        # Bin size for numerical feature with initial value = -1
        bin_size = [-1] * self.n_features
        
        # !!! Check which features are numerical(continuous) or categorical, assumption:numerical - later TO DO distinguish between these 2 types
        is_numerical = [True] * self.n_features
        
        # Number of labels/samples belonging to class0 and class1
        classes_value, n_classlabels = np.unique(y, return_counts=True)
        
        # Calculate parent entropy (entropy of class labels)
        e_parent = entropy(n_classlabels[0]/len_y, n_classlabels[1]/len_y, base=2)
        
        # number of samples in every bin, for every feature, without division into classes
        bin_counts = []
        
        HDIG = []
        n_bins = self.n_bins
   
        # Calculate child entropy (entropy for every feature)
        for i in range(self.n_features):
            if is_numerical[i]:
                if bin_size[i] == -1:
                    # Split into bins for every feature; 3D array: axis0-features, axis1-bins, axis2-classes; this array contains number of samples, which belong to these categories
                    bin_classes = []
                    # Initial value for weighted average entropy
                    e_weighted_avg_f = 0
                    
                    # Feature column for c(current) and p(previous) chunk
                    feature_c = X[:,i]
                    feature_p = Xp[:,i]

                    minimum_c = np.amin(feature_c)
                    maximum_c = np.amax(feature_c)
                    minimum_p = np.amin(feature_p)
                    maximum_p = np.amax(feature_p)

                    minimum = min(minimum_c,minimum_p)
                    maximum = max(maximum_c,maximum_p)
                    
                    # !!! W razie zmiany na inny sposób liczenia binów
                    # bin_size[i] = (maximum - minimum)/n_bins
                    
                    # function binned_statistic split the set into n_bins equal width from minimum to maximum
                    # stat - return number of samples how many belongs to the given bin; bin_n - show for every sample, to which bin it belongs 
                    stat, bin_e, bin_n = binned_statistic(feature_c, feature_c, bins=n_bins, statistic='count', range=(minimum,maximum))
                    bin_counts = stat.astype(int).tolist()
                         
                    for index, bin_i in enumerate(np.unique(bin_n)):
                        bin_index = np.where(bin_n==bin_i)
                        classes, class_count = np.unique(y[bin_index], return_counts=True)
                        
                        # This if statement add one more value in classes and class_count, if the number of given class is 0
                        if len(classes) == 1: 
                            if classes[0] != 0:
                                class_count1 = class_count[0]
                                class_count[0] = 0
                                class_count = np.append(class_count, class_count1)
                            else:
                                classes[0] = 0
                                classes = np.append(classes, 1)
                                class_count = np.append(class_count, 0)
                            
                        bin_classes.append(class_count.tolist())

                        # print(bin_i-1)
                        # print(index)
                        
                        # print(bin_classes)
                        # print(bin_classes[index][0])
                        # print(bin_counts[bin_i-1])
                        # print(bin_classes[index][1])
                        # print("===")

                        # Calculate child entropy for every feature
                        e_child_f = entropy([bin_classes[index][0]/bin_counts[bin_i-1], bin_classes[index][1]/bin_counts[bin_i-1]], base=2)
                        # Weighted average entropy for all feature
                        e_weighted_avg_f += (bin_counts[bin_i-1]/len_y)*e_child_f

                    # Calculate Information Gain
                    inf_gain_f = e_parent-e_weighted_avg_f

                    # Count values in bins of previous chunk
                    stat, bin_e, bin_n = binned_statistic(feature_p, feature_p, bins=n_bins, statistic='count', range=(minimum,maximum))
                    bin_counts_p = stat.astype(int).tolist()

                    p = [a/len(Xp) for a in bin_counts_p]
                    q = [b/len(X) for b in bin_counts]
                    hellinger_dist_f = sqrt(sum((np.sqrt(p)-np.sqrt(q))**2))

                    # Calculate HDIG - Hellinger Distance with Information Gain
                    HDIG_f = hellinger_dist_f*(1+inf_gain_f)
                    HDIG.append(HDIG_f)
                    
                    
            # else:
                # !!! Tu znajdzie się kod, jeśli cecha jest kategorialna
            
        print(HDIG)
        # Calculate final distance HDIG
        dist_HDIG = sum(HDIG)/self.n_features
        print(dist_HDIG)

        # Calculate weights based on distance hdig and to the power of n_estimator - ensemble size
        weights = dist_HDIG**(-self.n_estimators)
        return weights
        


    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        
        # Binarization of classes into 0 and 1
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes
        y = self.label_encoder.transform(y)
        
        # # Give names for minority and majority classes
        # if self.minority_name is None or self.majority_name is None:
        #     self.minority_name, self.majority_name = minority_majority_name(y)
    
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self.weights = []
        
        # wersja z hdig, ale to nie ma sensu 
        # for instance in range(self.X_.shape[0]):
        #     if self.previous_X is None and self.previous_y is None:
        #         K = np.asarray([np.random.poisson(1, 1)[0] for i in range(self.n_estimators)])
        #         self.weights.append(K)
        #     else:
        #         weight_hdig = self.distance_hdig(self.previous_X, self.previous_y, X, y)
                
        #         # print(weight_hdig)
        #         # Normalize weights
        #         weight_hdig = normalize(np.reshape(weight_hdig, (-1, 1)))
        #         # print(weight_hdig)
        #         self.weights.append(weight_hdig)
        # self.weights = np.asarray(self.weights).T
        # for w, base_model in enumerate(self.ensemble_):
        #     base_model.partial_fit(
        #         self.X_, self.y_, self.classes_, sample_weight=self.weights[w]
        #     )
        
        
        # to ma być osobno dla klasyfikatorów w ensemble, trzeba by użyć self.ens_weights
        # weight_hdig = normalize(np.reshape(weight_hdig, (-1, 1)))
        
        # To było wcześniej do OB oryginalnie
        for instance in range(self.X_.shape[0]):
            K = np.asarray(
                [np.random.poisson(1, 1)[0] for i in range(self.n_estimators)]
            )
            self.weights.append(K)
        self.weights = np.asarray(self.weights).T
        for w, base_model in enumerate(self.ensemble_):
            base_model.partial_fit(
                self.X_, self.y_, self.classes_, sample_weight=self.weights[w]
            )
        
        self.previous_X = X
        self.previous_y = y

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, self.n_features)
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

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        
        # tu w prediction mogą byc te wagi dla ensemble, jeśli będzie się je dało jakoś użyć
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
    
