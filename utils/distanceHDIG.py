import numpy as np
from math import sqrt
from scipy.stats import entropy, binned_statistic

# !!! to jest niedokończone i może źle działać

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
            
        # Calculate final distance HDIG
        dist_HDIG = sum(HDIG)/self.n_features

        # Calculate weights based on distance hdig and to the power of n_estimator - ensemble size
        weights = dist_HDIG**(-self.n_estimators)
        return weights