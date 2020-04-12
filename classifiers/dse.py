from sklearn.base import BaseEstimator
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from utils import minority_majority_split, minority_majority_name
from imblearn.metrics import geometric_mean_score
from sklearn.base import clone


class DeterministicSamplingEnsemble(BaseEstimator):

    def __init__(self, base_classifier=KNeighborsClassifier(), number_of_classifiers=10, balance_ratio=0.50, oversampling=RandomOverSampler(), undersampling=RandomUnderSampler(sampling_strategy='majority')):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers

        self.undersampling = undersampling
        self.oversampling = oversampling
        self.balance_ratio = balance_ratio

        self.drift_detector = None

        self.metrics_array = []
        self.classifier_array = []
        self.stored_X = []
        self.stored_y = []
        self.number_of_features = None

        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.label_encoder = None
        self.iteration = 0

    def partial_fit(self, X, y, classes=None):

        # ________________________________________
        # Initial preperation

        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        if classes[0] is "positive":
            self.minority_name = self.label_encoder.transform(classes[0])
            self.majority_name = self.label_encoder.transform(classes[1])
        elif classes[1] is "positive":
            self.minority_name = self.label_encoder.transform(classes[1])
            self.majority_name = self.label_encoder.transform(classes[0])

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)
            self.number_of_features = len(X[0])

        # ________________________________________
        # Drift detector

        if(self.drift_detector is not None):
            dd_pred = self.drift_detector.predict(X)
            score = geometric_mean_score(dd_pred, y)
            if score / np.mean(self.metrics_array) < 0.7:
                self.drift_detector = None
                self.metrics_array = []
                self.classifier_array = []
                self.stored_X = []
                self.stored_y = []
            else:
                self.metrics_array.append(score)

        # ________________________________________
        # Get stored data

        new_X, new_y = [], []

        for tmp_X, tmp_y in zip(self.stored_X, self.stored_y):
            new_X.extend(tmp_X)
            new_y.extend(tmp_y)

        new_X.extend(X)
        new_y.extend(y)

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        # ________________________________________
        # Undersample and store new data

        und_X, und_y = self.undersampling.fit_resample(X, y)

        self.stored_X.append(und_X)
        self.stored_y.append(und_y)

        # ________________________________________
        # Oversample when below ratio

        minority, majority = minority_majority_split(new_X, new_y, self.minority_name, self.majority_name)
        ratio = len(minority)/len(majority)

        if ratio < self.balance_ratio:
            new_X, new_y = self.oversampling.fit_resample(new_X, new_y)

        # ________________________________________
        # Train new classifier

        self.classifier_array.append(clone(self.base_classifier).fit(new_X, new_y))
        if len(self.classifier_array) >= self.number_of_classifiers:
            del self.classifier_array[0]
            del self.stored_X[0]
            del self.stored_y[0]

        if self.drift_detector is None:
            self.drift_detector = MLPClassifier((10))
        self.drift_detector.partial_fit(new_X, new_y, np.unique(new_y))

        self.iteration += 1

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifier_array]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.classifier_array]
        return np.average(probas_, axis=0)
