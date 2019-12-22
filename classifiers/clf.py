from sklearn.base import BaseEstimator


class CLF(BaseEstimator):
    """Basic CLF to fulfill."""

    def __init__(self, arg):
        super(CLF, self).__init__()
        self.arg = arg

    # Method for learning
    def partial_fit(self, X, y):
        pass

    # Method for estimating
    def predict(self, X):
        pass

    # Method which return support vectors (wektory wsparc)
    def predict_proba(self, X):
        pass
