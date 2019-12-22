import strlearn as sl
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from classifiers import REA
from classifiers import OUSE

# Basic function to run experiments of classification
def classify(stream, clf):
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)

    return evaluator


clfs = [
    GaussianNB(),
    MLPClassifier(),
    REA(),
    OUSE(),
]

# scores_ - to jest accuracy
# print(eval.scores_, eval.scores_.shape)


n_chunks = 5
a = []
for i in range(1000):
    stream = sl.streams.StreamGenerator(
        n_chunks=n_chunks,
        chunk_size=500,
        weights=[0.3, 0.7],     # stationary imbalanced stream
        # weights=(2, 5, 0.9),    # dynamically imbalanced stream - dla mgr'ki?
    )

    eval = classify(stream, clfs)

    print("\n# TestThenTrain\n# %04i" % i, np.mean(eval.scores_, axis=1), "\n")

    a.append(np.mean(eval.scores_, axis=1))
    exit()

a = np.array(a)

a = np.std(a, axis=0)

print(a)
