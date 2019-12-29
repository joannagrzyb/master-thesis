import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classifiers import REA
from classifiers import OUSE
from classifiers import KMeanClustering
from classifiers import LearnppCDS
from classifiers import LearnppNIE


# List of classifiers from sklearn and others, but partial_fit() function is mandatory
clfs = [
    GaussianNB(),
    MLPClassifier(),
    REA(),
    OUSE(),
    KMeanClustering(),
    LearnppCDS(),
    LearnppNIE(),
    sl.ensembles.OOB(GaussianNB()),
    sl.ensembles.UOB(GaussianNB())
]

clf_names = [
    "GNB",
    "MLP",
    "REA",
    "OUSE",
    "KMeanClustering",
    "LearnppCDS",
    "LearnppNIE",
    "OOB",
    "UOB",
]

# Declaration of the data stream with given parameters
stream = sl.streams.StreamGenerator(
        n_chunks=50,
        chunk_size=250,
        n_features = 10,
        n_drifts = 4,
        weights=[0.1, 0.9],     # stationary imbalanced stream
        # weights=(2, 5, 0.9),    # dynamically imbalanced stream - do mgr'ki?
)

# Chosen metrics
metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.f1_score]

# Initialize evaluator with given metrics
evaluator = sl.evaluators.TestThenTrain(metrics)

evaluator.process(stream, clfs)
scores = evaluator.scores
# wymiary tablicy scores: 1) klasyfikator, 2) chunk, 3) metryka
# kazda kolumna to inna metryka, kazda tablica/macierz to inny klasyfikator, w ktorej wiersze to kolejne chunki testowe
print(scores)

# Plotting figures of chosen metrics
fig, ax = plt.subplots(1, len(metrics), figsize=(12, 4))
labels = clf_names
for m, metric in enumerate(metrics):
    ax[m].set_title(metric.__name__)
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=labels[i])
    ax[m].legend()
# plt.show()
fig.savefig("results/plots/1.png")
