import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classifiers import REA
from classifiers import OUSE
from classifiers import KMeanClustering
from classifiers import LearnppCDS
from classifiers import LearnppNIE
# from plotting import plot_stream


# List of classifiers from sklearn and others, but partial_fit() function is mandatory
clfs = [
    GaussianNB(),
    # MLPClassifier(),
    REA(),
    # OUSE(),
    KMeanClustering(),
    LearnppCDS(),
    # LearnppNIE(), # ValueError: Cannot take a larger sample than population when 'replace=False'
    # sl.ensembles.OOB(GaussianNB()),
    # sl.ensembles.UOB(GaussianNB())
]

clf_names = [
    "GNB",
    # "MLP",
    "REA",
    # "OUSE",
    "KMeanClustering",
    "LearnppCDS",
    # "LearnppNIE",
    # "OOB",
    # "UOB",
]

# Declaration of the data stream with given parameters
n_chunks = 100
concept_kwargs = {
    "n_chunks": n_chunks,
    "chunk_size": 250,
    "n_classes": 2,
    "random_state": 106,
    "n_features": 10,
    "n_drifts": 4,
    "n_informative": 2,
    "n_redundant": 0,
    "n_repeated": 0,
    # "weights": [0.1, 0.9],     # stationary imbalanced stream
    # "weights": (1, 5, 0.9),    # dynamically imbalanced stream - do mgr'ki
}
stream = sl.streams.StreamGenerator(**concept_kwargs, weights=(1, 5, 0.9))

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
    plt.ylabel("Quality")
    plt.xlabel("Chunk")
    ax[m].legend()
# plt.show()
fig.savefig("results/plots/1.png")

# Plotting stream
# plot_stream(stream, n_chunks, "dynamic-imbalanced", "Data stream with dynamically imbalanced drift") # TypeError: 'NoneType' object is not iterable
# blad, wiec robie to recznie w plotting.py - zawsze przekopiuj concept_kwargs i stream do tego pliku
