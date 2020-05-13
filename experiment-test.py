import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from classifiers import HDWE
from classifiers import HDDT


# List of classifiers from sklearn and others, but partial_fit() function is mandatory
clfs = [
    # sl.ensembles.AWE(GaussianNB()),
    # sl.ensembles.AWE(HDDT()),
    HDWE(GaussianNB(), pred_type="hard"),
    # HDDT(),
    # HDWE(HDDT(), pred_type="hard"),
    HDWE(DecisionTreeClassifier(), pred_type="hard")
]

clf_names = [
    # "AWE",
    # "AWE-HD"
    "HDWE-GNB",
    # "HDDT",
    # "HDWE-HDDT",
    "HDWE-CART",
]

# Declaration of the data stream with given parameters
n_chunks = 3
concept_kwargs = {
    "n_chunks": n_chunks,
    "chunk_size": 500,
    "n_classes": 2,
    "random_state": 157,
    "n_features": 20,
    "n_drifts": 2,
    # "incremental": True,   # incremental drift
    # "concept_sigmoid_spacing": 5,  # incremental drift
    "n_informative": 15,
    "n_redundant": 5,
    "n_repeated": 0,
    "weights": [0.1, 0.9],     # stationary imbalanced stream
    # "weights": (2, 5, 0.9),    # dynamically imbalanced stream
}

stream = sl.streams.StreamGenerator(**concept_kwargs)
stream_name = "nonstationary"

# Chosen metrics
# metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.f1_score]
# metrics = [sl.metrics.balanced_accuracy_score]
metrics = [sl.metrics.f1_score]
# Initialize evaluator with given metrics - stream learn evaluator
evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream, clfs)
scores = evaluator.scores
# Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric
# Every matrix is different classifier, every row is test chunks and every column is different metric
print(scores.shape)
# print(scores)

# Plotting figures of chosen metrics
fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
labels = clf_names

# # For few metrics
# for m, metric in enumerate(metrics):
#     ax[m].set_title(metric.__name__)
#     ax[m].set_ylim(0, 1)
#     for i, clf in enumerate(clfs):
#         ax[m].plot(evaluator.scores[i, :, m], label=labels[i])
#     plt.ylabel("Quality")
#     plt.xlabel("Chunk")
#     ax[m].legend()

# For one metric
ax.set_ylim(0, 1)
for i, clf in enumerate(clfs):
    ax.plot(evaluator.scores[i, :], label=labels[i])
plt.ylabel("Quality")
plt.xlabel("Chunk")
ax.legend()
# plt.show()
# fig.savefig("results/plots/test_result.png")
