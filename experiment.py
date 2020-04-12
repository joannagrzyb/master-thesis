import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classifiers import HDWE
from sklearn.tree import DecisionTreeClassifier
from utils.ploting import Ploting
from utils.DriftEvaluator import DriftEvaluator
from utils.TestThenTrainEvaluator import TestThenTrainEvaluator


# List of classifiers from sklearn and others, but partial_fit() function is mandatory
clfs = [
    MLPClassifier(),
    # sl.ensembles.OOB(GaussianNB()),
    # sl.ensembles.UOB(GaussianNB()),
    # sl.ensembles.OnlineBagging(GaussianNB()),
    sl.ensembles.AWE(GaussianNB()),
    # sl.ensembles.AWE(DecisionTreeClassifier()),
    HDWE(GaussianNB()),
    # HDWE(DecisionTreeClassifier()),
]

clf_names = [
    "MLP",
    # "OOB",
    # "UOB",
    # "OB",
    "AWE",
    # "AWEdt",
    "HDWE",
    # "HDWEdt",
]

# Declaration of the data stream with given parameters
n_chunks = 200
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
    # "weights": (1, 5, 0.9),    # dynamically imbalanced stream - !!! wtedy coś nie działa w prediction w funckji HDWE, i ze zwykłą predykcją, wykres trochę dziwny
}

stream = sl.streams.StreamGenerator(**concept_kwargs)
stream_name = "nonstationary"

# Chosen metrics
# metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.f1_score]
# metrics = [sl.metrics.balanced_accuracy_score]
metrics = [sl.metrics.f1_score]

# Initialize evaluator with given metrics
evaluator = sl.evaluators.TestThenTrain(metrics)

# This evaluator is need to DriftEvaluator
# evaluator = TestThenTrainEvaluator()


evaluator.process(stream, clfs)

# !!! Scores from Drift eval. - wychodzą trochę gorsze wyniki, czemu, jak to jest to samo co w strlearn tylko kilka rzeczy dodanych?
# drift_evaluator = DriftEvaluator(evaluator.scores, evaluator.drift_indices)
# print("Final scores:\n", evaluator.scores)
# Metryki dryftu ?
# max_performance_loss = drift_evaluator.get_max_performance_loss()
# recovery_lengths = drift_evaluator.get_recovery_lengths()
# accuracy = drift_evaluator.get_mean_acc()
# models_scores = evaluator.scores


scores = evaluator.scores
print(scores.shape)

# Shape of the score: 1st - classfier, 2nd - chunk, 3rd - metric
# Every matrix is different classifier, every row is test chunks and every column is different metric
print(scores)

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
fig.savefig("results/plots/test_result.png")
