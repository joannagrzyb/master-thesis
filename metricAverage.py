import strlearn as sl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classifiers import REA
from classifiers import LearnppCDS
from classifiers import LearnppNIE
# from plotting import plot_stream
# from utils.ploting import Ploting


# List of classifiers from sklearn and others, but partial_fit() function is mandatory
clfs = [
    # GaussianNB(),
    MLPClassifier(),
    REA(GaussianNB()),
    LearnppCDS(GaussianNB()),
    # LearnppNIE(GaussianNB()), # dla dynamic stream: ValueError: Cannot take a larger sample than population when 'replace=False'
    sl.ensembles.OOB(GaussianNB(), n_estimators=10),
    sl.ensembles.UOB(GaussianNB(), n_estimators=10)
]

clf_names = [
    # "GNB",
    "MLP",
    "REA",
    "LearnppCDS",
    # "LearnppNIE",
    "OOB",
    "UOB",
]

# Declaration of the data stream with given parameters
n_chunks = 100

# metrics = [sl.metrics.balanced_accuracy_score]
metrics = [sl.metrics.geometric_mean_score_1]
evaluator = sl.evaluators.TestThenTrain(metrics)

random_state = 111
n_streams = 10
i = 0
# Generating streams and write scores to the files
# while i < n_streams:
#     stream = sl.streams.StreamGenerator(
#             n_chunks=n_chunks,
#             chunk_size=250,
#             n_classes=2,
#             random_state=random_state,
#             n_features=2,
#             n_drifts=4,
#             n_informative=2,
#             n_redundant=0,
#             n_repeated=0,
#             # weights=[0.1, 0.9],     # stationary imbalanced stream
#             weights=(1, 5, 0.9),    # dynamically imbalanced stream
#     )
#     evaluator.process(stream, clfs)
#     scores = evaluator.scores
#
#     # Wybierz 1 z 4:
#
#     # ###Load into file-static-bac
#     # with open("results/text/scores%i.pkl" % (i),"wb") as f:
#     #     pickle.dump(scores,f)
#
#     # ###Load into file-dynamic-bac
#     # with open("results/text/dynamic_scores%i.pkl" % (i),"wb") as f:
#     #     pickle.dump(scores,f)
#
#     # ###Load into file-dynamic-g_mean
#     # with open("results/text/g_mean/static_scores%i.pkl" % (i),"wb") as f:
#     #     pickle.dump(scores,f)
#
#     ###Load into file-dynamic-g_mean
#     with open("results/text/g_mean/dynamic_scores%i.pkl" % (i),"wb") as f:
#         pickle.dump(scores,f)
#
#     print(scores)
#
#     random_state += 111
#     i += 1

# # Calculate mean and std and draw figures
i = 0
scores_temp = []
while i < 10:

      ## Wybierz 1 z 4:

    ###Extract from file-static
    # with open("results/text/scores%i.pkl" % (i),"rb") as f:
    #     scores_temp.append(np.squeeze(pickle.load(f)))

    # ###Extract from file-dynamic
    # with open("results/text/dynamic_scores%i.pkl" % (i),"rb") as f:
    #     scores_temp.append(np.squeeze(pickle.load(f)))
    # i += 1

    # ###Extract from file-static-g_mean
    # with open("results/text/g_mean/static_scores%i.pkl" % (i),"rb") as f:
    #     scores_temp.append(np.squeeze(pickle.load(f)))
    #     i += 1

    ###Extract from file-dynamic-g_mean
    with open("results/text/g_mean/dynamic_scores%i.pkl" % (i),"rb") as f:
        scores_temp.append(np.squeeze(pickle.load(f)))
        i += 1


scores_temp = np.asarray(scores_temp)
print(scores_temp.shape)

scores_mean = np.mean(scores_temp, axis=0)
scores_std = np.std(scores_temp, axis=0)
print(scores_std.shape)
print(scores_mean)
print(scores_std)


# Plotting figures - for mean and std
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
labels = clf_names

for i, clf in enumerate(clf_names):
    ax[0].plot(scores_mean[i], label=labels[i])
ax[0].set_ylabel("Mean")
ax[0].set_xlabel("Chunk")
ax[0].legend()

for i, clf in enumerate(clf_names):
    ax[1].plot(scores_std[i], label=labels[i])
ax[1].set_ylabel("Standard deviation")
ax[1].set_xlabel("Chunk")
ax[1].legend()

# plt.show()
# fig.savefig("results/plots/bac_mean_stat.png")
# fig.savefig("results/plots/bac_mean_dynamic.png")
# fig.savefig("results/plots/g_mean_static.png")
fig.savefig("results/plots/g_mean_dynamic.png")









# For one metric - only mean
# labels = clf_names
# fig, ax = plt.subplots(1, len(metrics), figsize=(12, 4))
# ax.set_ylim(0, 1)
# for i, clf in enumerate(clfs):
#     ax.plot(scores_mean[i], label=labels[i])
# ax.legend()
# plt.ylabel("Quality")
# plt.xlabel("Chunk")
# fig.savefig("results/plots/bac_mean_stat.png")
