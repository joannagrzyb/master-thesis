# import numpy as np

# X = np.random.rand(100,3)
# y = np.random.randint(2, size=100)

######################### Old clfs


# from classifiers import REA
# from classifiers import LearnppCDS
# from classifiers import LearnppNIE
# from classifiers import OnlineBaggingHDIG
# from classifiers import AWEhdig
# from classifiers import SEAhdig
# # List of classifiers from sklearn and others, but partial_fit() function is mandatory
# clfs = [
#     # GaussianNB(),
#     MLPClassifier(),
#     # REA(),
#     # LearnppCDS(),
#     # LearnppNIE(), # dla dynamic stream: ValueError: Cannot take a larger sample than population when 'replace=False'
#     # sl.ensembles.OOB(GaussianNB()),
#     # sl.ensembles.UOB(GaussianNB()),
#     # sl.ensembles.OnlineBagging(GaussianNB()),
#     # OnlineBaggingHDIG(GaussianNB()),
#     # =============================== NEW
#     # AWEhdig(GaussianNB()),
#     sl.ensembles.AWE(GaussianNB()),
#     # sl.ensembles.AWE(DecisionTreeClassifier()),
#     # SEAhdig(GaussianNB()),
#     # sl.ensembles.SEA(GaussianNB()),
#     HDWE(GaussianNB()),
#     # HDWE(DecisionTreeClassifier()),
# ]

# clf_names = [
#     # "GNB",
#     "MLP",
#     # "REA",
#     # "LearnppCDS",
#     # "LearnppNIE",
#     # "OOB",
#     # "UOB",
#     # "OB",
#     # "OBhdig",
#     # =============================== NEW
#     # "AWEhdig",
#     "AWE",
#     # "AWEdt",
#     # "SEAhdig",
#     # "SEA",
#     "HDWE",
#     # "HDWEdt",
# ]


################################ HDDT tests
# from hellinger_distance_criterion import HellingerDistanceCriterion
# from sklearn.ensemble import RandomForestClassifier

# hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
# clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
# clf.fit(X_train, y_train)
# print('hellinger distance score: ', clf.score(X_test, y_test))


# ############################## Drift Evaluator tests
# from utils.DriftEvaluator import DriftEvaluator
# from utils.TestThenTrainEvaluator import TestThenTrainEvaluator
# from sklearn.neural_network import MLPClassifier
# from classifiers import HDWE
# import strlearn as sl
# from sklearn.naive_bayes import GaussianNB
# import matplotlib.pyplot as plt

# def start():
#     random_states = [157, 158, 159]
#     stream_num = 1
#     n_chunks = 200
#     concept_kwargs = {
#         "n_chunks": n_chunks,
#         "chunk_size": 500,
#         "n_classes": 2,
#         "random_state": 157,
#         "n_features": 20,
#         "n_drifts": 2,
#         # "incremental": True,   # incremental drift
#         # "concept_sigmoid_spacing": 5,  # incremental drift
#         "n_informative": 15,
#         "n_redundant": 5,
#         "n_repeated": 0,
#         "weights": [0.1, 0.9],     # stationary imbalanced stream
#         # "weights": (2, 5, 0.9),    # dynamically imbalanced stream
#     }
    
#     for random_state in random_states:
#         print("Stream: ", stream_num)
#         clfs = [
#                 MLPClassifier(hidden_layer_sizes=(10)),
#                 # sl.ensembles.OOB(GaussianNB()),
#                 # sl.ensembles.UOB(GaussianNB()),
#                 # sl.ensembles.OnlineBagging(GaussianNB()),
#                 # sl.ensembles.AWE(GaussianNB()),
#                 # sl.ensembles.AWE(DecisionTreeClassifier()),
#                 HDWE(GaussianNB()),
#                 # HDWE(DecisionTreeClassifier()),
#             ]

#         clf_names = [
#             "MLP",
#             # "OOB",
#             # "UOB",
#             # "OB",
#             # "AWE",
#             # "AWEdt",
#             "HDWE",
#             # "HDWEdt",
#         ]

#         models_scores = []
        
#         for clf, name in zip(clfs, clf_names):
#             stream = sl.streams.StreamGenerator(**concept_kwargs)
            
#             evaluator = TestThenTrainEvaluator()

#             evaluator.process(stream, clf)

#             drift_evaluator = DriftEvaluator(evaluator.scores, evaluator.drift_indices)
#             print("Final scores:\n", evaluator.scores)

#             max_performance_loss = drift_evaluator.get_max_performance_loss()
#             recovery_lengths = drift_evaluator.get_recovery_lengths()
#             accuracy = drift_evaluator.get_mean_acc()
#             models_scores.append(evaluator.scores)
#             print(max_performance_loss)
#             print(recovery_lengths)
            
#         print(models_scores)
#         # Plotting figures of chosen metrics
#         # fig, ax = plt.subplots(1, 1, figsize=(24, 8))
#         # labels = clf_names
        
#         # # # For few metrics
#         # # for m, metric in enumerate(metrics):
#         # #     ax[m].set_title(metric.__name__)
#         # #     ax[m].set_ylim(0, 1)
#         # #     for i, clf in enumerate(clfs):
#         # #         ax[m].plot(evaluator.scores[i, :, m], label=labels[i])
#         # #     plt.ylabel("Quality")
#         # #     plt.xlabel("Chunk")
#         # #     ax[m].legend()
        
#         # # For one metric
#         # ax.set_ylim(0, 1)
#         # for i, clf in enumerate(clfs):
#         #     ax.plot(evaluator.scores[i, :], label=labels[i])
#         # plt.ylabel("Quality")
#         # plt.xlabel("Chunk")
#         # ax.legend()
#         # # plt.show()
#         # fig.savefig("results/plots/test_result.png")
#         # stream_num += 1


# if __name__ == '__main__':
#     start()
