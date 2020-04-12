import numpy as np

X = np.random.rand(100,3)
y = np.random.randint(2, size=100)

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



from hellinger_distance_criterion import HellingerDistanceCriterion
from sklearn.ensemble import RandomForestClassifier

hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
clf.fit(X_train, y_train)
print('hellinger distance score: ', clf.score(X_test, y_test))

