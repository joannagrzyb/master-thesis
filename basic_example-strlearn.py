# Example from the stream-learn, using basic classifiers

import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

stream = sl.streams.StreamGenerator()
clfs = [
    GaussianNB(),
    MLPClassifier()
]

evaluator = sl.evaluators.TestThenTrainEvaluator()
evaluator.process(stream, clfs)

print(evaluator.scores_, evaluator.scores_.shape)
