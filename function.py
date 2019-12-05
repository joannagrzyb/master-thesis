import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Basic function to run experiments of classification
def classify(stream, clf):
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)

    return evaluator



stream = sl.streams.StreamGenerator()
clfs = [
    GaussianNB(),
    MLPClassifier()
]

eval = classify(stream, clfs)

print(eval.scores_, eval.scores_.shape)
