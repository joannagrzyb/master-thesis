from strlearn.evaluators.TestThenTrain import *
from skmultiflow.drift_detection import DDM
from sklearn.metrics import accuracy_score, auc, f1_score

class TestThenTrainEvaluator(TestThenTrain):

    def __init__(self, metrics=[accuracy_score]):
        super(TestThenTrainEvaluator, self).__init__(metrics=metrics)
        self._ddm = DDM(min_num_instances=10, warning_level=1.0, out_control_level=1.5)
        self.drift_indices = []

    def process(self, stream, clfs):
        """
        Perform learning procedure on data stream.

        Parameters
        ----------
        clf : scikit-learn estimator
            Classifier implementing a `partial_fit()` method.
        stream : object
            Data stream as an object.
        """
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs_ = [clfs]
        else:
            self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        self.scores = np.zeros(
            (len(self.clfs_), ((self.stream_.n_chunks - 1)), len(self.metrics))
        )

        while True:
            X, y = stream.get_chunk()

            # Test
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs_):
                    y_pred = clf.predict(X)

                    self.scores[clfid, stream.chunk_id - 1] = [
                        metric(y, y_pred) for metric in self.metrics
                    ]

                    self._ddm.add_element(1 - self.scores[clfid, stream.chunk_id - 1][0])

                    if self._ddm.detected_change():
                        self.drift_indices.append(stream.chunk_id)
                        if hasattr(clf, 'dropout'):
                            clf.save_state()
                            outputs = clf.get_submodels_predictions(X)

                            if outputs is None:
                                clf.set_state(None)
                            else:
                                submodels_scores = []
                                for pred in outputs:
                                    submodels_scores.append([metric(y, pred) for metric in self.metrics])

                                mean_submodels_scores = [np.mean(score_group) for score_group in submodels_scores]

                                if max(mean_submodels_scores) > 0.7:
                                    idx = mean_submodels_scores.index(max(mean_submodels_scores))
                                    clf.set_state(idx)
                                else:
                                    clf.set_state(None)

            # Train
            [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]

            if stream.is_dry():
                break