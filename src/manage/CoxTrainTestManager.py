import numpy as np
from sklearn.model_selection import KFold

from src.models.CoxModel import CoxModel


class CoxTrainTestManager:
    """
    Train-test manager for Cox Model.
    """
    def __init__(self) -> None:
        """
        CoxTrainTestManager class builder.

        ### Parameters :
        None

        ### Returns :
        None
        """
        self.model = None

    def leave_one_out_cv(self,
                         X: np.ndarray[np.ndarray[float]],
                         y: np.ndarray[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes the leave one out cross validation to find test risk scores and
        risk classes.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing features of each sample
        - y (n_samples, ) : numpy array containing the event status and the time surviving of each sample.

        ### Returns :
        - risk_scores (n_samples, ) : numpy array containing the risk score of each sample
        - risk_classes (n_samples, ) : numpy array containing the risk class of each sample
        """
        # Initialize risk score and class arrays
        risk_classes = np.zeros(y.shape)
        risk_scores = np.zeros(y.shape)

        # Split the index to n_splits folds
        n_samples = X.shape[0]
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        for i, (train_index, test_index) in enumerate(folds):
            # Instanciate model
            self.model = CoxModel()

            # Train on training set
            self.model.train(X[train_index], y[train_index])

            # Predict train scores and find cutoff
            train_scores = self.model.predict_risk_score(X[train_index])
            cutoff = self.model.find_cutoff(train_scores)

            # Forward pass on test set
            risk_scores[test_index] = self.model.predict_risk_score(X[test_index])
            risk_classes[test_index] = self.model.predict_class(risk_scores[test_index], cutoff)

        return risk_scores, risk_classes
