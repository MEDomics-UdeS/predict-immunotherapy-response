from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np


class CoxModel:
    """
    Cox Proportional Hazard model for binary risk classification.
    """
    def __init__(self) -> None:
        """
        Define the Cox Model attribute.
        """
        self.model = CoxPHSurvivalAnalysis()

    def train(self,
              X: np.ndarray[np.ndarray[float]],
              y: np.ndarray[tuple[int, float]]) -> None:
        """
        Train the Cox Model with estimating hazard function parameters.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing features of each
        sample
        - y (n_samples, ) : numpy array containing the event status and the
        time surviving of each sample.

        ### Returns :
        None
        """
        self.model = self.model.fit(X, y)

    def predict_risk_score(self,
                           X: np.ndarray[np.ndarray[float]]) -> np.ndarray:
        """
        Predict the risk score.

        ### Parameters :
        - X : X (n_samples, n_features) : numpy array containing features of
        each sample

        ### Returns :
        numpy array (n_samples, ) containing the risk score of each sample
        """
        return self.model.predict(X)

    def find_cutoff(self,
                    risk_scores: np.ndarray) -> float:
        """
        Find cutoff between high risk and low risk with computing median risk
        scores.

        ### Parameters :
        - risk_scores (n_samples, ) : numpy array containing the risk score of
        each sample

        ### Returns :
        The cutoff between high risk and low risk score
        """
        return np.median(risk_scores)

    def predict_class(self,
                      risk_scores: np.ndarray,
                      cutoff: float) -> np.ndarray:
        """
        Predict the risk class, 1 if high, 0 otherwise.

        ### Parameters :
        - risk_scores (n_samples, ) : numpy array containing the risk score of
        each sample
        - cutoff : the cutoff between high risk and low risk

        ### Returns :
        risk_classes (n_samples, ) : risk class of each sample
        """
        # Copy risk scores array
        risk_classes = np.copy(risk_scores)

        # 1 aggregation for risk_scores >= cutoff
        risk_classes[risk_scores >= cutoff] = 1

        # 0 aggregation for risk_scores < cutoff
        risk_classes[risk_scores < cutoff] = 0

        return risk_classes