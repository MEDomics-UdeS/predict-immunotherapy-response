import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis


class CoxModel:
    """
    Cox Proportional Hazard model for binary risk classification.
    """
    def __init__(self) -> None:
        """Cox Model class builder.

        Args:
            None

        Returns:
            None
        """
        self.model = CoxPHSurvivalAnalysis()

    def train(self,
              X: np.ndarray[np.ndarray[float]],
              y: np.ndarray[tuple[int, float]]) -> None:
        """Trains the Cox Model.

        Args:
            X (n_samples, n_features) : numpy array containing features of each sample
            y (n_samples, ) : numpy array containing the event status and the time of survival of each sample.

        Returns:
            None
        """
        self.model = self.model.fit(X, y)

    def predict_risk_score(self,
                           X: np.ndarray[np.ndarray[float]]) -> np.ndarray:
        """Predicts the risk score.

        Args:
            X : X (n_samples, n_features) : numpy array containing features of each sample

        Returns:
            numpy array (n_samples, ) containing the risk score of each sample
        """
        return self.model.predict(X)

    def find_cutoff(self,
                    risk_scores: np.ndarray[float],
                    q: float = 0.5) -> float:
        """Finds cutoff between high risk and low risk with computing median risk scores.

        Args:
            risk_scores (n_samples, ) : numpy array containing the risk score of each sample
            q : the quantile used as threshold (between 0 and 1)

        Returns:
            The cutoff between high risk and low risk score
        """
        return np.quantile(risk_scores, q)

    def predict_class(self,
                      risk_scores: np.ndarray[float],
                      cutoff: float) -> np.ndarray[int]:
        """Predicts the risk class, 1 if high, 0 otherwise. If the risk score is more than cutoff, the patient
        is assigned to the class 1 (high risk).

        Args: :
            risk_scores (n_samples, ) : numpy array containing the risk score of each sample
            cutoff : the cutoff between high risk and low risk.

        Returns:
            risk_classes (n_samples, ) : risk class of each sample
        """
        # 1 aggregation for risk_scores >= cutoff, 0 otherwise
        risk_classes = np.array([1 if risk >= cutoff else 0 for risk in risk_scores])

        return risk_classes
