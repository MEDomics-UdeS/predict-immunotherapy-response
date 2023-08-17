import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival


class SurvivalMetrics:
    """
    Implementation of some survival analysis evaluation metrics.
    """
    @staticmethod
    def get_c_index(status: np.ndarray[int],
                    time: np.ndarray[int],
                    risk_scores: np.ndarray[float]) -> float:
        """
        Computes the concordance index.

        ### Parameters :
        - status (n_samples, ) : the event status for each sample (1 if event happened, 0 otherwise)
        - time (n_samples, ) : the duration before event for each sample.
        - risk_score (n_samples, ) : the risk score for each sample.

        ### Returns :
        The concordance index.
        """
        return np.round(concordance_index_censored(status, time, risk_scores)[0], 2)

    @staticmethod
    def estimate_survival_curve(status: np.ndarray[int],
                                time: np.ndarray[int]) -> tuple[np.ndarray[int], np.ndarray[float]]:
        """
        Estimates the survival curve using the Kaplan Meier Estimator.

        ### Parameters :
        - status (n_samples, ) : the event status for each sample (1 if event happened, 0 otherwise)
        - time (n_samples, ) : the duration before event for each sample.

        ### Returns :
        - x-axis : time points ;
        - y-axis : no event probability for each time point.
        """
        return kaplan_meier_estimator(status, time)

    @staticmethod
    def get_p_value_log_rank_test(status: np.ndarray[int],
                                  time: np.ndarray[int],
                                  risk_classes: np.ndarray[int]) -> float:
        """
        Executes the log rank test and return its p value.

        ### Parameters :
        - status (n_samples, ) : the event status for each sample (1 if event happened, 0 otherwise)
        - time (n_samples, ) : the duration before event for each sample
        - risk_classes (n_samples, ) : the risk class of each patient

        ### Returns :
        The log rank test p value
        """
        # Structured array for log rank tester input
        y = np.array(list(zip(status, time)), dtype=[('status', '?'), ('time surviving', '<f8')])

        return np.round(compare_survival(y, risk_classes)[1], 2)
