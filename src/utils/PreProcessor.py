import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreProcessor:
    """
    Implementation of some preprocessings.
    """
    @staticmethod
    def delete_nan_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deletes rows which contains at least 1 NaN value.

        ### Parameters :
        - df : the dataframe to update

        ### Returns :
        The dataframe without NaN values.
        """
        return df.dropna(axis=0)

    @staticmethod
    def normalize_data(X: np.ndarray[np.ndarray[float]]) -> np.ndarray[np.ndarray[float]]:
        """
        Normalizes the numpy array using the Standard Scaler.

        ### Parameters :
        - X : 2D numpy array to normalize

        ### Returns :
        The 2D normalized numpy array
        """
        return StandardScaler().fit_transform(X)

    @staticmethod
    def relabel_patients(df: pd.DataFrame,
                         status_name: str,
                         time_name: str,
                         t: float) -> pd.DataFrame:
        """
        Relabels patients depending on the event status, the time of event, and
        the time t when we look at.

        The 4 possibles cases are :
        - status = 1 & time < t : 1 : the event occured during the window [0,t]
        - status = 1 & time > t : 0 : the event occured but after time t. So at time t, the event has not occured yet.
        - status = 0 & time < t : x : we don't know what happened between time and t. We drop these censored patients
        - status = 0 & time > t : 0 : we know during the the window [0,t], the event has not occured.

        ### Parameters :
        - df : the dataframe to update
        - status_name : the name of the status event column in the dataframe
        - time_name : the name of the time event column in the dataframe
        - t : the time when we look at.

        ### Returns :
        The dataframe relabelled
        """
        # Drop censored patients
        to_drop = df.index[
            np.where((df[status_name] == 0) & (df[time_name] < t))[0]]
        df = df.drop(to_drop, axis=0)

        # Relabel patients. If time < t, 1, otherwise 0
        df[status_name] = np.where(df[time_name] < t, 1, 0)

        return df
