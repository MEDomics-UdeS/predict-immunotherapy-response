import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class featureSelection:
    """
    Implementation of features correlation and features selection.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def correlation(df: pd.DataFrame,
                    plot_heatmap: bool = False) -> pd.DataFrame:
        """
        Compute the Pearson correlation between each column.

        ### Parameters :
        - df : the dataframe to analyze
        - plot_heatmap (default False) : if True, print a heatmap of the correlation matrix.

        ### Returns :
        The correlation matrix.
        """
        # Compute the correlation matrix
        correl = np.round(df.corr(), 2)

        # Plot the heatmap
        if plot_heatmap:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax = sns.heatmap(correl,
                             linewidths=1,
                             square=True,
                             annot=False,
                             cmap=sns.color_palette("Spectral",
                                                    as_cmap=True))
            ax.set_title("Correlation between features")
            plt.show()
        return np.round(df.corr(), 2)

    @staticmethod
    def feature_importance(df: pd.DataFrame,
                           y: np.ndarray[int],
                           plot_hist: bool = False) -> np.ndarray[float]:
        """
        Compute the feature importance of the dataframe attributes using the Random Forest Feature Importance.

        ### Parameters :
        - df : the dataframe to analyze
        - y : the label data for df
        - plot_hist (default False) : if True, print a histogram of each feature importance.

        ### Returns :
        - biomarkers_per_importance (n_features, ) : feature names sorted per importance
        """
        # Fitting the RF classifier and compute the feature importances
        rf_clf = RandomForestClassifier(oob_score=True)
        rf_clf.fit(df, y)
        feat_importance = rf_clf.feature_importances_

        # Sort the result
        indices_feat_importance = feat_importance.argsort()
        feat_importance_sorted = feat_importance[indices_feat_importance]
        biomarkers_per_importance = np.array(df.columns)[indices_feat_importance]

        # Plot the result
        if plot_hist:
            plt.figure(figsize=(10, 15))
            plt.barh(biomarkers_per_importance, feat_importance_sorted)
            plt.xlabel("Importance")
            plt.title("Importance of each feature")
            plt.show()

        # First values are the most important features
        biomarkers_per_importance = np.flip(biomarkers_per_importance)

        return biomarkers_per_importance
