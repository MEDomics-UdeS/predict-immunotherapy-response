from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import from_networkx

from src.evaluation.SurvivalMetrics import SurvivalMetrics
from src.manage.CoxTrainTestManager import CoxTrainTestManager
from src.manage.GNNCoxTrainTestManager import GNNCoxTrainTestManager
from src.utils.FeatureSelector import FeatureSelector
from src.utils.FeaturesNames import get_features_names
from src.utils.PreProcessor import PreProcessor


def argument_parser():
    """
    Creates the command line parser to execute the pipeline.
    """
    # Create parser without argument
    parser = ArgumentParser()

    # Add sigmut argument
    parser.add_argument('--sigmut',
                        type=str,
                        default='comb',
                        choices=['no-sigmut', 'only-sigmut-sbs', 'only-sigmut-indel', 'only-sigmut-comb', 'comb'],
                        help='Integration of mutational signatures')

    # Add architecture argument
    parser.add_argument('--architecture',
                        type=str,
                        default='cox',
                        choices=['cox', 'gcn', 'gat'],
                        help="Model architecture")

    # Add n_features argument
    parser.add_argument('--n_features',
                        type=int,
                        default=5,
                        help='Number of features')

    # Add n_epochs argument
    parser.add_argument('--n_epochs',
                        type=int,
                        default=150,
                        help='Number of epochs')

    # Add learning rate argument
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help='Learning rate')

    # Add regularization rate argument
    parser.add_argument('--reg',
                        type=float,
                        default=0.005,
                        help='Regularization factor')

    # Add max_neighbors argument (for GNN)
    parser.add_argument('--max_neighbors',
                        type=int,
                        default=2,
                        help='Max of neighbors per node (for GNN)')

    # Add quantile argument
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Quantile used as cutoff between high risk and low risk patients.')

    arguments = parser.parse_args()

    return arguments


def Preprocess() -> tuple[pd.DataFrame,
                          np.ndarray[int],
                          np.ndarray[tuple[int, float]],
                          np.ndarray[int],
                          np.ndarray[tuple[int, float]]]:
    """Reads the dataset, drops non naive patients and NaN values, relabels patients, and normalizes initial biomarkers.

    Args:
        None

    Returns:
        The preprocessed dataframe containing the dataset.
        The label of each patient for classification for TTP
        The label of each patient for survival analysis for TTP
        The label of each patient for classification for OS
        The label of each patient for survival analysis for OS
    """
    # Reading dataset
    df = pd.read_excel('data/cohort-dataset.xlsx')

    # Drop non naive patients
    df = df.loc[df["Cohort"] == "Naive"]

    # Drop NaN values
    df = PreProcessor.delete_nan_values(df)

    # Relabel patients (t = 183 days = 6 months)
    t = 183
    df = PreProcessor.relabel_patients(df, "Progression_1", "Time to progression (days)", t)
    df = PreProcessor.relabel_patients(df, "Alive_0", "Overall survival (days)", t)

    # Normalize initial biomarkers
    features_to_normalize = ["Age at advanced disease diagnosis",
                             "CD8+ T cell score",
                             "Genome mut per mb",
                             "Exome mut per mb",
                             "CD274 expression",
                             "M1M2 expression"]

    df.loc[:, features_to_normalize] = PreProcessor.normalize_data(df.loc[:, features_to_normalize])

    # Extract labels
    # TTP :
    y_clf_ttp = df["Progression_1"].to_numpy()
    y_cox_ttp = np.array(list((df[['Progression_1', 'Time to progression (days)']].itertuples(index=False, name=None))),
                         dtype=[('Progression_1', '?'), ('Time to progression (days)', '<f8')])

    # OS :
    y_clf_os = df["Alive_0"].to_numpy()
    y_cox_os = np.array(list((df[['Alive_0', 'Overall survival (days)']].itertuples(index=False, name=None))),
                        dtype=[('Alive_0', '?'), ('Overall survival (days)', '<f8')])

    return df, y_clf_ttp, y_cox_ttp, y_clf_os, y_cox_os


def SelectFeatures(df: pd.DataFrame,
                   y_clf_ttp: np.ndarray[int],
                   y_clf_os: np.ndarray[int],
                   features_name: list[str],
                   n_features: int) -> list[str]:
    """Selects the n_features most important features, using the Random Forest feature importance.

    Args:
        df (n_samples, n_dataset) : the dataframe containing the whole dataset.
        y_clf_ttp (n_samples, ) : the label of each patient, for TTP.
        y_clf_os (n_samples, ) : the label of each patient, for OS.
        features_name : the name of each feature.
        n_features : the number of features to select.

    Returns:
        The list of features names to select, which are the most important, for TTP.
        The list of features names to select, which are the most important, for OS.
    """
    # Compute feature importance
    # TTP :
    features_name_ttp = FeatureSelector.feature_importance(df.loc[:, features_name], y_clf_ttp, False)
    features_name_os = FeatureSelector.feature_importance(df.loc[:, features_name], y_clf_os, False)

    # Select the most n_features important features
    if n_features < len(features_name):
        features_name_ttp = features_name_ttp[:n_features]
        features_name_os = features_name_os[:n_features]

    return features_name_ttp, features_name_os


def main() -> None:
    """
    Executes the whole pipeline, from reading to testing.
    """
    # Parse arguments
    args = argument_parser()
    sigmut = args.sigmut
    architecture = args.architecture
    n_features = args.n_features
    n_epochs = args.n_epochs
    lr = args.lr
    reg = args.reg
    max_neighbors = args.max_neighbors
    q = args.threshold

    assert q > 0 and q < 1, "Quantile must be between 0 and 1."

    # 1 : READING AND PREPROCESSING
    df, y_clf_ttp, y_cox_ttp, y_clf_os, y_cox_os = Preprocess()

    # 2 : FEATURE SELECTION
    dico_features_names = get_features_names()
    features_name = dico_features_names[sigmut]
    features_name_ttp, features_name_os = SelectFeatures(df, y_clf_ttp, y_clf_os, features_name, n_features)

    # 3 : EXECUTE LEAVE ONE OUT CROSS VALIDATION
    # Extract features
    X_ttp = df.loc[:, features_name_ttp].to_numpy()
    X_os = df.loc[:, features_name_os].to_numpy()

    print("Start leave one out CV...")
    # Instanciate train-test manager and make the leave one out CV
    if architecture == "cox":
        # TTP :
        manager_ttp = CoxTrainTestManager()
        risk_scores_ttp, risk_classes_ttp = manager_ttp.leave_one_out_cv(X_ttp, y_cox_ttp, q)
        # OS :
        manager_os = CoxTrainTestManager()
        risk_scores_os, risk_classes_os = manager_os.leave_one_out_cv(X_os, y_cox_os, q)

    else:
        group = df["Tumour type"].to_numpy()
        # TTP :
        manager_ttp = GNNCoxTrainTestManager(architecture)
        risk_scores_ttp, risk_classes_ttp, nx_graph_ttp = manager_ttp.leave_one_out_cv(X_ttp,
                                                                                       y_clf_ttp,
                                                                                       y_cox_ttp,
                                                                                       group,
                                                                                       n_epochs,
                                                                                       lr,
                                                                                       reg,
                                                                                       max_neighbors,
                                                                                       q)

        # OS :
        manager_os = GNNCoxTrainTestManager(architecture)
        risk_scores_os, risk_classes_os, nx_graph_os = manager_os.leave_one_out_cv(X_os,
                                                                                   y_clf_os,
                                                                                   y_cox_os,
                                                                                   group,
                                                                                   n_epochs,
                                                                                   lr,
                                                                                   reg,
                                                                                   max_neighbors,
                                                                                   q)

    print("Finished leave one out CV !")
    print("Start computing evaluation metrics...")

    # 4 : EVALUATE PERFORMANCES
    status_ttp, time_ttp = df['Progression_1'].to_numpy().astype(bool), df['Time to progression (days)'].to_numpy()
    status_os, time_os = df['Alive_0'].to_numpy().astype(bool), df['Overall survival (days)'].to_numpy()

    # C-index
    c_index_ttp = SurvivalMetrics.get_c_index(status_ttp, time_ttp, risk_scores_ttp)
    c_index_os = SurvivalMetrics.get_c_index(status_os, time_os, risk_scores_os)

    # Log rank test p value
    p_value_ttp = SurvivalMetrics.get_p_value_log_rank_test(status_ttp.astype(int), time_ttp, risk_classes_ttp)
    p_value_os = SurvivalMetrics.get_p_value_log_rank_test(status_os.astype(int), time_os, risk_classes_os)

    # Survival curves
    # TTP, low risk :
    status_low_risk_ttp = status_ttp[np.where(risk_classes_ttp == 0)]
    time_low_risk_ttp = time_ttp[np.where(risk_classes_ttp == 0)]
    time_axis_low_risk_ttp, prob_axis_low_risk_ttp = SurvivalMetrics.estimate_survival_curve(status_low_risk_ttp,
                                                                                             time_low_risk_ttp)
    # TTP, high risk :
    status_high_risk_ttp = status_ttp[np.where(risk_classes_ttp == 1)]
    time_high_risk_ttp = time_ttp[np.where(risk_classes_ttp == 1)]
    time_axis_high_risk_ttp, prob_axis_high_risk_ttp = SurvivalMetrics.estimate_survival_curve(status_high_risk_ttp,
                                                                                               time_high_risk_ttp)

    # OS, low risk :
    status_low_risk_os = status_os[np.where(risk_classes_os == 0)]
    time_low_risk_os = time_os[np.where(risk_classes_os == 0)]
    time_axis_low_risk_os, prob_axis_low_risk_os = SurvivalMetrics.estimate_survival_curve(status_low_risk_os,
                                                                                           time_low_risk_os)

    # OS, high risk :
    status_high_risk_os = status_os[np.where(risk_classes_os == 1)]
    time_high_risk_os = time_os[np.where(risk_classes_os == 1)]
    time_axis_high_risk_os, prob_axis_high_risk_os = SurvivalMetrics.estimate_survival_curve(status_high_risk_os,
                                                                                             time_high_risk_os)
    fig = plt.figure()

    # 5 : PLOT RESULTS
    # Survival curves
    # TTP :
    ax0 = fig.add_subplot(121)
    ax0.step(time_axis_low_risk_ttp, prob_axis_low_risk_ttp, where='post', color='green', label='low risk')
    ax0.step(time_axis_high_risk_ttp, prob_axis_high_risk_ttp, where='post', color='red', label='high risk')
    ax0.set_xlabel('Time (days)')
    ax0.set_ylabel('Proportion no progression')
    ax0.set_title(f'{architecture};{sigmut}; % no progression through days')
    ax0.text(0, 0.1, f'C-index : {c_index_ttp}')
    ax0.text(0, 0.05, f'p value : {p_value_ttp}')
    ax0.set_ylim(0, 1)
    ax0.legend()

    # OS :
    ax1 = fig.add_subplot(122)
    ax1.step(time_axis_low_risk_os, prob_axis_low_risk_os, where='post', color='green', label='low risk')
    ax1.step(time_axis_high_risk_os, prob_axis_high_risk_os, where='post', color='red', label='high risk')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Proportion survival')
    ax1.set_title(f'{architecture};{sigmut}; % survival through days')
    ax1.text(0, 0.1, f'C-index : {c_index_os}')
    ax1.text(0, 0.05, f'p value : {p_value_os}')
    ax1.set_ylim(0, 1)
    ax1.legend()

    plt.gcf().set_size_inches(30, 15)
    plt.subplots_adjust(wspace=0.4)

    plt.savefig(f"results/perfs-survivalAnalysis-{sigmut}-{architecture}.png")

    # SHAP values
    if architecture == "cox":
        # TTP :
        explainer_ttp = shap.Explainer(manager_ttp.model.model.predict, X_ttp)
        shap_values_ttp = explainer_ttp(X_ttp)

        # OS :
        explainer_os = shap.Explainer(manager_os.model.model.predict, X_os)
        shap_values_os = explainer_os(X_os)

        fig = plt.figure()

        ax0 = fig.add_subplot(121)
        shap.summary_plot(shap_values_ttp, plot_type='violin', feature_names=features_name_ttp, show=False)
        ax0.set_title("SHAP values - TTP")
        ax0.set_xlim(-1.5, 1.5)
        ax1 = fig.add_subplot(122)
        shap.summary_plot(shap_values_os, plot_type='violin', feature_names=features_name_os, show=False)
        ax1.set_title("SHAP values - OS")
        ax1.set_xlim(-1.5, 1.5)

        plt.gcf().set_size_inches(30, 15)
        plt.subplots_adjust(wspace=0.4)

        plt.savefig(f"results/shap-survivalAnalysis-{sigmut}-{architecture}.png")

    else:
        # SHAP values
        pyg_graph_ttp = from_networkx(nx_graph_ttp)
        pyg_graph_os = from_networkx(nx_graph_os)

        # TTP :
        explainer_ttp = Explainer(
            model=manager_ttp.gnn_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        explanation_ttp = explainer_ttp(pyg_graph_ttp.x, pyg_graph_ttp.edge_index)

        # OS :
        explainer_os = Explainer(
            model=manager_os.gnn_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        explanation_os = explainer_os(pyg_graph_os.x, pyg_graph_os.edge_index)

        explanation_ttp.visualize_feature_importance(
            path=f"results/shap-survivalAnalysis-{sigmut}-{architecture}-ttp.png",
            feat_labels=features_name_ttp)
        explanation_os.visualize_feature_importance(
            path=f"results/shap-survivalAnalysis-{sigmut}-{architecture}-os.png",
            feat_labels=features_name_os)

        # Vizualize graph
        fig, ax = plt.subplots()
        nx.draw(nx_graph_ttp)
        plt.savefig(f"results/graph-survivalAnalysis-{architecture}-ttp.png")
        fig, ax = plt.subplots()
        nx.draw(nx_graph_os)
        plt.savefig(f"results/graph-survivalAnalysis-{architecture}-os.png")

    print("Finished computing evaluation metrics !")


if __name__ == '__main__':
    main()
