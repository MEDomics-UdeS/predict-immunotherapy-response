import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from argparse import ArgumentParser
from evaluation.SurvivalMetrics import SurvivalMetrics
from manage.CoxTrainTestManager import CoxTrainTestManager
from manage.GNNCoxTrainTestManager import GNNCoxTrainTestManager
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import from_networkx
from utils.featureSelection import featureSelection
from utils.preProcessing import preProcessing


def argument_parser():
    """
    Create the command line parser to execute the pipeline.
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

    arguments = parser.parse_args()

    return arguments


def main() -> None:
    """
    Execute the whole pipeline, from reading to testing.
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

    # 1 : READING AND PREPROCESSING
    # Reading dataset
    df = pd.read_excel('data/cohort-dataset.xlsx')

    # Drop non naive patients
    df = df.loc[df["Cohort"] == "Naive"]

    # Drop NaN values
    df = preProcessing.delete_nan_values(df)

    # Relabel patients (t = 183 days = 6 months)
    t = 183
    df = preProcessing.relabel_patients(df, "Progression_1", "Time to progression (days)", t)
    df = preProcessing.relabel_patients(df, "Alive_0", "Overall survival (days)", t)

    # Normalize initial biomarkers
    features_to_normalize = ["Age at advanced disease diagnosis",
                             "CD8+ T cell score",
                             "Genome mut per mb",
                             "Exome mut per mb",
                             "CD274 expression",
                             "M1M2 expression"]

    df.loc[:, features_to_normalize] = preProcessing.normalize_data(df.loc[:, features_to_normalize])

    # Extract labels
    # TTP :
    y_clf_ttp = df["Progression_1"].to_numpy()
    y_cox_ttp = np.array(list((df[['Progression_1', 'Time to progression (days)']].itertuples(index=False, name=None))),
                         dtype=[('Progression_1', '?'), ('Time to progression (days)', '<f8')])

    # OS :
    y_clf_os = df["Alive_0"].to_numpy()
    y_cox_os = np.array(list((df[['Alive_0', 'Overall survival (days)']].itertuples(index=False, name=None))),
                        dtype=[('Alive_0', '?'), ('Overall survival (days)', '<f8')])

    # 2 : FEATURE SELECTION
    if sigmut == "no-sigmut":
        features_name = ["Age at advanced disease diagnosis",
                         "CD8+ T cell score",
                         "Genome mut per mb",
                         "Exome mut per mb",
                         "CD274 expression",
                         "M1M2 expression"]

    elif sigmut == "only-sigmut-sbs":
        features_name = ["SBS1",
                         "SBS2",
                         "SBS3",
                         "SBS4",
                         "SBS5",
                         "SBS7a",
                         "SBS7b",
                         "SBS7c",
                         "SBS7d",
                         "SBS8",
                         "SBS10a",
                         "SBS10b",
                         "SBS10c",
                         "SBS13",
                         "SBS15",
                         "SBS17a",
                         "SBS17b",
                         "SBS18",
                         "SBS31",
                         "SBS35",
                         "SBS36",
                         "SBS37",
                         "SBS38",
                         "SBS40",
                         "SBS44",
                         "SBS4426"]

    elif sigmut == "only-sigmut-indel":
        features_name = ["ID1",
                         "ID2",
                         "ID3",
                         "ID4",
                         "ID5",
                         "ID6",
                         "ID7",
                         "ID8",
                         "ID9",
                         "ID10",
                         "ID11",
                         "ID12",
                         "ID13",
                         "ID14",
                         "ID15",
                         "ID16",
                         "ID17",
                         "ID18"]

    elif sigmut == "only-sigmut-comb":
        features_name = ["SBS1",
                         "SBS2",
                         "SBS3",
                         "SBS4",
                         "SBS5",
                         "SBS7a",
                         "SBS7b",
                         "SBS7c",
                         "SBS7d",
                         "SBS8",
                         "SBS10a",
                         "SBS10b",
                         "SBS10c",
                         "SBS13",
                         "SBS15",
                         "SBS17a",
                         "SBS17b",
                         "SBS18",
                         "SBS31",
                         "SBS35",
                         "SBS36",
                         "SBS37",
                         "SBS38",
                         "SBS40",
                         "SBS44",
                         "SBS4426",
                         "ID1",
                         "ID2",
                         "ID3",
                         "ID4",
                         "ID5",
                         "ID6",
                         "ID7",
                         "ID8",
                         "ID9",
                         "ID10",
                         "ID11",
                         "ID12",
                         "ID13",
                         "ID14",
                         "ID15",
                         "ID16",
                         "ID17",
                         "ID18"]

    elif sigmut == "comb":
        features_name = ["Age at advanced disease diagnosis",
                         "CD8+ T cell score",
                         "Genome mut per mb",
                         "Exome mut per mb",
                         "CD274 expression",
                         "M1M2 expression",
                         "SBS1",
                         "SBS2",
                         "SBS3",
                         "SBS4",
                         "SBS5",
                         "SBS7a",
                         "SBS7b",
                         "SBS7c",
                         "SBS7d",
                         "SBS8",
                         "SBS10a",
                         "SBS10b",
                         "SBS10c",
                         "SBS13",
                         "SBS15",
                         "SBS17a",
                         "SBS17b",
                         "SBS18",
                         "SBS31",
                         "SBS35",
                         "SBS36",
                         "SBS37",
                         "SBS38",
                         "SBS40",
                         "SBS44",
                         "SBS4426",
                         "ID1",
                         "ID2",
                         "ID3",
                         "ID4",
                         "ID5",
                         "ID6",
                         "ID7",
                         "ID8",
                         "ID9",
                         "ID10",
                         "ID11",
                         "ID12",
                         "ID13",
                         "ID14",
                         "ID15",
                         "ID16",
                         "ID17",
                         "ID18"]

    else:
        raise ValueError("Invalid name of problem. The valid choices are : no-sigmut, only-sigmut, and comb.")

    # Compute feature importance
    # TTP :
    features_name_ttp = featureSelection.feature_importance(df.loc[:, features_name], y_clf_ttp, False)
    features_name_os = featureSelection.feature_importance(df.loc[:, features_name], y_clf_os, False)

    # Select the most n_features important features
    if n_features < len(features_name):
        features_name_ttp = features_name_ttp[:n_features]
        features_name_os = features_name_os[:n_features]

    # 3 : EXECUTE LEAVE ONE OUT CROSS VALIDATION
    # Extract features
    X_ttp = df.loc[:, features_name_ttp].to_numpy()
    X_os = df.loc[:, features_name_os].to_numpy()

    print("Start leave one out CV...")
    # Instanciate train-test manager and make the leave one out CV
    if architecture == "cox":
        # TTP :
        manager_ttp = CoxTrainTestManager()
        risk_scores_ttp, risk_classes_ttp = manager_ttp.leave_one_out_cv(X_ttp, y_cox_ttp)
        # OS :
        manager_os = CoxTrainTestManager()
        risk_scores_os, risk_classes_os = manager_os.leave_one_out_cv(X_os, y_cox_os)

    elif architecture == "gcn" or "gat":
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
                                                                                       max_neighbors)

        # OS :
        manager_os = GNNCoxTrainTestManager(architecture)
        risk_scores_os, risk_classes_os, nx_graph_os = manager_os.leave_one_out_cv(X_os,
                                                                                   y_clf_os,
                                                                                   y_cox_os,
                                                                                   group,
                                                                                   n_epochs,
                                                                                   lr,
                                                                                   reg,
                                                                                   max_neighbors)

    else:
        raise ValueError("Invalid value of architecture. The valid choices are gcn and gat")

    print("Finished !")

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
    ax0.set_title(f'{architecture} ; % no progression along days')
    ax0.text(0, 0.1, f'C-index : {c_index_ttp}')
    ax0.text(0, 0.05, f'p value : {p_value_ttp}')
    ax0.set_ylim(0, 1)
    ax0.legend()

    # OS :
    ax1 = fig.add_subplot(122)
    ax1.step(time_axis_low_risk_os, prob_axis_low_risk_os, where='post', color='green', label='low risk')
    ax1.step(time_axis_high_risk_os, prob_axis_high_risk_os, where='post', color='red', label='high risk')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Proportion surviving progression')
    ax1.set_title(f'{architecture} ; % surviving along days')
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


if __name__ == '__main__':
    main()
