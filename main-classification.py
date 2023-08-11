import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
import torch
from argparse import ArgumentParser
from evaluation.ClassificationMetrics import ClassificationMetrics
from manage.LogisticRegressionTrainTestManager import LogisticRegressionTrainTestManager
from manage.GNNClassifierTrainTestManager import GNNClassifierTrainTestManager
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
                        default='no-sigmut',
                        choices=['no-sigmut', 'only-sigmut-sbs', 'only-sigmut-indel', 'only-sigmut-comb', 'comb'],
                        help='Integration of mutational signatures')

    # Add architecture argument
    parser.add_argument('--architecture',
                        type=str,
                        default='logistic-regression',
                        choices=['logistic-regression', 'gcn', 'gat'],
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
    df = preProcessing.relabel_patients(df,
                                        "Progression_1",
                                        "Time to progression (days)",
                                        t)

    # Normalize initial biomarkers
    features_to_normalize = ["Age at advanced disease diagnosis",
                             "CD8+ T cell score",
                             "Genome mut per mb",
                             "Exome mut per mb",
                             "CD274 expression",
                             "M1M2 expression"]

    df.loc[:, features_to_normalize] = preProcessing.normalize_data(df.loc[:, features_to_normalize])

    # Extract labels
    y = df["Progression_1"].to_numpy()

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
    features_name = featureSelection.feature_importance(df.loc[:, features_name],
                                                        y,
                                                        False)

    # Select the most n_features important features
    if n_features < len(features_name):
        features_name = features_name[:n_features]

    # 3 : EXECUTE LEAVE ONE OUT CROSS VALIDATION

    # Extract features
    X = df.loc[:, features_name].to_numpy()

    print("Start leave one out CV...")
    # Instanciate train-test manager and make the leave one out CV
    if architecture == "logistic-regression":
        manager = LogisticRegressionTrainTestManager()
        scores, classes, train_loss, val_loss = manager.leave_one_out_cv(X,
                                                                         y,
                                                                         n_epochs,
                                                                         lr,
                                                                         reg)

    elif architecture == "gcn" or "gat":
        manager = GNNClassifierTrainTestManager(architecture)
        group = df["Tumour type"].to_numpy()
        scores, classes, train_loss, val_loss, nx_graph = manager.leave_one_out_cv(X,
                                                                                   y,
                                                                                   group,
                                                                                   n_epochs,
                                                                                   lr,
                                                                                   reg,
                                                                                   max_neighbors)

    else:
        raise ValueError("Invalid value of architecture. The valid choices are gcn and gat")

    # Write output in .xlsx file, for LogReg and sigmut + other case
    if architecture == 'logistic-regression' and sigmut == 'comb':
        df["LogReg output sigmut+others"] = scores
        df.to_excel("data/cohort-dataset-with-logreg-output.xlsx")

    print("Finished !")

    # 4 : EVALUATE PERFORMANCES

    # Classification metrics
    fpr, tpr, thresholds = ClassificationMetrics.compute_roc_curve(y,
                                                                   scores)
    sensitivity, specificity = ClassificationMetrics.compute_sensitivity_specificity(y,
                                                                                     classes)
    auc = ClassificationMetrics.compute_auc(y,
                                            scores)

    if architecture == "logistic-regression":
        # SHAP values
        X_torch = torch.from_numpy(X).float()
        explainer = shap.DeepExplainer(manager.model,
                                       X_torch)
        shap_values = explainer.shap_values(X_torch)
        expected_values = explainer.expected_value

        # 5 : PLOT RESULTS
        fig = plt.figure()
        x = np.linspace(0, 1, 100)

        ax0 = fig.add_subplot(121)
        ax0.plot(fpr, tpr, color='red', label=sigmut)
        ax0.plot(x, x, linestyle='--', color='black', label='y=x')
        ax0.set_xlabel("FPR")
        ax0.set_ylabel("TPR")
        ax0.set_title(f"{architecture} ; n_features={n_features} ; n_epochs = {n_epochs} ; lr={lr} ; reg={reg}")
        ax0.text(0, 0.8, f"sensitivity : {sensitivity}")
        ax0.text(0, 0.7, f"specificity : {specificity}")
        ax0.text(0, 0.6, f"AUC : {auc}")
        ax0.legend()

        ax1 = fig.add_subplot(122)
        shap.decision_plot(expected_values, shap_values, features_name, show=False)
        ax1.set_title("SHAP values")
        ax1.set_xlim(0, 1)

        plt.gcf().set_size_inches(30, 15)
        plt.subplots_adjust(wspace=0.4)
        plt.savefig(f"results/perfs-classif-{sigmut}-{architecture}.png")

    else:
        # SHAP values
        pyg_graph = from_networkx(nx_graph)
        explainer = Explainer(
            model=manager.model,
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

        # Generate explanation for the node at index `10`:
        explanation = explainer(pyg_graph.x, pyg_graph.edge_index)

        fig = plt.figure()
        x = np.linspace(0, 1, 100)

        ax0 = fig.add_subplot(111)
        ax0.plot(fpr, tpr, color='red', label=sigmut)
        ax0.plot(x, x, linestyle='--', color='black', label='y=x')
        ax0.set_xlabel("FPR")
        ax0.set_ylabel("TPR")
        ax0.set_title(f"{architecture} ; n_features={n_features} ; n_epochs = {n_epochs} ; lr={lr} ; reg={reg}")
        ax0.text(0, 0.8, f"sensitivity : {sensitivity}")
        ax0.text(0, 0.7, f"specificity : {specificity}")
        ax0.text(0, 0.6, f"AUC : {auc}")
        ax0.legend()

        plt.savefig(f"results/perfs-classif-{sigmut}-{architecture}.png")

        explanation.visualize_feature_importance(path=f"results/shap-classif-{sigmut}-{architecture}.png",
                                                 feat_labels=features_name)

        fig, ax = plt.subplots()
        nx.draw(nx_graph)
        plt.savefig(f"results/graph-classif-{architecture}.png")


if __name__ == '__main__':
    main()
