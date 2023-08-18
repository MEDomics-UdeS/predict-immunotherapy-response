from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import from_networkx

from src.evaluation.ClassificationMetrics import ClassificationMetrics
from src.manage.GNNClassifierTrainTestManager import GNNClassifierTrainTestManager
from src.manage.LogisticRegressionTrainTestManager import LogisticRegressionTrainTestManager
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


def Preprocess() -> tuple[pd.DataFrame, np.ndarray[int]]:
    """
    Reads the dataset, drops non naive patients and NaN values, relabels patients, and normalizes initial biomarkers.

    ### Parameters :
    None

    ### Returns :
    - The preprocessed dataframe containing the dataset.
    - The label of each patient.
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

    # Normalize initial biomarkers
    features_to_normalize = ["Age at advanced disease diagnosis",
                             "CD8+ T cell score",
                             "Genome mut per mb",
                             "Exome mut per mb",
                             "CD274 expression",
                             "M1M2 expression"]

    df.loc[:, features_to_normalize] = PreProcessor.normalize_data(df.loc[:, features_to_normalize])

    # Extract labels
    y = df["Progression_1"].to_numpy()

    return df, y


def SelectFeatures(df: pd.DataFrame,
                   y: np.ndarray[int],
                   features_name: list[str],
                   n_features: int) -> list[str]:
    """
    Selects the n_features most important features, using the Random Forest feature importance.

    ### Parameters :
    - df (n_samples, n_dataset) : the dataframe containing the whole dataset.
    - y (n_samples, ) : the label of each patient.
    - features_name : the name of each feature.
    - n_features : the number of features to select.

    ### Returns :
    The list of features names to select, which are the most important.
    """
    # Compute feature importance
    features_name = FeatureSelector.feature_importance(df.loc[:, features_name], y, False)

    # Select the most n_features important features
    if n_features < len(features_name):
        features_name = features_name[:n_features]

    return features_name


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

    # 1 : READING AND PREPROCESSING
    df, y = Preprocess()

    # 2 : FEATURE SELECTION
    dico_features_names = get_features_names()
    features_name = dico_features_names[sigmut]
    features_name = SelectFeatures(df, y, features_name, n_features)

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

    else:
        manager = GNNClassifierTrainTestManager(architecture)
        group = df["Tumour type"].to_numpy()
        scores, classes, train_loss, val_loss, nx_graph = manager.leave_one_out_cv(X,
                                                                                   y,
                                                                                   group,
                                                                                   n_epochs,
                                                                                   lr,
                                                                                   reg,
                                                                                   max_neighbors)

    # Write output in .xlsx file, for LogReg and sigmut + other case
    if architecture == 'logistic-regression' and sigmut == 'comb':
        df["LogReg output sigmut+others"] = scores
        df.to_excel("data/cohort-dataset-with-logreg-output.xlsx")

    print("Finished leave one out CV !")
    print("Start computing evaluation metrics...")

    # 4 : EVALUATE PERFORMANCES
    # Classification metrics
    fpr, tpr = ClassificationMetrics.compute_roc_curve(y, scores)
    sensitivity, specificity = ClassificationMetrics.compute_sensitivity_specificity(y, classes)
    auc = ClassificationMetrics.compute_auc(y, scores)

    if architecture == "logistic-regression":
        # SHAP values
        X_torch = torch.from_numpy(X).float()
        explainer = shap.DeepExplainer(manager.model, X_torch)
        shap_values = explainer.shap_values(X_torch)
        expected_values = explainer.expected_value

        # 5 : PLOT RESULTS
        fig = plt.figure()
        x = np.linspace(0, 1, 100)

        # Plot ROC curve with sensitivity, specificity, and AUC
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

        # Plot SHAP values
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

        # Generate explanation all patients
        explanation = explainer(pyg_graph.x, pyg_graph.edge_index)

        # Plot ROC curve with printing sensitivity, specificity, and AUC
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

        # Plot SHAP values
        explanation.visualize_feature_importance(path=f"results/shap-classif-{sigmut}-{architecture}.png",
                                                 feat_labels=features_name)

        # Plot graph
        fig, ax = plt.subplots()
        nx.draw(nx_graph)
        plt.savefig(f"results/graph-classif-{architecture}.png")

    print("Finished computing evaluation metrics !")


if __name__ == '__main__':
    main()
