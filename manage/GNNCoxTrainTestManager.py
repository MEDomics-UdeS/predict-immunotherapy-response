import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import KFold
from models.GraphBuilder import GraphBuilder
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from sklearn.metrics import euclidean_distances
from models.GCNClassifier import GCNClassifier
from models.GATClassifier import GATClassifier
from models.CoxModel import CoxModel


class GNNCoxTrainTestManager:
    """
    Train-test manager for gnn Cox Model.
    """
    def __init__(self, gnn_architecture: str) -> None:
        """
        gnnCoxTrainTestManager class builder.

        ### Parameters :
        - gnn_architecture ('gcn' or 'gat') : the type of GNN architecture

        ### Returns :
        None
        """
        self.gnn_model = None
        self.cox_model = None
        self.gnn_architecture = gnn_architecture

    def train_gnn(self,
                  nx_graph: nx.DiGraph,
                  n_epochs: int,
                  lr: float,
                  reg: float,
                  train_index: list[int]) -> tuple[list[float], list[float]]:
        """
        Trains the GNN model for n_epochs with 80% train 20% validation.

        ### Parameters :
        - nx_graph : the networkx graph containing the features and the label of each sample, and the graph connectivity
        - n_epochs : the number of epochs.
        - lr : the learning rate for the gradient descent
        - reg : the regularization factor in optimizer
        - train_index : the dataset index of each train sample

        ### Returns :
        - train_loss (n_epochs, ) : list containing the train loss for each epoch
        - val_loss (n_epochs, ) : list containing the validation loss for each epoch
        """

        # Initialize train loss and validation loss lists
        train_loss_list, val_loss_list = [], []

        # Define loss function and optimizer
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=lr, weight_decay=reg)

        # Set train-validation set
        index_train, index_val, node_train, node_val = train_test_split([i for i in range(len(train_index))],
                                                                        train_index,
                                                                        test_size=0.2)

        # Build train graph with removing validation nodes
        nx_graph_train = nx_graph.copy()
        for val in node_val:
            nx_graph_train.remove_node(val)

        # Convert Networkx graph to PyTorch geometric graph
        pyg_graph_train = from_networkx(nx_graph_train)

        # Convert validation networkx graph to PyTorch Geometric graph
        pyg_graph = from_networkx(nx_graph)

        for epoch in range(n_epochs):

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass on training graph
            out_train = self.gnn_model.forward(pyg_graph_train.x, pyg_graph_train.edge_index)

            # Compute train loss
            loss_train = loss_function(out_train, pyg_graph_train.y)

            # Add train loss to train_loss list
            train_loss_list.append(loss_train.item())

            # Backward pass (gradients computation)
            loss_train.backward()

            # Update parameters
            optimizer.step()

            # Forward pass on validation set
            out_val = self.gnn_model.forward(pyg_graph.x, pyg_graph.edge_index)[index_val]

            # Compute validation loss
            loss_val = loss_function(out_val, pyg_graph.y[index_val])

            # Add validation loss to val_loss list
            val_loss_list.append(loss_val.item())

        return train_loss_list, val_loss_list

    def train_cox_model(self,
                        X: np.ndarray[np.ndarray[float]],
                        y: np.ndarray[tuple[int, float]]) -> None:
        """
        Trains the Cox Model.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing features of each sample
        - y (n_samples, ) : numpy array containing the event status and the time surviving of each sample.

        ### Returns :
        None
        """
        self.cox_model.train(X, y)

    def leave_one_out_cv(self,
                         X: np.ndarray[np.ndarray[float]],
                         y_clf: np.ndarray[int],
                         y_cox: np.ndarray[tuple[int, float]],
                         group: np.ndarray[int | str],
                         n_epochs: int,
                         lr: float,
                         reg: float,
                         max_neighbors: int) -> tuple[np.ndarray[float], np.ndarray[int], nx.DiGraph]:
        """
        Executes the leave one out cross validation to find test risk scores and risk classes.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing features of each sample
        - y_clf (n_samples, ) : Classifier labels, numpy array containing the classifier label of each patient
        - y_cox (n_samples, ) : Cox Model labels, numpy array containing the event status and the time surviving of
        each sample.
        - group (n_samples, ) : numpy array containing the group of each sample
        - n_epochs : the number of epochs.
        - lr : the learning rate for the gradient descent
        - reg : the regularization factor in optimizer
        - max_neighbors : the maximum number of neighbors per sample

        ### Returns :
        - risk_scores (n_samples, ) : numpy array containing the risk score of each sample
        - risk_classes (n_samples, ) : numpy array containing the risk class of each sample
        - nx_graph : the networkx graph used for training containing the eatures and the label of each sample, and the
        graph connectivity
        """

        # Split dataframe in n_samples groups
        n_samples, n_features = X.shape
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        # Initialize test risk scores and classes arrays
        risk_scores = np.zeros(X.shape[0])
        risk_classes = np.zeros(X.shape[0])

        # Initialize train loss and val loss lists
        train_losses, val_losses = [], []

        # Build networkx graph
        distance_matrix = euclidean_distances(X)
        build_graph = GraphBuilder(X, y_clf, group)
        build_graph.build_graph(distance_matrix, max_neighbors, True)
        nx_graph = build_graph.nx_graph

        for i, (train_index, test_index) in enumerate(folds):

            # Instanciate GNN classifier model and Cox Model
            if self.gnn_architecture == "gcn":
                self.gnn_model = GCNClassifier(n_features)
            elif self.gnn_architecture == "gat":
                self.gnn_model = GATClassifier(n_features)
            else:
                raise ValueError("Invalid value of architecture. The valid choices are gcn and gat")
            self.cox_model = CoxModel()

            # Build train graph with removing test nodes
            nx_graph_train = nx_graph.copy()
            for test in test_index:
                nx_graph_train.remove_node(test)

            # Training gnn on train set
            train_loss, val_loss = self.train_gnn(nx_graph_train,
                                                  n_epochs,
                                                  lr,
                                                  reg,
                                                  train_index)

            # Add train and validation losses to lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Convert graph to networkx format
            pyg_graph = from_networkx(nx_graph)

            # Predict new embeddings of each sample on train set
            X_cox_train = self.gnn_model.forward_conv(pyg_graph.x,
                                                      pyg_graph.edge_index).detach().numpy()[train_index]

            # Select train set labels
            y_train_cox = y_cox[train_index]

            # Training Cox Model on train set
            self.cox_model.train(X_cox_train, y_train_cox)

            # Find risk score cutoff between high risk and low risk
            risk_scores_train = self.cox_model.predict_risk_score(X_cox_train)
            risk_cutoff = self.cox_model.find_cutoff(risk_scores_train)

            # Predict new embedding of test set
            X_cox_test = self.gnn_model.forward_conv(pyg_graph.x,
                                                     pyg_graph.edge_index).detach().numpy()[test_index]

            # Predict test risk score and risk class
            risk_score_test = self.cox_model.predict_risk_score(X_cox_test)
            risk_class_test = self.cox_model.predict_class(risk_score_test, risk_cutoff)

            # Add risk score and risk class to arrays
            risk_scores[test_index] = risk_score_test
            risk_classes[test_index] = risk_class_test

        return risk_scores, risk_classes, nx_graph
