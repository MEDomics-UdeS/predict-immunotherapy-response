import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.utils import from_networkx

from src.models.GATClassifier import GATClassifier
from src.models.GCNClassifier import GCNClassifier
from src.models.GraphBuilder import GraphBuilder


class GNNClassifierTrainTestManager:
    """
    Train-test manager for Graph Neural Network classification model.
    """
    def __init__(self, architecture: str) -> None:
        """
        GNNClassifierTrainTestManager class builder.

        ### Parameters :
        - architecture ('gcn' or 'gat'): GNN architecture : gcn or gat

        ### Returns :
        None
        """
        self.model = None
        self.architecture = architecture

    def train(self,
              nx_graph: nx.DiGraph,
              n_epochs: int,
              lr: float,
              reg: float,
              train_index: list[int]) -> tuple[list[float], list[float]]:
        """
        Trains the model for n_epochs with 80% train 20% validation.

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)

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
            out_train = self.model.forward(pyg_graph_train.x, pyg_graph_train.edge_index)

            # Compute train loss
            loss_train = loss_function(out_train, pyg_graph_train.y)

            # Add train loss to train_loss list
            train_loss_list.append(loss_train.item())

            # Backward pass (gradients computation)
            loss_train.backward()

            # Update parameters
            optimizer.step()

            # Forward pass on validation set
            out_val = self.model.forward(pyg_graph.x, pyg_graph.edge_index)[index_val]

            # Compute validation loss
            loss_val = loss_function(out_val, pyg_graph.y[index_val])

            # Add validation loss to val_loss list
            val_loss_list.append(loss_val.item())

        return train_loss_list, val_loss_list

    def leave_one_out_cv(self,
                         X: np.ndarray[np.ndarray[float]],
                         y: np.ndarray[int],
                         group: np.ndarray[int | str],
                         n_epochs: int,
                         lr: float,
                         reg: float,
                         max_neighbors: int) -> tuple[np.ndarray[float],
                                                      np.ndarray[int],
                                                      list[float],
                                                      list[float],
                                                      nx.DiGraph]:
        """
        Executes the leave one out cross validation to find test scores and
        labels.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing the features of each sample
        - y (n_samples,) : numpy array containing the label of each sample
        - group (n_samples, ) : numpy array containing the group of each sample
        - n_epochs : the number of epochs.
        - lr : the learning rate for the gradient descent
        - reg : the regularization factor in optimizer
        - max_neighbors : the maximum number of neighbors per sample

        ### Returns :
        - test_scores (n_samples, ) : numpy array containing the test score of each sample
        - test_classes (n_samples, ) : numpy array containing the test class of each sample
        - train_losses (n_samples, n_epochs) : list containing the train loss for each sample and epoch
        - val_losses (n_samples, n_epochs) : list containing the validation loss for each sample and epoch
        - nx_graph : the networkx graph used for training containing the features and the label of each sample, and the
        graph connectivity
        """

        # Split dataframe in n_samples groups
        n_samples, n_features = X.shape
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        # Initialize test scores and classes arrays
        test_scores = np.zeros(X.shape[0])
        test_classes = np.zeros(X.shape[0])

        # Initialize train losses and validation losses lists
        train_losses, val_losses = [], []

        # Build networkx graph with pruning
        distance_matrix = euclidean_distances(X)
        build_graph = GraphBuilder(X, y, group)
        build_graph.build_graph(distance_matrix, max_neighbors, True)
        nx_graph = build_graph.nx_graph

        for i, (train_index, test_index) in enumerate(folds):

            # Build train graph with removing test nodes
            nx_graph_train = nx_graph.copy()
            for test in test_index:
                nx_graph_train.remove_node(test)

            # Instanciate model
            if self.architecture == "gcn":
                self.model = GCNClassifier(n_features)
            elif self.architecture == "gat":
                self.model = GATClassifier(n_features)
            else:
                raise ValueError("Invalid value of architecture. The valid choices are gcn and gat")

            # Training in train set
            train_loss, val_loss = self.train(nx_graph_train,
                                              n_epochs,
                                              lr,
                                              reg,
                                              train_index)

            # Add train and validation loss to lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Convert test networkx graph to PyTorch Geometric graph
            pyg_graph = from_networkx(nx_graph)

            # Forward pass on test set
            score_test = self.model.forward(pyg_graph.x,
                                            pyg_graph.edge_index).detach().numpy().reshape((1, -1))[0]

            class_test = self.model.predict_class(pyg_graph.x,
                                                  pyg_graph.edge_index).detach().numpy().reshape((1, -1))[0]

            # Add score and class to scores and classes arrays
            test_scores[test_index] = score_test[test_index]
            test_classes[test_index] = class_test[test_index]

        return test_scores, test_classes, train_losses, val_losses, nx_graph
