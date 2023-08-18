import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx


class GraphBuilder:
    """
    Graph builder model for Graph Neural Network. Each node of the graph represents a patient, with its features and
    label. The graph is the combination of distinct subgraphs, where each subgraph contains patient of the same
    group.
    """
    def __init__(self,
                 X: np.ndarray[np.ndarray[float]],
                 y: np.ndarray[int],
                 group: np.ndarray[int | str]) -> None:
        """
        GraphBuilder class builder.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing features of each sample
        - y (n_samples, ) : numpy array containing the label of each sample
        - group (n_samples, ) : numpy array containing the group of each sample : tumour type, or cluster.
        """
        self.X = X
        self.y = y
        self.group = group

        # Adjacency matrix
        self.adjacency_matrix = np.zeros((X.shape[0], X.shape[0]))

        # Networkx graph
        self.nx_graph = None

        # PyTorch geometric graph
        self.pyg_graph = None

    def compute_adjacency_matrix(self) -> None:
        """
        Computes the graph adjacency matrix of the graph where a connection is established between each pair of
        patients within the same group.

        ### Parameters :
        None

        ### Returns :
        None
        """
        shape_A = self.X.shape[0]

        for i in range(shape_A):
            group = self.group[i]

            # Assign 1 to patients within the same group
            self.adjacency_matrix[i] = np.where(self.group == group, 1, 0)

    def create_nx_graph_from_adjacency_matrix(self) -> None:
        """
        Creates the networkx graph from the adjacency matrix.

        ### Parameters :
        None

        ### Returns :
        None
        """
        # Formatting data to tensors
        X = torch.from_numpy(self.X).float()
        y = torch.from_numpy(self.y).float().unsqueeze(1)

        # Initialize with empty graph
        self.nx_graph = nx.Graph()

        # Add nodes, with its features and label
        for i in range(self.adjacency_matrix.shape[0]):
            self.nx_graph.add_nodes_from([(i, {"x": X[i], "y":y[i]})])

        # Add edges
        rows, cols = np.where(self.adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        self.nx_graph.add_edges_from(edges)

    def prune_graph(self,
                    distance_matrix: np.ndarray[np.ndarray[float]],
                    max_neighbors: int) -> None:
        """
        Prunes the graph with keeping maximum the max_neighbors closest samples.

        ### Parameters :
        - distance_matrix (n_samples, n_samples): numpy array containing the distance between each sample
        - max_neighbors : the maximum number of neighbors per sample

        ### Returns :
        None
        """
        for i in range(distance_matrix.shape[0]):
            # Get neighbors of node i
            neighbors_i = [n for n in self.nx_graph[i]]

            # Distances from i
            distance_i = distance_matrix[i][neighbors_i]

            # Get the number of i neighbors to drop
            number_to_drop = len(neighbors_i)-max_neighbors

            while number_to_drop > 0:

                # Get the furthest node from i
                to_drop = np.argmax(distance_i)

                # Remove edge between i and to_drop
                self.nx_graph.remove_edge(i, neighbors_i[to_drop])

                # Update adjacency matrix
                self.adjacency_matrix = nx.to_numpy_array(self.nx_graph)

                # Update neighbors list, distance_i, to_drop
                neighbors_i = [n for n in self.nx_graph[i]]
                distance_i = distance_matrix[i][neighbors_i]
                number_to_drop = len(neighbors_i) - max_neighbors

    def build_graph(self,
                    distance_matrix: np.ndarray[np.ndarray[float]],
                    max_neighbors: int,
                    pruning: bool = True) -> None:
        """
        Implements the whole pipeline of building Networkx and PyTorch geometric graphs.

        ### Parameters :
        - distance_matrix (n_samples, n_samples): numpy array containing the distance between each sample
        - max_neighbors : the maximum number of neighbors per sample
        - pruning (default True): True if graph pruning, False otherwise

        ### Returns :
        None
        """
        # Compute adjacency matrix
        self.compute_adjacency_matrix()

        # Create the Networkx graph
        self.create_nx_graph_from_adjacency_matrix()

        # Prune the graph
        if pruning:
            self.prune_graph(distance_matrix, max_neighbors)

        # Create the PyTorch geometric graph
        self.pyg_graph = from_networkx(self.nx_graph)

    def show_graph(self) -> None:
        """
        Displays the networkx graph.

        ### Parameters :
        None

        ### Returns :
        None
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw(self.nx_graph)
        plt.show()
