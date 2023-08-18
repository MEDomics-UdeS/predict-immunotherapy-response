import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import GCNConv


class GCNClassifier(torch.nn.Module):
    """
    Graph Convolutional Network model for binary classification.
    """
    def __init__(self,
                 n_features: int) -> None:
        """
        Sets the layers of the Graph Convolutional Network model.

        Args:
            n_features : the number of features for each sample

        Returns:
            None
        """
        # Parent's constructor
        super().__init__()

        # Convolutive layer
        self.conv = GCNConv(in_channels=n_features, out_channels=n_features//2)

        # ReLU activation
        self.relu = ReLU()

        # Classifier layer
        self.linear = Linear(in_features=n_features//2, out_features=1)

        # Sigmoid activation
        self.sigmoid = Sigmoid()

    def forward(self,
                x: Tensor,
                edge_index: Tensor) -> Tensor:
        """Executes the forward pass.

        Args:
            x (n_samples, n_features) : tensor containing features of each sample
            edge_index (2, n_edges) : tensor containing the graph connectivity

        Returns:
            output (n_samples, ) : tensor with positive class probability of each sample
        """
        # Convolution layer
        h = self.conv(x, edge_index)

        # Activation function
        h = self.relu(h)

        # Classification layer
        h = self.linear(h)

        # Activation function
        output = self.sigmoid(h)

        return output

    def predict_class(self,
                      x: Tensor,
                      edge_index: Tensor,
                      threshold: float = 0.5) -> Tensor:
        """Predicts class of each sample.

        Args:
            x (n_samples, n_features) : tensor containing features of each sample
            edge_index (2, n_edges) : tensor containing the graph connectivity
            threshold : the probability used as threshold between 0 and 1 classes

        Returns:
            (n_samples, ) tensor with class of each sample
        """
        # Model output
        output = self.forward(x, edge_index)

        # Class prediction
        return torch.where(output >= threshold, 1, 0)

    def forward_conv(self,
                     x: Tensor,
                     edge_index: Tensor) -> Tensor:
        """Predicts new samples embeddings obtained after convolution.

        Args:
            x (n_samples, n_features) : tensor containing features of each sample
            edge_index (2, n_edges) : tensor containing the graph connectivity

        Returns:
            (n_samples, n_features) tensor containing the new features of each sample
        """
        return self.conv(x, edge_index)
