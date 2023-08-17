import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv


class GATClassifier(torch.nn.Module):
    """
    Graph Attention Network model for binary classification.
    """
    def __init__(self,
                 n_features: int) -> None:
        """
        Set the layers of the Graph Attention Network model.

        ### Parameters :
        - n_features : the number of features for each sample
        """
        # Parent's constructor
        super().__init__()

        # Attention-convolution layer
        self.att_conv = GATv2Conv(in_channels=n_features, out_channels=n_features//2)

        # Classifier layer
        self.linear = Linear(in_features=n_features//2, out_features=1)

    def forward(self,
                x: Tensor,
                edge_index: Tensor) -> Tensor:
        """
        Execute the forward pass.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample
        - edge_index (2, n_edges) : tensor containing the graph connectivity

        ### Returns :
        - output (n_samples, ) : tensor with positive class probability of each sample
        """
        # Attention layer
        h = self.att_conv(x, edge_index)

        # Activation function
        h = F.relu(h)

        # Linear layer
        h = self.linear(h)

        # Activation function
        output = F.sigmoid(h)

        return output

    def predict_class(self,
                      x: Tensor,
                      edge_index: Tensor) -> Tensor:
        """
        Predict class of each sample.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample
        - edge_index (2, n_edges) : tensor containing the graph connectivity

        ### Returns :
        (n_samples, ) tensor with class of each sample
        """
        # Model output
        output = self.forward(x, edge_index)

        # Class prediction
        return torch.where(output >= 0.5, 1, 0)

    def forward_conv(self,
                     x: Tensor,
                     edge_index: Tensor) -> Tensor:
        """
        Predict new samples embeddings obtained after attention-convolution.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample
        - edge_index (2, n_edges) : tensor containing the graph connectivity

        ### Returns :
        (n_samples, n_features) tensor containing the new features of each sample
        """
        return self.att_conv(x, edge_index)
