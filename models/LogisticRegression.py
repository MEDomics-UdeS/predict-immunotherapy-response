import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression model for binary classification.
    """
    def __init__(self,
                 n_features: int) -> None:
        """
        Set the layers of the Logistic Regression model.

        ### Parameters :
        - n_features : the number of features for each sample
        """
        # Parent's constructor
        super().__init__()

        # Classifier layer
        self.linear = Linear(in_features=n_features,
                             out_features=1)

    def forward(self,
                x: Tensor) -> Tensor:
        """
        Execute the forward pass.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample

        ### Returns :
        - output (n_samples, ) : tensor with positive class probability of each
         sample
        """
        # Classifier layer
        h = self.linear(x)

        # Activation function
        output = F.sigmoid(h)

        return output

    def predict_class(self,
                      x: Tensor) -> Tensor:
        """
        Predict class of each sample.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample

        ### Returns :
        (n_samples, ) tensor with class of each sample
        """
        # Model output
        output = self.forward(x)

        # Class prediction
        return torch.where(condition=output >= 0.5,
                           self=1,
                           other=0)
