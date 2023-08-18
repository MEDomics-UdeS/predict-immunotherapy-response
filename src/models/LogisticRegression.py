import torch
from torch import Tensor
from torch.nn import Linear, Sigmoid


class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression model for binary classification.
    """
    def __init__(self,
                 n_features: int) -> None:
        """
        Sets the layers of the Logistic Regression model.

        ### Parameters :
        - n_features : the number of features for each sample
        """
        # Parent's constructor
        super().__init__()

        # Classifier layer
        self.linear = Linear(in_features=n_features, out_features=1)

        # Sigmoid activation
        self.sigmoid = Sigmoid()

    def forward(self,
                x: Tensor) -> Tensor:
        """
        Executes the forward pass.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample

        ### Returns :
        - output (n_samples, ) : tensor with positive class probability of each sample
        """
        # Classifier layer
        h = self.linear(x)

        # Activation function
        output = self.sigmoid(h)

        return output

    def predict_class(self,
                      x: Tensor,
                      threshold: float = 0.5) -> Tensor:
        """
        Predicts class of each sample.

        ### Parameters :
        - x (n_samples, n_features) : tensor containing features of each sample
        - threshold : the probability used as threshold between 0 and 1 classes

        ### Returns :
        (n_samples, ) tensor with class of each sample
        """
        # Model output
        output = self.forward(x)

        # Class prediction
        return torch.where(condition=output >= threshold, self=1, other=0)
