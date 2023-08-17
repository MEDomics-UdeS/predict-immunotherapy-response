import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from models.LogisticRegression import LogisticRegression


class LogisticRegressionTrainTestManager:
    """
    Train-test manager for Logistic Regression model.
    """
    def __init__(self) -> None:
        """
        LogisticRegressionTrainTestManager class constructor.

        ### Parameters :
        None

        ### Returns :
        None
        """
        self.model = None

    def train(self,
              X: np.ndarray[np.ndarray[float]],
              y: np.ndarray[int],
              n_epochs: int,
              lr: float,
              reg: float) -> tuple[list[float], list[float]]:
        """
        Train the model for n_epochs with 80% train 20% validation.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing the features of each sample
        - y (n_samples,) : numpy array containing the label of each sample
        - n_epochs : the number of epochs.
        - lr : the learning rate for the gradient descent
        - reg : the regularization factor in optimizer

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
        X_train, X_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          test_size=0.2)

        # Convert in tensor
        X_train_torch = torch.from_numpy(X_train).float()
        X_val_torch = torch.from_numpy(X_val).float()
        y_train_torch = torch.from_numpy(y_train).float().unsqueeze(1)
        y_val_torch = torch.from_numpy(y_val).float().unsqueeze(1)

        for epoch in range(n_epochs):

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass for train set
            out_train = self.model.forward(X_train_torch)

            # Compute train loss
            loss_train = loss_function(out_train, y_train_torch)

            # Add train loss to train loss list
            train_loss_list.append(loss_train.item())

            # Backward pass (gradients computation)
            loss_train.backward()

            # Update parameters
            optimizer.step()

            # Forward pass for validation set
            out_val = self.model.forward(X_val_torch)

            # Compute validation loss
            loss_val = loss_function(out_val, y_val_torch)

            # Add validation loss to validation loss list
            val_loss_list.append(loss_val.item())

        return train_loss_list, val_loss_list

    def leave_one_out_cv(self,
                         X: np.ndarray[np.ndarray[float]],
                         y: np.ndarray[int],
                         n_epochs: int,
                         lr: float,
                         reg: float) -> tuple[np.ndarray[float], np.ndarray[int], list[float], list[float]]:
        """
        Execute the leave one out cross validation to find test scores and labels.

        ### Parameters :
        - X (n_samples, n_features) : numpy array containing the features of each sample
        - y (n_samples,) : numpy array containing the label of each sample
        - n_epochs : the number of epochs.
        - lr : the learning rate for the gradient descent
        - reg : the regularization factor in optimizer

        ### Returns :
        - test_scores (n_samples, ) : numpy array containing the test score of each sample
        - test_classes (n_samples, ) : numpy array containing the test class of each sample
        - train_losses (n_samples, n_epochs) : list containing the train loss for each sample and epoch
        - val_losses (n_samples, n_epochs) : list containing the validation loss for each sample and epoch
        """

        # Split dataframe in n_samples groups
        n_samples, n_features = X.shape
        folds = KFold(n_splits=n_samples, shuffle=True).split(X)

        # Initialize test scores and test classes arrays
        test_scores = np.zeros(X.shape[0])
        test_classes = np.zeros(X.shape[0])

        # Initialize train losses and validation losses lists
        train_losses_list, val_losses_list = [], []

        for i, (train_index, test_index) in enumerate(folds):

            # Set train-test set
            X_train, y_train = X[train_index], y[train_index]
            X_test = X[test_index]

            # Instanciate model
            self.model = LogisticRegression(n_features)

            # Training on train set
            train_loss, val_loss = self.train(X_train,
                                              y_train,
                                              n_epochs,
                                              lr,
                                              reg)

            # Add to train loss and validation loss lists
            train_losses_list.append(train_loss)
            val_losses_list.append(val_loss)

            # Convert test set in tensor
            X_test_torch = torch.from_numpy(X_test).float()

            # Forward pass on test set (score and class)
            score_test = self.model.forward(X_test_torch).detach().numpy()[0]
            class_test = self.model.predict_class(X_test_torch).detach().numpy()[0]

            # Add score and class to scores and classes arrays
            test_scores[test_index] = score_test
            test_classes[test_index] = class_test

        return test_scores, test_classes, train_losses_list, val_losses_list
