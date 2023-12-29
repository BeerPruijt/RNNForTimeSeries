from torch.utils import data
import numpy as np
import pandas as pd
import torch

def train_test_split(X, y, test_size=0.1):
    """
    Splits time series data into training and test sets.

    This function divides a given dataset into two subsets: training and validation/test sets.
    The division is based on a specified proportion of the dataset for the validation/test set. 
    It is designed specifically for univariate time series data, ensuring that the 
    chronological order is maintained in the splits.

    Parameters:
    - X (torch.Tensor): Input data to be split. Should be a time series dataset.
    - y (torch.Tensor): Target values corresponding to the input data.
    - test_size (float, optional): Proportion of the dataset to include in the validation/test set. Defaults to 0.1.

    Returns:
    - X_train, y_train (torch.Tensor): Training data and labels.
    - X_test, y_test (torch.Tensor): Test data and labels.

    Raises:
    - AssertionError: If the length of X and y are not equal.

    Note:
    The function assumes that the dataset represents a time series, and the order of data points is crucial.
    """
    # Check that the length of X and y are equal
    assert X.shape[0] == y.shape[0]

    # Determine lengths of each subset
    train_size = int(len(y) * (1 - test_size))

    # Create datasets for training and test
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test

def generate_data_loaders(X_train, y_train, X_test, y_test, batch_size, shuffle=False):
    """
    Creates DataLoader objects for training and test datasets.

    This function takes in the training and test datasets and their corresponding labels,
    and creates PyTorch DataLoader objects for each. These DataLoaders can then be used to iterate over
    the datasets in batches during the training and evaluation process.

    Parameters:
    - X_train, X_test (torch.Tensor): Input data for the training and test sets, respectively.
    - y_train, y_test (torch.Tensor): Target values (labels) for the training and test sets, respectively.
    - batch_size (int): The size of the batches in which the data should be loaded.
    - shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to False.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.

    Note:
    The test DataLoader is always created with shuffle set to False, to ensure the test data order is preserved.
    """
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=shuffle, batch_size=batch_size) 
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)

    return train_loader, test_loader

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, patience, verbose=False):
    """
    Trains a model using given data loaders and evaluates its performance on validation data.

    The training process includes forward and backward passes, optimization steps, and evaluation 
    on validation data at the end of each epoch. The function implements early stopping based on 
    validation loss. If the validation loss does not improve for a specified number of consecutive 
    epochs (defined by 'patience'), training is stopped early. The state of the model with the best 
    validation loss is retained.

    Parameters:
    - model (torch.nn.Module): The neural network model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    - loss_fn (callable): Loss function to be used for training.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
    - device (torch.device): Device to which the model and data should be moved (CPU or GPU).
    - num_epochs (int): Maximum number of epochs for training the model.
    - patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    - verbose (bool, optional): If True, prints progress of training after each epoch. Defaults to False.

    Returns:
    - float: The best validation loss achieved during training.
    """
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Reset the gradients
            optimizer.zero_grad()
            # Backpropagate
            loss.backward()
            # Update the parameters
            optimizer.step()

        # Evaluation phase
        val_loss = evaluate_model(model, val_loader, loss_fn, device)

        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Best Validation Loss: {best_val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            best_model = model.state_dict()
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch + 1} as validation loss has not improved for {patience} consecutive epochs.')
            break

    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return best_val_loss

def evaluate_model(model, loader, loss_fn, device):
    """
    Evaluates the model's performance on a given dataset.

    The function computes the loss of the model predictions against the actual data. It is used 
    during training to evaluate the model on training and validation datasets. This function does 
    not perform backpropagation and is used only for assessing the model's performance.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - loader (torch.utils.data.DataLoader): DataLoader for the data on which the model is to be evaluated.
    - loss_fn (callable): Loss function to compute the error between predictions and actual values.
    - device (torch.device): Device to which the model and data should be moved (CPU or GPU).

    Returns:
    - mean_loss (float): The average loss of the model over the entire dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_loss += loss_fn(y_pred, y_batch).item()

    mean_loss = total_loss / len(loader)
    return np.sqrt(mean_loss)

def create_dataset(timeseries, lookback, device='cpu'):
    """
    Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    # Check if the input is a numpy ndarray
    if not isinstance(timeseries, np.ndarray):
        raise ValueError("Input is not a numpy ndarray")

    # Check if the input is 2D and the second dimension is 1
    if timeseries.ndim != 2 or timeseries.shape[1] != 1:
        raise ValueError("Input is not a 2D array with second dimension equal to 1")

    X, y = [], []
    for i in range(len(timeseries)-lookback):
        feature = timeseries[i:i+lookback] # e.g. [0, 1] for lookback=2 and i=0
        target = timeseries[i+lookback] # e.g. [2] for lookback=2 and i=0
        X.append(feature)
        y.append(target)

    return torch.tensor(np.array(X)).to(device), torch.tensor(np.array(y)).to(device)

def recursive_forecast(model, initial_input, n_steps):
    """
    Perform recursive forecasting using the trained model.

    Args:
    model: The trained RNN model.
    initial_input: A tensor containing the initial input data (last part of the time series).
    n_steps: Number of future steps to forecast.

    Returns:
    A list of predicted values, should probably still be inverse transformed using something like scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).
    """

    model.eval()  # Set the model to evaluation mode
    predictions = []
    input_seq = initial_input

    with torch.no_grad():
        for _ in range(n_steps):
            # Predict the next step
            prediction = model(input_seq)
            # Add the prediction to the list
            predictions.append(prediction.cpu().numpy().item())
            # Concatenate the torch.Size([1, 1]) prediction with the torch.Size([1, lookback, 1]) input
            input_seq = torch.cat((input_seq[:, 1:], prediction.unsqueeze(0)), dim=1)

    # Inverse transform the predictions to original scale
    return predictions