import torch.optim as optim
import torch.nn as nn
import models

# Variable parameters for which we investigate the effect
experiment_parameters = {
    'datasets': ['linear', 'sinus', 'gasoline', 'hicp'],
    'logdiff': [False],
    'models': [models.RnnModel, models.GRUModel, models.LSTMModel],
    'lookbacks': [1, 4, 8],
    'batch_sizes': [1, 4, 16],
    'patiences': [10, 25, 50],
    'learning_rates': [0.0001, 0.001, 0.01],
    'loss_functions': [nn.MSELoss],
    'optimizers': [optim.Adam]
}

# Fixed parameters, that should be set the same for all experiments
n_epochs = 1000 # Early stopping will take care of the actual number of epochs, should just be set sufficiently high
verbose = False # Set to True to see training progress after each epoch
val_size = 0.1 # Validation set percentage
test_size = 0.3 # Test set percentage

