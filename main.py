import torch
from itertools import product
import models
import torch.nn as nn
import torch.optim as optim
from experiment import run_experiment, create_param_dict
import config  # Assuming config contains experiment_parameters

def main():
    # Retrieve experiment parameters
    exp_params = config.experiment_parameters

    i=0
    
    # Iterate over all combinations of parameters
    for dataset, model_class, lookback, batch_size, patience, learning_rate, loss_fn_class, optimizer_class, log_diff in product(
        exp_params['datasets'], exp_params['models'], exp_params['lookbacks'], 
        exp_params['batch_sizes'], exp_params['patiences'], exp_params['learning_rates'], 
        exp_params['loss_functions'], exp_params['optimizers'], exp_params['logdiff']):

        # Store the parameters of this iteration in a new dictionary
        settings_dict = create_param_dict(dataset, model_class, lookback, batch_size, patience, learning_rate, loss_fn_class, optimizer_class, log_diff)

        # Initialize model, loss function, optimizer
        model = model_class().to(device)
        loss_fn = loss_fn_class()
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        i += 1
        print(i)

        # Run the experiment with the current settings and save the results
        run_experiment(model=model, 
                       loss_fn=loss_fn, 
                       optimizer=optimizer, 
                       device=device, 
                       n_epochs=config.n_epochs, 
                       patience=patience, 
                       batch_size=batch_size, 
                       lookback=lookback, 
                       verbose=config.verbose, 
                       val_size=config.val_size, 
                       test_size=config.test_size, 
                       dataset=dataset, 
                       log_diff=log_diff, 
                       settings_dict=settings_dict)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()