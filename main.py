import torch
from itertools import product
import src.models
import torch.nn as nn
import torch.optim as optim
from src.experiment import run_experiment, create_param_dict, select_settings
import src.config as config  # Assuming config contains experiment_parameters

def main():
    # Retrieve experiment parameters
    exp_params = config.experiment_parameters

    # Prompt the user whether she wants to run all experiments
    selected_settings, run_one = select_settings(exp_params)

    # Iterate over all combinations of parameters 
    # NOTE: IF YOU CHANGE THIS MAKE SURE THE ORDER IS CORRECT
    for dataset, model_class, lookback, batch_size, patience, learning_rate, loss_fn_class, optimizer_class in selected_settings:

        # Store the parameters of this iteration in a new dictionary
        settings_dict = create_param_dict(dataset, model_class, lookback, batch_size, patience, learning_rate, loss_fn_class, optimizer_class)

        # Initialize model, loss function, optimizer
        model = model_class().to(device)
        loss_fn = loss_fn_class()
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        # Print the current iteration
        if not run_one:
            try:
                i += 1
                print(f'({i}/{len(selected_settings)})')
            except:
                i = 1
                print(f'({i}/{len(selected_settings)})')
        else:
            print('\nRunning experiment...')

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
                       settings_dict=settings_dict,
                       show_plot=run_one)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()