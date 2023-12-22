from utils import train_val_test_split, create_dataset, generate_data_loaders, train_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import mean_squared_error
from datetime import datetime
import json
from itertools import product

def report_performance(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, timeseries, settings_dict, show_plot=False):
    """
    Evaluates and reports the performance of a trained model on training, validation, and test datasets.

    This function makes one step ahead predictions using the provided model on the training, validation, and test sets. 
    It then calculates and reports the Root Mean Squared Error (RMSE) for each set. Optionally, it can 
    also plot the original timeseries data along with the predictions for each set.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - X_train, X_val, X_test (torch.Tensor): Input data for the training, validation, and test sets.
    - y_train, y_val, y_test (torch.Tensor): Target values for the training, validation, and test sets.
    - scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to inverse-transform the model's predictions.
    - timeseries (np.array): Original timeseries data for plotting.
    - settings_dict (dict): A dictionary containing the settings used for the experiment.

    Returns:
    None. The function directly stores the outputs and plots in the output directory (and creates one if it doesn't exist yet).
    """
    model.eval()

    # Retrieve settings
    lookback = settings_dict['lookback']

    with torch.no_grad():
        # Predictions for Training Set
        train_preds = model(X_train).cpu()
        train_preds = scaler.inverse_transform(train_preds.numpy())
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[lookback:lookback + len(train_preds)] = train_preds

        # Predictions for Validation Set
        val_preds = model(X_val).cpu()
        val_preds = scaler.inverse_transform(val_preds.numpy())
        val_plot = np.ones_like(timeseries) * np.nan
        val_start_index = len(train_preds) + lookback
        val_plot[val_start_index:val_start_index + len(val_preds)] = val_preds

        # Predictions for Test Set
        test_preds = model(X_test).cpu()
        test_preds = scaler.inverse_transform(test_preds.numpy())
        test_plot = np.ones_like(timeseries) * np.nan
        test_start_index = len(train_preds) + len(val_preds) + lookback
        test_plot[test_start_index:test_start_index + len(test_preds)] = test_preds

        train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.cpu().numpy()), train_preds))
        val_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_val.cpu().numpy()), val_preds))
        test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.cpu().numpy()), test_preds))

        # Saving
        true_vals = scaler.inverse_transform(timeseries)
        experiment_name = '_'.join([f"{key}={value}" for key, value in settings_dict.items()])
        data = {
            'true_vals': true_vals.tolist(),
            'train_plot': train_plot.tolist(),
            'val_plot': val_plot.tolist(),
            'test_plot': test_plot.tolist()
        }

        append_to_csv(train_rmse, val_rmse, test_rmse, settings_dict, output_dir='output')

        with open(os.path.join('output', f"{experiment_name}_data.json"), 'w') as file:
            json.dump(data, file)

        # Plotting
        if show_plot:
            plt.figure(figsize=(15, 8))
            plt.plot(true_vals, label='Original Data', color='blue')
            plt.plot(train_plot, label='Training Predictions', color='red')
            plt.plot(val_plot, label='Validation Predictions', color='orange')
            plt.plot(test_plot, label='Test Predictions', color='green')
            plt.legend()
            plt.show()

def append_to_csv(train_rmse, val_rmse, test_rmse, settings_dict, output_dir='output'):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define CSV file path
    csv_file = os.path.join(output_dir, 'experiment_results.csv')
    
    # Start with Date/Time
    data = {
        'Date/Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Merge parameter dictionary
    data.update(settings_dict)  # Merging parameter data

    # Add experiment name and RMSE values (I save it like this because my csv otherwise gets messed up when I open it)
    data.update({
        'Training RMSE': round(train_rmse, 5),
        'Validation RMSE': round(val_rmse, 5),
        'Test RMSE': round(test_rmse, 5)
    })

    # Append data to CSV file
    df = pd.DataFrame([data])
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)


def generate_series(dataset='linear', log_diff=False):
    """
    The datasets look something like this:

    df
         date            data
    0    1949-01         112
    1    1949-02         118
    2    1949-03         132
    3    1949-04         129
    4    1949-05         121
    ..       ...         ...
    139  1960-08         606
    140  1960-09         508
    141  1960-10         461
    142  1960-11         390
    143  1960-12         432

    [144 rows x 2 columns]
    """
    # Linear trend
    if dataset == 'linear':
        date_range = pd.date_range(start='1995-01-01', periods=300, freq='MS')
        values = [i/2 for i in range(len(date_range))] 
        df = pd.DataFrame({
            "date": date_range,
            "data": values
        })

    # Sinusoidal trend
    elif dataset == 'sinus':
        date_range = pd.date_range(start='1995-01-01', periods=300, freq='MS')
        values = [np.sin(i/2) for i in range(len(date_range))] 
        df = pd.DataFrame({
            "date": date_range,
            "data": values
        })
    
    # Gasoline prices in dollar per gallon
    elif dataset == 'gasoline':
        df = pd.read_csv('data/gasoline_data.txt', sep='\t', parse_dates=True).rename(columns={'oil': 'data'})

    # American HICP (inflation) index
    elif dataset == 'hicp':
        df = pd.read_csv('data/hicp_data.txt', sep='\t', parse_dates=True).rename(columns={'hicp': 'data'})

    start_idx = 0
    if log_diff:
        df.loc[:, 'data'] = np.log(df.loc[:, 'data']).diff() 
        start_idx = 1

    timeseries_original = df.loc[df.index[start_idx::], ['data']].values.astype('float32')

    return timeseries_original

def run_experiment(model, loss_fn, optimizer, device, n_epochs, patience, batch_size, lookback, verbose, val_size, test_size, dataset, log_diff, settings_dict, show_plot):
    # Data preparation
    timeseries_original = generate_series(dataset, log_diff)

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    timeseries_scaled = scaler.fit_transform(timeseries_original)

    # Dataset creation
    X, y = create_dataset(timeseries_scaled, lookback=lookback, device=device)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, val_size=val_size, test_size=test_size)

    # DataLoader creation
    train_loader, val_loader, _ = generate_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, shuffle=False)

    # Model training and evaluation
    train_model(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs, patience, verbose)
    report_performance(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, timeseries_scaled, settings_dict, show_plot=show_plot)

def create_param_dict(dataset, model_class, lookback, batch_size, patience, learning_rate, loss_fn_class, optimizer_class, log_diff):
    """
    Creates a dictionary of parameters with their respective names as keys.

    Parameters:
    - dataset: The dataset being used.
    - model_class: The class of the model.
    - lookback: The lookback period for the model.
    - batch_size: The size of the batches used in training.
    - patience: The patience level for early stopping.
    - learning_rate: The learning rate for the optimizer.
    - loss_fn_class: The class of the loss function.
    - optimizer_class: The class of the optimizer.
    - log_diff: Whether logarithmic differencing is applied.

    Returns:
    - A dictionary containing the parameters and their values.
    """
    param_dict = {
        'dataset': dataset,
        'model_class': model_class.__name__ if model_class else None,
        'lookback': lookback,
        'batch_size': batch_size,
        'patience': patience,
        'learning_rate': learning_rate,
        'loss_fn_class': loss_fn_class.__name__ if loss_fn_class else None,
        'optimizer_class': optimizer_class.__name__ if optimizer_class else None,
        'log_diff': log_diff
    }

    return param_dict

def select_settings(exp_params):
    """
    Interactively prompts the user to select experiment settings or choose to run all experiments.

    This function asks the user if they want to run all possible combinations of experiments. If the user 
    chooses not to run all experiments, they are prompted to select specific settings for each parameter 
    category defined in `exp_params`. The function returns a list of tuples representing the selected 
    experiment configurations and a boolean indicating whether the user chose to run one or all experiments.

    Parameters:
    exp_params (dict): A dictionary where keys are parameter categories (like 'datasets', 'models', etc.) 
                       and values are lists of options for each category.

    Returns:
    (list of tuples, bool): The first element is a list with one tuple (such that I can use it as an interable with 1 element) 
                            representing the selected experiment configurations. Each tuple is a combination of parameters. 
                            The second element is a boolean, True if only one experiment is to be run, and False otherwise.

    Raises:
    ValueError: If the user's input is not 'y' or 'n' when asked to run all experiments or if the selection 
                is out of the valid range for a parameter category.
    """
    run_all = input("Do you want to run all experiments? (y/n): ").strip().lower()

    if run_all not in ['y', 'n']:
        raise ValueError("Come on you got this! Either a 'y' (for yes) or an 'n' (for no).")

    selected_settings = list(product(*[exp_params[key] for key in exp_params.keys()]))

    if run_all == 'n':
        user_selections = {}
        for category in exp_params:
            print(f"\n{category.capitalize()} options:")
            for i, option in enumerate(exp_params[category]):
                print(f"{i + 1}. {option}")
            print('')

            selection = int(input(f"Number next to your choice: ")) - 1

            if selection not in range(len(exp_params[category])):
                raise ValueError(f"Let's try again! I just need a number between 1 and {len(exp_params[category])}.")

            user_selections[category] = exp_params[category][selection]

        selected_settings = [tuple(user_selections.values())]

    return selected_settings, run_all == 'n'