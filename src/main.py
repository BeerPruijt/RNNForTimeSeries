import optuna
import torch
import models
import torch.nn as nn
import torch.optim as optim
from experiment import run_experiment
import config  

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_classes = {'RnnModel': models.RnnModel, 'GRUModel': models.GRUModel, 'LSTMModel': models.LSTMModel}
    loss_fn_classes = {'MSELoss': nn.MSELoss}
    optimizer_classes = {'Adam': optim.Adam}
    
    # Suggest parameters using Optuna
    model_class = model_classes[trial.suggest_categorical('model_class', config.experiment_parameters['models'])]
    lookback = trial.suggest_int('lookback', config.experiment_parameters['min_lookback'], config.experiment_parameters['max_lookback'])
    batch_size = trial.suggest_int('batch_size', config.experiment_parameters['min_batch_size'], config.experiment_parameters['max_batch_size'])
    patience = trial.suggest_int('patience', config.experiment_parameters['min_patience'], config.experiment_parameters['max_patience'])
    learning_rate = trial.suggest_float('learning_rate', config.experiment_parameters['min_learning_rate'], config.experiment_parameters['max_learning_rate'], log=True)
    loss_fn_class = loss_fn_classes[trial.suggest_categorical('loss_fn_class', ['MSELoss'])]
    optimizer_class = optimizer_classes[trial.suggest_categorical('optimizer_class', ['Adam'])]
    num_layers = trial.suggest_int('num_layers', config.experiment_parameters['min_num_layers'], config.experiment_parameters['max_num_layers'])
    hidden_size = trial.suggest_int('hidden_size', config.experiment_parameters['min_hidden_size'], config.experiment_parameters['max_hidden_size'])

    # Initialize model, loss function, optimizer
    model = model_class(num_layers, hidden_size).to(device)
    loss_fn = loss_fn_class()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Run the experiment
    val_loss = run_experiment(model=model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              device=device,
                              n_epochs=config.n_epochs,
                              patience=patience,
                              batch_size=batch_size,
                              lookback=lookback,
                              verbose=config.verbose,
                              val_size=config.val_size,
                              dataset='removed')

    return val_loss

def main():
    # Create a study object
    study = optuna.create_study(study_name=config.study_name, storage=config.storage_name, direction='minimize', load_if_exists=True)
    
    # Run the optimization
    study.optimize(objective, n_trials=config.n_optuna_trials)  

    # Print the best hyperparameters
    print('Best hyperparameters:', study.best_params)

if __name__ == "__main__":
    main()
