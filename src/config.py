# Variable parameters for which we investigate the effect
experiment_parameters = {
    'models': ['RnnModel'],
    'loss_functions': ['MSELoss'],
    'optimizers': ['Adam'],
    'min_lookback': 6,
    'max_lookback': 12,
    'min_batch_size': 1,
    'max_batch_size': 50,
    'min_patience': 10,
    'max_patience': 100,
    'min_num_layers': 3,
    'max_num_layers': 4,
    'min_hidden_size': 8,
    'max_hidden_size': 64,
    'min_learning_rate': 0.00001,
    'max_learning_rate': 0.001
}

# Fixed parameters, that should be set the same for all experiments
n_epochs = 2000 # Early stopping will take care of the actual number of epochs, should just be set sufficiently high
verbose = False # Set to True to see training progress after each epoch
val_size = 0.3 # Validation set percentage

# The cpi categories that are used for the full analysis
cpi_categories = [
    'all_items_less_energy',
    'all_items_less_food',
    'all_items_less_medical_care',
    'all_items_less_shelter',
    'apparel',
    'commodities',
    'commodities_less_food',
    'energy',
    'food_and_beverages',
    'housing',
    'medical_care',
    'nondurables',
    'nondurables_less_food',
    'nondurables_less_food_and_apparel',
    'other_goods_and_services',
    'services',
    'services_less_medical_care_services',
    'services_less_rent_of_shelter',
    'transportation',
    'all_items_less_food_and_energy',
    'durables',
    'food',
    'fuels_and_utilities',
    'household_furnishings_and_operations',
    'other_services'
]

# Optuna setup
study_name = 'promising_rnns'
storage_name = 'sqlite:///../output/{}.db'.format(study_name)
n_optuna_trials = 300