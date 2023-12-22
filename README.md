# RNNForTimeSeries

The `RNNForTimeSeries` project implements various Recurrent Neural Networks (RNNs) for univariate time series forecasting. The implementation focuses on the following aspects:

## Models
- **Simple RNN**: A basic recurrent neural network architecture.
- **GRU (Gated Recurrent Unit)**: An advanced RNN architecture that handles long-range dependencies more effectively.
- **LSTM (Long Short-Term Memory)**: Another advanced RNN type known for its capability to remember information over long periods.

## Forecasting Tasks
- **Linear Pattern Forecasting**: Analyzing the performance of RNN models in forecasting a linear time series pattern.
- **Sinusoidal Pattern Forecasting**: Evaluating how these models perform with sinusoidal time series patterns.

## Real-world Datasets
- **US HICP (Harmonized Index of Consumer Prices)**: Monthly data on consumer price indices in the United States.
- **Gasoline Prices (in USD per gallon)**: Monthly data on gasoline prices.

This project aims to explore and compare the effectiveness of different RNN architectures in handling both stylized and real-world time series data.

## Project Structure

- `config.py`: Contains the configuration parameters for the experiments.
- `data/`: Contains the data files used in the experiments.
- `experiment.py`: Contains the code to run a single experiment.
- `main.py`: The main entry point of the application. It creates an output folder, sets up- and runs the experiments.
- `models.py`: Contains the model definitions.
- `utils.py`: Contains utility functions that I expect to re-use when taking these models to production elsewhere.

## Setup

1. **Clone the Repository**:
   - Clone the repository to your local machine using `git clone https://github.com/BeerPruijt/RNNForTimeSeries.git`.

2. **Create a Python Environment**:
   - Create a new Python environment with Python 3.9. You can use virtualenv or conda to do this.
     - With virtualenv: `virtualenv venv --python=python3.9`
     - With conda: `conda create -n myenv python=3.9`
   - Activate the environment.
     - With virtualenv: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
     - With conda: `conda activate myenv`

3. **Install Required Packages**:
   - Install the required packages using `pip install -r requirements.txt`.
   - The `requirements.txt` file contains the necessary Python packages. Note that this file does not include specifications for CUDA compatibility. This is to ensure that the setup is compatible with systems that do not have CUDA.

4. **CUDA Compatibility (Optional)**:
   - If you wish to run experiments with CUDA support and your system supports it, you will need to manually install CUDA-enabled versions of relevant packages (e.g., TensorFlow or PyTorch).
   - Check your CUDA version and install the corresponding packages.

5. **Run the Experiments**:
   - Run the `main.py` script to start the experiments: `python main.py`

Please follow these steps to ensure a smooth setup of the project environment.

## Usage

To run an experiment, modify the parameters in `config.py` as needed, then run the `main.py` script.

During the execution of `main.py`, you'll be interactively prompted to decide how to proceed with the experiments:
- **Run All Combinations**: You can choose to run all combinations of experiments as defined in `config.py`. Simply respond with 'y' when prompted.
- **Run a Specific Experiment**: If you prefer to run a specific experiment, respond with 'n'. You'll then be guided through a series of prompts to select the specific settings for each parameter category, such as 'datasets', 'models', etc. This allows for a tailored experiment based on your selected configurations.

Once the experiment(s) start, an `output/` directory will be created where the output of the experiments will be stored. This folder will contain the results (train, validation, and test RMSE) appended to a CSV file for hyperparameter analysis on the specified datasets. Additionally, it will include JSON files with the train, validation, and test predictions for each iteration.

## Contributing

Any comments, suggestions, or contributions are welcome. Feel free to submit a pull request or open an issue. I'm especially interested in deriving/learning about heuristics concerning model architecture and always excited for discussions on implementations and extensions of this project.
