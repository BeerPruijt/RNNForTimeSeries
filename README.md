# RNNForTimeSeries

This project provides a basic implementation of several Recurrent Neural Networks (RNNs) for univariate time series. Specifically, the project considers a simple RNN, a GRU (Gated Recurrent Unit) and LSTM (Long-Short-Term Memory) and analyses their performance on two stylized forecasting tasks (a linear and a sinusoidal pattern), as well as two monthly datasets (US HICP and Gasoline prices in dollar per gallon). 

## Project Structure

- `config.py`: Contains the configuration parameters for the experiments.
- `data/`: Contains the data files used in the experiments.
- `experiment.py`: Contains the code to run a single experiment.
- `main.py`: The main entry point of the application. It creates an output folder, sets up- and runs the experiments.
- `models.py`: Contains the model definitions.
- `utils.py`: Contains utility functions that I expect to re-use when taking these models to production elsewhere.

## Setup

1. **Clone the Repository**:
   - Clone the repository to your local machine using `git clone [repository URL]`.

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

The `output/` directory is where the output of the experiments will be stored. This folder will contain the results (train, validation, and test RMSE) appended to a CSV file for hyperparameter analysis on the specified datasets. Additionally, it will include plots of the train, validation, and test predictions for each iteration.

## Contributing

Any comments, suggestions, or contributions are welcome. Feel free to submit a pull request or open an issue. I'm especially interested in deriving/learning about heuristics concerning model architecture and always excited for discussions on implementations and extensions of this project.
