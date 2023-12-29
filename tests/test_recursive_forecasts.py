from src.utils import recursive_forecast 
import torch
import pytest
from src.utils import recursive_forecast 
import torch

# Define a theoretical model that takes in something like torch.Size([1, lookback, 1]) and outputs torch.Size([1, 1]), always adding 1 to the final input
class TheoreticalModel(torch.nn.Module):
    def __init__(self):
        super(TheoreticalModel, self).__init__()

    def forward(self, input_seq):
        return input_seq[:, -1, :] + 1

# Test that the output is a list of length n_steps
@pytest.fixture
def model():
    return TheoreticalModel()

@pytest.fixture
def initial_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOOKBACK_EXAMPLE = 3
    return torch.arange(1, LOOKBACK_EXAMPLE + 1).unsqueeze(0).unsqueeze(-1).to(device)

# Test that it is a list
def test_recursive_forecast_type(model, initial_input):
    N_STEPS = 4
    output = recursive_forecast(model, initial_input, N_STEPS)
    assert isinstance(output, list)

# Test that the first element of the output is the last element of the initial input + 1
def test_recursive_forecast_first(model, initial_input):
    N_STEPS = 4
    output = recursive_forecast(model, initial_input, N_STEPS)
    assert output[0] == initial_input[:, -1, :] + 1

# Test that the final element of the output is the last element of the initial input + N_STEPS
def test_recursive_forecast_final(model, initial_input):
    N_STEPS = 4
    output = recursive_forecast(model, initial_input, N_STEPS)
    assert output[-1] == initial_input[:, -1, :] + N_STEPS  

# Test that the output increases by 1 for each step
def test_recursive_forecast_increment(model, initial_input):
    N_STEPS = 4
    output = recursive_forecast(model, initial_input, N_STEPS)
    for i in range(len(output)-1):
        assert output[i+1] == output[i] + 1
