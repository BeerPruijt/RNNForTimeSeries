import torch.nn as nn

class RnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.rnn(x)
        # Select only the last time step's output for each sequence in the batch
        x = x[:, -1, :]
        x = self.linear(x)
        return x

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        # Select only the last time step's output for each sequence in the batch
        x = x[:, -1, :]
        x = self.linear(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        # Select only the last time step's output for each sequence in the batch
        x = x[:, -1, :]
        x = self.linear(x)
        return x
