# +
import typing as typ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(0)

# Check if MPS is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using MPS")
# else:
#     device = torch.device("cpu")
#     print("MPS not available, using CPU")

device = torch.device("cpu")
print("Device:", device)
# -

# Generate sine wave data
def generate_sine_wave(seq_length: int, num_samples: int) -> np.ndarray:
    x = np.linspace(0, 10 * np.pi, num_samples)
    y = np.sin(x)
    return y.reshape(-1, seq_length)


bbl_df = pd.read_csv("../time_series_data/SET_DLY_BBL, 5_d2141.csv")

bbl_df


def generate_price_sequence(bbl_df: pd.DataFrame, seq_length: int) -> np.ndarray:
    # Ensure the 'close' column exists
    if 'close' not in bbl_df.columns:
        raise ValueError("The dataframe must have a 'close' column")

    # Get the close prices as a numpy array
    prices = bbl_df['close'].values

    # Calculate the number of sequences
    num_sequences = len(prices) - seq_length + 1

    # Create the sequences
    sequences = np.array([prices[i:i + seq_length] for i in range(num_sequences)])

    return sequences


# Parameters
seq_length = 20
# num_samples = 1000
input_size = 1
hidden_size = 10
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Generate data
# data = generate_sine_wave(seq_length, num_samples)
data = generate_price_sequence(bbl_df, seq_length)
data = torch.FloatTensor(data).unsqueeze(2).to(device)

# Split data into train and test sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


# +
# LSTM model
class SineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Configuration:
    model: object
    criterion: object

    def __init__(
        self,
        model: object,
        criterion: object,
        optimizer: typ.Callable,
        **kwargs
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), **kwargs)


my_configs = [
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD,
        lr=learning_rate
    ),
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.L1Loss(),
        optimizer=torch.optim.Adam,
        lr=learning_rate
    ),
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.SmoothL1Loss(),
        optimizer=torch.optim.Adam,
        lr=learning_rate
    ),
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.RMSprop,
        lr=learning_rate
    ),
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adagrad,
        lr=learning_rate
    ),
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD,
        lr=learning_rate,
        momentum=0.9
    ),
    Configuration(
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.NAdam,
        lr=learning_rate
    )
]


# Initialize model, loss function, and optimizer
# model = SineLSTM(input_size, hidden_size, num_layers, output_size).to(device)

# This pair is not work. NN does not learn after 2nd epoch.
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# criterion = nn.L1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# criterion = nn.SmoothL1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)


# -
def run_experiment(_config: Configuration):
    model = _config.model
    criterion = _config.criterion
    optimizer = _config.optimizer

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i in range(len(train_data)):
            seq = train_data[i, :-1, :].unsqueeze(0)
            target = train_data[i, -1, :].unsqueeze(0)

            output = model.forward(seq)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Generate sine curve
    model.eval()
    with torch.no_grad():
        test_seq = test_data[0, :-1, :].unsqueeze(0)  # Add batch dimension
        true_vals = test_data[0, -seq_length:, 0].cpu().numpy()
        predicted = []

        for _ in range(seq_length):
            out = model(test_seq)
            predicted.append(out.cpu().item())
            test_seq = torch.cat((test_seq[:, 1:, :], out.unsqueeze(1)), dim=1)

            if _ % 5 == 0:
                plt.plot(test_seq.squeeze().cpu())
                plt.plot(predicted)
                plt.savefig(f"extrapolation/predicted/try_real_data_lstm_{str(model)}_{str(criterion)}_{optimizer}_{_}.png")


if __name__ == "__main__":
    from multiprocessing import Pool
    import os

    cpu_count = os.cpu_count()
    with Pool(processes=cpu_count) as pool:
        for _ in tqdm(pool.imap_unordered(run_experiment, my_configs), total=len(my_configs)):
            pass


