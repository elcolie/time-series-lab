# +
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

bbl_df = pd.read_csv("../time_series_data/SET_DLY_BBL, 5_d2141.csv")


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
seq_length = 40
# num_samples = 1000
input_size = 1
hidden_size = 100
num_layers = 10
output_size = 1
num_epochs = 1000
learning_rate = 0.01

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

    def __init__(
        self,
        id: str,
        model: object,
        criterion: object,
        optimizer: typ.Callable,
        **kwargs
    ):
        self.id = id
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), **kwargs)


my_configs = [
    Configuration(
        id="1",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD,
        lr=learning_rate
    ),
    Configuration(
        id="2",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.L1Loss(),
        optimizer=torch.optim.Adam,
        lr=learning_rate
    ),
    Configuration(
        id="3",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.SmoothL1Loss(),
        optimizer=torch.optim.Adam,
        lr=learning_rate
    ),
    Configuration(
        id="4",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.RMSprop,
        lr=learning_rate
    ),
    Configuration(
        id="5",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adagrad,
        lr=learning_rate
    ),
    Configuration(
        id="6",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD,
        lr=learning_rate,
        momentum=0.9
    ),
    Configuration(
        id="7",
        model=SineLSTM(input_size, hidden_size, num_layers, output_size).to(device),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.NAdam,
        lr=learning_rate
    )
]


# -
def run_experiment(_config: Configuration):
    _id = _config.id
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
        chunk_test_data = test_data[0, :-1, :]
        test_seq = chunk_test_data.unsqueeze(0)  # Add batch dimension
        y_bottom, y_top = min(chunk_test_data) - 0.15, max(chunk_test_data) + 0.15  # Use in .ylim()
        # true_vals = test_data[0, -seq_length:, 0].cpu()
        predicted = []

        for _ in range(seq_length):
            out = model(test_seq)
            predicted.append(out.cpu().item())
            test_seq = torch.cat(
                (test_seq[:, 1:, :], out.unsqueeze(1)),
                dim=1
            )
        plt.plot(chunk_test_data, label="Test", color="orange")
        plt.plot(chunk_test_data + predicted, label="Predicted", color="blue", linestyle='--')
        plt.ylim((y_bottom, y_top))
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.title("Test VS predicted value from neural network")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"predicted/try_real_data_lstm_{_id}_{_}.png")
        plt.clf()


if __name__ == "__main__":
    """
    Run the experiments. I need this if __name__ otherwise it will raises error regarding the Pool.
    Ref: https://stackoverflow.com/questions/65859890/python-multiprocessing-with-m1-mac
    """
    from multiprocessing import Pool
    import os

    cpu_count = os.cpu_count()
    with Pool(processes=cpu_count) as pool:
        for _ in tqdm(pool.imap_unordered(run_experiment, my_configs), total=len(my_configs)):
            pass
    # run_experiment(my_configs[0])



