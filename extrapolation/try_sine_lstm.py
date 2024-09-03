# +
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# -

# Generate sine wave data
def generate_sine_wave(seq_length: int, num_samples: int) -> np.ndarray:
    x = np.linspace(0, 10 * np.pi, num_samples)
    y = np.sin(x)
    return y.reshape(-1, seq_length)


# Parameters
seq_length = 50
num_samples = 1000
input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1
num_epochs = 100
learning_rate = 0.01


# Generate data
data = generate_sine_wave(seq_length, num_samples)
data = torch.FloatTensor(data).unsqueeze(2)

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

# Initialize model, loss function, and optimizer
model = SineLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Generate sine curve
model.eval()
with torch.no_grad():
    test_seq = test_data[0, :-1, :].unsqueeze(0)  # Add batch dimension
    # true_vals = test_data[0, -seq_length:, 0].numpy()
    predicted = []

    for _ in range(seq_length):
        out = model(test_seq)
        predicted.append(out.item())
        test_seq = torch.cat((test_seq[:, 1:, :], out.unsqueeze(1)), dim=1)


plt.plot(test_seq.squeeze())

plt.plot(predicted)


