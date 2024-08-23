import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Define the LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Prepare data
step = 0.1
X = np.arange(0, 10, step)
real_sin = np.sin(X)
X_train, y_train = [], []
start = 10  # training history
for i in range(0, len(real_sin) - start):
    X_train.append(real_sin[i:i + start])
    y_train.append(real_sin[i + start])

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

# Initialize and train the model
input_size = 1
hidden_size = 100
num_layers = 10
output_size = 1

model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train.unsqueeze(-1))
    optimizer.zero_grad()
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# +
# Generate predictions
model.eval()
X_new, Y_new = [X[-1]], [real_sin[-1]]
X_in = torch.FloatTensor(y_train[-start:]).unsqueeze(0).unsqueeze(-1)  # Shape: [1, start, 1]

with torch.no_grad():
    for i in range(200):
        X_new.append(X_new[-1] + step)
        next_y = model(X_in).item()
        Y_new.append(next_y)
        # Reshape next_y to match X_in dimensions
        next_y_tensor = torch.FloatTensor([[[next_y]]])  # Shape: [1, 1, 1]
        X_in = torch.cat([X_in[:, 1:, :], next_y_tensor], dim=1)
# -

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(X, real_sin, label='Real Sin')
plt.plot(X_new, Y_new, label='Predicted')
plt.legend()
plt.show()


