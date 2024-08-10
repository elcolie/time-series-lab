import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import get_sliding_dataframe


# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using MPS for acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found. Using CUDA for acceleration.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())


df = get_sliding_dataframe()


# Prepare your data
# Assuming df is your DataFrame and 'target' is your target column
X = df.drop('label', axis=1).values
y = df['label'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to device
X_train = torch.FloatTensor(X_train).unsqueeze(1).to(device)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_val = torch.FloatTensor(X_val).unsqueeze(1).to(device)
y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
input_size = X_train.shape[2]  # Number of features
hidden_size = 64
num_layers = 2
output_size = 1  # Assuming single target variable

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Create a TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="my_lstm")

# Train the model
trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices=1, logger=logger)
trainer.fit(model, train_loader, val_loader)


