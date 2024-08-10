import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.test_outputs = []  # Initialize the list here

    def forward(self, x) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.long().squeeze())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.long().squeeze())
        self.log('val_loss', loss)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self(x)
        self.test_outputs.append({'y_true': y, 'y_pred': y_hat})

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []  # Reset the list at the start of each test epoch

    def on_test_epoch_end(self):
        y_true = torch.cat([x['y_true'] for x in self.test_outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in self.test_outputs]).cpu()

        # Convert predictions to class labels
        y_pred_labels = torch.argmax(y_pred, dim=1).numpy()
        y_true_labels = y_true.numpy()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.png')
        plt.close()

        print("Confusion matrix saved as 'confusion_matrix.png'")

        self.test_outputs.clear()

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())


df = get_sliding_dataframe()
num_classes = df['label'].nunique()

# Prepare your data
# Assuming df is your DataFrame and 'target' is your target column
X = df.drop('label', axis=1).values
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to device
X_train = torch.FloatTensor(X_train).unsqueeze(1).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_val = torch.FloatTensor(X_val).unsqueeze(1).to(device)
y_val = torch.FloatTensor(y_val).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
input_size = X_train.shape[2]  # Number of features
hidden_size = 64
num_layers = 2

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# Create a TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="my_lstm")

# Train the model
trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices=1, logger=logger)
trainer.fit(model, train_loader, val_loader)

# Create a test set (you can use the validation set if you don't have a separate test set)
test_dataset = TensorDataset(X_val, y_val)
test_loader = DataLoader(test_dataset, batch_size=32)

# Make predictions
trainer.test(model, test_loader)
predictions = trainer.predict(model, test_loader)

# Concatenate all predictions
y_true = torch.cat([x['y_true'] for x in model.test_outputs]).cpu().numpy()
y_pred = torch.cat([x['y_pred'] for x in model.test_outputs]).cpu()

# Convert predictions to class labels
y_pred_labels = torch.argmax(y_pred, dim=1).numpy()

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix saved as 'confusion_matrix.png'")
