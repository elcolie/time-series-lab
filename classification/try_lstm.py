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
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from collections import Counter

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
device = torch.device("cpu")    # Small data, so using CPU

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.test_outputs = []  # Initialize the list here

    def forward(self, x) -> torch.Tensor:
        # print(f"Input type: {type(x)}, shape: {x.shape if isinstance(x, torch.Tensor) else None}")
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
        y_hat = self.forward(x)
        self.test_outputs.append({'y_true': y, 'y_pred': y_hat})
        return {'y_true': y, 'y_pred': y_hat}

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []  # Reset the list at the start of each test epoch

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())


window_size: int = 32
batch_size: int = 32
max_epochs: int = 100
df = get_sliding_dataframe(
    "/Users/sarit/mein-codes/time-series-lab/time_series_data/SET_DLY_BBL, 5_d2141.csv",
    window_size=window_size
)
print(f"DataFrame shape after sliding window: {df.shape}")
print(f"Number of samples for each label:\n{df['label'].value_counts()}")
min_samples_per_class = 2  # or whatever minimum you deem appropriate
label_counts = df['label'].value_counts()
if (label_counts < min_samples_per_class).any():
    print(f"Warning: Some classes have fewer than {min_samples_per_class} samples.")
    print(label_counts[label_counts < min_samples_per_class])

num_classes = df['label'].nunique()

# Prepare your data
# Assuming df is your DataFrame and 'target' is your target column
X_raw = df.drop('label', axis=1).values
le = LabelEncoder()
y_raw = le.fit_transform(df['label'])

ros = RandomOverSampler(random_state=0, sampling_strategy='not majority')
X, y = ros.fit_resample(X_raw, y_raw)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
result = Counter(y)
print(result)

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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Check for empty batches
for loader in [train_loader, val_loader]:
    for batch in loader:
        if len(batch[0]) == 0 or len(batch[1]) == 0:
            print(f"Warning: Empty batch found in {'train' if loader == train_loader else 'validation'} loader")

# Initialize the model
input_size = X_train.shape[2]  # Number of features
hidden_size = 64
num_layers = 2

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# Create a TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="my_lstm")

# Train the model
trainer = pl.Trainer(max_epochs=max_epochs, accelerator='auto', devices=1, logger=logger)
trainer.fit(model, train_loader, val_loader)

# Create a test set (you can use the validation set if you don't have a separate test set)
test_dataset = TensorDataset(X_val, y_val)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Make predictions
trainer.test(model, dataloaders=test_loader)

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
plt.savefig(f'./classification/confusion_matrix_{window_size}_{max_epochs}.png')
plt.close()

# After calculating y_true and y_pred_labels
accuracy = accuracy_score(y_true, y_pred_labels)
precision = precision_score(y_true, y_pred_labels, average='weighted')
recall = recall_score(y_true, y_pred_labels, average='weighted')
f1 = f1_score(y_true, y_pred_labels, average='weighted')
mcc = matthews_corrcoef(y_true, y_pred_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
with open(f"./classification/metrics_{window_size}_{max_epochs}.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n")

print(f"Confusion matrix saved as 'confusion_matrix_{window_size}_{max_epochs}.png'")
