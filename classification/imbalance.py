# +
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import get_sliding_dataframe
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# -

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

df = get_sliding_dataframe()

n_size = len(df.columns) - 1 # Omit `label` column

X = df[list(range(n_size))] # n = 5
X = X.astype(np.float32) # MPS does not support float64
y = df['label'] # Otherwise label must be int not float

le = LabelEncoder()
le.fit(y)

# +
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, le.transform(y), test_size=0.2, random_state=42)

# Apply SMOTE to the training data
# Apply Random Oversampling to the training data
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
# -

output_dim = len(Counter(y_train_res))

# Train a classifier on the resampled data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the LSTM network
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(x.device).squeeze()
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(x.device).squeeze()

        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))
        out1 = self.relu(out)
        out2 = self.fc(out1)
        out3 = torch.softmax(out2, dim=-1)
        return out3

# Define model parameters
input_dim = n_size
hidden_dim = 50
output_dim = output_dim
n_layers = 2

# Initialize the skorch NeuralNetClassifier
net = NeuralNetClassifier(
    LSTMNet,
    module__input_dim=input_dim,
    module__hidden_dim=hidden_dim,
    module__output_dim=output_dim,
    module__n_layers=n_layers,
    max_epochs=20,
    lr=0.001,
    batch_size=1,
    iterator_train__shuffle=True,
    device='mps' if torch.backends.mps.is_available() else 'cpu'
)

# Create a pipeline with a scaler and the neural network
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('net', net)
])

# -

# Train the model
clf.fit(X_train_res, y_train_res)


# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test.astype(np.int64), y_pred.astype(np.int64)))


