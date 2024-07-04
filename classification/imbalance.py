# +
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import get_sliding_dataframe
from collections import Counter
# -

df = get_sliding_dataframe()

n_size = len(df.columns) - 1 # Omit `label` column
print(n_size)

X = df[list(range(n_size))] # n = 5
y = df['label'] * 10  # Otherwise label must be int not float

# +
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
# Apply Random Oversampling to the training data
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Train a classifier on the resampled data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
# -


