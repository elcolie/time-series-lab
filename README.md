# Time series experiments
1. Time series classifiers. 5 classes
2. Time series extrapolation. Seq2Seq

# How to run the image
For some reason, the image can not be built from docker compose command. Need to `docker pull` first.
1. `docker pull quay.io/jupyter/pytorch-notebook`
1. `docker compose build`
1. `docker compose run --service-ports jupyter-pytorch /bin/bash`. Start container with `--service-ports` to enable port forwarding.
1. `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`

# Tensorboard
1. `tensorboard --logdir=tb_logs --bind_all`

# MPS does not work on docker
https://github.com/pytorch/pytorch/issues/81224

# Plan for the project
1. Choose the close-price VS time series data. `Open, High, Low` are not used.
2. Digitize the close-price data into 5 classes.
3. Assume the data is sequential. Use LSTM to classify the data.

# Imbalance data
When dealing with imbalanced data like this, there are several techniques you can use to address the issue. Here are some common approaches:

Oversampling the minority classes:

Random Oversampling
SMOTE (Synthetic Minority Over-sampling Technique)
Undersampling the majority class:

Random Undersampling
Cluster Centroids
Combination of Oversampling and Undersampling:

SMOTEENN
SMOTETomek
Adjust class weights:

Many machine learning algorithms allow you to specify class weights
Use ensemble methods:

Random Forest with balanced subsample
BalancedRandomForestClassifier


# Classifications
1. Vary window size from 5, 10, 15, 20
2. Vary neural network architecture, LSTM, GRU, RNN
3. Vary the number of layers
4. Vary the number of neurons in each layer
5. Vary the number of epochs
6. X is the close price of the stock. Y is the class of the stock price. 7 classes. Y is calculated from the latest rate of change of closed-price data.
