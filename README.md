# Time series experiments
1. Time series classifiers. 5 classes
2. Time series extrapolation. Seq2Seq

# How to run the image
For some reason, the image can not be built from docker compose command. Need to `docker pull` first.
1. `docker pull quay.io/jupyter/pytorch-notebook`
1. `docker compose build`
1. `docker compose run --service-ports jupyter-pytorch /bin/bash`. Start container with `--service-ports` to enable port forwarding.
1. `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`

# Plan for the project
1. Choose the close-price VS time series data. `Open, High, Low` are not used.
2. Digitize the close-price data into 5 classes.
3. Assume the data is sequential. Use LSTM to classify the data.
