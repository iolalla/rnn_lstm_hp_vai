import tensorflow as tf
import pandas as pd
import numpy as np

filedata="data/reall-complete-2000-2020.csv"
print("Loading data...\n")
data = pd.read_csv(filedata)
# Filter for SAN.MC ticker to get a single time series
data = data[data['Ticker'] == 'SAN.MC']
print("Data Head summary: %s \n", data.head())
# Selecting only the columns we need
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]), format='%Y-%m-%d')  # Selecting only date
data.set_index('Date', drop=True, inplace=True)  # Setting date column as index

datoz = data[['Close']].to_numpy()
input_data = datoz[:-10]
targets = datoz[10:]

dataset = tf.keras.utils.timeseries_dataset_from_array(
    input_data, targets, sequence_length=10)
for batch in dataset:
    inputs, targets = batch
    assert np.array_equal(inputs[0], datoz[:10])  # First sequence: steps [0-9]
    # Corresponding target: step 10
    assert np.array_equal(targets[0], datoz[10])
    break
