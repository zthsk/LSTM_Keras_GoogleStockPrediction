
__author__ = "Kshitiz Tiwari"
__copyright__ = "kshitiz Tiwari2019"
__version__ = "1.0.0"
__license__ = "MIT"

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Importing the training set
dataset_train = pd.read_csv('google.csv')
#making the training set a numpy array
training_set = dataset_train.iloc[:,[1,2,3,4,6]].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
sc_predict = MinMaxScaler(feature_range = (0, 1))
sc_predict.fit_transform(training_set[:,0:1])

# Creating a data structure with n timesteps and 1 output
X_train = []
y_train = []
n_past = 120  # Number of past days data to be remembered by LSTM to predict the future
n_data = len(training_set) #Number of rows in the training set
for i in range(n_past, n_data): #i will start from the nth days and go up to the last day in the training set
    X_train.append(training_set_scaled[i-n_past:i, 0:5])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 120, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 120, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 120, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 120, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 120, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a sixth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 120))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 150, batch_size = 32)


# Getting the real stock price of Google in January 2019
dataset_test = pd.read_csv('google_test.csv')
real_stock_price = dataset_test.iloc[:, 0:1].values

#Removing unwanted column in training and testing dataframe
dataset_test = dataset_test.drop(['Date', 'Adj Close'], axis=1)
dataset_train = dataset_train.drop(['Date', 'Adj Close'], axis=1)

# Getting the predicted stock price of 2019
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - n_past:].values
inputs = inputs.reshape(-1,5)
inputs = sc.transform(inputs)
X_test = []
for i in range(n_past, 124):
    X_test.append(inputs[i-n_past:i, 0:5])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc_predict.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
