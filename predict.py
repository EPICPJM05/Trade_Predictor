import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

def prepare_data(stock_data, look_back=60):
    """
    Prepares data for training by scaling and creating sequences for LSTM.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(stock_data_scaled)):
        X.append(stock_data_scaled[i-look_back:i, 0])
        y.append(stock_data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshaping for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(X_train):
    """
    Builds and compiles the LSTM model for stock price prediction.
    """
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_lstm_model(stock_data):
    """
    Train the LSTM model on stock data and save the trained model.
    """
    X, y, scaler = prepare_data(stock_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build and train the model
    model = build_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    model.save('models/stock_lstm_model.h5')
    print("LSTM model saved!")

    return model, scaler

def predict_stock_price(model, scaler, recent_data):
    """
    Use the trained LSTM model to predict the next stock price.
    """
    recent_data_scaled = scaler.transform(recent_data[-60:].reshape(-1, 1))
    recent_data_scaled = np.reshape(recent_data_scaled, (1, recent_data_scaled.shape[0], 1))
    predicted_price = model.predict(recent_data_scaled)

    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

import pickle

def train_lstm_model(stock_data):
    """
    Train the LSTM model on stock data and save the trained model.
    """
    X, y, scaler = prepare_data(stock_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build and train the model
    model = build_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    model.save('models/stock_lstm_model.h5')
    print("LSTM model saved!")

    # Save the scaler
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    return model, scaler
