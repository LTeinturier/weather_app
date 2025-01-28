import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lag_features(df, lag=1):
    for i in range(1, lag + 1):
        df[f'temp_lag_{i}'] = df['temperature'].shift(i)
    return df

def prepare_data_for_lstm(df, target, lag=1):
    df = create_lag_features(df, lag)
    df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []
    for i in range(lag, len(scaled_data)):
        X.append(scaled_data[i-lag:i, :-1])
        y.append(scaled_data[i, -1])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1)

    return model

def train_model(df, target):
    X, y, scaler = prepare_data_for_lstm(df, target, lag=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = train_lstm_model(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_scaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
    y_pred_scaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]

    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    plot_url = plot_results(y_test_scaled, y_pred_scaled)

    return X_train, X_test, y_train, y_test, model, y_pred_scaled, mse, plot_url

def plot_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Temperature')
    plt.plot(y_pred, label='Predicted Temperature', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Weather Pattern Prediction in Paris')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

