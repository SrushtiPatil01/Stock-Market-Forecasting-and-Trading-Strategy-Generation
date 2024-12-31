import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries


# Data collection and preparation
def get_stock_data(symbol="META", api_key="YOUR_API_KEY"):
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="full")
    return data["4. close"].sort_index()


def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back : i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


# Build LSTM model
def create_model(look_back):
    model = Sequential(
        [
            LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Main execution
def predict_stock_returns():
    look_back = 60

    # Get data
    data = get_stock_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(data, look_back)

    # Create and train model
    model = create_model(look_back)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    return train_predict, test_predict


if __name__ == "__main__":
    train_predictions, test_predictions = predict_stock_returns()
