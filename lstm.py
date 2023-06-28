import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape, units=50):
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    input_shape = (10, 1)
    model = create_lstm_model(input_shape)
    model.summary()
