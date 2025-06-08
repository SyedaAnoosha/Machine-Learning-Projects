# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

# Header
st.title("📈 Stock Price Predictor (LSTM)")
st.markdown("Predict tomorrow's closing price using LSTM.")

# Select stock and date range
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
start_date = st.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.date_input("End Date", datetime.today())

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df[['Close']].dropna()

df = load_data(stock)

# Plot raw closing price
st.subheader("📊 Closing Price Over Time")
st.line_chart(df['Close'])

# Prepare data for LSTM
window_size = 60
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # last 60 days to predict next day
X, y = create_dataset(scaled_data, look_back)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',)
    return model

if st.button("Train & Predict"):
    with st.spinner("Training the model..."):
        model = build_model()
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        preds = model.predict(X_test)
        actual = scaler.inverse_transform(y_test.reshape(-1,1))
        preds = scaler.inverse_transform(preds)

        # Plot actual vs predicted
        st.subheader("📈 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(actual, label="Actual")
        ax.plot(preds, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # Predict tomorrow
        last_60 = scaled_data[-window_size:].reshape(1, window_size, 1)
        next_day = model.predict(last_60)
        tomorrow_price = scaler.inverse_transform(next_day)[0][0]
        st.subheader(f"🧾 Predicted Closing Price for Tomorrow: **${tomorrow_price:.2f}**")
