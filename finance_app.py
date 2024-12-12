# Streamlit App for Financial Forecasting
# This Streamlit app predicts the prices of stocks for the upcoming week using machine learning models like LSTM networks.
# It integrates with Yahoo Finance API to fetch historical stock data and provides interactive visualizations.

# Import required libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# Set up the app title
st.title("Stock Price Prediction App")

# "How it works" section
st.write("""
### How it works
- Select a stock ticker symbol and specify a historical data range.
- Configure machine learning model parameters.
- The app fetches data from Yahoo Finance, trains an LSTM model, and predicts the next week's stock prices.
- Visualize historical and predicted prices, and download the results as a CSV file.
""")

# Input section for user parameters
st.sidebar.header("Adjustable Parameters")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL):", value="AAPL")
start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("2023-01-01"))
epochs = st.sidebar.number_input("Training Epochs:", min_value=1, max_value=100, value=10)
lstm_layers = st.sidebar.number_input("LSTM Layers:", min_value=1, max_value=5, value=2)
batch_size = st.sidebar.number_input("Batch Size:", min_value=1, max_value=256, value=32)

# Fetch historical stock data with caching
@st.cache_data
def fetch_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Load data
stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
if stock_data is not None:
    st.write(f"### Historical Data for {stock_symbol}")
    st.dataframe(stock_data.tail())

    # Plot historical data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))
    fig.update_layout(title="Historical Stock Prices", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    prediction_days = 60

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    for _ in range(lstm_layers - 1):
        model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Predict next week's prices
    test_data = scaled_data[-prediction_days:]
    x_test = []
    x_test.append(test_data)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Display predictions
    st.write("### Predicted Prices for the Next Week")
    st.write(predicted_prices)

    # Overlay predictions on historical data
    future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=7)
    predicted_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices.flatten()})
    combined_df = pd.concat([stock_data[['Close']].reset_index(), predicted_df.set_index('Date')], axis=1).fillna(method='ffill')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Close'], mode='lines', name='Historical Prices'))
    fig2.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Predicted Price'], mode='lines', name='Predicted Prices'))
    fig2.update_layout(title="Historical vs Predicted Prices", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig2)

    # Download results as CSV
    st.sidebar.write("### Download Results")
    csv_data = combined_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(label="Download CSV", data=csv_data, file_name="stock_predictions.csv", mime="text/csv")
else:
    st.error("No data available. Please check the stock ticker symbol or date range.")
