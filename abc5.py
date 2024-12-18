import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from tkinter import Tk, filedialog, Button, Label, Frame
import tkinter as tk

# Function to Filter Data Based on Timeframe
def filter_timeframe(data, timeframe):
    if timeframe == "1M":
        return data[-30:]
    elif timeframe == "6M":
        return data[-180:]
    elif timeframe == "1Y":
        return data[-365:]
    else:
        return data  # Full dataset

# Trend Detection Function using Moving Averages
def detect_trend(data):
    data['Short_MA'] = data['Close'].rolling(window=10).mean()
    data['Long_MA'] = data['Close'].rolling(window=50).mean()

    # Trend detection logic
    data['Trend'] = 'Neutral'
    data.loc[data['Short_MA'] > data['Long_MA'], 'Trend'] = 'Bullish'
    data.loc[data['Short_MA'] < data['Long_MA'], 'Trend'] = 'Bearish'
    return data

# Metrics Calculation
def calculate_metrics(data):
    data['Daily_Return'] = data['Close'].pct_change()

    avg_daily_return = data['Daily_Return'].mean()
    volatility = data['Daily_Return'].std()
    sharpe_ratio = avg_daily_return / volatility if volatility != 0 else 0

    return avg_daily_return, volatility, sharpe_ratio

# Price Prediction Function
def predict_prices(data):
    model = LinearRegression()

    # Preparing the data
    data['Day'] = np.arange(len(data))  # Create day index for training
    X = data[['Day']]
    y = data['Close']

    model.fit(X, y)  # Train the model

    # Predict for next 30 days
    future_days = np.arange(len(data), len(data) + 30).reshape(-1, 1)
    predictions = model.predict(future_days)

    # Create a DataFrame for predictions
    prediction_dates = pd.date_range(start=data['Date'].iloc[-1], periods=31, freq='D')[1:]
    prediction_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Close': predictions})

    return prediction_df

# Plotting Function
def plot_trend(data, stock_name, metrics, prediction=None):
    plt.close('all')  # Clear previous figures

    avg_daily_return, volatility, sharpe_ratio = metrics
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting Close Price and Moving Averages
    ax.plot(data['Date'], data['Close'], color='green', label='Close Price', linewidth=2)
    if 'Short_MA' in data.columns:
        ax.plot(data['Date'], data['Short_MA'], label='10-Day MA', color='orange', linewidth=1.5)
    if 'Long_MA' in data.columns:
        ax.plot(data['Date'], data['Long_MA'], label='50-Day MA', color='blue', linewidth=1.5)

    # Highlight Bullish/Bearish Trends
    if 'Trend' in data.columns:
        bullish = data[data['Trend'] == 'Bullish']
        bearish = data[data['Trend'] == 'Bearish']
        ax.scatter(bullish['Date'], bullish['Close'], color='lime', label='Bullish', s=10)
        ax.scatter(bearish['Date'], bearish['Close'], color='red', label='Bearish', s=10)

    # Plot Predictions
    if prediction is not None:
        ax.plot(prediction['Date'], prediction['Predicted_Close'], linestyle='--', color='purple', label='Predicted Prices')

    # Add Metrics
    plt.text(data['Date'].iloc[0], data['Close'].max(),
             f"Avg Daily Return: {avg_daily_return:.5f} ({avg_daily_return*100:.2f}%)\n"
             f"Volatility: {volatility:.5f} ({volatility*100:.2f}%)\n"
             f"Sharpe Ratio: {sharpe_ratio:.4f}",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    plt.title(f"{stock_name} - Trend Analysis and Predictions")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# GUI Application
def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            root.stock_name = file_path.split('/')[-1].split('.')[0]
            root.data = pd.read_csv(file_path)
            required_columns = {'Date', 'Close'}
            if not required_columns.issubset(root.data.columns):
                raise ValueError(f"Missing required columns: {required_columns - set(root.data.columns)}")
            root.data['Date'] = pd.to_datetime(root.data['Date'])
            root.data = detect_trend(root.data)
            update_chart("Max")  # Default view
        except Exception as e:
            print(f"Error loading file: {e}")

def update_chart(timeframe, prediction=None):
    if hasattr(root, 'data'):
        try:
            filtered_data = filter_timeframe(root.data, timeframe)
            metrics = calculate_metrics(filtered_data)
            fig = plot_trend(filtered_data, root.stock_name, metrics, prediction=prediction)
            display_chart(fig)
        except Exception as e:
            print(f"Error updating chart: {e}")

def display_chart(fig):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def predict_future_prices():
    if hasattr(root, 'data'):
        try:
            prediction_df = predict_prices(root.data)
            timeframe = "Max"
            filtered_data = filter_timeframe(root.data, timeframe)
            metrics = calculate_metrics(filtered_data)
            fig = plot_trend(filtered_data, root.stock_name, metrics, prediction=prediction_df)
            display_chart(fig)
        except Exception as e:
            print(f"Error predicting prices: {e}")
    else:
        print("Load a file first.")

# GUI Window
root = Tk()
root.title("Stock Trend and Price Prediction")
root.geometry("800x600")

# File Upload
label = Label(root, text="Upload Historical OHLCV Data", font=("Helvetica", 14))
label.pack(pady=10)

upload_button = Button(root, text="Upload File", command=load_file, font=("Helvetica", 12), bg="lightblue")
upload_button.pack(pady=10)

# Timeframe Buttons
frame_buttons = Frame(root)
frame_buttons.pack()

for timeframe in ["1M", "6M", "1Y", "Max"]:
    Button(frame_buttons, text=timeframe, command=lambda tf=timeframe: update_chart(tf), 
           font=("Helvetica", 10), bg="lightgray", width=10).pack(side="left", padx=5)

# Predict Button
predict_button = Button(root, text="Predict Prices", command=predict_future_prices, font=("Helvetica", 12), bg="lightgreen")
predict_button.pack(pady=10)

# Chart Display Frame
chart_frame = Frame(root)
chart_frame.pack(fill=tk.BOTH, expand=True)

# Exit Button
exit_button = Button(root, text="Exit", command=root.quit, font=("Helvetica", 12), bg="lightcoral")
exit_button.pack(pady=10)

root.mainloop()
