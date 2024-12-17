import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
import numpy as np

# Trend Detection Function using Moving Averages
def detect_trend(data):
    data['Short_MA'] = data['Close'].rolling(window=10).mean()  # Short-term Moving Average
    data['Long_MA'] = data['Close'].rolling(window=50).mean()  # Long-term Moving Average

    # Trend detection logic
    data['Trend'] = 'Neutral'
    data.loc[data['Short_MA'] > data['Long_MA'], 'Trend'] = 'Bullish'
    data.loc[data['Short_MA'] < data['Long_MA'], 'Trend'] = 'Bearish'
    return data

# Function to Calculate High, Low, and Returns
def calculate_metrics(data):
    high = data['Close'].max()
    low = data['Close'].min()
    returns = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    return high, low, returns

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

# Function to Predict Future Prices
def predict_prices(data, days_ahead=30):
    model = LinearRegression()

    # Prepare data: Convert Date to numerical value
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]
    y = data['Close']

    # Train the model
    model.fit(X, y)

    # Predict future days
    future_days = np.arange(data['Days'].max() + 1, data['Days'].max() + days_ahead + 1).reshape(-1, 1)
    predicted_prices = model.predict(future_days)

    # Create future date range
    future_dates = pd.date_range(data['Date'].max(), periods=days_ahead + 1)[1:]
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices})
    return prediction_df

# Plotting Function
# Plotting Function
def plot_trend(data, stock_name, metrics, prediction=None):
    plt.close('all')  # Close all previously opened figures
    
    high, low, returns = metrics

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], color='green', label='Close Price', linewidth=2)
    ax.fill_between(data['Date'], data['Close'], color='lightgreen', alpha=0.3)

    # Moving Averages
    ax.plot(data['Date'], data['Short_MA'], label='10-Day MA', color='orange', linewidth=1.5)
    ax.plot(data['Date'], data['Long_MA'], label='50-Day MA', color='blue', linewidth=1.5)

    # Highlight Bullish/Bearish Trends
    bullish = data[data['Trend'] == 'Bullish']
    bearish = data[data['Trend'] == 'Bearish']
    ax.scatter(bullish['Date'], bullish['Close'], color='lime', label='Bullish Trend', s=10)
    ax.scatter(bearish['Date'], bearish['Close'], color='red', label='Bearish Trend', s=10)

    # Plot Predictions if available
    if prediction is not None:
        ax.plot(prediction['Date'], prediction['Predicted_Close'], color='purple', linestyle='--', label='Predicted Prices')

    # Add Annotations for Metrics
    plt.text(data['Date'].iloc[0], low, 
             f"High: {high:.2f}\nLow: {low:.2f}\nReturns: {returns:.2f}%", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"{stock_name} - Trend Analysis and Predictions")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


# GUI Application
class TrendAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Trend Detection & Prediction")
        self.root.geometry("600x450")

        # UI Components
        label = tk.Label(root, text="Upload Historical OHLCV Data", font=("Helvetica", 12))
        label.pack(pady=10)

        upload_button = tk.Button(root, text="Upload File", command=self.load_file, font=("Helvetica", 12), bg="lightblue")
        upload_button.pack(pady=5)

        self.timeframe_frame = tk.Frame(root)
        self.timeframe_frame.pack(pady=5)

        # Timeframe Buttons
        self.timeframes = ["1M", "6M", "1Y", "Max"]
        for tf in self.timeframes:
            btn = tk.Button(self.timeframe_frame, text=tf, command=lambda t=tf: self.update_chart(t), 
                            font=("Helvetica", 10), bg="lightgray")
            btn.pack(side="left", padx=5)

        self.predict_button = tk.Button(root, text="Predict Prices", command=self.predict_and_plot, font=("Helvetica", 12), bg="lightgreen")
        self.predict_button.pack(pady=5)

        # Canvas for Chart Display
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill="both", expand=True)

        self.data = None
        self.stock_name = ""
        self.current_fig = None
        self.prediction = None

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.stock_name = file_path.split('/')[-1].split('.')[0]
            self.data = pd.read_csv(file_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = detect_trend(self.data)

            # Default chart for full data
            self.update_chart("Max")

    def update_chart(self, timeframe):
        if self.data is not None:
            filtered_data = filter_timeframe(self.data, timeframe)
            metrics = calculate_metrics(filtered_data)

            if self.current_fig:
                self.current_fig.clf()  # Clear previous figure

            self.current_fig = plot_trend(filtered_data, self.stock_name, metrics, self.prediction)
            self.display_chart(self.current_fig)

    def predict_and_plot(self):
        if self.data is not None:
            self.prediction = predict_prices(self.data, days_ahead=30)
            self.update_chart("Max")  # Update chart with predictions

    def display_chart(self, fig):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# Main Function
if __name__ == "__main__":
    root = tk.Tk()
    app = TrendAnalysisApp(root)
    root.mainloop()
