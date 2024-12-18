import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

# Trend Detection Function using Moving Averages
def detect_trend(data):
    data['Short_MA'] = data['Close'].rolling(window=10).mean()  # Short-term Moving Average
    data['Long_MA'] = data['Close'].rolling(window=50).mean()  # Long-term Moving Average

    # Trend detection logic
    data['Trend'] = 'Neutral'
    data.loc[data['Short_MA'] > data['Long_MA'], 'Trend'] = 'Bullish'
    data.loc[data['Short_MA'] < data['Long_MA'], 'Trend'] = 'Bearish'
    
    return data

# Plotting Function
def plot_trend(data, stock_name):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], data['Short_MA'], label='10-Day MA', color='orange')
    plt.plot(data['Date'], data['Long_MA'], label='50-Day MA', color='green')

    # Highlight Bullish/Bearish trends
    bullish = data[data['Trend'] == 'Bullish']
    bearish = data[data['Trend'] == 'Bearish']
    plt.scatter(bullish['Date'], bullish['Close'], color='lime', label='Bullish Trend', s=10)
    plt.scatter(bearish['Date'], bearish['Close'], color='red', label='Bearish Trend', s=10)

    plt.title(f'Trend Detection for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# GUI Application
def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        stock_name = file_path.split('/')[-1].split('.')[0]
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Detect trend and plot
        data_with_trend = detect_trend(data)
        plot_trend(data_with_trend, stock_name)

# GUI Window
root = tk.Tk()
root.title("Trend Detection and Trade Strategy Builder")
root.geometry("400x200")

label = tk.Label(root, text="Upload Historical OHLCV Data", font=("Helvetica", 12))
label.pack(pady=20)

upload_button = tk.Button(root, text="Upload File", command=load_file, font=("Helvetica", 12), bg="lightblue")
upload_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Helvetica", 12), bg="lightcoral")
exit_button.pack(pady=10)

root.mainloop()
