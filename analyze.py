import pandas as pd
import matplotlib.pyplot as plt

def analyze_stock_data(stock_data):
    """
    Performs basic stock analysis such as calculating moving averages
    and price change percentage.
    """
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()  # Simple Moving Average
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()  # Long-Term SMA
    
    stock_data['Price_Change'] = stock_data['Close'].pct_change() * 100  # Daily Price Change in percentage

    print("Stock Data Analysis Summary:")
    print(stock_data.tail())  # Displaying the last 5 rows with analysis

    # Plotting the price and moving averages
    stock_data[['Close', 'SMA_50', 'SMA_200']].plot(figsize=(10, 6))
    plt.title("Stock Price and Moving Averages")
    plt.show()

    return stock_data
