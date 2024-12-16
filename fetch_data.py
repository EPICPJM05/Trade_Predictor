import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol, period="7d", interval="1m"):
    """
    Fetches stock data from Yahoo Finance
    """
    stock_data = yf.download(symbol, period=period, interval=interval)
    stock_data.to_csv(f"data/{symbol}_stock_data.csv")
    print(f"Stock data for {symbol} saved to 'data/{symbol}_stock_data.csv'")
    return stock_data
