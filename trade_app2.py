import pandas as pd
import numpy as np
import joblib  # For loading the trained model
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from datetime import datetime
import matplotlib.pyplot as plt

# --- Step 1: Load the Pretrained Model ---
def load_model(filepath):
    """Load the pretrained model from a file."""
    try:
        model = joblib.load(filepath)
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Model file not found. Please train and save your model as 'trained_model.pkl'.")
        exit()

# Load the saved model
model = load_model('trained_model.pkl')

# --- Step 2: Open File Dialog to Load CSV ---
def load_data_from_csv():
    """Ask the user to upload a CSV file and load the data."""
    Tk().withdraw()  # Hide the root Tkinter window
    filename = askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    
    if filename:
        df = pd.read_csv(filename)
        print(f"Data loaded from {filename}")
        return df
    else:
        print("No file selected. Exiting...")
        exit()

# Load the CSV file data
df = load_data_from_csv()

# --- Step 3: Feature Engineering ---
# Ensure the necessary columns exist in the CSV
required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in df.columns for col in required_columns):
    print("Error: CSV file does not contain the required columns.")
    exit()

# Convert 'Date' to datetime and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)

# Add technical indicators
def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Feature calculations
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = calculate_rsi(df['Close'])
df['Price_Change'] = df['Close'].pct_change()

# Drop rows with missing values caused by rolling calculations
df.dropna(inplace=True)

# --- Step 4: Predict Signals ---
# Prepare the feature set
features = ['SMA_20', 'SMA_50', 'RSI', 'Price_Change']
new_features = df[features]

# Predict signals
df['Signal'] = model.predict(new_features)
df['Signal_Label'] = df['Signal'].map({1: 'Buy', -1: 'Sell', 0: 'Hold'})

# Display recent predictions
print("Recent predictions:")
print(df[['Date', 'Close', 'Signal', 'Signal_Label']].tail())

# --- Step 5: Save Output ---
# Save the predictions to a CSV file
output_file = f"company_signals_{datetime.today().strftime('%Y%m%d')}.csv"
df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}.")

# --- Step 6: Plot Results (Optional) ---
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.scatter(df['Date'][df['Signal'] == 1], df['Close'][df['Signal'] == 1], label='Buy Signal', marker='^', color='green')
plt.scatter(df['Date'][df['Signal'] == -1], df['Close'][df['Signal'] == -1], label='Sell Signal', marker='v', color='red')
plt.title(f"Trading Signals")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
