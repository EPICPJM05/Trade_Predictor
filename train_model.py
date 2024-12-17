# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import yfinance as yf

# --- Step 1: Load Data ---
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

df = load_data('AAPL', '2015-01-01', '2023-01-01')  # Example: Apple stock
print("Data loaded successfully!")

# --- Step 2: Feature Engineering ---
# Moving Averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# RSI Calculation
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Close'])
df['Price_Change'] = df['Close'].pct_change()
df['Future_Change'] = df['Close'].shift(-5) / df['Close'] - 1

# Target Labels
df['Target'] = np.where(df['Future_Change'] > 0.05, 1, 
                        np.where(df['Future_Change'] < -0.05, -1, 0))
df.dropna(inplace=True)

# --- Step 3: Train Model ---
features = ['SMA_20', 'SMA_50', 'RSI', 'Price_Change']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Step 4: Save Model ---
joblib.dump(model, 'trained_model.pkl')
print("Model saved as 'trained_model.pkl'.")
