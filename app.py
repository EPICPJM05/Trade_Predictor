from fetch_data import fetch_stock_data
from analyze import analyze_stock_data
from predict import train_lstm_model, predict_stock_price
import os

def main():
    symbol = "AAPL"  # You can change this to any stock symbol
    stock_data = fetch_stock_data(symbol)

    # Step 1: Analyze stock data
    analyzed_data = analyze_stock_data(stock_data)

    # Step 2: Train the LSTM model (if it hasn't been trained)
    if not os.path.exists('models/stock_lstm_model.h5'):
        print("Training the LSTM model...")
        model, scaler = train_lstm_model(analyzed_data)
    else:
        print("LSTM model already trained. Loading model...")
        from tensorflow.keras.models import load_model
        model = load_model('models/stock_lstm_model.h5')

        # Load the scaler used during training
        import pickle
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

    # Step 3: Make a prediction
    recent_data = stock_data['Close'].values
    predicted_price = predict_stock_price(model, scaler, recent_data)
    print(f"Predicted next price for {symbol}: {predicted_price[0][0]}")

if __name__ == "__main__":
    main()
