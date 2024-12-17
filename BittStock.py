import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from tkinter import Canvas, Image, Tk, filedialog, Button, Label, Frame
import tkinter as tk
from datetime import timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import webbrowser
from plotly.io import to_html
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image, ImageTk
import os
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import io


    
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
    high = data['Close'].max()
    low = data['Close'].min()
    returns = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    return avg_daily_return, volatility, sharpe_ratio, high, low, returns

# Price Prediction Function
def predict_prices(data):
    model = LinearRegression()
    data['Day'] = np.arange(len(data))
    X = data[['Day']]
    y = data['Close']
    model.fit(X, y)

    # Predict for next 30 days
    future_days = np.arange(len(data), len(data) + 30).reshape(-1, 1)
    predictions = model.predict(future_days)

    prediction_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(1), periods=30)
    return pd.DataFrame({'Date': prediction_dates, 'Predicted_Close': predictions})

# Plot Interactive Trend
def plot_interactive_trend(data, prediction=None):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines+markers', name='Close Price',
                             line=dict(color='green', width=2)))
    if 'Short_MA' in data.columns:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Short_MA'], mode='lines', name='10-Day MA', line=dict(color='orange')))
    if 'Long_MA' in data.columns:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Long_MA'], mode='lines', name='50-Day MA', line=dict(color='blue')))
    if prediction is not None:
        fig.add_trace(go.Scatter(x=prediction['Date'], y=prediction['Predicted_Close'], mode='lines',
                                 name='Predicted Prices', line=dict(dash='dot', color='purple')))
    fig.update_layout(title='Stock Price Trend and Prediction', xaxis_title='Date', yaxis_title='Price',
                      hovermode='x unified', template='simple_white')
    return fig



# GUI Functions
def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            root.data = pd.read_csv(file_path)
            root.data['Date'] = pd.to_datetime(root.data['Date'])
            root.data = detect_trend(root.data)
            update_chart("Max")  # Default view
        except Exception as e:
            print(f"Error loading file: {e}")

def update_chart(timeframe):
    if hasattr(root, 'data'):
        filtered_data = filter_timeframe(root.data, timeframe)
        fig = plot_interactive_trend(filtered_data)
        display_chart(fig)

def predict_future_prices():
    if hasattr(root, 'data'):
        prediction_df = predict_prices(root.data)
        filtered_data = filter_timeframe(root.data, "Max")
        fig = plot_interactive_trend(filtered_data, prediction=prediction_df)
        display_chart(fig)

        is_profitable = prediction_df['Predicted_Close'].iloc[-1] > prediction_df['Predicted_Close'].iloc[0]
        profitability_label.config(
            text="Profitable to Buy" if is_profitable else "Not Profitable",
            fg="green" if is_profitable else "red"
        )

def generate_stock_pdf(output_path, data):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    stock_name = data['Stock Name'].iloc[0] if 'Stock Name' in data.columns else "Unknown Stock"
    title = Paragraph(f"Stock Trend Analysis Report for {stock_name}", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    headers = ["Stock Name", "Entry Datetime", "Entry Price", "Exit Datetime",
               "Exit Price", "Quantity", "Profit/Loss Value", "Sharpe Ratio"]
    table_data = [headers]

    profit_loss_values = []
    for _, row in data.iterrows():
        entry_price = row.get('Entry Price', 0)
        exit_price = row.get('Exit Price', 0)
        quantity = row.get('Quantity', 1)
        profit_loss = (exit_price - entry_price) * quantity
        profit_loss_values.append(profit_loss)

        table_row = [
            row.get('Stock Name', "N/A"),
            row.get('Entry Datetime', "N/A"),
            row.get('Entry Price', "N/A"),
            row.get('Exit Datetime', "N/A"),
            row.get('Exit Price', "N/A"),
            row.get('Quantity', "N/A"),
            f"{profit_loss:.2f}",
            "Pending"
        ]
        table_data.append(table_row)

    sharpe_ratio = (np.mean(profit_loss_values) / np.std(profit_loss_values)) if np.std(profit_loss_values) != 0 else 0
    for i in range(1, len(table_data)):
        table_data[i][-1] = f"{sharpe_ratio:.4f}"

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10)
    ]))

    elements.append(table)
    doc.build(elements)

def generate_pdf_from_csv():
    if hasattr(root, 'data'):  # Check if data is loaded
        output_pdf_path = "Stock_Trend_Report.pdf"
        
        # Ensure necessary columns exist
        if not {'Stock Name','Entry Datetime', 'Entry Price', 'Exit Datetime', 'Exit Price', 'Quantity'}.issubset(root.data.columns):
            print("Error: Missing required columns in the CSV file.")
            return
        
        # Generate PDF
        generate_stock_pdf(output_pdf_path, root.data)
        print("PDF generated successfully!")


def display_chart(fig):
    img_bytes = fig.to_image(format="png", width=800, height=400)
    img = Image.open(io.BytesIO(img_bytes))
    img_tk = ImageTk.PhotoImage(img)

    chart_canvas.delete("all")  # Clear previous chart
    chart_canvas.create_image(0, 0, anchor="nw", image=img_tk)
    chart_canvas.image = img_tk

# Metrics Calculation
def calculate_metrics(data):
    # Ensure 'Close' is numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])

    # Check for sufficient and non-constant data
    if len(data) < 2 or data['Close'].nunique() <= 1:
        print("Error: Insufficient or constant 'Close' data.")
        return 0, 0, 0, 0, 0, 0

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()

    avg_daily_return = data['Daily_Return'].mean()
    volatility = data['Daily_Return'].std()
    sharpe_ratio = avg_daily_return / volatility if volatility != 0 else 0

    high = data['Close'].max()
    low = data['Close'].min()
    returns = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100

    print(f"Avg Daily Return: {avg_daily_return}, Volatility: {volatility}, Sharpe: {sharpe_ratio}")
    print(f"High: {high}, Low: {low}, Total Returns: {returns}%")

    return avg_daily_return, volatility, sharpe_ratio, high, low, returns

# GUI Window
root = Tk()
root.title("Stock Trend and Price Prediction")
root.geometry("800x600")

# GUI Functions
def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            root.data = pd.read_csv(file_path, parse_dates=['Date'])
            if not {'Date', 'Close'}.issubset(root.data.columns):
                raise ValueError("Data must contain 'Date' and 'Close' columns")
            root.data = detect_trend(root.data)
            update_chart("Max")

            # Display metrics
            avg_daily_return, volatility, sharpe_ratio, high, low, returns = calculate_metrics(root.data)
            metrics_label.config(
                text=f"High: {high:.2f}, Low: {low:.2f}, Returns: {returns:.2f}%\n"
                     f"Sharpe Ratio: {sharpe_ratio:.4f}, Volatility: {volatility*100:.2f},avg_daily_return: {avg_daily_return*100:.2f}%")
        except Exception as e:
            metrics_label.config(text=f"Error: {e}")

    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Extract stock name from file name
            stock_name = os.path.basename(file_path).split('.')[0]

            # Load CSV
            root.data = pd.read_csv(file_path)
            root.data['Date'] = pd.to_datetime(root.data['Date'], errors='coerce')

            # Print actual column names
            print("Columns in uploaded CSV:", root.data.columns)

            # Map columns to required names if needed
            column_mapping = {
                "Date": "Entry Datetime",     # Example mapping
                "Close": "Exit Price",
                "Volume": "Quantity"          # Map to Quantity
            }

            # Rename columns if they match the expected mapping
            root.data.rename(columns=column_mapping, inplace=True)

            # Add 'Stock Name' column dynamically
            root.data['Stock Name'] = stock_name

            # Check for required columns
            required_columns = {"Stock Name", "Entry Datetime", "Entry Price",
                                "Exit Datetime", "Exit Price", "Quantity"}
            if not required_columns.issubset(root.data.columns):
                print("Error: Missing required columns in the CSV file.")
                return

            # Detect trends
            root.data = detect_trend(root.data)
            update_chart("Max")  # Default view

            print(f"Data loaded for stock: {stock_name}")

        except Exception as e:
            print(f"Error loading file: {e}")
# File Upload Button
upload_button = Button(root, text="Upload File", command=load_file, font=("Helvetica", 12), bg="lightblue")
upload_button.pack(pady=10)

# Metrics Label
metrics_label = Label(root, text="", font=("Helvetica", 12))
metrics_label.pack(pady=5)
# Timeframe Buttons
frame_buttons = Frame(root)
frame_buttons.pack()
for timeframe in ["1M", "6M", "1Y", "Max"]:
    Button(frame_buttons, text=timeframe, command=lambda tf=timeframe: update_chart(tf),
           font=("Helvetica", 10), bg="lightgray", width=10).pack(side="left", padx=5)

# Predict Button
# Predict Prices Button
Button(root, text="Predict Future Prices", command=predict_future_prices, font=("Helvetica", 12), bg="lightgreen").pack(pady=10)

# Profitability Label
profitability_label = Label(root, text="", font=("Helvetica", 12))
profitability_label.pack(pady=5)

# Chart Display Canvas
chart_canvas = Canvas(root, width=800, height=400, bg="white")
chart_canvas.pack(pady=10)
# Generate PDF Report Button
pdf_button = Button(root, text="Generate PDF Report", command=generate_pdf_from_csv, font=("Helvetica", 12), bg="lightcoral")
pdf_button.pack(pady=10)

root.mainloop()
