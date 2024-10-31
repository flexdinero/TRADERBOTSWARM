import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
import os
import requests
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import sys
import signal
import logging
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
import optuna
import joblib
import ta 
from typing import List
import tkinter as tk
from tkinter import scrolledtext
import time
import alpaca_trade_api as tradeapi

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from datetime import datetime
import random


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from collections import deque
print("TensorFlow and Keras imports are working correctly.")

class TradingWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Live Trading")
        self.window.geometry("1000x800")

        # Alpaca API configuration
        self.ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
        self.ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
        self.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

        self.api = tradeapi.REST(self.ALPACA_API_KEY, self.ALPACA_SECRET_KEY, self.ALPACA_BASE_URL, api_version='v2')
        self.trading_client = TradingClient(self.ALPACA_API_KEY, self.ALPACA_SECRET_KEY, paper=True)
        self.data_client = StockHistoricalDataClient(self.ALPACA_API_KEY, self.ALPACA_SECRET_KEY)

        self.setup_ui()
        self.is_trading = False

    def setup_ui(self):
        control_frame = ttk.Frame(self.window)
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, padx=5)
        self.symbol_entry = ttk.Entry(control_frame)
        self.symbol_entry.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Asset Type:").grid(row=0, column=2, padx=5)
        self.asset_type_var = tk.StringVar(value="stock")
        asset_type_combo = ttk.Combobox(control_frame, textvariable=self.asset_type_var, values=["stock", "crypto"])
        asset_type_combo.grid(row=0, column=3, padx=5)

        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=4, padx=5)
        self.timeframe_var = tk.StringVar(value="1Min")
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var, values=["1Min", "5Min", "15Min"])
        timeframe_combo.grid(row=0, column=5, padx=5)

        self.start_button = ttk.Button(control_frame, text="Start Trading", command=self.start_trading)
        self.start_button.grid(row=0, column=6, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading)
        self.stop_button.grid(row=0, column=7, padx=5)

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

        self.log_text = scrolledtext.ScrolledText(self.window, width=100, height=10)
        self.log_text.pack(pady=10)

    def update_log(self, message):
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)

    def update_plot(self, data):
        self.ax.clear()
        data['close'].plot(ax=self.ax)
        self.ax.set_title(f"{self.symbol_entry.get()} - {self.timeframe_var.get()} Chart")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.canvas.draw()

    def start_trading(self):
        if not self.is_trading:
            self.is_trading = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            threading.Thread(target=self.trading_process, daemon=True).start()

    def stop_trading(self):
        self.is_trading = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def trading_process(self):
        symbol = self.symbol_entry.get()
        asset_type = self.asset_type_var.get()
        timeframe = self.timeframe_var.get()
        self.update_log(f"Started live trading for {symbol} ({asset_type}) on {timeframe} timeframe")

        while self.is_trading:
            try:
                # Fetch real-time data
                bars = self.fetch_data(symbol, asset_type, timeframe)
                if bars.empty:
                    self.update_log(f"No data received for {symbol}")
                    time.sleep(60)
                    continue

                latest_bar = bars.iloc[-1]

                # Generate trading signal
                signal = self.generate_trading_signal(bars)

                if signal == 'BUY':
                    self.place_order(symbol, 'buy', 1, asset_type)
                elif signal == 'SELL':
                    self.place_order(symbol, 'sell', 1, asset_type)

                # Update the chart
                self.window.after(0, self.update_plot, bars)

                # Check positions
                self.check_positions()

                # Update log with latest price
                self.update_log(f"Latest price for {symbol}: ${latest_bar['close']:.2f}")

                # Wait for the next bar
                time.sleep(60)  # Adjust based on your timeframe

            except Exception as e:
                self.update_log(f"Error in trading process: {str(e)}")
                time.sleep(60)  # Wait before retrying

    def fetch_data(self, symbol, asset_type, timeframe):
        end = datetime.now()
        start = end - timedelta(days=1)
        timeframe_val = TimeFrame.Minute

        if asset_type == "stock":
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe_val,
                start=start,
                end=end
            )
            bars = self.data_client.get_stock_bars(request_params)
        elif asset_type == "crypto":
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe_val,
                start=start,
                end=end
            )
            bars = self.data_client.get_crypto_bars(request_params)
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

        return bars.df

    def generate_trading_signal(self, data):
        if len(data) < 2:
            return 'HOLD'
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return 'BUY'
        elif data['close'].iloc[-1] < data['close'].iloc[-2]:
            return 'SELL'
        else:
            return 'HOLD'

    def place_order(self, symbol, side, qty, asset_type):
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data=order_data)
            self.update_log(f"Order placed: {order}")
        except Exception as e:
            self.update_log(f"Error placing order: {str(e)}")

    def check_positions(self):
        try:
            positions = self.trading_client.get_all_positions()
            self.update_log(f"Current positions: {len(positions)}")
            for position in positions:
                self.update_log(f"Position: {position.symbol} - {position.qty} - {position.current_price}")
        except Exception as e:
            self.update_log(f"Error checking positions: {str(e)}")

    def fetch_latest_quote(self, symbol):
        url = f"{self.ALPACA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
        headers = {
            "PKKTH5ONSE1UBXDXYIOX": self.ALPACA_API_KEY,
            "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi": self.ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['quote']
        else:
            self.update_log(f"Failed to fetch latest quote: {response.text}")
            return None

    def fetch_latest_trade(self, symbol):
        url = f"{self.ALPACA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
        headers = {
            "PKKTH5ONSE1UBXDXYIOX": self.ALPACA_API_KEY,
            "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi": self.ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['trade']
        else:
            self.update_log(f"Failed to fetch latest trade: {response.text}")
            return None

    def fetch_snapshot(self, symbol):
        url = f"{self.ALPACA_BASE_URL}/v2/stocks/{symbol}/snapshot"
        headers = {
            "PKKTH5ONSE1UBXDXYIOX": self.ALPACA_API_KEY,
            "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi": self.ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            self.update_log(f"Failed to fetch snapshot: {response.text}")
            return None

class TradingBotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Trading Bot")
        self.master.geometry("300x200")

        self.trading_window = None

        ttk.Button(self.master, text="Open Trading Window", command=self.open_trading_window).pack(pady=10)

    def open_trading_window(self):
        if self.trading_window is None or not self.trading_window.window.winfo_exists():
            self.trading_window = TradingWindow()
        else:
            self.trading_window.window.lift()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotApp(root)
    root.mainloop()


class TrainingWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Training")
        self.window.geometry("800x600")
        
        self.log_text = scrolledtext.ScrolledText(self.window, width=80, height=20)
        self.log_text.pack(pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.window, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=10, fill=tk.X, padx=20)
        
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Data", command=self.save_data)
        self.save_button.grid(row=0, column=2, padx=5)
        
        self.is_training = False

    def update_log(self, message):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)

    def start_training(self):
        if not self.is_training:
            self.is_training = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            threading.Thread(target=self.training_process).start()

    def stop_training(self):
        self.is_training = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def save_data(self):
        # Implement data saving logic here
        self.update_log("Data saved successfully")

    def training_process(self):
        episode = 0
        while self.is_training:
            episode += 1
            self.update_log(f"Starting episode {episode}")
            for step in range(100):  # Simulating 100 steps per episode
                if not self.is_training:
                    break
                time.sleep(0.1)  # Simulating work
                self.progress_var.set((step + 1) / 100 * 100)
                if step % 10 == 0:
                    self.update_log(f"Episode {episode}, Step {step + 1}: Simulated action")
            self.update_log(f"Episode {episode} completed")
        self.is_training = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.update_log("Training stopped")

class TradingWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Trading")
        self.window.geometry("1000x800")

        # Alpaca API configuration
        self.ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
        self.ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
        self.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

        self.api = tradeapi.REST(self.ALPACA_API_KEY, self.ALPACA_SECRET_KEY, self.ALPACA_BASE_URL, api_version='v2')

        self.setup_ui()
        self.is_trading = False

    def setup_ui(self):
        control_frame = ttk.Frame(self.window)
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Ticker:").grid(row=0, column=0, padx=5)
        self.ticker_entry = ttk.Entry(control_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=2, padx=5)
        self.timeframe_var = tk.StringVar(value="1min")
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var, values=["1min", "5min", "15min"])
        timeframe_combo.grid(row=0, column=3, padx=5)

        self.start_button = ttk.Button(control_frame, text="Start Trading", command=self.start_trading)
        self.start_button.grid(row=0, column=4, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading)
        self.stop_button.grid(row=0, column=5, padx=5)

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

        self.log_text = scrolledtext.ScrolledText(self.window, width=100, height=10)
        self.log_text.pack(pady=10)

    def update_log(self, message):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)

    def update_plot(self, data):
        self.ax.clear()
        data['Close'].plot(ax=self.ax)
        self.ax.set_title(f"{self.ticker_entry.get()} - {self.timeframe_var.get()} Chart")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.canvas.draw()

    def start_trading(self):
        if not self.is_trading:
            self.is_trading = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            threading.Thread(target=self.trading_process).start()

class TradingBot:
    def __init__(self, api_key, secret_key, base_url, ticker, timeframe='1Min'):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.ticker = ticker
        self.timeframe = timeframe
        self.is_trading = True

    def update_log(self, message):
        print(message)

    def update_plot(self, data):
        plt.clf()
        data['Close'].plot()
        plt.title(f"Live {self.ticker} Data")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.draw()
        plt.pause(0.1)

    def fetch_data(self):
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(minutes=100)
        barset = self.api.get_barset(self.ticker, self.timeframe, start=start_time.isoformat(), end=end_time.isoformat())
        return barset[self.ticker]

    def make_trade_decision(self, current_price):
        action = random.choice(["BUY", "SELL"])
        return action

    def trade(self):
        plt.ion()  # Enable interactive mode for live updating charts
        while self.is_trading:
            # Fetch live data
            data = self.fetch_data()
            prices = [bar.c for bar in data]
            dates = [bar.t for bar in data]

            # Create a DataFrame
            df = pd.DataFrame({
                'Close': prices
            }, index=dates)

            # Make trading decision
            current_price = prices[-1]
            action = self.make_trade_decision(current_price)
            self.update_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action} {self.ticker} at ${current_price}")

            # Update the chart
            self.update_plot(df)

            time.sleep(5)  # Wait for 5 seconds before next update

        self.update_log("Trading stopped")

if __name__ == "__main__":
    api_key = 'PKKTH5ONSE1UBXDXYIOX'
    secret_key = 'gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi'
    base_url = 'https://paper-api.alpaca.markets'
    ticker = 'AAPL'

    bot = TradingBot(api_key, secret_key, base_url, ticker)
    bot.trade()

class TradingBotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Trading Bot")
        self.master.geometry("300x200")

        self.training_window = None
        self.trading_window = None

        ttk.Button(self.master, text="Open Training Window", command=self.open_training_window).pack(pady=10)
        ttk.Button(self.master, text="Open Trading Window", command=self.open_trading_window).pack(pady=10)

    def open_training_window(self):
        if self.training_window is None or not self.training_window.window.winfo_exists():
            self.training_window = TrainingWindow()
        else:
            self.training_window.window.lift()

    def open_trading_window(self):
        if self.trading_window is None or not self.trading_window.window.winfo_exists():
            self.trading_window = TradingWindow()
        else:
            self.trading_window.window.lift()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotApp(root)
    root.mainloop()

class TradingBotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Trading Bot")
        self.master.geometry("300x200")

        self.training_window = None
        self.trading_window = None

        ttk.Button(self.master, text="Open Training Window", command=self.open_training_window).pack(pady=10)
        ttk.Button(self.master, text="Open Trading Window", command=self.open_trading_window).pack(pady=10)

    def open_training_window(self):
        if self.training_window is None or not self.training_window.window.winfo_exists():
            self.training_window = TrainingWindow()
        else:
            self.training_window.window.lift()

    def open_trading_window(self):
        if self.trading_window is None or not self.trading_window.window.winfo_exists():
            self.trading_window = TradingWindow()
        else:
            self.trading_window.window.lift()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotApp(root)
    root.mainloop()

    def generate_trading_signal(self, data):
        # Implement your high-frequency trading strategy here
        # This is a simple example and should be replaced with your actual strategy
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return 'BUY'
        elif data['close'].iloc[-1] < data['close'].iloc[-2]:
            return 'SELL'
        else:
            return 'HOLD'

class TradingBotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Trading Bot")
        self.master.geometry("300x200")

        self.training_window = None
        self.trading_window = None

        ttk.Button(self.master, text="Open Training Window", command=self.open_training_window).pack(pady=10)
        ttk.Button(self.master, text="Open Trading Window", command=self.open_trading_window).pack(pady=10)

    def open_training_window(self):
        if self.training_window is None or not self.training_window.window.winfo_exists():
            self.training_window = TrainingWindow()
        else:
            self.training_window.window.lift()

    def open_trading_window(self):
        if self.trading_window is None or not self.trading_window.window.winfo_exists():
            self.trading_window = TradingWindow()
        else:
            self.trading_window.window.lift()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotApp(root)
    root.mainloop()
def main():
    dashboard = RealTimeDashboard()
    
    # Thread to run the Tkinter GUI
    def run_dashboard():
        dashboard.run()
    
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.start()
    
    while True:
        print("\nChoose an option:")
        print("1. Train model")
        print("2. Start trading")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            train_and_save_model(dashboard)
        elif choice == '2':
            start_trading(dashboard)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("Exiting the program...")
    dashboard.root.quit()

def train_and_save_model(dashboard):
    # Preprocess data
    logger.info("Preprocessing training data")
    train_data = preprocess_data('train_data.csv', '2020-01-01', '2022-12-30')
    
    # Train model
    logger.info("Training model")
    agent = train_model(episodes=10, batch_size=32, csv_file_path='train_data.csv', balance=100000, risk_per_trade=0.02, dashboard=dashboard)
    
    # Save trained model
    if agent:
        agent.save('dqn_model.h5')
        logger.info("Model saved as dqn_model.h5")

def start_trading(dashboard):
    # Load the trained model
    try:
        agent = DQNAgent(state_size=18, action_size=5)
        agent.load('dqn_model.h5')
        logger.info("Loaded trained model")
    except:
        logger.error("Failed to load trained model. Please train the model first.")
        return
    
    # Fetch real-time data
    logger.info("Fetching real-time data")
    real_time_data = fetch_real_time_data('BTC-USD', '1m')  # Use 1-minute interval data for Bitcoin
    dashboard.update_plot(real_time_data)
    logger.info(f"Real-time data fetched: {real_time_data.head()}")
    
    # Implement trading strategy
    signals = momentum_strategy(real_time_data)
    portfolio = apply_risk_management(signals, real_time_data, balance=100000, risk_per_trade=0.02)
    dashboard.update_plot(portfolio['total'])
    
    # Start trading loop
    trading_loop(agent, dashboard)

def trading_loop(agent, dashboard):
    while True:
        # Fetch latest data
        latest_data = fetch_real_time_data('BTC-USD', '1m').iloc[-1]
        
        # Prepare state
        state = prepare_state(latest_data)
        
        # Get action from agent
        action = agent.act(state)
        
        # Execute trade based on action
        execute_trade(action, latest_data, dashboard)
        
        # Wait for 1 minute before next iteration
        time.sleep(60)

ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use this for paper trading

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

def execute_trade(action, data, dashboard):
    ticker = 'BTCUSD'
    qty = 0.001

    if action == 1:  # Buy
        order_response = api.submit_order(
            symbol=ticker,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        dashboard.update_log(f"BUY order placed: {order_response}")
    elif action == 2:  # Sell
        order_response = api.submit_order(
            symbol=ticker,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        dashboard.update_log(f"SELL order placed: {order_response}")
    else:
        dashboard.update_log("No trade action taken")

# The rest of your script goes here...

if __name__ == "__main__":
    # Your script initialization and main loop goes here
    pass

def prepare_state(data):
    # Prepare the state based on the latest data
    # This function should return a state vector that matches the input shape of your model
    # You may need to adjust this based on your model's input requirements
    return np.array([
        data['Open'], data['High'], data['Low'], data['Close'], 
        data['Volume'], data['Open'], data['High'], data['Low'], 
        data['Close'], data['Volume'], data['Open'], data['High'], 
        data['Low'], data['Close'], data['Volume'], data['Open'], 
        data['High'], data['Low']
    ]).reshape(1, 18)



class RealTimeDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Trading Dashboard")
        
        self.log_text = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.log_text.pack()
        
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        self.trade_info = tk.StringVar()
        self.trade_label = tk.Label(self.root, textvariable=self.trade_info)
        self.trade_label.pack()
        
    def update_log(self, message):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        
    def update_plot(self, data):
        self.ax.clear()
        data['Close'].plot(ax=self.ax)
        self.ax.set_title("BTC-USD Price")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.canvas.draw()
        
    def update_trade_info(self, ticker, buy_price, sell_price):
        info = f"Ticker: {ticker}, Buy Price: ${buy_price:.2f}, Sell Price: ${sell_price:.2f}"
        self.trade_info.set(info)
        
    def run(self):
        self.root.mainloop()





class HighFrequencyTradingSystem:   
    def __init__(self, exchange: str, symbol: str, timeframes: List[str], risk_per_trade: float = 0.01):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframes = timeframes
        self.risk_per_trade = risk_per_trade

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Script started")

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Alpaca API configuration
ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use this for paper trading
# ALPACA_BASE_URL = "https://api.alpaca.markets"  # Use this for live trading

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

headers = {
    "accept": "application/json",
    "PKKTH5ONSE1UBXDXYIOX": ALPACA_API_KEY,
    "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi": ALPACA_SECRET_KEY
}

def check_account_info(self):
    try:
        account = self.api.get_account()
        logger.info(f"Account information: {account}")
        self.update_log(f"Account status: {account.status}")
        self.update_log(f"Buying power: ${float(account.buying_power):.2f}")
        self.update_log(f"Cash: ${float(account.cash):.2f}")
        self.update_log(f"Portfolio value: ${float(account.portfolio_value):.2f}")
    except Exception as e:
        logger.error(f"Error checking account information: {e}")
        self.update_log(f"Error checking account information: {e}")
def get_account_info():
    url = f"{ALPACA_BASE_URL}/v2/account"
    response = requests.get(url, headers=headers)
    return json.loads(response.text)

def fetch_real_time_data(symbol):
    # This is a placeholder. In a real implementation, you would fetch data from a real-time API
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    return data

def place_order_with_stop_loss(self, ticker, side, qty, stop_loss_percent=0.01):
    try:
        # Place the main order
        order = self.api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        
        logger.info(f"Order placed: {order}")
        
        # Get the fill price
        fill_price = float(order.filled_avg_price)
        
        # Calculate stop loss price
        stop_loss_price = fill_price * (1 - stop_loss_percent) if side == 'buy' else fill_price * (1 + stop_loss_percent)
        
        # Place stop loss order
        stop_loss_order = self.api.submit_order(
            symbol=ticker,
            qty=qty,
            side='sell' if side == 'buy' else 'buy',
            type='stop',
            stop_price=stop_loss_price,
            time_in_force='gtc'
        )
        
        logger.info(f"Stop loss order placed: {stop_loss_order}")
        
        self.update_log(f"{side.upper()} order placed for {ticker} at {fill_price} with stop loss at {stop_loss_price}")
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        self.update_log(f"Error placing order: {e}")

def calculate_performance_metrics(returns):
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming daily returns
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252)
    return {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio
    }

def preprocess_data(data_dir, start_date, end_date):
    try:
        logger.info(f"Attempting to read data from directory: {data_dir}")
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"The path '{data_dir}' is not a directory.")
        
        logger.info(f"Checking permissions for directory: {data_dir}")
        if not os.access(data_dir, os.R_OK):
            raise PermissionError(f"Read permission denied for directory: {data_dir}")
        
        all_data = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Attempting to read file: {file_path}")
                
                # Check if the file is empty or too small
                if os.path.getsize(file_path) == 0 or (filename in ['test_data.csv', 'train_data.csv'] and os.path.getsize(file_path) < 1000):
                    logger.warning(f"File {filename} is empty or too small. Generating synthetic data.")
                    if filename in ['test_data.csv', 'train_data.csv']:
                        synthetic_data = generate_synthetic_price_data(start_date, end_date, file_name=file_path)
                    else:
                        synthetic_data = generate_synthetic_price_data(start_date, end_date)
                    synthetic_data.to_csv(file_path)
                    all_data.append(synthetic_data)
                else:
                    try:
                        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                        logger.info(f"Successfully read file {filename}")
                        logger.info(f"Columns in {filename}: {data.columns.tolist()}")
                        logger.info(f"Shape of data in {filename}: {data.shape}")
                        all_data.append(data)
                    except Exception as e:
                        logger.error(f"Error reading file {filename}: {e}")
        
        if not all_data:
            raise ValueError(f"No valid CSV files found in {data_dir}")
        
        logger.info(f"Concatenating {len(all_data)} dataframes")
        data = pd.concat(all_data, axis=0)
        logger.info(f"Shape of concatenated data: {data.shape}")
        
        # Ensure all required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter data based on start and end dates
        data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
        
        if data.empty:
            raise ValueError(f"No data available between {start_date} and {end_date}")
        
        logger.info(f"Final preprocessed data shape: {data.shape}")
        return data

    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None
        
        # Filter data based on start and end dates
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if data.empty:
            raise ValueError(f"No data available between {start_date} and {end_date}")
        
        logger.info(f"Loaded {len(data)} rows of data")
        
        logger.info("Adding technical indicators")
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
        
        # Add custom features
        data['price_momentum'] = data['Close'].pct_change(5)
        data['volatility'] = data['Close'].rolling(window=20).std()
        
        # Multi-timeframe analysis
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Add Z-score
        data['close_zscore'] = zscore(data['Close'])
        
        # Add Bollinger Bands
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.volatility.bollinger_hband(data['Close']), ta.volatility.bollinger_mavg(data['Close']), ta.volatility.bollinger_lband(data['Close'])
        
        logger.info("Simulating options data")
        options_data = pd.DataFrame({
            'strike_price': data['Close'] + np.random.randn(len(data)) * 5,
            'time_to_expiry': np.linspace(30, 0, len(data)),
            'implied_volatility': np.random.rand(len(data)) * 0.5 + 0.2,
            'underlying_price': data['Close'],
            'option_price': data['Close'] * 0.1 + np.random.randn(len(data)) * 2,
            'delta': np.random.rand(len(data)) * 0.5 + 0.25,
            'gamma': np.random.rand(len(data)) * 0.1,
            'theta': -np.random.rand(len(data)) * 0.5,
            'vega': np.random.rand(len(data)) * 5,
        })
        
        logger.info("Combining historical and options data")
        data = pd.concat([data, options_data], axis=1)
        
        logger.info(f"Combined data shape: {data.shape}")
        logger.info("Columns with NaN values:")
        logger.info(data.isna().sum())
        
        logger.info("Handling NaN values")
        data = data.ffill().bfill()
        
        if data.empty:
            raise ValueError(f"After handling NaN values, no valid data remains")
        
        logger.info(f"Data shape after handling NaNs: {data.shape}")
        
        logger.info("Scaling data")
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        
        logger.info(f"Final preprocessed data shape: {data_scaled.shape}")
        return data_scaled
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None

class MarketRegimeDetector:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(self.clf, param_grid, cv=5)
        grid_search.fit(X, y)
        self.clf = grid_search.best_estimator_

    def predict(self, X):
        return self.clf.predict(X)


def log_trade(self, action, ticker, price):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {action} {ticker} at ${price:.2f}"
    self.update_log(log_entry)
    
    # Save to a CSV file for later analysis
    with open('trade_log.csv', 'a') as f:
        f.write(f"{timestamp},{action},{ticker},{price}\n")

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        return self.sia.polarity_scores(text)['compound']

class OptionsTrading:
    def __init__(self, data, initial_balance, risk_per_trade):
        self.data = data
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        return self._next_observation()
        
    def _next_observation(self):
        obs = self.data.iloc[self.current_step]
        return np.array([
            obs['Open'], obs['High'], obs['Low'], obs['Close'], 
            obs['Volume'], obs['Returns'] if 'Returns' in obs.index else 0,
            obs['Adj Close'] if 'Adj Close' in obs.index else obs['Close']
        ])
        
    def step(self, action):
        # Implement the step logic here
        # This is a placeholder implementation
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = 0  # Calculate the appropriate reward
        return self._next_observation(), reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # ... (rest of the methods remain the same)



def train_model(episodes, batch_size, data, balance, risk_per_trade):
    if data is None or data.empty:
        logger.error("Failed to preprocess data. Exiting training.")
        return None
    
    env = OptionsTrading(data, balance, risk_per_trade)
    state_size = 18  # Updated for new features
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        logger.info(f"Starting episode {e+1}/{episodes}")
        
        for time in range(len(data) - 1):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if time % 100 == 0:
                logger.info(f"  Step {time}/{len(data) - 1}")
            
            if done:
                agent.update_target_model()
                logger.info(f"Episode: {e}/{episodes}, Score: {env.balance}")
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    
    logger.info("Training completed")
    return agent

# Preprocess data
logger.info("Preprocessing training data")
train_data = preprocess_data('train_data.csv', '2020-01-01', '2022-12-30')

# Train model
logger.info("Training model")
agent = train_model(episodes=10, batch_size=32, data=train_data, balance=100000, risk_per_trade=0.02)

def retrain_model(self):
    # Load the trade log
    trade_log = pd.read_csv('trade_log.csv', names=['timestamp', 'action', 'ticker', 'price'])
    
    # Preprocess the data
    # ... (implement your preprocessing steps)
    
    # Retrain the model
    # ... (implement your model retraining logic)
    
    self.update_log("Model retrained with new trade data")

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint_')]
    if not checkpoints:
        return None
    return os.path.join(checkpoint_dir, max(checkpoints))
def backtest(agent, test_data):
    if test_data is None or test_data.empty:
        logger.error("Failed to preprocess test data. Exiting backtest.")
        return
    
    env = OptionsTrading(test_data)
    state = env.reset()
    done = False
    total_reward = 0
    returns = []
    
    while not done:
        state = np.reshape(state, [1, 18])  # Updated for new features
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        returns.append(reward)
        
    performance_metrics = calculate_performance_metrics(returns)
    logger.info(f"Backtest results: {performance_metrics}")

    return performance_metrics
if __name__ == "__main__":
    csv_file_path = r"C:\Users\elijah\TRADERMANDEV\data"
    start_date = "2020-01-01"
    end_date = "2023-12-31"  # Adjust as needed

    logger.info("Starting model training...")
    try:
        data = preprocess_data(csv_file_path, start_date, end_date)
        if data is None or data.empty:
            logger.error("Failed to preprocess data. Exiting training.")
        else:
            trained_agent = train_model(episodes=100, batch_size=32, data=data)
            if trained_agent:
                logger.info("Model training completed successfully.")


                # Add code here for saving the model, backtesting, etc.
            else:
                logger.warning("Training failed to produce a valid agent.")
    except Exception as e:
        logger.error(f"An error occurred during script execution: {str(e)}")
        logger.error(f"Error details: {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}")
    finally:
        logger.info("Script execution completed.")



