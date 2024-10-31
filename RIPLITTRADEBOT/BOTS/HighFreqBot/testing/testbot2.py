import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import alpaca_trade_api as tradeapi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# Alpaca API configuration
ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)


class TradingWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Live Trading")
        self.window.geometry("1000x800")

        # Initialize Alpaca clients
        self.stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.crypto_client = CryptoHistoricalDataClient()
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        self.trades = []  # To keep track of trades

        self.setup_ui()
        self.is_trading = False

    def setup_ui(self):
        control_frame = ttk.Frame(self.window)
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Ticker:").grid(row=0, column=0, padx=5)
        self.ticker_entry = ttk.Entry(control_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Asset Type:").grid(row=0, column=2, padx=5)
        self.asset_type_var = tk.StringVar(value="stock")
        asset_type_combo = ttk.Combobox(control_frame, textvariable=self.asset_type_var, values=["stock", "crypto"])
        asset_type_combo.grid(row=0, column=3, padx=5)

        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=4, padx=5)
        self.timeframe_var = tk.StringVar(value="1Min")
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var, values=["1Min", "5Min", "15Min"])
        timeframe_combo.grid(row=0, column=5, padx=5)

        ttk.Label(control_frame, text="Strategy:").grid(row=0, column=6, padx=5)
        self.strategy_var = tk.StringVar(value="momentum")
        strategy_combo = ttk.Combobox(control_frame, textvariable=self.strategy_var, values=["momentum", "other_strategy"])
        strategy_combo.grid(row=0, column=7, padx=5)

        ttk.Label(control_frame, text="Amount:").grid(row=0, column=8, padx=5)
        self.amount_entry = ttk.Entry(control_frame)
        self.amount_entry.grid(row=0, column=9, padx=5)

        self.start_button = ttk.Button(control_frame, text="Start Trading", command=self.start_trading)
        self.start_button.grid(row=0, column=10, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading)
        self.stop_button.grid(row=0, column=11, padx=5)

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
        mpf.plot(data, ax=self.ax, type='candle', style='charles', volume=True)
        for trade in self.trades:
            color = 'g' if trade['side'] == OrderSide.BUY else 'r'
            self.ax.axvline(trade['timestamp'], color=color, linestyle='--')
        self.ax.set_title(f"{self.ticker_entry.get()} - {self.timeframe_var.get()} Chart")
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
        ticker = self.ticker_entry.get().upper()
        asset_type = self.asset_type_var.get()
        timeframe = self.timeframe_var.get()
        strategy = self.strategy_var.get()
        self.update_log(f"Started live trading for {ticker} ({asset_type}) on {timeframe} timeframe using {strategy} strategy")

        while self.is_trading:
            try:
                # Fetch real-time data
                bars = self.fetch_data(ticker, asset_type, timeframe)
                if bars.empty:
                    self.update_log(f"No data received for {ticker}")
                    time.sleep(60)
                    continue

                latest_bar = bars.iloc[-1]

                # Generate trading signal based on the selected strategy
                if strategy == 'momentum':
                    signal = self.generate_trading_signal(bars)
                elif strategy == 'other_strategy':
                    signal = self.generate_other_trading_signal(bars)
                else:
                    signal = 'HOLD'

                if signal == 'BUY':
                    self.place_order(ticker, OrderSide.BUY, 1, asset_type)
                elif signal == 'SELL':
                    self.place_order(ticker, OrderSide.SELL, 1, asset_type)

                # Update the chart
                self.window.after(0, self.update_plot, bars)

                # Check positions
                self.check_positions()

                # Update log with latest price
                self.update_log(f"Latest price for {ticker}: ${latest_bar['close']:.2f}")

                # Wait for the next bar
                time.sleep(60)  # Adjust based on your timeframe

            except Exception as e:
                self.update_log(f"Error in trading process: {str(e)}")
                time.sleep(60)  # Wait before retrying

    def fetch_data(self, ticker, asset_type, timeframe):
        end = datetime.now()
        start = end - timedelta(days=1)
        timeframe_val = TimeFrame(1, TimeFrameUnit.Minute) if timeframe == "1Min" else TimeFrame(5, TimeFrameUnit.Minute) if timeframe == "5Min" else TimeFrame(15, TimeFrameUnit.Minute)

        if asset_type == "stock":
            bars = self.stock_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe_val,
                start=start,
                end=end
            )).df
        elif asset_type == "crypto":
            ticker = ticker + '/USD'  # Ensure ticker is in correct format for Alpaca API
            bars = self.crypto_client.get_crypto_bars(CryptoBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe_val,
                start=start,
                end=end
            )).df
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

        return bars

    def generate_trading_signal(self, data):
        if len(data) < 2:
            return 'HOLD'
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return 'BUY'
        elif data['close'].iloc[-1] < data['close'].iloc[-2]:
            return 'SELL'
        else:
            return 'HOLD'

    def generate_other_trading_signal(self, data):
        # Implement your other strategy here
        return 'HOLD'

    def place_order(self, ticker, side, qty, asset_type):
        try:
            ticker = ticker + '/USD' if asset_type == "crypto" else ticker  # Ensure correct ticker format
            amount = float(self.amount_entry.get())
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=ticker,
                    notional=amount,
                    side=side,
                    time_in_force=TimeInForce.GTC
                )
            )
            self.update_log(f"Order placed: {order}")
            # Log trade for plotting
            self.trades.append({
                'timestamp': datetime.now(),
                'side': side
            })
            self.log_trade(ticker, side, qty, order.filled_avg_price)
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

    def log_trade(self, ticker, side, qty, price):
        trade_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'side': side,
            'quantity': qty,
            'price': price
        }
        trades_df = pd.DataFrame([trade_data])
        trades_file = r"C:\Users\elijah\TRADERMANDEV\data\trades\trades.csv"
        if not os.path.isfile(trades_file):
            trades_df.to_csv(trades_file, index=False)
        else:
            trades_df.to_csv(trades_file, mode='a', header=False, index=False)


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
        # Save training log to CSV
        log_data = self.log_text.get("1.0", tk.END).strip()
        log_lines = log_data.split("\n")
        training_log = []
        for line in log_lines:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                training_log.append(parts)

        training_df = pd.DataFrame(training_log, columns=["timestamp", "message"])
        training_file = r"C:\Users\elijah\TRADERMANDEV\data\training\training.csv"
        if not os.path.isfile(training_file):
            training_df.to_csv(training_file, index=False)
        else:
            training_df.to_csv(training_file, mode='a', header=False, index=False)
        
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
            self.save_data()  # Save after each episode
        self.is_training = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.update_log("Training stopped")


class HighFrequencyTradingSystem:
    from typing import List
    def __init__(self, exchange: str, symbol: str, timeframes: List[str], risk_per_trade: float = 0.01):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframes = timeframes
        self.risk_per_trade = risk_per_trade
        self.position_size = 0
        self.stop_loss = 0.2
        self.take_profit = 0.8
        self.trade_log = []

    def calculate_atr(self, data, period=14):
        """Calculate the Average True Range (ATR) for volatility measurement."""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    def calculate_dynamic_stop_loss(self, atr, multiplier=1.5):
        """Calculate dynamic stop-loss based on ATR."""
        return atr * multiplier

    def calculate_take_profit(self, price, atr, multiplier=2):
        """Calculate dynamic take profit based on ATR."""
        return price - (atr * multiplier)

    def determine_position_size(self, account_balance, stop_loss_amount):
        """Determine the position size based on account balance and stop-loss amount."""
        return (account_balance * self.risk_per_trade) / stop_loss_amount

    def generate_signals(self, data):
        """Generate trading signals based on your strategy."""
        # Example strategy: Simple moving average crossover for short positions
        short_window = 10
        long_window = 50
        signals = pd.DataFrame(index=data.index)
        signals['short_mavg'] = data['close'].rolling(window=short_window, min_periods=1).mean()
        signals['long_mavg'] = data['close'].rolling(window=long_window, min_periods=1).mean()
        signals['signal'] = 0
        signals.loc[short_window:, 'signal'] = np.where(
            signals['short_mavg'][short_window:] < signals['long_mavg'][short_window:], -1, 0)
        signals['positions'] = signals['signal'].diff()
        return signals

    def execute_trade(self, signal, price, account_balance):
        """Execute the trade with calculated position size, stop-loss, and take-profit."""
        atr = self.calculate_atr(price)
        if signal == 'short':
            self.stop_loss = price + self.calculate_dynamic_stop_loss(atr)
            self.take_profit = self.calculate_take_profit(price, atr)
            self.position_size = self.determine_position_size(account_balance, price - self.stop_loss)
            self.trade_log.append({'signal': 'short', 'price': price, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit, 'position_size': self.position_size})
            # Code to place sell order here
        elif signal == 'cover':
            # Code to cover the short position here
            self.trade_log.append({'signal': 'cover', 'price': price})

    def update_trade(self, current_price):
        """Update trade by checking if stop-loss or take-profit is hit."""
        if current_price >= self.stop_loss or current_price <= self.take_profit:
            # Code to cover the short position here
            self.trade_log.append({'signal': 'cover', 'price': current_price})
            # Reset stop_loss and take_profit
            self.stop_loss = 0.2
            self.take_profit = 0.8
    def monitor_trades(self, live_data):
        """Monitor live data to manage open positions."""
        for index, row in live_data.iterrows():
            self.update_trade(row['close'])
            # Check for new signals and execute trades if conditions are met
            signals = self.generate_signals(live_data.loc[:index])
            for idx, signal in signals['positions'].iteritems():
                if signal == -1:
                    self.execute_trade('short', row['close'], account_balance)
                elif signal == 1:
                    self.execute_trade('cover', row['close'], account_balance)

    def place_order(self, ticker, side, qty, asset_type, price):
        try:
            ticker = ticker + '/USD' if asset_type == "crypto" else ticker  # Ensure correct ticker format
            amount = float(self.amount_entry.get().replace('$', '').replace(',', ''))
            order = self.alpaca.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            self.update_log(f"{ticker} {side} at {amount} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", side, price)
            self.trades.append({
                'timestamp': datetime.now(),
                'side': side,
                'price': price
            })
            self.log_trade({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': ticker,
                'units': qty,
                'buy_price': price if side == 'buy' else '',
                'sell_price': price if side == 'sell' else '',
                'live_price': price
            })
            self.update_dashboard(ticker, qty, price, price if side == 'buy' else '', price if side == 'sell' else '')
        except Exception as e:
            self.update_log(f"Error placing order: {str(e)}")

    def check_positions(self):
        try:
            positions = self.alpaca.list_positions()
            self.update_log(f"Current positions: {len(positions)}")
            for position in positions:
                self.update_log(f"Position: {position.symbol} - {position.qty} - {position.current_price}")
        except Exception as e:
            self.update_log(f"Error checking positions: {str(e)}")

    def log_trade(self, trade):
        trades_file = r"C:\Users\elijah\TRADERMANDEV\data\trades.csv"
        if not os.path.isfile(trades_file):
            pd.DataFrame([trade]).to_csv(trades_file, index=False, columns=['timestamp', 'ticker', 'units', 'buy_price', 'sell_price', 'live_price'])
        else:
            pd.DataFrame([trade], index=[len(pd.read_csv(trades_file))]).to_csv(trades_file, mode='a', header=False, index=False)

    def fetch_balance(self):
        try:
            account = self.alpaca.get_account()
            return float(account.cash)
        except Exception as e:
            self.update_log(f"Error fetching balance: {str(e)}")
            return self.current_balance

    def update_balance(self):
        self.current_balance = self.fetch_balance()
        self.balance_label.setText(f"Balance: ${self.current_balance:.2f}")


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

# The following parts are additional and need to be properly integrated.

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

# Additional functions needed for completeness

def preprocess_data(data_dir, start_date, end_date):
    # Preprocess the data and return it
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
        
def generate_synthetic_price_data(start_date, end_date, file_name=None):
    dates = pd.date_range(start=start_date, end=end_date, freq='T')
    prices = np.cumsum(np.random.randn(len(dates))) + 100
    data = pd.DataFrame(data={'Date': dates, 'Close': prices})
    data.set_index('Date', inplace=True)
    return data

def momentum_strategy(data):
    data['momentum'] = data['Close'] - data['Close'].shift(1)
    data['signal'] = 0
    data.loc[data['momentum'] > 0, 'signal'] = 1
    data.loc[data['momentum'] < 0, 'signal'] = -1
    return data

def apply_risk_management(signals, data, balance, risk_per_trade):
    positions = pd.DataFrame(index=data.index)
    positions['positions'] = signals['signal'] * (balance * risk_per_trade / data['Close'])
    portfolio = positions.cumsum()
    portfolio['total'] = portfolio['positions'] * data['Close']
    return portfolio

def prepare_state(data):
    # Prepare the state based on the latest data
    return np.array([
        data['Open'], data['High'], data['Low'], data['Close'], 
        data['Volume'], data['Open'], data['High'], data['Low'], 
        data['Close'], data['Volume'], data['Open'], data['High'], 
        data['Low'], data['Close'], data['Volume'], data['Open'], 
        data['High'], data['Low']
    ]).reshape(1, 18)

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

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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