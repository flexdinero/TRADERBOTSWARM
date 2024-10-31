import sys
import os
import logging
import PyQt5
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
import requests
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import signal
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
import optuna
import joblib
import threading
import time
import alpaca_trade_api as tradeapi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import mplfinance as mpf
import pyqtgraph as pg
import lightweight_charts as lwc
from alpaca_trade_api.rest import REST 
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QMainWindow
from config import APIKEY
from config import SECRETKEY



# Alpaca API configuration
ALPACA_API_KEY = APIKEY
ALPACA_SECRET_KEY = SECRETKEY
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)


class TradingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
        self.trades = []
        self.is_trading = False
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        self.balance = 100000.0
        self.chart = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Trading Bot')
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        self.start_button = QPushButton('Start Trading', self)
        self.start_button.clicked.connect(self.start_trading)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Trading', self)
        self.stop_button.clicked.connect(self.stop_trading)
        self.layout.addWidget(self.stop_button)

        self.log_area = QTextEdit(self)
        self.layout.addWidget(self.log_area)

        self.balance_label = QLabel(f'Balance: ${self.balance:.2f}', self)
        self.layout.addWidget(self.balance_label)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.show()

    def update_log(self, message):
        self.log_area.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}")

    def update_balance(self):
        self.balance_label.setText(f"Balance: ${self.balance:.2f}")

    def start_trading(self):
        self.is_trading = True
        self.update_log("Started trading...")
        self.timer = threading.Timer(60.0, self.trading_process)
        self.timer.start()

    def stop_trading(self):
        self.is_trading = False
        self.update_log("Stopped trading...")
        if hasattr(self, 'timer'):
            self.timer.cancel()

    def trading_process(self):
        if not self.is_trading:
            return

        ticker = 'AAPL'
        asset_type = 'stock'
        timeframe = '1Min'

        self.update_log(f"Fetching data for {ticker} ({asset_type}) on {timeframe} timeframe")
        bars = self.fetch_data(ticker, asset_type, timeframe)

        if bars is not None and not bars.empty:
            self.data = pd.concat([self.data, bars]).drop_duplicates().sort_values('timestamp').reset_index(drop=True)
            self.update_plot(self.data)

            latest_bar = bars.iloc[-1]
            self.update_log(f"Latest price for {ticker}: ${latest_bar['close']:.2f}")

            # Generate trading signal based on the selected strategy
            signal = self.generate_trading_signal(bars)

            self.update_log(f"Generated signal: {signal}")

            if signal == 'BUY':
                self.place_order(ticker, 'buy', 1, asset_type)
            elif signal == 'SELL':
                self.place_order(ticker, 'sell', 1, asset_type)

        if self.is_trading:
            self.timer = threading.Timer(60.0, self.trading_process)
            self.timer.start()

    def fetch_data(self, ticker, asset_type, timeframe):
        try:
            end = datetime.now()
            start = end - timedelta(days=1)
            timeframe_val = TimeFrame.Minute if timeframe == "1Min" else TimeFrame.FiveMinute if timeframe == "5Min" else TimeFrame.FifteenMinute

            if asset_type == "stock":
                stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
                bars_request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=timeframe_val,
                    start=start.strftime('%Y-%m-%dT%H:%M:%S'),
                    end=end.strftime('%Y-%m-%dT%H:%M:%S')
                )
                bars = stock_client.get_stock_bars(bars_request).df
            elif asset_type == "crypto":
                crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
                ticker = f"{ticker}/USD"
                bars_request = CryptoBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=timeframe_val,
                    start=start.strftime('%Y-%m-%dT%H:%M:%S'),
                    end=end.strftime('%Y-%m-%dT%H:%M:%S')
                )
                bars = crypto_client.get_crypto_bars(bars_request).df
            else:
                raise ValueError(f"Unsupported asset type: {asset_type}")

            if bars.empty:
                self.update_log(f"No data received for {ticker}")
                return None

            bars.reset_index(inplace=True)
            bars.rename(columns={'timestamp': 'timestamp'}, inplace=True)
            self.update_log(f"Fetched {len(bars)} bars for {ticker}")
            return bars

        except Exception as e:
            self.update_log(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def place_order(self, ticker, side, qty, asset_type):
        try:
            ticker = f"{ticker}/USD" if asset_type == "crypto" else ticker
            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            trade_price = float(order.filled_avg_price) if order.filled_avg_price else 0
            self.update_log(f"Order placed: {order}")
            trade = {
                'timestamp': datetime.now(),
                'side': side,
                'price': trade_price
            }
            self.trades.append(trade)
            self.update_dashboard(ticker, qty, trade_price if side == 'buy' else '-', trade_price if side == 'sell' else '-', trade_price)
            self.update_plot(self.data)

        except Exception as e:
            self.update_log(f"Error placing order: {str(e)}")

    def update_plot(self, data):
        if not data.empty:
            data['date'] = pd.to_datetime(data['timestamp']).map(datetime.timestamp)
            candlestick_data = data[['date', 'open', 'high', 'low', 'close']].to_dict('records')
            self.chart.set_data(candlestick_data)
            self.chart.show()

    def update_dashboard(self, ticker, units, buy_price, sell_price, live_price):
        self.update_log(f"Ticker: {ticker}, Units: {units}, Buy Price: {buy_price}, Sell Price: {sell_price}, Live Price: {live_price}, Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def generate_trading_signal(self, data):
        if len(data) < 2:
            return 'HOLD'
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return 'BUY'
        elif data['close'].iloc[-1] < data['close'].iloc[-2]:
            return 'SELL'
        else:
            return 'HOLD'


if __name__ == "__main__":
    app = QApplication(sys.argv)
    trading_window = TradingWindow()
    sys.exit(app.exec_())
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
        self.master.setWindowTitle("Trading Bot")
        self.master.setGeometry(300, 200, 300, 200)

        self.training_window = None
        self.trading_window = None

        main_layout = QVBoxLayout()

        training_button = QPushButton("Open Training Window")
        training_button.clicked.connect(self.open_training_window)
        main_layout.addWidget(training_button)

        trading_button = QPushButton("Open Trading Window")
        trading_button.clicked.connect(self.open_trading_window)
        main_layout.addWidget(trading_button)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.master.setCentralWidget(central_widget)

    def open_training_window(self):
        if self.training_window is None or not self.training_window.isVisible():
            self.training_window = QMainWindow()
            self.training_window.setWindowTitle("Training Window")
            self.training_window.setGeometry(300, 200, 800, 600)
            self.training_window.show()
        else:
            self.training_window.raise_()

    def open_trading_window(self):
        if self.trading_window is None or not self.trading_window.isVisible():
            self.trading_window = TradingWindow()
            self.trading_window.show()
        else:
            self.trading_window.raise_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = QMainWindow()
    tradingApp = TradingBotApp(mainWin)
    mainWin.show()
    sys.exit(app.exec_())
# The following parts are additional and need to be properly integrated.

class RealTimeDashboard:
    def __init__(self):
        self.chart = Chart()

    def update_log(self, message):
        print(f"LOG: {message}")

    def update_plot(self, data):
        if not data.empty:
            data['time'] = pd.to_datetime(data['timestamp']).map(datetime.timestamp)
            candlestick_data = data[['time', 'open', 'high', 'low', 'close']].to_dict('records')
            self.chart.set(candlestick_data)
            self.chart.show()

    def update_trade_info(self, ticker, buy_price, sell_price):
        print(f"Ticker: {ticker}, Buy Price: ${buy_price:.2f}, Sell Price: ${sell_price:.2f}")

def main():
    dashboard = RealTimeDashboard()
    
    # Thread to run the dashboard
    def run_dashboard():
        while True:
            time.sleep(1)  # Placeholder for real dashboard update logic

    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.start()
    
    trading_window = TradingWindow()
    trading_window.set_trading_parameters(ticker='AAPL', asset_type='stock', timeframe='5Min', strategy='momentum')
    trading_window.start_trading()

if __name__ == '__main__':
    main()

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