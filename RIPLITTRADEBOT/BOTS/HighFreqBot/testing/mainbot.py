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

def generate_synthetic_price_data(start_date, end_date, initial_price=100, file_name=None):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # for reproducibility
    
    prices = [initial_price]
    for _ in range(1, len(date_range)):
        change = np.random.normal(0, 2)  # Random daily change, mean 0, std dev 2
        new_price = max(prices[-1] * (1 + change/100), 0.01)  # Ensure price doesn't go negative
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': date_range,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'Volume': np.random.randint(100000, 1000000, len(date_range))
    })
    
    df.set_index('Date', inplace=True)
    
    if file_name:
        df.to_csv(file_name)
        logger.info(f"Synthetic data written to {file_name}")
    
    return df

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
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

def get_account_info():
    url = f"{ALPACA_BASE_URL}/v2/account"
    response = requests.get(url, headers=headers)
    return json.loads(response.text)

def fetch_real_time_data(symbol):
    # This is a placeholder. In a real implementation, you would fetch data from a real-time API
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    return data

def place_order(symbol, qty, side, order_type, time_in_force="day", limit_price=None):
    url = f"{ALPACA_BASE_URL}/v2/orders"
    
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force
    }
    
    if limit_price:
        data["limit_price"] = limit_price
    
    response = requests.post(url, headers=headers, json=data)
    return json.loads(response.text)

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

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        return self.sia.polarity_scores(text)['compound']

class OptionsTrading:
    def __init__(self, data, initial_balance=100000, risk_per_trade=0.02):
        self.data = data
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.reset()

    def reset(self):
        self.index = 0
        self.positions = []
        self.balance = self.initial_balance
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.iloc[self.index]
        return np.array([
            obs['Open'], obs['High'], obs['Low'], obs['Close'], 
            obs['Volume'], 
            obs['Returns'] if 'Returns' in obs.index else 0,
            obs['Adj Close'] if 'Adj Close' in obs.index else obs['Close']
        ])

    def step(self, action):
        current_price = self.data.iloc[self.index]['Close']
        
        if action == 1:  # Buy
            position_size = self._calculate_position_size(current_price)
            self.positions.append(('long', current_price, position_size))
            self.balance -= current_price * position_size
        elif action == 2:  # Sell
            position_size = self._calculate_position_size(current_price)
            self.positions.append(('short', current_price, position_size))
            self.balance += current_price * position_size
        elif action == 3 and self.positions:  # Close position
            position_type, bought_price, position_size = self.positions.pop(0)
            if position_type == 'long':
                self.balance += current_price * position_size
            else:  # short
                self.balance -= current_price * position_size
        # Action 4 is "Hold", so we do nothing for that

        self.index += 1
        if self.index >= len(self.data) - 1:
            self.done = True

        return self._next_observation(), self._get_reward(), self.done

    def _calculate_position_size(self, price):
        return max(1, int((self.balance * self.risk_per_trade) / price))

    def _get_reward(self):
        return (self.balance - self.initial_balance) / self.initial_balance

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    # ... (rest of the class remains the same)

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
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = keras.models.load_model(file_path)

def train_model(episodes, batch_size, data):
    env = OptionsTrading(data)
    state_size = 7  # Updated to match the number of features in _next_observation
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    
    try:
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
        
        logger.info("Training completed successfully")
        return agent
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Error details: {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}")
        return None

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