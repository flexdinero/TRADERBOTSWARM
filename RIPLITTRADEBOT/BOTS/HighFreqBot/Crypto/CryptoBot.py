import pandas as pd
import alpaca_trade_api as tradeapi
import logging
import numpy as np
import time

# Alpaca API configuration
ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neural Network Functions
def preprocess_data(data):
    data['price'] = (data['price'] - data['price'].mean()) / data['price'].std()
    return data[['price']].values.T

def init_params(input_size, hidden_size=82, output_size=1):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / exp_Z.sum(axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = np.eye(A2.shape[0])[Y.reshape(-1)].T
    dZ2 = A2 - one_hot_Y
    dW2 = (1. / m) * np.dot(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = (1. / m) * np.dot(dZ1, X.T)
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def gradient_descent(X, Y, alpha, iterations, input_size, hidden_size, output_size):
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            loss = compute_loss(A2, Y)
            predictions = np.argmax(A2, axis=0)
            accuracy = get_accuracy(predictions, Y)
            logger.info(f"Iteration {i}: Loss = {loss}, Training Accuracy = {accuracy}")
    return W1, b1, W2, b2

def compute_loss(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    loss = -1 / m * np.sum(logprobs)
    return loss

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, axis=0)
    return predictions

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Trading Bot Class
class HighFrequencyScalpingBot:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
        self.data = pd.DataFrame(columns=['timestamp', 'price'])
        self.position = 0
        self.symbol = 'BTC/USD'  # Correct symbol format
        self.order_size = 0.01  # Example order size
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.alpha = 0.01
        self.hidden_size = 82
        self.output_size = 1

    def start(self):
        self.W1, self.b1, self.W2, self.b2 = init_params(1, self.hidden_size, self.output_size)
        self.run_trading_loop()

    def fetch_latest_price(self):
        bars = self.api.get_crypto_bars(self.symbol, tradeapi.TimeFrame.Minute, limit=1).df
        return bars.iloc[0]['close']

    def run_trading_loop(self):
        while True:
            latest_price = self.fetch_latest_price()
            if pd.notna(latest_price):
                new_data = pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'price': [latest_price]})
                if not new_data.empty:
                    self.data = pd.concat([self.data, new_data], ignore_index=True)
                if len(self.data) > 100:
                    self.data = self.data.iloc[-100:]
                self.evaluate_strategy()
            time.sleep(60)  # Adjust sleep time as needed

    def evaluate_strategy(self):
        if len(self.data) < 10:  # Ensure sufficient data points for evaluation
            return

        X = preprocess_data(self.data)
        predictions = make_predictions(X, self.W1, self.b1, self.W2, self.b2)
        action = self.develop_strategy(predictions)

        if action == 1 and self.position == 0:
            self.place_order('buy')
            self.position = 1
        elif action == 0 and self.position == 1:
            self.place_order('sell')
            self.position = 0

    def place_order(self, side):
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=self.order_size,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"Order placed: {order}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")

    def preprocess_data(self, data):
        data['price'] = (data['price'] - data['price'].mean()) / data['price'].std()
        return data[['price']].values.T

    def develop_strategy(self, predictions):
        threshold = 0.5
        return 1 if predictions[-1] > threshold else 0

    def risk_management(self):
        if self.position == 1:
            last_price = self.data['price'].iloc[-1]
            buy_price = self.data['price'][self.data['position'] == 1].iloc[-1]

            if last_price <= buy_price * (1 - self.risk_management['stop_loss']):
                self.place_order('sell')
                self.position = 0
                logger.info(f"Stop loss triggered at {last_price}")

            if last_price >= buy_price * (1 + self.risk_management['take_profit']):
                self.place_order('sell')
                self.position = 0
                logger.info(f"Take profit triggered at {last_price}")

if __name__ == "__main__":
    bot = HighFrequencyScalpingBot()
    bot.start()