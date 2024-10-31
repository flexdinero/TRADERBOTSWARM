import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QTreeWidget, QTreeWidgetItem, QWidget, QTextEdit
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pandas as pd
from alpaca_trade_api import REST, Stream
from datetime import datetime

ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

class TradingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Trading")
        self.setGeometry(100, 100, 1400, 900)
        self.alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
        self.trades = []

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Control Panel
        control_panel = QHBoxLayout()

        self.ticker_label = QLabel("Ticker:")
        self.ticker_input = QLineEdit()
        control_panel.addWidget(self.ticker_label)
        control_panel.addWidget(self.ticker_input)

        self.asset_type_label = QLabel("Asset Type:")
        self.asset_type_combo = QComboBox()
        self.asset_type_combo.addItems(["stock", "crypto"])
        control_panel.addWidget(self.asset_type_label)
        control_panel.addWidget(self.asset_type_combo)

        self.timeframe_label = QLabel("Timeframe:")
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1Min", "5Min", "15Min"])
        control_panel.addWidget(self.timeframe_label)
        control_panel.addWidget(self.timeframe_combo)

        self.strategy_label = QLabel("Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["momentum", "other_strategy", "HighFrequencyTradingSystem"])
        control_panel.addWidget(self.strategy_label)
        control_panel.addWidget(self.strategy_combo)

        self.amount_label = QLabel("Buy Amount:")
        self.amount_input = QLineEdit()
        control_panel.addWidget(self.amount_label)
        control_panel.addWidget(self.amount_input)

        self.units_label = QLabel("Units:")
        self.units_input = QLineEdit()
        control_panel.addWidget(self.units_label)
        control_panel.addWidget(self.units_input)

        self.start_button = QPushButton("Start Trading")
        self.start_button.clicked.connect(self.start_trading)
        control_panel.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Trading")
        self.stop_button.clicked.connect(self.stop_trading)
        control_panel.addWidget(self.stop_button)

        self.balance_label = QLabel("Balance: $100000.00")
        control_panel.addWidget(self.balance_label)

        main_layout.addLayout(control_panel)

        # Plot Area
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        main_layout.addWidget(self.plot_widget)

        # Log Area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        # Dashboard Area
        self.dashboard_tree = QTreeWidget()
        self.dashboard_tree.setColumnCount(6)
        self.dashboard_tree.setHeaderLabels(["Ticker", "Units", "Buy Price", "Sell Price", "Live Price", "Timestamp"])
        main_layout.addWidget(self.dashboard_tree)

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])

    def start_trading(self):
        self.log_text.append("Started trading...")
        self.timer.start(60000)  # Update plot every minute

    def stop_trading(self):
        self.log_text.append("Stopped trading...")
        self.timer.stop()

    def update_log(self, message):
        self.log_text.append(message)

    def update_balance(self, balance):
        self.balance_label.setText(f"Balance: ${balance:.2f}")

    def update_plot(self):
        ticker = self.ticker_input.text().upper()
        asset_type = self.asset_type_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        bars = self.fetch_data(ticker, asset_type, timeframe)

        if bars is not None:
            self.data = self.data.append(bars, ignore_index=True)
            self.plot_candlestick(self.data)

    def plot_candlestick(self, data):
        self.plot_widget.clear()

        candlesticks = []

        for idx, row in data.iterrows():
            timestamp = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            candlesticks.append((timestamp, open_price, close_price, low_price, high_price))

        item = pg.PlotItem()
        candlestick_item = pg.graphicsItems.CandlestickItem.CandlestickItem(candlesticks)
        item.addItem(candlestick_item)
        self.plot_widget.addItem(item)

        for trade in self.trades:
            color = 'g' if trade['side'] == 'buy' else 'r'
            self.plot_widget.plot([trade['timestamp']], [trade['price']], symbol='o', symbolBrush=color, symbolSize=10)

    def update_dashboard(self, ticker, units, buy_price, sell_price, live_price):
        item = QTreeWidgetItem([ticker, str(units), str(buy_price), str(sell_price), str(live_price), datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        self.dashboard_tree.addTopLevelItem(item)

    def fetch_data(self, ticker, asset_type, timeframe):
        try:
            end = datetime.now()
            start = end - pd.DateOffset(days=1)
            start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')

            if asset_type == "stock":
                bars = self.alpaca.get_barset(ticker, timeframe, start=start_str, end=end_str).df[ticker]
            elif asset_type == "crypto":
                ticker = ticker + '/USD'  # Ensure ticker is in correct format for Alpaca API
                bars = self.alpaca.get_crypto_bars(ticker, timeframe, start=start_str, end=end_str).df[ticker]
            else:
                raise ValueError(f"Unsupported asset type: {asset_type}")

            if bars.empty:
                return None

            bars.reset_index(inplace=True)
            bars.rename(columns={'time': 'timestamp'}, inplace=True)
            bars['timestamp'] = bars['timestamp'].apply(lambda x: x.to_pydatetime())
            return bars

        except Exception as e:
            self.update_log(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def place_order(self, ticker, side, qty, asset_type, price):
        try:
            ticker = ticker + '/USD' if asset_type == "crypto" else ticker  # Ensure correct ticker format
            amount = float(self.amount_input.text().replace('$', '').replace(',', ''))
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
            self.update_dashboard(ticker, qty, price, price if side == 'buy' else '', price if side == 'sell' else '')
        except Exception as e:
            self.update_log(f"Error placing order: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingWindow()
    window.show()
    sys.exit(app.exec_())