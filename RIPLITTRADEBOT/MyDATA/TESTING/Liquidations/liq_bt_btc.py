'''
# Backtest results and parameters
Start                               2024-06-04 13:13:00
End                                 2024-06-20 12:13:00
Duration                            15 days 23:00:00
Exposure Time [%]                   100
Equity Final [$]                    104352.4348674
Equity Peak [$]                     104352.4348444
Return [%]                          4.352435
Buy & Hold Return [%]               -5.411047
Return (Ann.) [%]                   149.60968
Volatility (Ann.) [%]               23.755338
Sharpe Ratio                        6.297939
Sortino Ratio                       602.357841
Calmar Ratio                        116.033975
Max. Drawdown [%]                   -1.718893
Avg. Drawdown [%]                   -0.711823
Max. Drawdown Duration              1 days 18:41:00
Avg. Drawdown Duration              0 days 00:06:00
# Trades                            9
Win Rate [%]                        44.444444
Best Trade [%]                      2.294093
Worst Trade [%]                     -0.977541
Avg. Trade [%]                      0.720826
Max. Trade Duration                 2 days 23:11:00
Avg. Trade Duration                 0 days 09:09:00
Profit Factor                       4.688766
Expectancy [%]                      1.672857
SQN                                 1.621587
_strategy                           LiquidationStrat
_equity_curve                       ...
_trades                             Size  EntryB...
dtype: object
Best Parameters:
Liquidation Threshold: 4,000,000
Time Window (minutes): 22
Take Profit: 0.02
Stop Loss: 0.01
'''

import numpy as np
from backtesting import Backtest, Strategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
class LiquidationStrategy(Strategy):

    Liquidation_thresh = 500000 # Default Liquidation threshold
    time_window_mins = 30 # Default time window in minutes
    take_profit = 0.10 # Default take profit as 10% (0.10)
    stop_loss = 0.05 # Default stop loss as 5% (0.05), 

    def init(self):
        self.liquidations = self.data.liquidations

    def next(self):
        current_idx = len(self.data.Close) - 1  
        current_time = self.data.index[current_idx] # Current time

        # Define the start time of the window
        start_time = current_time - pd.Timedelta(minutes=self.time_window_mins)

        # Find the index of the start time
        start_idx = np.searchsorted(self.data. index, start_time, side='left')

        # Count the number of liquidations in the window
        recent_liquidations = self. liquidations [start_idx:current_idx]

        # Buy if liquidations exceed the threshold and we are not in a position
        if recent_liquidations >= self.Liquidation_thresh and self.position.size == 0:
        self.buy(sl=self.data.Close[-1] * (1 - self.stop_loss), tp=self.data.Close[-1] * (1 + self.take_profit))

        # Load the data
        data_path = "C:\\Users\\elijah\\TRADERMANDEV\\data\\BTC_liq_data.csv" # Data file
        data = pd.read_csv(data_path)


        # convert 'datetime' column to datetime format and set as index 
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime',inplace=True)

        # Ensure necessary columns are present
        data = data[['symbol', 'LIQ SIDE', 'price', 'usd_size']]

        # Run the backtest