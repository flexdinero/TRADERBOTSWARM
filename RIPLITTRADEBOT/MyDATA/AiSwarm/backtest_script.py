from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

# Define a strategy that might contain errors
class SmaCross(Strategy):
    import pandas as pd


#functio to calculate
def calculate_sma(arr, n):
    return pd.Series(arr).rolling(n).mean()

# Using the standalone function to calculate SMA
price = [1, 2, 3, 4, 5]
sma_result = calculate_sma(price, 10)
print(sma_result)

def run_faulty_backtest():
    bt = Backtest(GOOG, SmaCross, cash=10000, commission=.002)
    stats = bt.run()
    print(stats)

import pandas as pd
