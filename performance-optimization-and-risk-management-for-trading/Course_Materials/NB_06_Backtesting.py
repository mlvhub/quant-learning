# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Creating and Backtesting simple Momentum/Contrarian Strategies

# ## Getting started

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time")

data

data.info()

data.plot(figsize = (12, 8), title = "EUR/USD", fontsize = 12)
plt.show()

data.loc["2019-06"].plot(figsize = (12, 8), title = "EUR/USD", fontsize = 12)
plt.show()

data["returns"] = np.log(data.div(data.shift(1)))

data.dropna(inplace = True)

data



# ## Intro to Backtesting: a Buy-and-Hold "Strategy"

# Assumption: Invest 1 [Dollar] in Instrument EURUSD on 2018-01-02 and hold until 2019-12-30 (no further trades).

data

data[["returns"]].cumsum().apply(np.exp).plot(figsize = (12 , 8), fontsize = 12) # normalized price with Base == 1
plt.show()

multiple = data[["returns"]].sum().apply(np.exp)
multiple

data.returns.mean() # 6h mean return

data.returns.std() # std of 6h returns



# ## Defining a simple Contrarian Strategy (window = 3)

window = 3

data

data["returns"].rolling(window).mean()

data["position"] = -np.sign(data["returns"].rolling(window).mean()) # contrarian (minus sign)

data



# ## Vectorized Strategy Backtesting

data

data["strategy"] = data.position.shift(1) * data["returns"] # position to take for the next bar - use shift(1)
data

data.dropna(inplace = True)

data

data[["returns", "strategy"]].sum().apply(np.exp) # multiple for buy-and-hold and strategy

data["creturns"] = data["returns"].cumsum().apply(np.exp)  # normalized price with base = 1 for buy-and-hold
data["cstrategy"] = data["strategy"].cumsum().apply(np.exp) # normalized price with base = 1 for strategy

data

data[["creturns", "cstrategy"]].plot(figsize = (12 , 8),
                                     title = "EUR/USD | Window = {}".format(window), fontsize = 12)
plt.show()

tp_year = data.returns.count() / ((data.index[-1] - data.index[0]).days / 365.25) # 6h trading periods per year
tp_year

data[["returns", "strategy"]].mean() * tp_year # annualized returns

data[["returns", "strategy"]].std() * np.sqrt(tp_year) # annualized std

# -> __All long/short trading strategies__ (either -1 or 1) based on the same underlying instrument have the __same risk (std)__. <br>
# -> Risk (std) can be reduced with __neutral positions (0)__.



# ## Changing the window parameter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time")

data

data["returns"] = np.log(data.div(data.shift(1)))

data.dropna(inplace = True)
data

to_plot = ["returns"]

for w in [1, 2, 3, 5, 10]:
    data["position{}".format(w)] = -np.sign(data["returns"].rolling(w).mean())
    data["strategy{}".format(w)] = data["position{}".format(w)].shift(1) * data["returns"]
    to_plot.append("strategy{}".format(w))

data

to_plot

data[to_plot].dropna().cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.title("DJI Intraday - 6h bars", fontsize = 12)
plt.legend(fontsize = 12)
plt.show()

data



# ## Trades and Trading Costs (Part 1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time")

window = 3

data["returns"] = np.log(data.div(data.shift(1)))

data["position"] = -np.sign(data["returns"].rolling(window).mean())

data["strategy"] = data.position.shift(1) * data["returns"]

data

data.dropna(inplace = True)

data["creturns"] = data["returns"].cumsum().apply(np.exp)
data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

data

data.loc[:, "position"].plot(figsize = (12 , 8))
plt.show()

data.loc["2019-06", "position"].plot(figsize = (12 , 8))
plt.show()

data.position.diff().fillna(0).abs() # absolute change in position

data["trades"] = data.position.diff().fillna(0).abs()

data.trades.value_counts()

# -> __553 full trades__ (from short to long or from long to short) <br>
# -> each trade __triggers trading costs__, don´t ignore them!!! <br>
# -> Trading Costs __must be included__ in Backtesting!!! <br>



# ## Trades and Trading Costs (Part 2)

# __Trading/Transaction Costs__ (simplified but good approximation)

commissions = 0

spread = 1.5 * 0.0001 # pips == fourth price decimal
spread

half_spread = spread / 2 # absolute costs per trade (position change +-1)
half_spread

half_spread * 100000 # absolute costs in USD when buying 100,000 units of EUR/USD

# __Proportional trading costs are more useful than absolute trading costs.__ <br>
# __Goal: Deduct proportional trading costs from strategy returns before costs.__

ptc = half_spread / data.EURUSD.mean() # proportional costs per trade (position change +-1)
ptc

ptc = 0.00007 # conservative approx.

data

data["strategy_net"] = data.strategy - data.trades * ptc # strategy returns net of costs

data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)

data

data[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8))
plt.show()



# ## Generalization with OOP: the ConBacktester Class

# __Why using OOP and creating a class?__

# - Organizing/Storing/Linking all Functionalities and the Code in one Place/Class (managing/reducing complexity)
# - Reusability of Code
# - Simple to use for external Users (complex operations in one line of code)

# __How Important is OOP for this Course?__
# - Priority #1: Understand the Trading Concepts (Backtesting, Costs, Optimization, Smoothing, SL & TP, Leverage, etc.)
# - Priority #2: Understand the Python Code behind the Concepts
# - __Priority #3: Understand OOP / Class__ -> can help to adjust/use the class for your own strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


# Version 1.0
class ConBacktester():
    ''' Class for the vectorized backtesting of simple contrarian trading strategies.
    
    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade
    
    
    Methods
    =======
    get_data:
        imports the data.
        
    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).
        
    prepare_data:
        prepares the data for backtesting.
    
    run_backtest:
        runs the strategy backtest.
        
    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.
    '''    
    
    def __init__(self, filepath, symbol, start, end, tc):
        
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        
    def __repr__(self):
        return "ConBacktester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)
        
    def get_data(self):
        ''' Imports the data.
        '''
        raw = pd.read_csv(self.filepath, parse_dates = ["time"], index_col = "time")
        raw = raw[self.symbol].to_frame().fillna(method = "ffill") 
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.data = raw
        
    def test_strategy(self, window = 1):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).
         
        Parameters
        ============
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        self.window = window
                                
        self.prepare_data(window)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        multiple = round(self.results.cstrategy[-1], 6)
        print("Strategy-Multiple: {}".format(multiple))
    
    def prepare_data(self, window):
        
        ''' Prepares the Data for Backtesting.
        '''
        data = self.data.copy()
        data["roll_return"] = data["returns"].rolling(window).mean()
        data["position"] = -np.sign(data["roll_return"])
        self.results = data
    
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
        
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc
        
        self.results = data
    
    def plot_results(self):
        '''  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | Window = {} | TC = {}".format(self.symbol, self.window, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))


symbol = "EURUSD"
start = "2018-01-01"
end = "2019-12-31"
tc = 0.00007

tester = ConBacktester(filepath = "intraday.csv", symbol = symbol, start = start, end = end, tc = tc)
tester

tester.data

tester.test_strategy(window = 3)

tester.plot_results()

tester.results

tester.symbol

tester.start

tester.end

tester.tc


