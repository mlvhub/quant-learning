# generated with Cursor

import backtrader as bt
from datetime import time

class LuxorStrategy(bt.Strategy):
    params = (
        ('fast_length', 3),      # Fast MA period
        ('slow_length', 30),     # Slow MA period
        ('tset', time(16, 0)),   # Start time (16:00)
        ('window_dist', 100),    # Window distance in minutes
    )

    def __init__(self):
        # Initialize moving averages
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_length)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_length)
        
        # Trading signals
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Initialize variables
        self.buy_stop = None
        self.sell_stop = None
        self.buy_limit = None
        self.sell_limit = None
        
        # Calculate end time
        minutes = self.p.window_dist
        hour = self.p.tset.hour
        minute = self.p.tset.minute + minutes
        self.tend = time(hour + minute // 60, minute % 60)

    def next(self):
        # Time window filter
        current_time = self.data.datetime.time()
        if not (self.p.tset <= current_time < self.tend):
            return

        # Entry conditions
        go_long = self.fast_ma > self.slow_ma
        go_short = self.fast_ma < self.slow_ma

        # Set up entry prices on crossover
        if self.crossover > 0:  # Fast crosses above Slow
            self.buy_stop = self.data.high[0] + 1
            self.buy_limit = self.data.high[0] + 5
            self.sell_stop = None  # Reset opposite signals
            self.sell_limit = None
            
        if self.crossover < 0:  # Fast crosses below Slow
            self.sell_stop = self.data.low[0] - 1
            self.sell_limit = self.data.low[0] - 5
            self.buy_stop = None  # Reset opposite signals
            self.buy_limit = None

        # Entry orders
        if self.buy_limit and go_long and self.data.close[0] < self.buy_limit and not self.position:
            self.buy(exectype=bt.Order.Stop, price=self.buy_stop)

        if self.sell_limit and go_short and self.data.close[0] > self.sell_limit and not self.position:
            self.sell(exectype=bt.Order.Stop, price=self.sell_stop)

        # Note: The original exit conditions are commented out in the source code,
        # so they are not implemented here. You can add custom exit conditions as needed.
