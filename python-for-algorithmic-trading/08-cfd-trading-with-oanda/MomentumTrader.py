#
# Python Script
# with Momentum Trading Class
# for Oanda v20
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import tpqoa
import numpy as np
import pandas as pd


class MomentumTrader(tpqoa.tpqoa):
    def __init__(
        self, conf_file, instrument, bar_length, momentum, units, *args, **kwargs
    ):
        super(MomentumTrader, self).__init__(conf_file)
        self.position = 0
        self.instrument = instrument
        self.momentum = momentum
        self.bar_length = bar_length
        self.units = units
        self.raw_data = pd.DataFrame()
        self.min_length = self.momentum + 1

    def on_success(self, time, bid, ask):
        """Takes actions when new tick data arrives."""
        print(self.ticks, end=" ")
        self.raw_data = self.raw_data.append(
            pd.DataFrame({"bid": bid, "ask": ask}, index=[pd.Timestamp(time)])
        )
        self.data = (
            self.raw_data.resample(self.bar_length, label="right")
            .last()
            .ffill()
            .iloc[:-1]
        )
        self.data["mid"] = self.data.mean(axis=1)
        self.data["returns"] = np.log(self.data["mid"] / self.data["mid"].shift(1))
        self.data["position"] = np.sign(
            self.data["returns"].rolling(self.momentum).mean()
        )

        if len(self.data) > self.min_length:
            self.min_length += 1
            if self.data["position"].iloc[-1] == 1:
                if self.position == 0:
                    self.create_order(self.instrument, self.units)
                elif self.position == -1:
                    self.create_order(self.instrument, self.units * 2)
                self.position = 1
            elif self.data["position"].iloc[-1] == -1:
                if self.position == 0:
                    self.create_order(self.instrument, -self.units)
                elif self.position == 1:
                    self.create_order(self.instrument, -self.units * 2)
                self.position = -1
