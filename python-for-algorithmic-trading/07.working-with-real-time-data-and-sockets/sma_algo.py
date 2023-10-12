import zmq
import datetime
import numpy as np
import pandas as pd

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://0.0.0.0:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "SYMBOL")

df = pd.DataFrame()
sma1 = 3
sma2 = 9
min_length = sma2 + 1

while True:
    data = socket.recv_string()
    t = datetime.datetime.now()
    sym, value = data.split()
    df = df.append(pd.DataFrame({sym: float(value)}, index=[t]))
    dr = df.resample("5s", label="right").last()
    dr["returns"] = np.log(dr / dr.shift(1))
    if len(dr) > min_length:
        min_length += 1
        dr['SMA1'] = dr['returns'].rolling(sma1).mean()
        dr['SMA2'] = dr['returns'].rolling(sma2).mean()
        dr['position'] = np.where(dr['SMA1'] > dr['SMA2'], 1, -1)
        dr['strategy'] = dr['position'].shift(1) * dr['returns']
        dr.dropna(inplace=True)
        dr['creturns'] = dr['returns'].cumsum().apply(np.exp)
        dr['cstrategy'] = dr['strategy'].cumsum().apply(np.exp)
        print("\n" + "=" * 51)
        print("NEW SIGNAL | {}".format(datetime.datetime.now()))
        print("=" * 51)
        print(dr.iloc[:-1].tail())
        if dr["position"].iloc[-2] == 1.0:
            print("\nLong market position.")
            # take some action (e.g., place buy order)
        elif dr["position"].iloc[-2] == -1.0:
            print("\nShort market position.")
            # take some action (e.g., place sell order)
