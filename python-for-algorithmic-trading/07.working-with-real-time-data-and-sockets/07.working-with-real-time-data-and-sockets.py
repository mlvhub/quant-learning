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

# ## Visualizing Streaming Data with Plotly

# ### The Basics

import zmq
from datetime import datetime
import plotly.graph_objects as go 

symbol = 'SYMBOL'

fig = go.FigureWidget()
fig.add_scatter()

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://0.0.0.0:5555')
socket.setsockopt_string(zmq.SUBSCRIBE, symbol)

# +
times = list()
prices = list()

for _ in range(50):
    msg = socket.recv_string()
    t = datetime.now()
    times.append(t)
    _, price = msg.split()
    prices.append(float(price))
    fig.data[0].x = times
    fig.data[0].y = prices
# -

fig

# ### Three Real-Time Streams

fig = go.FigureWidget()
fig.add_scatter(name=symbol)
fig.add_scatter(name='SMA1', line=dict(width=1, dash='dot'), mode='lines+markers')
fig.add_scatter(name='SMA2', line=dict(width=1, dash='dash'), mode='lines+markers')

import pandas as pd

df = pd.DataFrame()
for _ in range(75):
    msg = socket.recv_string()
    t = datetime.now()
    sym, price = msg.split()
    df = df.append(pd.DataFrame({sym: float(price)}, index=[t]))
    df['SMA1'] = df[sym].rolling(5).mean()
    df['SMA2'] = df[sym].rolling(10).mean()
    fig.data[0].x = df.index
    fig.data[1].x = df.index
    fig.data[2].x = df.index
    fig.data[0].y = df[sym]
    fig.data[1].y = df['SMA1']
    fig.data[2].y = df['SMA2']

fig

# ### Three Sub-Plots for Three Streams

from plotly.subplots import make_subplots

f = make_subplots(rows=3, cols=1, shared_xaxes=True)
f.append_trace(go.Scatter(name='SYMBOL'), row=1, col=1)
f.append_trace(go.Scatter(name='RETURN', line=dict(width=1, dash='dot'),
                  mode='lines+markers', marker={'symbol': 'triangle-up'}),
                  row=2, col=1)
f.append_trace(go.Scatter(name='MOMENTUM', line=dict(width=1, dash='dash'),
                  mode='lines+markers', marker={'symbol': 'x'}), row=3, col=1)

fig = go.FigureWidget(f)
fig

import numpy as np

df = pd.DataFrame()
for _ in range(75):
    msg = socket.recv_string()
    t = datetime.now()
    sym, price = msg.split()
    df = df.append(pd.DataFrame({sym: float(price)}, index=[t]))
    df['RET'] = np.log(df[sym] / df[sym].shift(1))
    df['MOM'] = df['RET'].rolling(10).mean()
    fig.data[0].x = df.index
    fig.data[1].x = df.index
    fig.data[2].x = df.index
    fig.data[0].y = df[sym]
    fig.data[1].y = df['RET']
    fig.data[2].y = df['MOM']

# ### Streaming Data as Bars

socket = context.socket(zmq.SUB)
socket.connect('tcp://0.0.0.0:5556')
socket.setsockopt_string(zmq.SUBSCRIBE, '')

for _ in range(5):
    msg = socket.recv_string()
    print(msg)

fig = go.FigureWidget()
fig.add_bar()
fig

x = list('abcdefgh')
fig.data[0].x = x
for _ in range(25):
    msg = socket.recv_string()
    y = msg.split()
    y = [float(n) for n in y]
    fig.data[0].y = y


