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

# ## Reading Financial Data From Different Sources

# ### The Data Set

fn = '../data/AAPL.csv'

with open(fn, 'r') as f:
    for _ in range(5):
        print(f.readline(), end='')

# ### Reading from a CSV File with Python

import csv

csv_reader = csv.reader(open(fn, 'r'))

data = list(csv_reader)

data[:5]

csv_reader = csv.DictReader(open(fn, 'r'))

data = list(csv_reader)

data[:5]

mean = sum([float(l['CLOSE']) for l in data]) / len(data)
mean

# ### Reading from a CSV File with pandas

import pandas as pd

data = pd.read_csv(fn, index_col=0, parse_dates=True)

data.info()

data.tail()

data.index

data['CLOSE'].mean()

# ### Exporting to Excel and JSON

# +
#data.to_excel('data/aapl.xls', 'AAPL')
# -

data.to_json('./aapl.json')

# ### Reading from Excel and JSON

# +
#excel_data = pd.read_excel('data/aapl.xls', 'AAPL', index_col=0)
#excel_data.head()
# -

json_data = pd.read_json('./aapl.json')
json_data.head()

# ## Working with Open Data Sources

# %load_ext dotenv
# %dotenv
import os

quandl_api_key = os.environ.get("QUANDL_API_KEY")

import quandl as q

data = q.get('BCHAIN/MKPRU', api_key=quandl_api_key)

data.info()

data['Value'].resample('A').last()

data = q.get('FSE/SAP_X', start_date='2018-1-1',
                      end_date='2020-05-01',
                      api_key=quandl_api_key)

data.info()

data.head()

q.ApiConfig.api_key = quandl_api_key

vol = q.get_table('QUANTCHA/VOL', date='2018-12-31', ticker='MSFT')

vol.iloc[:, :10].info()

vol[['ivmean30', 'ivmean60', 'ivmean90']].tail()

# ## Storing Financial Data Efficiently

# +
#
# Python Module to Generate a
# Sample Financial Data Set
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd

r = 0.05  # constant short rate
sigma = 0.5  # volatility factor


def generate_sample_data(rows, cols, freq='1min'):
    '''
    Function to generate sample financial data.

    Parameters
    ==========
    rows: int
        number of rows to generate
    cols: int
        number of columns to generate
    freq: str
        frequency string for DatetimeIndex

    Returns
    =======
    df: DataFrame
        DataFrame object with the sample data
    '''
    rows = int(rows)
    cols = int(cols)
    # generate a DatetimeIndex object given the frequency
    index = pd.date_range('2021-1-1', periods=rows, freq=freq)
    # determine time delta in year fractions
    dt = (index[1] - index[0]) / pd.Timedelta(value='365D')
    # generate column names
    columns = ['No%d' % i for i in range(cols)]
    # generate sample paths for geometric Brownian motion
    raw = np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                 sigma * np.sqrt(dt) *
                 np.random.standard_normal((rows, cols)), axis=0))
    # normalize the data to start at 100
    raw = raw / raw[0] * 100
    # generate the DataFrame object
    df = pd.DataFrame(raw, index=index, columns=columns)
    return df


# -

rows = 5  # number of rows
columns = 3  # number of columns
freq = 'D'  # daily frequency
generate_sample_data(rows, columns, freq)

# %time data = generate_sample_data(rows=5e6, cols=10).round(4)

data.info()

# ### Storing DataFrame Objects

# #### Writing directly to an HDFS file

# overwrites an existing file with the same name
h5 = pd.HDFStore('./data.h5', 'w')

# %time h5['data'] = data

h5

# %ls -n ./data.*

h5.close()

h5 = pd.HDFStore('./data.h5', 'r')

# %time data_copy = h5['data']

data_copy.info()

h5.close()

# %rm ./data.h5

# #### Writing to HDFS through a DataFrame

# %time data.to_hdf('./data.h5', 'data', format='table')

# %time data_copy = pd.read_hdf('./data.h5', 'data')

data_copy.info()

# ### Using TsTables

# %time data = generate_sample_data(rows=2.5e6, cols=5, freq='1s').round(4)

data.info()

import tstables
import tables as tb


class desc(tb.IsDescription):
    ''' Description of TsTables table structure.
    '''
    timestamp = tb.Int64Col(pos=0)
    No0 = tb.Float64Col(pos=1)
    No1 = tb.Float64Col(pos=2)
    No2 = tb.Float64Col(pos=3)
    No3 = tb.Float64Col(pos=4)
    No4 = tb.Float64Col(pos=5)


h5 = tb.open_file('./data.h5ts', 'w')

ts = h5.create_ts('/', 'data', desc)

h5

# %time ts.append(data)

h5

import datetime

start = datetime.datetime(2021, 1, 2)
end = datetime.datetime(2021, 1, 3)

# %time subset = ts.read_range(start, end)

start = datetime.datetime(2021, 1, 2, 12, 30, 0)

end = datetime.datetime(2021, 1, 5, 17, 15, 30)

# %time subset = ts.read_range(start, end)

subset.info()

h5.close()

# rm data*

# ## Storing Data with SQLite3

# %time data = generate_sample_data(1e6, 5, '1min').round(4)

data.info()

import sqlite3 as sq3

con = sq3.connect('./data.sql')

# %time data.to_sql('data', con)

# ls -n ./data.*

query = 'SELECT * FROM data WHERE No1 > 105 and No2 < 108'

# %time res = con.execute(query).fetchall()

res[:5]

len(res)

con.close()

# rm data*


