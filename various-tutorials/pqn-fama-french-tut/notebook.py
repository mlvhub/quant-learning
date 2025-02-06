# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# %%
tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = "2020-01-01"
end_date = "2024-12-31"

# %%
stock_data = yf.download(
    tickers, 
    start=start_date, 
    end=end_date
)['Close']

stock_data

# %%
port_returns = (
    stock_data
    .pct_change()
    .sum(axis=1)
)

port_returns.name = "port_returns"
port_returns

# %%
# Fetch Fama French factors
fama_french = web.DataReader(
    "F-F_Research_Data_Factors_daily",
    "famafrench",
    start_date,
    end_date
)[0]

fama_french

# %%
# Preprocess Fama French factors

fama_french = fama_french / 100  # Convert to decimals
fama_french.index = fama_french.index.tz_localize(None)

data = fama_french.join(port_returns, how="inner")

excess_returns = data.port_returns - data.RF
excess_returns

# %%
# Model excess returns using Fama French factors

X = data[["SMB", "HML"]]
X = sm.add_constant(X)

model = sm.OLS(excess_returns, X).fit()

hedge_weights = -model.params[1:]
hedge_weights


# %%
# Simulate and analyze the hedged portfolio

hedge_portfolio = (data[["SMB", "HML"]] @ hedge_weights).dropna()

hedged_portfolio_returns = port_returns.loc[hedge_portfolio.index] + hedge_portfolio

hedge = pd.DataFrame({
    "unhedged_returns": port_returns.loc[hedged_portfolio_returns.index],
    "hedged_returns": hedged_portfolio_returns
})

hedge.mean() / hedge.std()

# %%
