import importlib.metadata as metadata
import time
import requests
import pandas as pd
import os
import urllib.parse
import hashlib
import hmac
import base64
import streamlit as st
import matplotlib.pyplot as plt

api_key = "jU4wL3COeMEn7rnxLsNVXi8SwgZUYqr+nY4EKXGPf39jvGCT9QqLPBes"
api_sec = "CD1XL1klofm++Xv5kqRy7YOTveBFOadhlxsumqKOQ7FCtUhlUaN9wQsK2HMBHnG3q7fMWUPun8UaqOqKbO/IWQ=="


def get_kraken_signature(urlpath, data, secret):
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()
def kraken_request(uri_path, data, api_key, api_sec):
    api_url = "https://api.kraken.com"
    headers = {}
    headers['API-Key'] = api_key
    headers['API-Sign'] = get_kraken_signature(uri_path, data, api_sec)
    req = requests.post((api_url + uri_path), headers=headers, data=data)
    return req


def get_open_orders(open_trades):
    data = {
        "nonce": str(int(1000*time.time())),
        "trades": True
    }
    # Send the request and print the result
    open_orders = kraken_request('/0/private/OpenOrders', data, api_key, api_sec).json()['result']['open']

    prices_resp = requests.get('https://api.kraken.com/0/public/Ticker').json()['result']
    for order_id, order_data in open_orders.items():
        pair = order_data['descr']['pair']
        stop_price = order_data['stopprice']
        if pair in open_trades['Pair'].values:
            open_trades.loc[open_trades['Pair'] == pair, 'Stop Price'] = stop_price
            recent_bid = round(float(prices_resp[pair]["b"][0]), 2)
            open_trades.loc[open_trades['Pair'] == pair, 'Current Price'] = recent_bid

    open_trades['Stop Price'] = open_trades['Stop Price'].astype(float)
    open_trades['Open Price'] = open_trades['Open Price'].astype(float)

    open_trades['@Stop PnL %'] = ((open_trades['Stop Price'] - open_trades['Open Price']) / open_trades['Open Price']) * 100
    open_trades['@Current PnL %'] = ((open_trades['Current Price'] - open_trades['Open Price']) / open_trades['Open Price']) * 100
    open_trades['% Dist from Stop'] = ((open_trades['Stop Price'] - open_trades['Current Price']) / open_trades['Current Price']) * 100
    column_order = ['Pair', '@Stop PnL %', '@Current PnL %', '% Dist from Stop', 'Stop Price', 'Current Price', 'Volume', 'Date Opened', 'Open Price']
    open_trades = open_trades.reindex(columns=column_order)
    open_trades.drop(columns=['Date Opened', 'Open Price'], inplace=True)
    return open_trades
def get_trades_history_df():
    data = {
        "nonce": str(int(1000 * time.time()))
    }
    resp = kraken_request('/0/private/TradesHistory', data, api_key, api_sec).json()['result']['trades']

    extracted_data = []
    for key, value in resp.items():
        pair = value['pair']
        MTS = value['time']
        ordertype = value['ordertype']
        price = float(value['price'])
        vol = float(value['vol'])

        extracted_data.append((pair, MTS, ordertype, price, vol))

    df = pd.DataFrame(extracted_data, columns=['Pair', 'MTS', 'Order Type', 'Price', 'Volume'])
    df['MTS'] = pd.to_datetime(df['MTS'], unit='s')
    df = df[df["MTS"].dt.year >= 2024]
    return df


def match_up_open_and_closed_trades():
    # All trade opens
    df = get_trades_history_df()

    # Filter trailing stop market orders
    ts_market = df[df['Order Type'] == 'trailing stop market'].reset_index(drop=True)

    # Filter market orders
    market = df[df['Order Type'] == 'market']
    market = market.iloc[:len(market)-3].reset_index(drop=True)

    # Merge trailing stop market orders with market orders based on 'Pair' and 'Volume'
    merged_df = pd.merge(ts_market, market, on=['Pair', 'Volume'], suffixes=('_tsl', '_market'))

    # Add merged records to closed_trades DataFrame
    closed_trades = merged_df.copy()
    closed_trades['PnL %'] = ((closed_trades['Price_tsl'] - closed_trades['Price_market']) / closed_trades['Price_market']) *100
    closed_trades = closed_trades[['Pair', 'PnL %', 'Volume', 'MTS_tsl', 'MTS_market', 'Price_market', 'Price_tsl']]
    closed_trades.columns = ['Symbol', 'PnL %', 'Volume', 'Close Date', 'Open Date', 'Open Price', 'Close Price',]
    closed_trades['Close Date'] = pd.to_datetime(closed_trades['Close Date'])
    closed_trades['Open Date'] = pd.to_datetime(closed_trades['Open Date'])

    # Calculate the number of days open
    closed_trades['Periods Open'] = (closed_trades['Close Date'] - closed_trades['Open Date']).dt.days
    closed_trades.drop(columns=['Open Date'], inplace=True)
    closed_trades = closed_trades[['Symbol', 'PnL %', 'Volume', 'Periods Open', 'Close Date', 'Open Price', 'Close Price']]
    closed_trades['Close Date'] = closed_trades['Close Date'].dt.date
    closed_trades['PnL %'] = closed_trades['PnL %'].round(2)

    # Filter out records without a corresponding trailing stop market order
    open_trades = market[~market.set_index(['Pair', 'Volume']).index.isin(ts_market.set_index(['Pair', 'Volume']).index)].reset_index(drop=True)
    open_trades.rename(columns={'Price': 'Open Price', 'MTS': 'Date Opened'}, inplace=True)
    open_trades['Date Opened'] = open_trades['Date Opened'].dt.date
    open_trades.drop(columns=['Order Type'], inplace=True)
    return open_trades, closed_trades

def plus_minus_colorize(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}'


def calculate_trading_metrics(pnls):
    # Summarise trade counts
    num_trades = len(pnls)
    winning_trades = len(pnls[pnls > 0])
    losing_trades = len(pnls[pnls <= 0])

    # Ensure non-zero denominators
    if num_trades == 0:
        winning_probability = 0
        losing_probability = 0
    else:
        winning_probability = (winning_trades / num_trades) * 100
        losing_probability = (losing_trades / num_trades) * 100

    # Calculate total PnL
    total_pnl = pnls.sum()

    # Calculate max/min pnls
    max_profit = pnls.max() if pnls.max() > 0 else 0
    max_loss = pnls.min() if pnls.min() < 0 else 0

    # Calculate average PnL per trade
    if num_trades == 0:
        average_pnl = 0
        average_winning_pnl = 0
        average_losing_pnl = 0
    else:
        average_pnl = total_pnl / num_trades
        average_winning_pnl = sum(pnls[pnls > 0]) / winning_trades if winning_trades > 0 else 0
        average_losing_pnl = sum(pnls[pnls <= 0]) / losing_trades if losing_trades > 0 else 0

    # Calculate profit factor
    total_profit = pnls[pnls > 0].sum()
    total_loss = pnls[pnls <= 0].sum()
    if total_loss != 0:
        profit_factor = total_profit / abs(total_loss)
    else:
        profit_factor = float('inf') if total_profit > 0 else 0.0

    metrics = {
        "Trades": num_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Winning Probability": winning_probability,
        "Losing Probability": losing_probability,
        "Total PnL %": total_pnl,
        "Average PnL %": average_pnl,
        "Average Winning PnL %": average_winning_pnl,
        "Average Losing PnL %": average_losing_pnl,
        "Profit Factor": profit_factor,
        "Max Profit %": max_profit,
        "Max Loss %": max_loss
    }
    for key in metrics: metrics[key] = round(metrics[key], 2)
    metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    return metrics


# prep datasets
open_trades, closed_trades = match_up_open_and_closed_trades()
open_trades = get_open_orders(open_trades)

st.header("🌌 Strategy Dashboard 🌌")



st.header("Open Trades")
open_trades = open_trades.style.applymap(plus_minus_colorize, subset=['@Stop PnL %', '@Current PnL %'])
st.dataframe(open_trades, use_container_width=True)
st.markdown("<hr>", unsafe_allow_html=True)


st.header("Closed Trades")

pnls = closed_trades['PnL %']
periods_open = closed_trades['Periods Open']
closed_trades = closed_trades.style.applymap(plus_minus_colorize, subset=['PnL %'])
st.dataframe(closed_trades, use_container_width=True)


st.markdown("<hr>", unsafe_allow_html=True)


st.header("Strategy Metrics")
strategy_stats = calculate_trading_metrics(pnls).transpose()
strategy_stats = strategy_stats.style.applymap(plus_minus_colorize, subset=["Total PnL %", "Average PnL %", "Average Winning PnL %", "Average Losing PnL %"])

# Create histogram
fig1, ax = plt.subplots()
ax.hist(pnls, bins=10, color='skyblue', edgecolor='black')
ax.set_xlabel('PnL %')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Closed Trade PnLs')
plt.axvline(x=0, color='k', linestyle='--')


# Create scatter plot
fig2, ax = plt.subplots()
colors = ['red' if pnl < 0 else 'green' for pnl in pnls]
ax.scatter(periods_open, pnls, c=colors, alpha=0.5)
ax.set_xlabel('Periods Open')
ax.set_ylabel('PnL %')
ax.set_title('Periods Open vs PnL')

st.table(strategy_stats)
st.pyplot(fig1)
st.pyplot(fig2)
