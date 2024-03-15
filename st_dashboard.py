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
from datetime import datetime, timedelta

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

def get_closed_trades_history_df():
    """Returns df of closed trades with pnl % and market open orders without tsl closes"""
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

    df = pd.DataFrame(extracted_data, columns=['Symbol', 'MTS', 'Order Type', 'Price', 'Volume'])
    df['MTS'] = pd.to_datetime(df['MTS'], unit='s')
    df = df[df["MTS"].dt.date >= datetime(st.session_state.year, st.session_state.month, st.session_state.day).date()]

    # Filter trailing stop & market orders
    ts_market = df[df['Order Type'] == 'trailing stop market'].reset_index(drop=True)
    market = df[df['Order Type'] == 'market']

    # Merge trailing stop market orders with market orders based on 'Pair' and 'Volume'
    closed_trades = pd.merge(ts_market, market, on=['Symbol', 'Volume'], suffixes=('_tsl', '_market'))
    closed_trades['PnL %'] = ((closed_trades['Price_tsl'] - closed_trades['Price_market']) / closed_trades['Price_market']) * 100
    closed_trades = closed_trades[['Symbol', 'PnL %', 'Volume', 'MTS_tsl', 'MTS_market', 'Price_market', 'Price_tsl']]
    closed_trades.columns = ['Symbol', 'PnL %', 'Volume', 'Close Date', 'Open Date', 'Open Price', 'Close Price', ]
    closed_trades['Close Date'] = pd.to_datetime(closed_trades['Close Date'])
    closed_trades['Open Date'] = pd.to_datetime(closed_trades['Open Date'])

    # Calculate the number of days open
    closed_trades['Periods Open'] = (closed_trades['Close Date'] - closed_trades['Open Date']).dt.days
    closed_trades = closed_trades[['Symbol', 'PnL %', 'Periods Open', 'Close Date', 'Close Price']]
    closed_trades['Close Date'] = closed_trades['Close Date'].dt.date
    closed_trades['PnL %'] = closed_trades['PnL %'].round(4)

    # Filter out records without a corresponding trailing stop market order
    market_opens = market[~market.set_index(['Symbol', 'Volume']).index.isin(ts_market.set_index(['Symbol', 'Volume']).index)].reset_index(drop=True)
    market_opens.rename(columns={'Symbol': 'Symbol', 'Price': 'Open Price'}, inplace=True)
    market_opens.drop(columns=['Order Type', 'MTS', 'Volume'], inplace=True)
    return market_opens, closed_trades

def get_open_orders(market_opens):
    data = {
        "nonce": str(int(1000*time.time())),
        "trades": True
    }
    prices_resp = requests.get('https://api.kraken.com/0/public/Ticker').json()['result']
    try:
        open_orders = kraken_request('/0/private/OpenOrders', data, api_key, api_sec).json()['result']['open']
        for order_id, order_data in open_orders.items():
            pair = order_data['descr']['pair']
            stop_price = order_data['stopprice']
            if pair in market_opens['Symbol'].values:
                market_opens.loc[market_opens['Symbol'] == pair, 'Stop Price'] = stop_price
                recent_bid = round(float(prices_resp[pair]["c"][0]), 2)
                market_opens.loc[market_opens['Symbol'] == pair, 'Current Price'] = recent_bid

        for col in market_opens.columns:
            if col != 'Symbol': market_opens[col] = market_opens[col].astype(float)
        market_opens['@Stop PnL %'] = ((market_opens['Stop Price'] - market_opens['Open Price']) / market_opens['Open Price']) * 100
        market_opens['% Dist from Stop'] = ((market_opens['Stop Price'] - market_opens['Current Price']) / market_opens['Current Price']) * 100
        market_opens['@Current PnL %'] = ((market_opens['Current Price'] - market_opens['Open Price']) / market_opens['Open Price']) * 100

        market_opens = market_opens[['Symbol', '@Stop PnL %', '@Current PnL %', '% Dist from Stop']]
    except:
        market_opens = pd.DataFrame(columns=['Symbol', '@Stop PnL %', '% Dist from Stop', '@Current PnL %'])
    return market_opens.dropna()



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
def plot_pnls_charts(pnls):
    # Create histogram
    fig1, ax = plt.subplots()
    ax.hist(pnls, bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel('PnL %')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Closed Trade PnLs')
    plt.axvline(x=0, color='k', linestyle='--')

    return fig1



st.header("🌌 Strategy Dashboard 🌌")
col1, col2, col3 = st.columns([1,1,1])
with col1: st.number_input("Day Start", value=15, min_value=1, key='day')
with col2: st.number_input("Month Start", value=3, min_value=1, max_value=12, key='month')
with col3: st.number_input("Year Start", value=2024, min_value=1, key='year')
st.markdown("<hr>", unsafe_allow_html=True)

market_opens, closed_trades = get_closed_trades_history_df()


def plus_minus_colorize(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}'

st.header("Open Trades")
open_trades = get_open_orders(market_opens)
open_symbols = open_trades['Symbol'].values
open_trades = open_trades.style.applymap(plus_minus_colorize, subset=['@Stop PnL %', '@Current PnL %'])
st.dataframe(open_trades, use_container_width=True, hide_index=True)

st.header("Closed Trades")
fig1 = plot_pnls_charts(closed_trades['PnL %'])
closed_trades = closed_trades.style.applymap(plus_minus_colorize, subset=['PnL %'])
st.dataframe(closed_trades, use_container_width=True, hide_index=True)
st.pyplot(fig1)
