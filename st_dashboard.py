import time
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from math import pi
import urllib.parse
import hashlib
import hmac
import base64
import streamlit as st
from datetime import datetime, timedelta
from bokeh.plotting import figure, column
from bokeh.palettes import Category10
from bokeh.models import HoverTool, CrosshairTool

api_key = "klQfL6/kNRHtkRbm0Jd/ZDAE8ct5jmpzyv2fEG5DaSkDnqz5XRUmQfW+"
api_sec = "vbrZantDeqz9YAuKeWw8pvFOKVaMnRLyp7Y+6ggTleuaubju5uVtO+dUxfcIV9tVvO+ifhfE7q1HmuHRuOpvUQ=="

def get_kraken_daily_price_data(symbol=None, interval_hours=None, since=None):
    interval_hours = interval_hours * 60
    if since is not None: resp = requests.get(f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval_hours}&since={since}")
    else: resp = requests.get(f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval_hours}")
    if resp.status_code == 200:
        resp = resp.json()['result']
        resp = resp[list(resp.keys())[0]]
        price_df = pd.DataFrame(resp, columns=['MTS', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        for col in price_df.columns: price_df[col] = price_df[col].astype(float)
        price_df = price_df.iloc[:len(price_df) - 1].reset_index(drop=True)
        price_df['MTS'] = pd.to_datetime(price_df['MTS'], unit='s')
        return price_df
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
    data = {
        "nonce": str(int(1000 * time.time()))
    }
    resp = kraken_request('/0/private/TradesHistory', data, api_key, api_sec).json()['result']['trades']

    extracted_data = []
    for key, value in resp.items():
        pair = value['pair']
        MTS = value['time']
        ordertype = value['type']
        price = float(value['price'])
        vol = float(value['vol'])

        extracted_data.append((pair, MTS, ordertype, price, vol))

    df = pd.DataFrame(extracted_data, columns=['Symbol', 'MTS', 'Order Type', 'Price', 'Volume'])
    df['MTS'] = pd.to_datetime(df['MTS'], unit='s')
    df = df[df['MTS'] >= '2024-03-18'] #'2024-03-18']
    df['MTS'] = df['MTS'].dt.date

    # Merge buy and sell orders based on matching 'Symbol' and 'Volume'
    buy_orders = df[df['Order Type'] == 'buy']
    sell_orders = df[df['Order Type'] == 'sell']
    closed_trades = pd.merge(buy_orders, sell_orders, on=['Symbol', 'Volume'], suffixes=('', '_close'), how='left')


    # Identify open trades (buys without corresponding sells)
    open_trades = closed_trades[closed_trades['Price_close'].isnull()]
    open_trades = open_trades.drop(columns=[col for col in open_trades.columns if '_close' in col])
    open_trades.columns = ['Symbol', 'Open Date', 'Order Type', 'Trade Open Price', 'Volume']

    closed_trades.dropna(inplace=True)
    closed_trades = closed_trades.reset_index(drop=True)
    closed_trades['PnL'] = ((closed_trades['Price_close'] - closed_trades['Price']) / closed_trades['Price']) * 100
    closed_trades.columns = ['Symbol', 'Open Date', 'Order Type', 'Trade Open Price', 'Volume', 'Close Date', 'Close Order Type', 'Trade Close Price', 'PnL']
    closed_trades = closed_trades.drop(['Order Type', 'Close Order Type'], axis=1)

    return open_trades, closed_trades

def get_wallet_balances():
    # Construct the request data
    data = {
        "nonce": str(int(1000 * time.time()))
    }

    # Send the request and print the result
    balances = kraken_request('/0/private/Balance', data, api_key, api_sec).json()['result']

    # Filter balances greater than 0
    positive_balances = {symbol: float(balance) for symbol, balance in balances.items() if float(balance) > 0}

    # Construct DataFrame
    trade_balances = pd.DataFrame(list(positive_balances.items()), columns=['Symbol', 'Wallet Balance'])
    usd_balance = trade_balances.loc[trade_balances['Symbol'] == 'ZUSD', 'Wallet Balance'].values[0]
    trade_balances = trade_balances[trade_balances['Symbol'] != 'ZUSD']
    return trade_balances, usd_balance

def get_current_prices():
    current_prices_data = []  # Initialize a list to store data
    resp = requests.get('https://api.kraken.com/0/public/Ticker').json()['result']
    for symbol in resp:
        if 'USD' in symbol and 'USDT' not in symbol and 'EUR' not in symbol and 'GBP' not in symbol and 'CUSD' not in symbol and 'TUSD' not in symbol and 'USDC' not in symbol and 'PYUSD' not in symbol:
            todays_open = float(resp[symbol]["o"])
            last_trade_closed_price = float(resp[symbol]["c"][0])
            hours24_volume = float(resp[symbol]["v"][1])
            hours24_vwap = float(resp[symbol]["p"][1])
            hours24_transactions = int(resp[symbol]["t"][1])
            if todays_open != 0: current_prices_data.append({'Symbol': symbol, 'Todays Open': todays_open, 'Last Traded Price': last_trade_closed_price, '24h Volume': hours24_volume, '24h VWAP': hours24_vwap, '24h Transactions': hours24_transactions})
    current_prices_df = pd.DataFrame(current_prices_data)  # Convert list to DataFrame
    current_prices_df['24h USD Volume (VWAP est)'] = current_prices_df['24h VWAP'] * current_prices_df['24h Volume']
    current_prices_df['Current Day PnL'] = ((current_prices_df['Last Traded Price'] - current_prices_df['Todays Open']) / current_prices_df['Todays Open']) * 100
    current_prices_df.dropna()
    current_prices_df = current_prices_df.sort_values(by='Current Day PnL', ascending=False).reset_index(drop=True)
    return current_prices_df

def create_master_df(open_trades, trade_balances_df, usd_balance, current_prices_df):
    # merge dfs (except trade open) together on Symbol
    trade_balances_df['Symbol'] = trade_balances_df['Symbol'] + 'USD'
    master_df = current_prices_df.merge(trade_balances_df, on='Symbol', how='left').merge(open_trades, on='Symbol', how='left')

    wallet_values = master_df['Wallet Balance'] * master_df['Last Traded Price']
    master_df.insert(5, 'Wallet %', wallet_values)
    master_df.loc[len(master_df)] = {'Symbol': 'ZUSD', 'Todays Open': 1, 'Last Traded Price': 1, 'Current Day PnL': 0,
                                     'Wallet Balance': usd_balance, 'Wallet %': usd_balance}
    master_df['Wallet %'] = (master_df['Wallet %'] / master_df['Wallet %'].sum()) * 100
    master_df['Open Trade PnL'] = ((master_df['Last Traded Price'] - master_df['Trade Open Price']) / master_df['Trade Open Price']) * 100
    master_df.set_index('Symbol')
    return master_df

def calculate_trading_metrics(pnls):
    # Summarise trade counts
    num_trades = len(pnls)
    if num_trades == 0:
        # Handle the case when there are no trades
        return pd.DataFrame(columns=['Value'])  # Return an empty DataFrame or handle it accordingly

    winning_trades = len(pnls[pnls > 0])
    losing_trades = len(pnls[pnls <= 0])

    if num_trades != 0:
        winning_probability = (winning_trades / num_trades) * 100
        losing_probability = (losing_trades / num_trades) * 100
    else:
        winning_probability = 0
        losing_probability = 0

    # Calculate total PnL
    total_pnl = pnls.sum()

    # Calculate max/min pnls
    max_profit = pnls.max()
    max_loss = pnls.min()

    # Calculate average PnL per trade
    average_pnl = total_pnl / num_trades if num_trades != 0 else 0
    average_winning_pnl = sum(pnls[pnls > 0]) / winning_trades if winning_trades != 0 else 0
    average_losing_pnl = sum(pnls[pnls <= 0]) / losing_trades if losing_trades != 0 else 0

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
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    return metrics_df

# data sources
open_trades, closed_trades = get_closed_trades_history_df()
trade_balances_df, usd_balance = get_wallet_balances()
current_prices_df = get_current_prices()
master_df = create_master_df(open_trades, trade_balances_df, usd_balance, current_prices_df)


col1, col2, col3, col4 = st.columns([1,1,1,1])
tabs = ["Today", "Inspect Trades", "Portfolio", "Strategy"]
with st.sidebar:
    st.header("Hide Tab Content")
    tab1_toggle = st.toggle(f"Hide {tabs[0]}")
    tab2_toggle = st.toggle(f"Hide {tabs[1]}")
    tab3_toggle = st.toggle(f"Hide {tabs[2]}")
    tab4_toggle = st.toggle(f"Hide {tabs[3]}")
    st.markdown("<hr>", unsafe_allow_html=True)

if 'symbol' not in st.session_state: st.session_state.symbol = "None"
# Display the open trades
with st.expander("Open Trades ♻️"):
    filtered_df = master_df[master_df['Order Type'] == 'buy']
    filtered_df = filtered_df[['Symbol', 'Open Date', 'Trade Open Price', 'Open Trade PnL']]
    open_trades_table = st.dataframe(filtered_df, hide_index=True, use_container_width=True)
# Display the closed trades
with st.expander("Closed Trades ♻️"):
    filtered_df = closed_trades[['Symbol', 'Open Date', 'Close Date', 'PnL']]
    closed_open_trades_table = st.dataframe(filtered_df, hide_index=True, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(tabs)
# df colouriser function which highlights row based on symbol value or takes columns and turns value into green or red text based on if > or < than 0


with tab1:
    if not tab1_toggle:
        # Calculate statistics
        statistics = {
            "Today's Mean": master_df['Current Day PnL'].mean(),
            "Today's Median": master_df['Current Day PnL'].median(),
            "Today's  Std": master_df['Current Day PnL'].std(),
            "Today's  Skew": master_df['Current Day PnL'].skew(),
            "Today's Kurtosis": master_df['Current Day PnL'].kurtosis()
        }
        current_day_ct_statistics = pd.DataFrame(statistics, index=['value'])

        st.header("Current Day PnL Distribution 📊")
        # Plot current day's PnL histogram
        p = figure(title="Histogram of Current Day PnLs", x_axis_label='PnL', y_axis_label='Frequency')
        hist, edges = np.histogram(master_df['Current Day PnL'], bins=20)
        colors = ["green" if val >= 0 else "red" for val in edges[:-1]]
        bars = p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=colors, line_color="white", alpha=0.5)
        todays_mean = statistics["Today's Mean"]
        p.line([todays_mean, todays_mean], [0, max(hist)+3], line_color='blue', line_width=2, legend_label=f"Mean: {todays_mean.round(2)}%")

        if st.session_state.symbol != 'None':
            symbol_pnl = master_df.loc[master_df['Symbol'] == st.session_state.symbol, 'Current Day PnL'].values[0]
            p.line([symbol_pnl, symbol_pnl], [0, max(hist)+3], line_color='black', line_dash='dashed', line_width=2, legend_label=f"{st.session_state.symbol}: {symbol_pnl.round(2)}%")

        hover = HoverTool(tooltips=[("Range", "@left{0.00} to @right{0.00}"), ("Frequency", "@top")], renderers=[bars])
        p.add_tools(hover)

        st.bokeh_chart(p, use_container_width=True)
        st.dataframe(current_day_ct_statistics, hide_index=True, use_container_width=True)

        # from master_df
        filtered_df = master_df[['Symbol', 'Last Traded Price', 'Current Day PnL']].dropna().sort_values(by='Current Day PnL', ascending=False)
        with st.expander("Today's Top 20 Gainers 📈"): st.dataframe(filtered_df.head(20), hide_index=True, use_container_width=True, height=740)
        with st.expander("Today's Top 20 Losers 📉"): st.dataframe(filtered_df.tail(20)[::-1], hide_index=True, use_container_width=True, height=740)


        filtered_df = master_df[['Symbol', 'Todays Open', '24h VWAP', '24h Volume', '24h Transactions', '24h USD Volume (VWAP est)']].dropna().sort_values(by='24h USD Volume (VWAP est)', ascending=False)
        with st.expander("Today's Top 20 USD Volume (VWAP estimated) 🔊"): st.dataframe(filtered_df.head(20), hide_index=True, use_container_width=True, height=740)
        with st.expander("Today's Bottom 20 USD Volume (VWAP estimated) 🔊"): st.dataframe(filtered_df.tail(20)[::-1], hide_index=True, use_container_width=True, height=740)

with tab2:
    if not tab2_toggle:
        if st.session_state.symbol == "None": st.session_state.symbol = "XXBTZUSD"
        price_df = get_kraken_daily_price_data(st.session_state.symbol, 24, None)
        price_df['candle_col'] = np.where(price_df['close'] - price_df['open'] > 0, 'green', 'red')
        candle_width = 86400000 # 1 day in mts

        price_df['sma'] = ta.sma(price_df['close'], timeperiod=20)
        price_df['rsi'] = ta.rsi(price_df['close'], timeperiod=20)
        macd_data = ta.macd(price_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        macd_data.columns = ['macd', 'macdhist', 'macdsignal']
        price_df['macd'], price_df['macdsignal'], price_df['macdhist'] = macd_data['macd'], macd_data['macdsignal'], \
        macd_data['macdhist']


        st.header(f"Currently Showing: {st.session_state.symbol}")

        # date filters
        # Convert timedelta objects to datetime objects for comparison
        st.subheader(f"Quick-Access Date Filters")
        today = datetime.today()
        two_years_ago = datetime.combine(today - timedelta(days=365 * 2), datetime.min.time())
        one_year_ago = datetime.combine(today - timedelta(days=365), datetime.min.time())
        six_months_ago = datetime.combine(today - timedelta(days=30 * 6), datetime.min.time())
        three_months_ago = datetime.combine(today - timedelta(days=30 * 3), datetime.min.time())
        one_month_ago = datetime.combine(today - timedelta(days=30), datetime.min.time())
        fourteen_days_ago = datetime.combine(today - timedelta(days=14), datetime.min.time())
        price_df['MTS'] = pd.to_datetime(price_df['MTS'])
        open_trades['Open Date'] = pd.to_datetime(open_trades['Open Date'])
        closed_trades['Close Date'] = pd.to_datetime(closed_trades['Close Date'])

        # Create six columns to display buttons horizontally
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        if col1.button("2y"):
            price_df = price_df[price_df['MTS'] > two_years_ago]
            open_trades = open_trades[open_trades['Open Date'] > two_years_ago]
            closed_trades = closed_trades[closed_trades['Close Date'] > two_years_ago]
        if col2.button("1y"):
            price_df = price_df[price_df['MTS'] > one_year_ago]
            open_trades = open_trades[open_trades['Open Date'] > one_year_ago]
            closed_trades = closed_trades[closed_trades['Close Date'] > one_year_ago]
        if col3.button("6m"):
            price_df = price_df[price_df['MTS'] > six_months_ago]
            open_trades = open_trades[open_trades['Open Date'] > six_months_ago]
            closed_trades = closed_trades[closed_trades['Close Date'] > six_months_ago]
        if col4.button("3m"):
            price_df = price_df[price_df['MTS'] > three_months_ago]
            open_trades = open_trades[open_trades['Open Date'] > three_months_ago]
            closed_trades = closed_trades[closed_trades['Close Date'] > three_months_ago]
        if col5.button("1m"):
            price_df = price_df[price_df['MTS'] > one_month_ago]
            open_trades = open_trades[open_trades['Open Date'] > one_month_ago]
            closed_trades = closed_trades[closed_trades['Close Date'] > one_month_ago]
        if col6.button("14d"):
            price_df = price_df[price_df['MTS'] > fourteen_days_ago]
            open_trades = open_trades[open_trades['Open Date'] > fourteen_days_ago]
            closed_trades = closed_trades[closed_trades['Close Date'] > fourteen_days_ago]

        # show indicator values & signal checks
        with st.expander("Indicator Values & Signals"):
            today_indicators = price_df.iloc[-1][['close', 'sma', 'rsi', 'macd', 'macdsignal', 'macdhist', 'candle_col']]
            today_indicators.name = 'Value Today'
            yesterday_indicators = price_df.iloc[-2][['close', 'sma', 'rsi', 'macd', 'macdsignal', 'macdhist', 'candle_col']]
            yesterday_indicators.name = 'Value Yesterday'
            indicators = pd.concat([today_indicators, yesterday_indicators], axis=1)

            conditions = {
                'Close > SMA': indicators['Value Today']['close'] > indicators['Value Today']['sma'],
                'RSI between 30 and 70': (indicators['Value Today']['rsi'] > 30) & (indicators['Value Today']['rsi'] < 70),
                'Green Candle': indicators['Value Today']['candle_col'] == 'green',
                'MACD Histogram > 0': indicators['Value Today']['macdhist'] > 0
            }

            # # Create a new DataFrame using the conditions dictionary
            results_df = pd.DataFrame(conditions, index=[0])
            st.text("Indicator Values")
            st.dataframe(indicators, use_container_width=True)
            st.text("Most Recent Signals")
            st.dataframe(results_df, hide_index=True, use_container_width=True)


        # Create a candlestick chart
        crosshair = CrosshairTool(dimensions='both', line_color='grey', line_width=0.5)
        candlestick_chart = figure(x_axis_type='datetime', height=400, width=700, toolbar_location='above')
        candlestick_chart.segment(x0='MTS', y0='low', x1='MTS', y1='high', color='candle_col', source=price_df, legend_label='Price') # plot candles
        candlestick_chart.vbar(x='MTS', width=candle_width, top='open', bottom='close', fill_color='candle_col', line_color='candle_col', source=price_df, legend_label='Price') # plot candles
        candlestick_chart.xaxis.major_label_orientation = pi / 4
        candlestick_chart.add_tools(crosshair)
        candlestick_chart.legend.click_policy = "hide"
        candlestick_chart.legend.orientation = "horizontal"
        candlestick_chart.legend.location = "top_left"
        candlestick_chart.line(x='MTS', y='sma', color='blue', legend_label='SMA', source=price_df)


        # plot MACD
        crosshair = CrosshairTool(dimensions='both', line_color='grey', line_width=0.5)
        macd_chart = figure(x_axis_type='datetime', height=200, width=700, toolbar_location=None, x_range=candlestick_chart.x_range)
        macd_chart.line(x='MTS', y='macd', color='blue', source=price_df, legend_label='MACD')
        macd_chart.line(x='MTS', y='macdsignal', color='orange', source=price_df, legend_label='Signal')
        price_df['hist_color'] = price_df['macdhist'].apply(lambda x: 'green' if x > 0 else 'red')
        macd_chart.vbar(x='MTS', width=candle_width, top='macdhist', color='hist_color', source=price_df, fill_alpha=0.2, line_alpha=0.2)
        macd_chart.add_tools(crosshair)
        macd_chart.legend.orientation = "horizontal"
        macd_chart.legend.location = "top_left"

        # plot rsi
        rsi_chart = figure(x_axis_type='datetime', height=200, width=700, toolbar_location=None, x_range=candlestick_chart.x_range)
        rsi_chart.line(x='MTS', y='rsi', color='purple', source=price_df)
        rsi_chart.varea(x=price_df['MTS'], y1=30, y2=price_df['rsi'].min(), fill_color='orange', fill_alpha=0.2)
        rsi_chart.varea(x=price_df['MTS'], y1=70, y2=price_df['rsi'].max(), fill_color='orange', fill_alpha=0.2)

        # plot trades on all plots - I -1 FROM OPEN DATE HERE - THINK ITS CORRECT...
        filtered_trades = closed_trades[closed_trades['Symbol'] == st.session_state.symbol]
        for index, trade in filtered_trades.iterrows():
            if trade['PnL'] > 0:
                candlestick_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=trade['Close Date']), y1=price_df['low'].min(), y2=price_df['high'].max(), fill_color='green', fill_alpha=0.5)
                macd_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=trade['Close Date']), y1=price_df['macd'].min(), y2=price_df['macd'].max(), fill_color='green', fill_alpha=0.5)
                rsi_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=trade['Close Date']), y1=price_df['rsi'].min(), y2=price_df['rsi'].max(), fill_color='green', fill_alpha=0.5)
            else:
                candlestick_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=trade['Close Date']), y1=price_df['low'].min(), y2=price_df['high'].max(), fill_color='red', fill_alpha=0.5)
                macd_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=trade['Close Date']), y1=price_df['macd'].min(), y2=price_df['macd'].max(), fill_color='red', fill_alpha=0.5)
                rsi_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=trade['Close Date']), y1=price_df['rsi'].min(), y2=price_df['rsi'].max(), fill_color='green', fill_alpha=0.5)

        filtered_trades = open_trades[open_trades['Symbol'] == st.session_state.symbol]
        for index, trade in filtered_trades.iterrows():
            candlestick_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=price_df['MTS'].iloc[-1]), y1=price_df['low'].min(), y2=price_df['high'].max(), fill_color='grey', fill_alpha=0.5)
            macd_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=price_df['MTS'].iloc[-1]), y1=price_df['macd'].min(), y2=price_df['macd'].max(), fill_color='grey', fill_alpha=0.5)
            rsi_chart.varea(x=pd.date_range(start=trade['Open Date'] - timedelta(days=1), end=price_df['MTS'].iloc[-1]), y1=price_df['rsi'].min(), y2=price_df['rsi'].max(), fill_color='grey', fill_alpha=0.5)

        # Show the column of plots
        plots = column(candlestick_chart, macd_chart, rsi_chart)
        st.bokeh_chart(plots, use_container_width=True)


with tab3:
    if not tab3_toggle:
        # Plot portfolio allocations pie chart
        st.header("Live Portfolio Allocations 🏷️")
        filtered_df = master_df[master_df['Wallet %'] > 0][['Symbol', 'Wallet %', 'Open Trade PnL']].sort_values(by='Wallet %', ascending=False).reset_index(drop=True)
        filtered_df['Wallet %'] = filtered_df['Wallet %'].astype(float)
        p2 = figure(title="Portfolio Allocations", toolbar_location=None, height=400)
        total_wallet_percentage = filtered_df['Wallet %'].sum()
        angles = filtered_df['Wallet %'] / total_wallet_percentage * 2 * 3.1416
        start_angle = 0
        palette = Category10[len(filtered_df)]
        for index, row in filtered_df.iterrows():
            end_angle = start_angle + angles[index]
            wedge = p2.wedge(x=0, y=0, radius=0.8, start_angle=start_angle, end_angle=end_angle, color=palette[index % len(palette)], legend_label=row['Symbol'])
            start_angle = end_angle

        p2.axis.visible = False

        col1, col2 = st.columns([4, 4])
        with col1: st.bokeh_chart(p2, use_container_width=True)
        with col2: st.dataframe(filtered_df, hide_index=True, use_container_width=True)

with tab4:
    if not tab4_toggle:
        col1, col2 = st.columns([1,2])
        pnls = closed_trades['PnL']
        stats = calculate_trading_metrics(pnls)
        with col1: st.dataframe(stats, height=460)

        # plot closed pnl histogram
        mean_value = pnls.mean()
        hist, edges = np.histogram(pnls, bins=30)
        closed_trades_hist = figure(title='Closed Trades PnL Histogram', background_fill_color="#fafafa", height=460)
        colors = ["green" if val >= 0 else "red" for val in edges[:-1]]
        closed_trades_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=colors, line_color="white", alpha=0.5)
        closed_trades_hist.line([mean_value, mean_value], [0, max(hist)], line_color="orange", line_dash='dashed', legend_label=f'Mean PnL: {mean_value:.2f}')
        closed_trades_hist.legend.location = "top_right"
        closed_trades_hist.legend.click_policy = "hide"
        with col2: st.bokeh_chart(closed_trades_hist, use_container_width=True)



with st.sidebar:
    st.header("Enquire Symbol")
    symbol_options = pd.concat([pd.Series(['None']), master_df['Symbol'].sort_values(ascending=True)], ignore_index=True)
    st.selectbox("", options=symbol_options, key='symbol')
