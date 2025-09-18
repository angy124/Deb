import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import threading
import time

# -----------------------------
# 1️⃣ Symbols and Timeframes
# -----------------------------
segments = {
    "Equity Indices": {f"Index{i}": f"IDX{i}" for i in range(1,101)},
    "Stocks": {f"Stock{i}": f"STK{i}.NS" for i in range(1,1001)},
    "Forex": {f"FX{i}": f"FX{i}=X" for i in range(1,101)},
    "Crypto": {f"Crypto{i}": f"CRY{i}-USD" for i in range(1,101)},
    "Commodities": {f"Comm{i}": f"COM{i}=F" for i in range(1,101)}
}
timeframes = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1wk","1mo"]

# -----------------------------
# 2️⃣ Data Structures
# -----------------------------
strategies = {}
trades = []
ai_signals = {}

# -----------------------------
# 3️⃣ LSTM Model
# -----------------------------
class PriceDataset(Dataset):
    def __init__(self, prices, seq_len=10):
        self.prices = prices
        self.seq_len = seq_len
    def __len__(self):
        return len(self.prices)-self.seq_len
    def __getitem__(self, idx):
        x = self.prices[idx:idx+self.seq_len]
        y = self.prices[idx+self.seq_len]
        return torch.tensor(x,dtype=torch.float32).unsqueeze(-1), torch.tensor(y,dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=50,num_layers=2,output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        h,_ = self.lstm(x)
        return self.fc(h[:,-1,:])

models = {}
optimizers = {}
criterions = {}

def train_symbol_lstm(symbol_code, prices, epochs=1):
    if len(prices)<10: return
    if symbol_code not in models:
        models[symbol_code] = LSTMModel()
        optimizers[symbol_code] = torch.optim.Adam(models[symbol_code].parameters(), lr=0.001)
        criterions[symbol_code] = nn.MSELoss()
    model = models[symbol_code]
    optimizer = optimizers[symbol_code]
    criterion = criterions[symbol_code]
    dataset = PriceDataset(prices)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
    model.eval()

def predict_symbol_signal(symbol_code, prices):
    if len(prices) < 10 or symbol_code not in models: return "HOLD"
    seq = torch.tensor(prices[-10:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    pred = models[symbol_code](seq).item()
    if pred > prices[-1]: return "BUY"
    elif pred < prices[-1]: return "SELL"
    else: return "HOLD"

# -----------------------------
# 4️⃣ Trading Functions
# -----------------------------
def fibonacci_levels(prices):
    max_p, min_p = max(prices), min(prices)
    diff = max_p - min_p
    return {"0%": max_p, "23.6%": max_p-0.236*diff, "38.2%": max_p-0.382*diff,
            "50%": max_p-0.5*diff, "61.8%": max_p-0.618*diff, "100%": min_p}

def support_resistance(prices):
    return {"support": min(prices), "resistance": max(prices)}

def smc_zone(prices):
    recent_high = max(prices[-20:])
    recent_low = min(prices[-20:])
    last_close = prices[-1]
    if last_close > recent_high*0.995: return "Liquidity Grab - Bullish"
    elif last_close < recent_low*1.005: return "Liquidity Grab - Bearish"
    else: return "Neutral Zone"

def calculate_target_stop(price, fib_levels, signal):
    if signal=="BUY":
        target = fib_levels["61.8%"]
        stoploss = fib_levels["38.2%"]
    elif signal=="SELL":
        target = fib_levels["38.2%"]
        stoploss = fib_levels["61.8%"]
    else:
        target = price
        stoploss = price
    return target, stoploss

def calculate_trade_accuracy(symbol):
    relevant_trades = [t for t in trades if t["symbol"]==symbol]
    if not relevant_trades: return 0
    correct = 0
    for t in relevant_trades:
        if t["signal"]=="BUY" and t["price"] < t["target"]: correct+=1
        elif t["signal"]=="SELL" and t["price"] > t["target"]: correct+=1
    return round(correct/len(relevant_trades)*100,2)

def execute_trade(strategy_id, price, symbol_code):
    fib = fibonacci_levels(strategies[strategy_id]["historical_prices"])
    signal = predict_symbol_signal(symbol_code, strategies[strategy_id]["historical_prices"])
    target, stoploss = calculate_target_stop(price, fib, signal)
    trades.append({"strategy":strategy_id,"signal":signal,"price":price,"target":target,
                   "stoploss":stoploss,"time":datetime.now(),"symbol":symbol_code})
    accuracy = calculate_trade_accuracy(symbol_code)
    return signal, target, stoploss, accuracy

# -----------------------------
# 5️⃣ Fetch Live Data
# -----------------------------
def fetch_live_data(symbol, period="1d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        df = df.reset_index()
        return df
    except:
        return pd.DataFrame()

# -----------------------------
# 6️⃣ Background AI Update
# -----------------------------
def update_ai_signals_batch(batch_symbols):
    for sym in batch_symbols:
        df = fetch_live_data(sym, period="1d", interval="5m")
        if df.empty: continue
        prices = df['Close'].values[-50:]
        train_symbol_lstm(sym, prices, epochs=1)
        ai_signals[sym] = predict_symbol_signal(sym, prices)

def background_ai_update(all_symbols, batch_size=50, interval=10):
    while True:
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i+batch_size]
            threading.Thread(target=update_ai_signals_batch, args=(batch,), daemon=True).start()
        time.sleep(interval)

# -----------------------------
# 7️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Multi-Symbol Trading", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Live Chart","AI Signals","Trading Dashboard","Learn AI"])

# Segment & Symbol selection
segment_choice = st.sidebar.selectbox("Segment", list(segments.keys()))
symbols_list = segments[segment_choice]
symbol_choice = st.sidebar.selectbox("Symbol", list(symbols_list.keys()))
symbol_code = symbols_list[symbol_choice]

# Timeframe
tf_choice = st.sidebar.selectbox("Timeframe", timeframes)

# Strategy
strategy_id = st.text_input("Strategy ID", value="default")
if strategy_id not in strategies:
    strategies[strategy_id] = {"historical_prices":[100,102,101,103]}

# Auto-refresh
AUTORELOAD_INTERVAL = 5000
st_autorefresh(interval=AUTORELOAD_INTERVAL, key="refresh")

# Start background AI thread
if "ai_thread_started" not in st.session_state:
    all_symbols = []
    for seg in segments:
        all_symbols += list(segments[seg].values())
    threading.Thread(target=background_ai_update, args=(all_symbols,), daemon=True).start()
    st.session_state.ai_thread_started = True

# -----------------------------
# Live Chart
# -----------------------------
if menu=="Live Chart":
    st.title(f"📈 {symbol_choice} Live Chart ({tf_choice})")
    df = fetch_live_data(symbol_code, period="1d", interval=tf_choice)
    if not df.empty:
        prices = df['Close'].values[-50:]
        strategies[strategy_id]["historical_prices"] = prices.tolist()
        signal, target, stoploss, accuracy = execute_trade(strategy_id, prices[-1], symbol_code)
        fib = fibonacci_levels(prices)
        sr = support_resistance(prices)
        smc = smc_zone(prices)

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name="Price"))
        for lvl in fib.values(): fig.add_hline(y=lvl,line_dash="dot",line_color="purple")
        fig.add_hline(y=sr["support"], line_color="green", line_dash="solid")
        fig.add_hline(y=sr["resistance"], line_color="red", line_dash="solid")
        for t in trades[-10:]:
            if t["symbol"]==symbol_code:
                color = "green" if t["signal"]=="BUY" else "red"
                fig.add_scatter(x=[t["time"]], y=[t["price"]], mode="markers",
                                marker=dict(color=color, size=12), name=t["signal"])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Signal: **{signal}**, Target: **{target:.2f}**, Stoploss: **{stoploss:.2f}**, Accuracy: **{accuracy}%**")
        st.write(f"SMC Zone: {smc}")

# -----------------------------
# AI Signals
# -----------------------------
elif menu=="AI Signals":
    st.title("🤖 AI Signals Across Symbols")
    if ai_signals:
        df_signals = pd.DataFrame(list(ai_signals.items()), columns=["Symbol","Signal"])
        st.dataframe(df_signals)

# -----------------------------
# Dashboard
# -----------------------------
elif menu=="Trading Dashboard":
    st.title("💰 Trading Dashboard")
    st.write("Active Strategies:", strategies)
    st.write("Recent Trades:", trades[-20:])

# -----------------------------
# Learn AI
# -----------------------------
elif menu=="Learn AI":
    st.title("🧠 Learn AI Trading Step by Step")
    lessons = ["Technical Analysis","Fibonacci & Indicators","Support & Resistance",
               "Smart Money Concepts","AI for Trading","Strategy Backtesting","Real Trading Deployment"]
    for i, lesson in enumerate(lessons,1):
        st.write(f"{i}. {lesson}")