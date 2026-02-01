import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
import time
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Stock predictor and analysis", page_icon="📈", layout="wide")
st.title("Stock predictor and analysis")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="TCS.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=730), max_value=datetime.now())
end_date = st.sidebar.date_input("End Date", value=datetime.now(), max_value=datetime.now())
pred_days = st.sidebar.slider("Days to Predict", 1, 30, 7)
refresh_interval = 30

def create_features(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Mid'] = df['SMA_20']
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']
    
    for lag in [1, 2, 3, 5]:
        df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
        df[f'Vol_Lag_{lag}'] = df['Vol_Ratio'].shift(lag)
    
    return df.dropna()

@st.cache_data(ttl=300)
def get_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)
        return stock.info, hist
    except Exception as e:
        return {}, pd.DataFrame()

def get_live_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        live_data = stock.history(period="1d", interval="1m")
        return live_data
    except:
        return pd.DataFrame()

def predict_prices(df, days):
    feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower', 
                    'Vol_Ratio', 'Ret_Lag_1', 'Ret_Lag_2', 'Ret_Lag_3', 'Ret_Lag_5']
    
    df_model = df.dropna()
    X = df_model[feature_cols].values
    y = df_model['Log_Ret'].values 
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    score = model.score(X_scaled, y)
    
    future_prices = []
    history_df = df.copy()
    last_close = history_df['Close'].iloc[-1]
    
    for _ in range(days):
        subset = history_df.tail(60).copy()
        subset = create_features(subset)
        if subset.empty: break
            
        last_features = subset.iloc[[-1]][feature_cols].values
        last_features_scaled = scaler.transform(last_features)
        
        pred_log_ret = model.predict(last_features_scaled)[0]
        next_price = last_close * np.exp(pred_log_ret)
        future_prices.append(next_price)
        
        avg_vol = history_df['Volume'].iloc[-5:].mean()
        next_date = history_df.index[-1] + timedelta(days=1)
        
        new_row = pd.DataFrame({
            'Open': [next_price], 'High': [next_price], 'Low': [next_price], 
            'Close': [next_price], 'Volume': [avg_vol]
        }, index=[next_date])
        
        history_df = pd.concat([history_df, new_row])
        last_close = next_price
    
    return future_prices, score

if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'running' not in st.session_state:
    st.session_state.running = False

run_analysis = st.sidebar.button("Start Analysis & Live Tracking", type="primary")

if run_analysis:
    st.session_state.running = True

main_layout = st.empty()

if st.session_state.running:
    try:
        with main_layout.container():
            st.success(f"LIVE MODE - Auto-refreshing every 30s | Last update: {datetime.now().strftime('%H:%M:%S')}")
            with st.spinner(f"Fetching data for {ticker}..."):
                info, hist = get_data(ticker, start_date, end_date)
                live_data = get_live_data(ticker)
                if hist.empty:
                    st.error(f"No data found for {ticker}. Check the symbol or date range.")
                else:
                    processed_data = create_features(hist)
                    st.header(f"{ticker} - Live Dashboard")
                    if not live_data.empty:
                        current_price = live_data['Close'].iloc[-1]
                        prev_close = info.get('previousClose', live_data['Close'].iloc[0])
                        day_change = current_price - prev_close
                        day_change_pct = (day_change / prev_close) * 100
                    else:
                        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
                        prev_close = info.get('previousClose', hist['Close'].iloc[-2])
                        day_change = 0
                        day_change_pct = 0
                    cols = st.columns(6)
                    def safe_format(val, prefix="", suffix=""):
                        return f"{prefix}{val}{suffix}" if val is not None and val != 'N/A' else "N/A"
                    cols[0].metric("Current Price", f"₹{current_price:,.2f}" if isinstance(current_price, (int, float)) else current_price, 
                                  f"{day_change:+.2f} ({day_change_pct:+.2f}%)")
                    cols[1].metric("High", safe_format(info.get('dayHigh'), "₹"))
                    cols[2].metric("Low", safe_format(info.get('dayLow'), "₹"))
                    cols[3].metric("Volume", f"{info.get('volume', 0):,.0f}")
                    cols[4].metric("Prev Close", f"₹{prev_close}" if isinstance(prev_close, (int, float)) else prev_close)
                    mkt_cap = info.get('marketCap')
                    cols[5].metric("Market Cap", f"₹{mkt_cap/1e9:.2f}B" if mkt_cap else 'N/A')
                    st.markdown("---")
                    graph_option = st.radio(
                        "Select Graph to Display:",
                        ["Live Intraday", " Historical Price", " RSI Indicator", " Price Prediction"],
                        index=0,
                        horizontal=True,
                        key="graph_selector_live"
                    )
                    st.markdown("---")
                    if graph_option == "Live Intraday":
                        if not live_data.empty and len(live_data) > 1:
                            st.subheader("Today's Intraday Movement (Candlestick)")
                            fig_live = go.Figure()
                            fig_live.add_trace(go.Candlestick(
                                x=live_data.index,
                                open=live_data['Open'], high=live_data['High'],
                                low=live_data['Low'], close=live_data['Close'],
                                name='Price'
                            ))
                            min_time = live_data.index.min()
                            max_time = live_data.index.max()
                            range_padding = timedelta(minutes=5)
                            fig_live.update_layout(height=500, template='plotly_dark', title=f"{ticker} Intraday", xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig_live, width="stretch")
                        else:
                            st.info("Live intraday data not available.")
                    elif graph_option == " Historical Price":
                        st.subheader("Historical Price Analysis")
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='#1f77b4')))
                        fig_hist.add_trace(go.Scatter(x=processed_data.index, y=processed_data['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')))
                        fig_hist.add_trace(go.Scatter(x=processed_data.index, y=processed_data['SMA_50'], name='SMA 50', line=dict(color='green', dash='dot')))
                        fig_hist.update_layout(height=600, title=f"{ticker} Historical Trend")
                        st.plotly_chart(fig_hist, width="stretch")
                    
                    elif graph_option == "RSI Indicator":
                        st.subheader("RSI (Relative Strength Index)")
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=processed_data.index, y=processed_data['RSI'], name='RSI', line=dict(color='#9467bd')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(height=500, title=f"{ticker} RSI", yaxis=dict(range=[0, 100]))
                        st.plotly_chart(fig_rsi, width="stretch")
                        
                    elif graph_option == "Price Prediction":
                        st.subheader(f"XGBoost AI Prediction ({pred_days} Days)")
                        if len(processed_data) < 60:
                            st.error("Not enough data to generate predictions.")
                        else:
                            predictions, score = predict_prices(processed_data, pred_days)
                            cols = st.columns(2)
                            cols[0].metric("Model Confidence (R²)", f"{score:.4f}")
                            last_date = hist.index[-1]
                            future_dates = pd.bdate_range(start=last_date + timedelta(1), periods=pred_days)
                            fig_pred = go.Figure()
                            recent = hist.tail(90)
                            fig_pred.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name='Historical', line=dict(color='#1f77b4', width=2)))
                            fig_pred.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast', line=dict(color='red', width=3, dash='dash'), marker=dict(symbol='diamond', size=6)))
                            fig_pred.update_layout(title=f"{ticker} - Future Price Forecast", height=600, hovermode='x unified')
                            st.plotly_chart(fig_pred, width="stretch")
                            
                            current_close = hist['Close'].iloc[-1]
                            final_pred = predictions[-1]
                            chg = ((final_pred - current_close) / current_close) * 100
                            
                            if chg > 0:
                                st.success(f"Bullish: Model predicts price will move to **₹{final_pred:.2f}** ({chg:+.2f}%)")
                            else:
                                st.error(f"Bearish: Model predicts price will move to **₹{final_pred:.2f}** ({chg:+.2f}%)")

        time.sleep(refresh_interval)
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        time.sleep(refresh_interval)
        st.rerun()
else:
    with main_layout.container():
        st.info(" Enter a Stock Ticker (e.g., TCS.NS) and click 'Start Analysis'")
        
        st.markdown("""
        ###Live Tracking:
        - **Real-time Price**: refreshs every 30 seconds
        - **Live**: Current price, day high/low, volume updates
        
        ### Technical Indicators:
        - **(Relative Strength Index**: shows if the stock is overbought or not 
          - RSI > 70: Overbought
          - RSI < 30: Oversold 
          - RSI ≈ 50: Neutral market
        
        ### How It Works:
        1. Click 'Start Analysis & Live Tracking'
        2. select between 4 graphs
        """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Done by Rohith and Adithya</div>", unsafe_allow_html=True)
