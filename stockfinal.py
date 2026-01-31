import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock predictor and analysis", page_icon="📈", layout="wide")
st.title("Stock predictor and analysis")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="TCS.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=260), max_value=datetime.now())
end_date = st.sidebar.date_input("End Date", value=datetime.now(), max_value=datetime.now())
pred_days = st.sidebar.slider("Days to Predict", 1, 30, 5)

refresh_interval = 30

def create_features(df):
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
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
    
    for lag in [1, 2, 3, 5]:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    
    return df.dropna()

@st.cache_data(ttl=300)
def get_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    return stock.info, hist

def get_live_data(ticker):
    stock = yf.Ticker(ticker)
    live_data = stock.history(period="1d", interval="1m")
    return live_data

def predict_prices(df, days):
    feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
    df_model = df[feature_cols + ['Close']].dropna()
    
    X = df_model[feature_cols].values
    y = df_model['Close'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    future_preds = []
    current_features = X_scaled[-1].reshape(1, -1)
    
    for _ in range(days):
        pred = model.predict(current_features)[0]
        future_preds.append(pred)
        current_features = np.roll(current_features, -1)
        current_features[0, -1] = pred
    
    return future_preds, model.score(X_scaled, y)

if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'running' not in st.session_state:
    st.session_state.running = False

run_analysis = st.sidebar.button("🚀 Start Analysis & Live Tracking", type="primary")

if run_analysis:
    st.session_state.running = True

if st.session_state.running:
    try:
        status_placeholder = st.empty()
        status_placeholder.success(f"🔴 LIVE MODE - Auto-refreshing every 30s | Last update: {datetime.now().strftime('%H:%M:%S')}")
        
        with st.spinner(f"Fetching data for {ticker}..."):
            info, hist = get_data(ticker, start_date, end_date)
            live_data = get_live_data(ticker)
            
            if hist.empty:
                st.error(f"No data for {ticker}")
            else:
                st.header(f"📊 {ticker} - Live Dashboard")
                
                if not live_data.empty:
                    current_price = live_data['Close'].iloc[-1]
                    prev_close = info.get('previousClose', live_data['Close'].iloc[0])
                    day_change = current_price - prev_close
                    day_change_pct = (day_change / prev_close) * 100
                else:
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                    prev_close = info.get('previousClose', 'N/A')
                    day_change = current_price - prev_close if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)) else 0
                    day_change_pct = (day_change / prev_close * 100) if prev_close and prev_close != 'N/A' else 0
                
                cols = st.columns(6)
                
                if isinstance(current_price, (int, float)):
                    cols[0].metric("Current Price", f"₹{current_price:.2f}", 
                                  f"{day_change:+.2f} ({day_change_pct:+.2f}%)")
                else:
                    cols[0].metric("Current Price", current_price)
                
                cols[1].metric("High", f"₹{info.get('dayHigh', 'N/A')}")
                cols[2].metric("Low", f"₹{info.get('dayLow', 'N/A')}")
                cols[3].metric("Volume", f"{info.get('volume', 0):,.0f}")
                cols[4].metric("Prev Close", f"₹{prev_close}" if isinstance(prev_close, (int, float)) else prev_close)
                cols[5].metric("Market Cap", f"₹{info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else 'N/A')
                
                st.markdown("---")
                
                graph_option = st.radio(
                    "Select Graph to Display:",
                    ["📊 Live Intraday", "📉 Historical Price", "📈 RSI Indicator", "🔮 Price Prediction"],
                    index=0,
                    horizontal=True
                )
                
                st.markdown("---")
                
                graph_container = st.container()
                
                with graph_container:
                    if graph_option == "📊 Live Intraday":
                        if not live_data.empty and len(live_data) > 1:
                            st.subheader("Today's Intraday Movement")
                            
                            min_price = live_data['Close'].min()
                            max_price = live_data['Close'].max()
                            price_range = max_price - min_price
                            y_min = min_price - (price_range * 0.05)
                            y_max = max_price + (price_range * 0.05)
                            
                            fig_live = go.Figure()
                            fig_live.add_trace(go.Scatter(
                                x=live_data.index,
                                y=live_data['Close'],
                                mode='lines+markers',
                                name='Price',
                                line=dict(color='#00ff00' if day_change >= 0 else '#ff0000', width=2),
                                marker=dict(size=3)
                            ))
                            
                            fig_live.update_layout(
                                title=f"{ticker} - Live Intraday (1-minute intervals)",
                                xaxis_title="Time",
                                yaxis_title="Price (₹)",
                                height=600,
                                template='plotly_dark',
                                hovermode='x unified',
                                yaxis=dict(range=[y_min, y_max], autorange=False)
                            )
                            
                            st.plotly_chart(fig_live, use_container_width=True, key="live_chart")
                        else:
                            st.info("Live intraday data not available. Market may be closed.")
                    
                    elif graph_option == "📉 Historical Price":
                        st.subheader("Historical Price Analysis")
                        hist_with_features = create_features(hist)
                        
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', 
                                                     line=dict(color='#1f77b4', width=2)))
                        fig_hist.add_trace(go.Scatter(x=hist_with_features.index, y=hist_with_features['SMA_20'], name='SMA 20', 
                                                     line=dict(color='orange', dash='dash')))
                        fig_hist.add_trace(go.Scatter(x=hist_with_features.index, y=hist_with_features['SMA_50'], name='SMA 50', 
                                                     line=dict(color='green', dash='dot')))
                        
                        fig_hist.update_layout(
                            title=f"{ticker} - Historical Price with Moving Averages",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            height=600,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_hist, use_container_width=True, key="hist_chart")
                    
                    elif graph_option == "📈 RSI Indicator":
                        st.subheader("RSI (Relative Strength Index)")
                        
                        hist_with_features = create_features(hist)
                        
                        fig_rsi = go.Figure()
                        
                        fig_rsi.add_trace(go.Scatter(
                            x=hist_with_features.index, 
                            y=hist_with_features['RSI'], 
                            name='RSI', 
                            line=dict(color='#9467bd', width=2)
                        ))
                        
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                         annotation_text="Overbought (70)", 
                                         annotation_position="right")
                        
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                         annotation_text="Oversold (30)", 
                                         annotation_position="right")
                        
                        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                         annotation_text="Neutral (50)", 
                                         annotation_position="right")
                        
                        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
                        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
                        
                        fig_rsi.update_layout(
                            title=f"{ticker} - RSI Indicator (14-period)",
                            xaxis_title="Date",
                            yaxis_title="RSI Value",
                            height=600,
                            hovermode='x unified',
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_rsi, use_container_width=True, key="rsi_chart")
                        
                        current_rsi = hist_with_features['RSI'].iloc[-1]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current RSI", f"{current_rsi:.2f}")
                        
                        with col2:
                            if current_rsi > 70:
                                st.warning("🔴 **Overbought** - May indicate potential reversal down")
                            elif current_rsi < 30:
                                st.success("🟢 **Oversold** - May indicate potential reversal up")
                            else:
                                st.info("⚪ **Neutral** - No extreme conditions")
                        
                        with col3:
                            rsi_change = current_rsi - hist_with_features['RSI'].iloc[-5]
                            if abs(rsi_change) > 5:
                                trend = "Rising" if rsi_change > 0 else "Falling"
                                st.metric("5-Day Trend", trend, f"{rsi_change:+.2f}")
                            else:
                                st.metric("5-Day Trend", "Stable", f"{rsi_change:+.2f}")
                    
                    elif graph_option == "🔮 Price Prediction":
                        st.subheader("XGBoost Price Prediction")
                        
                        with st.spinner("Training XGBoost model..."):
                            hist_with_features = create_features(hist)
                            
                            if len(hist_with_features) < 100:
                                st.error("Need at least 6 months of data with sufficient history")
                            else:
                                predictions, r2_score_val = predict_prices(hist_with_features, pred_days)
                                
                                cols = st.columns(2)
                                cols[0].metric("R² Score", f"{r2_score_val:.4f}")
                                cols[1].metric("Model Accuracy", f"{r2_score_val * 100:.2f}%")
                                
                                last_date = hist.index[-1]
                                future_dates = pd.bdate_range(start=last_date + timedelta(1), periods=pred_days)
                                
                                fig_pred = go.Figure()
                                
                                recent = hist.tail(60)
                                fig_pred.add_trace(go.Scatter(
                                    x=recent.index, 
                                    y=recent['Close'], 
                                    name='Historical', 
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                fig_pred.add_trace(go.Scatter(
                                    x=future_dates, 
                                    y=predictions, 
                                    name='XGBoost Prediction', 
                                    line=dict(color='red', width=3, dash='dash'),
                                    marker=dict(size=10, symbol='diamond')
                                ))
                                
                                upper = [p * 1.05 for p in predictions]
                                lower = [p * 0.95 for p in predictions]
                                
                                fig_pred.add_trace(go.Scatter(
                                    x=future_dates, y=upper, 
                                    line=dict(width=0), 
                                    showlegend=False
                                ))
                                fig_pred.add_trace(go.Scatter(
                                    x=future_dates, y=lower, 
                                    fill='tonexty', 
                                    fillcolor='rgba(255,0,0,0.2)', 
                                    line=dict(width=0), 
                                    name='Confidence ±5%'
                                ))
                                
                                fig_pred.update_layout(
                                    title=f"{ticker} - {pred_days} Day XGBoost Forecast",
                                    xaxis_title="Date",
                                    yaxis_title="Price (₹)",
                                    height=600,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_pred, use_container_width=True, key="pred_chart")
                                
                                st.subheader("Detailed Predictions")
                                pred_df = pd.DataFrame({
                                    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                                    'Predicted Price': [f"₹{p:.2f}" for p in predictions],
                                    'Change from Current': [f"{((p/hist['Close'].iloc[-1] - 1) * 100):+.2f}%" for p in predictions]
                                })
                                st.dataframe(pred_df, hide_index=True, use_container_width=True)
                                
                                st.markdown("---")
                                st.subheader("📊 AI Insights")
                                
                                current = hist['Close'].iloc[-1]
                                final = predictions[-1]
                                change_pct = ((final - current) / current) * 100
                                
                                cols = st.columns(2)
                                cols[0].info(f"**Current Close**: ₹{current:.2f}\n\n**Predicted ({pred_days}d)**: ₹{final:.2f}")
                                
                                if change_pct > 0:
                                    cols[1].success(f"**Expected Change**: {change_pct:+.2f}%\n\n📈 **Trend**: Upward")
                                else:
                                    cols[1].error(f"**Expected Change**: {change_pct:+.2f}%\n\n📉 **Trend**: Downward")
                
                time.sleep(refresh_interval)
                st.rerun()
                    
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check ticker symbol and date range (6 months minimum)")
        
        time.sleep(refresh_interval)
        st.rerun()

else:
    st.info("👈 Click 'Start Analysis & Live Tracking' to begin")
    
    st.markdown("""
    ### 🔴 Live Tracking Features (Auto-Enabled):
    - **Real-time Price Updates**: Auto-refresh every 30 seconds
    - **Intraday Chart**: 1-minute interval price movements for today
    - **Live Metrics**: Current price, day high/low, volume updates
    - **XGBoost Predictions**: Model retrains with latest data each refresh
    
    ### Technical Indicators:
    - **RSI (Relative Strength Index)**: Momentum oscillator measuring overbought/oversold conditions
      - RSI > 70: Overbought (potential reversal down)
      - RSI < 30: Oversold (potential reversal up)
      - RSI ≈ 50: Neutral market
      
    ### Supported Markets:
    - 🇮🇳 NSE: `.NS` (e.g., TCS.NS, INFY.NS)
    - 🇮🇳 BSE: `.BO` (e.g., RELIANCE.BO)
    - 🇺🇸 US Stocks: No suffix (e.g., AAPL, TSLA, GOOGL)
    
    ### How It Works:
    1. Click 'Start Analysis & Live Tracking'
    2. App auto-fetches latest data and refreshes every 30 seconds
    3. Live intraday chart shows today's price movement (default view)
    4. Switch between 4 graph options anytime
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Done by Rohith and Adithya</div>", 
            unsafe_allow_html=True)