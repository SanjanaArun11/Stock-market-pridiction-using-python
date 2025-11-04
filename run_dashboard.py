# run_dashboard.py
# Setup script to install dependencies and run the dashboard

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    required_packages = [
        'streamlit',
        'plotly', 
        'yfinance',
        'pandas',
        'numpy',
        'matplotlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def create_dashboard_file():
    """Create the dashboard file"""
    dashboard_code = '''import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard", 
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Stock Analysis Dashboard")

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="e.g., AAPL, GOOGL, MSFT")

# Sidebar: Time Period
time_period = st.sidebar.selectbox(
    "Select Time Period", 
    ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"]
)

# Map time period to days
period_mapping = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

@st.cache_data
def load_stock_data(symbol, period):
    """Load stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return None
            
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
with st.spinner(f"Loading {symbol} data..."):
    df = load_stock_data(symbol.upper(), period_mapping[time_period])

if df is None or df.empty:
    st.error(f"Could not load data for {symbol}. Please check the symbol and try again.")
    st.stop()

# Create features for analysis
def create_technical_indicators(df):
    """Create technical indicators"""
    df = df.copy()
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

df = create_technical_indicators(df)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

with col1:
    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

with col2:
    st.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

with col3:
    high_52w = df['High'].max()
    st.metric("Period High", f"${high_52w:.2f}")

with col4:
    low_52w = df['Low'].min()
    st.metric("Period Low", f"${low_52w:.2f}")

# Main chart - Stock Price with Moving Averages
st.subheader(f"{symbol.upper()} Stock Price Chart")

fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name=symbol.upper()
))

# Moving averages
fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['MA_20'], 
    mode='lines',
    name='MA 20',
    line=dict(color='orange', width=1)
))

fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['MA_50'], 
    mode='lines',
    name='MA 50',
    line=dict(color='red', width=1)
))

fig.update_layout(
    title=f"{symbol.upper()} Price Chart with Moving Averages",
    yaxis_title="Price ($)",
    xaxis_title="Date",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Technical Analysis Charts
col1, col2 = st.columns(2)

with col1:
    # Volume chart
    st.subheader("Volume Analysis")
    fig_volume = px.bar(df, x='Date', y='Volume', title="Trading Volume")
    fig_volume.update_traces(marker_color='lightblue')
    st.plotly_chart(fig_volume, use_container_width=True)

with col2:
    # RSI chart
    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.update_layout(title="RSI Indicator", yaxis_title="RSI", xaxis_title="Date")
    st.plotly_chart(fig_rsi, use_container_width=True)

# Performance Metrics
st.subheader("Performance Metrics")

# Calculate metrics
total_return = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
volatility = df['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized
max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Return", f"{total_return:.2f}%")

with col2:
    st.metric("Volatility (Annualized)", f"{volatility:.2f}%")

with col3:
    st.metric("Max Drawdown", f"-{max_drawdown:.2f}%")

# Data Table
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(20))

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** Yahoo Finance  
**Disclaimer:** This dashboard is for educational purposes only. Not financial advice.
""")
'''
    
    with open('stock_dashboard_fixed.py', 'w', encoding='utf-8') as f:
        f.write(dashboard_code)
    print("✓ Dashboard file created: stock_dashboard_fixed.py")

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("\n" + "="*50)
    print("🚀 Starting Stock Dashboard...")
    print("="*50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "stock_dashboard_fixed.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("\nTry running manually:")
        print("streamlit run stock_dashboard_fixed.py")

if __name__ == "__main__":
    print("🔧 Setting up Stock Dashboard...")
    print("-" * 40)
    
    # Install requirements
    install_requirements()
    
    # Create dashboard file
    create_dashboard_file()
    
    # Ask user if they want to run the dashboard
    print("\n✅ Setup complete!")
    response = input("\nDo you want to run the dashboard now? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        run_dashboard()
    else:
        print("\nTo run the dashboard later, use:")
        print("streamlit run stock_dashboard_fixed.py")
        print("\nOr run this script again!")