import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import ta
import warnings
warnings.filterwarnings('ignore')

# Robust yfinance import with fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    st.success("✅ Live data mode: yfinance loaded successfully")
except ImportError as e:
    st.error(f"⚠️ yfinance import failed: {str(e)}")
    st.warning("🔄 Running in DEMO MODE with sample data")
    YFINANCE_AVAILABLE = False
    yf = None

# Page configuration
st.set_page_config(
    page_title="Live Stock Analysis & Future Projection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.positive {
    color: #00ff00;
}
.negative {
    color: #ff0000;
}
</style>
""", unsafe_allow_html=True)

def get_demo_data(symbol="AAPL", period="1y"):
    """Generate realistic demo stock data when yfinance is unavailable"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate realistic stock data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
    
    # Simulate stock price movement
    base_price = 150.0 if symbol == "AAPL" else 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create realistic OHLCV data
    data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.03) for p in prices],
        'Low': [p * np.random.uniform(0.97, 1.00) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data

# Title
st.markdown('<h1 class="main-header">📈 Live Stock Analysis & Future Projection</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("📊 Analysis Controls")

# Stock selection
default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
selected_stock = st.sidebar.selectbox(
    "Select Stock Symbol:",
    options=default_stocks + ['Custom'],
    index=0
)

if selected_stock == 'Custom':
    custom_stock = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL")
    selected_stock = custom_stock.upper()

# Time period selection
time_periods = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}

selected_period = st.sidebar.selectbox(
    "Historical Data Period:",
    options=list(time_periods.keys()),
    index=3
)

# Projection settings
st.sidebar.subheader("🔮 Projection Settings")
projection_days = st.sidebar.slider("Days to Project:", min_value=7, max_value=90, value=30)
projection_method = st.sidebar.selectbox(
    "Projection Method:",
    ['Linear Regression', 'Polynomial Regression', 'Moving Average Trend']
)

# Technical indicators
st.sidebar.subheader("📊 Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (20, 50)", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period):
    """Fetch stock data with fallback to demo data when yfinance is unavailable"""
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            
            # Try different approaches to fetch data
            data = None
            info = None
            
            # Method 1: Standard history call
            try:
                data = ticker.history(period=period, auto_adjust=True, prepost=True)
            except:
                pass
            
            # Method 2: If first method fails, try with different parameters
            if data is None or data.empty:
                try:
                    data = ticker.history(period=period, interval='1d', auto_adjust=True)
                except:
                    pass
            
            # Method 3: Try with explicit start and end dates
            if data is None or data.empty:
                try:
                    period_days = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
                    days = period_days.get(period, 365)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                except:
                    pass
            
            # Try to get company info
            try:
                info = ticker.info
            except:
                info = {'longName': symbol, 'sector': 'N/A', 'industry': 'N/A'}
            
            # Validate data
            if data is not None and not data.empty and len(data) > 0:
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in data.columns for col in required_columns):
                    return data, info
            
            # If yfinance fails, fall back to demo data
            st.warning(f"No live data available for {symbol}, using demo data")
            demo_data = get_demo_data(symbol, period)
            demo_info = {'longName': f'{symbol} (Demo)', 'sector': 'Demo', 'industry': 'Demo'}
            return demo_data, demo_info
            
        except Exception as e:
            st.error(f"Error fetching {symbol}: {str(e)}")
            st.info("Falling back to demo data")
            demo_data = get_demo_data(symbol, period)
            demo_info = {'longName': f'{symbol} (Demo)', 'sector': 'Demo', 'industry': 'Demo'}
            return demo_data, demo_info
    else:
        # yfinance not available, use demo data
        demo_data = get_demo_data(symbol, period)
        demo_info = {'longName': f'{symbol} (Demo)', 'sector': 'Demo', 'industry': 'Demo'}
        return demo_data, demo_info

def test_yfinance_connection():
    """Test yfinance connection with a simple query"""
    if not YFINANCE_AVAILABLE:
        return False, "yfinance not available - using demo data"
    
    try:
        # Test with a very reliable stock
        test_ticker = yf.Ticker("AAPL")
        test_data = test_ticker.history(period="5d")
        if test_data is not None and not test_data.empty:
            return True, "Connection successful"
        else:
            return False, "No data returned"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def create_sample_data():
    """Create sample stock data for demonstration when yfinance is unavailable"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate 252 days of sample data (1 trading year)
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    # Generate realistic stock price data
    np.random.seed(42)  # For reproducible results
    initial_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for return_rate in returns[1:]:
        prices.append(prices[-1] * (1 + return_rate))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
    data['Volume'] = np.random.randint(1000000, 10000000, len(data))
    
    # Fill any NaN values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Sample company info
    info = {
        'longName': 'Sample Technology Company',
        'sector': 'Technology',
        'industry': 'Software',
        'marketCap': 2500000000000,
        'trailingPE': 28.5,
        'dividendYield': 0.005,
        'beta': 1.2
    }
    
    return data, info

def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    
    return data

def project_future_prices(data, days, method):
    """Project future stock prices using different methods"""
    prices = data['Close'].values
    dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days, freq='D')
    
    if method == 'Linear Regression':
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
        future_prices = model.predict(future_X)
        
    elif method == 'Polynomial Regression':
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        future_X = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
        future_X_poly = poly_features.transform(future_X)
        future_prices = model.predict(future_X_poly)
        
    else:  # Moving Average Trend
        recent_trend = np.mean(np.diff(prices[-20:]))  # Last 20 days trend
        last_price = prices[-1]
        future_prices = [last_price + (i + 1) * recent_trend for i in range(days)]
    
    return dates, future_prices

# Main application
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()

# Add connection test section
with st.sidebar.expander("🔧 Debug Info", expanded=False):
    if st.button("Test yfinance Connection"):
        with st.spinner("Testing connection..."):
            is_connected, message = test_yfinance_connection()
            if is_connected:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                st.info("Try refreshing the page or check your internet connection.")
    
    use_demo_data = st.checkbox("Use Demo Data (if yfinance fails)", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Demo Data"):
            st.session_state['force_demo'] = True
    with col2:
        if st.button("Try Live Data"):
            st.session_state['force_demo'] = False
            st.cache_data.clear()

# Fetch and display data
with st.spinner(f"Fetching data for {selected_stock}..."):
    # Check if we should force demo mode
    force_demo = st.session_state.get('force_demo', False)
    
    if force_demo:
        stock_data, stock_info = create_sample_data()
        selected_stock = "DEMO"
        st.info("📊 Using demo data for demonstration purposes")
    else:
        stock_data, stock_info = fetch_stock_data(selected_stock, time_periods[selected_period])
        
        # If yfinance fails and demo mode is enabled, use sample data
        if (stock_data is None or stock_data.empty) and use_demo_data:
            st.warning(f"⚠️ Could not fetch live data for {selected_stock}. Using demo data instead.")
            stock_data, stock_info = create_sample_data()
            selected_stock = f"{selected_stock} (DEMO)"

if stock_data is not None and not stock_data.empty:
    # Calculate technical indicators
    stock_data = calculate_technical_indicators(stock_data)
    
    # Current stock info
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = stock_data['Close'].iloc[-1]
    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    with col1:
        st.metric(
            label=f"{selected_stock} Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Volume",
            value=f"{stock_data['Volume'].iloc[-1]:,.0f}"
        )
    
    with col3:
        high_52w = stock_data['High'].rolling(window=252).max().iloc[-1]
        low_52w = stock_data['Low'].rolling(window=252).min().iloc[-1]
        st.metric(
            label="52W High",
            value=f"${high_52w:.2f}"
        )
    
    with col4:
        st.metric(
            label="52W Low",
            value=f"${low_52w:.2f}"
        )
    
    # Stock info from yfinance
    if stock_info:
        st.subheader(f"📋 {stock_info.get('longName', selected_stock)} Overview")
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** ${stock_info.get('marketCap', 0):,.0f}")
        
        with info_col2:
            st.write(f"**P/E Ratio:** {stock_info.get('trailingPE', 'N/A')}")
            st.write(f"**Dividend Yield:** {stock_info.get('dividendYield', 0)*100:.2f}%" if stock_info.get('dividendYield') else "**Dividend Yield:** N/A")
            st.write(f"**Beta:** {stock_info.get('beta', 'N/A')}")
    
    # Main price chart with projections
    st.subheader("📈 Price Chart with Future Projections")
    
    # Generate future projections
    future_dates, future_prices = project_future_prices(stock_data, projection_days, projection_method)
    
    # Create the main chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Technical Indicators', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add technical indicators
    if show_sma:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
    
    if show_bollinger:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
    
    # Add future projections
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_prices,
            name=f'Projection ({projection_method})',
            line=dict(color='purple', dash='dot', width=3)
        ),
        row=1, col=1
    )
    
    # RSI
    if show_rsi:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if show_macd:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['MACD_Histogram'], name='Histogram'),
            row=3, col=1
        )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"{selected_stock} Technical Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    
    st.plotly_chart(fig, width='stretch')
    
    # Future projection summary
    st.subheader("🔮 Future Projection Summary")
    
    proj_col1, proj_col2, proj_col3 = st.columns(3)
    
    with proj_col1:
        projected_end_price = future_prices[-1]
        projected_change = projected_end_price - current_price
        projected_change_pct = (projected_change / current_price) * 100
        
        st.metric(
            label=f"Projected Price ({projection_days} days)",
            value=f"${projected_end_price:.2f}",
            delta=f"{projected_change:+.2f} ({projected_change_pct:+.2f}%)"
        )
    
    with proj_col2:
        max_projected = max(future_prices)
        st.metric(
            label="Projected High",
            value=f"${max_projected:.2f}"
        )
    
    with proj_col3:
        min_projected = min(future_prices)
        st.metric(
            label="Projected Low",
            value=f"${min_projected:.2f}"
        )
    
    # Recent performance table
    st.subheader("📊 Recent Performance")
    recent_data = stock_data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
    recent_data['Change %'] = recent_data['Close'].pct_change() * 100
    recent_data = recent_data.round(2)
    st.dataframe(recent_data, width='stretch')
    
    # Analysis insights
    st.subheader("🧠 AI Insights")
    
    insights = []
    
    # RSI analysis
    current_rsi = stock_data['RSI'].iloc[-1]
    if current_rsi > 70:
        insights.append("⚠️ RSI indicates the stock may be overbought (RSI > 70)")
    elif current_rsi < 30:
        insights.append("📈 RSI indicates the stock may be oversold (RSI < 30) - potential buying opportunity")
    else:
        insights.append(f"✅ RSI is in neutral territory ({current_rsi:.1f})")
    
    # Price vs Moving Averages
    if current_price > stock_data['SMA_20'].iloc[-1] and current_price > stock_data['SMA_50'].iloc[-1]:
        insights.append("📈 Price is above both 20-day and 50-day moving averages - bullish trend")
    elif current_price < stock_data['SMA_20'].iloc[-1] and current_price < stock_data['SMA_50'].iloc[-1]:
        insights.append("📉 Price is below both moving averages - bearish trend")
    
    # Volatility analysis
    volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
    if volatility > 30:
        insights.append(f"⚡ High volatility detected ({volatility:.1f}% annualized)")
    elif volatility < 15:
        insights.append(f"😌 Low volatility ({volatility:.1f}% annualized)")
    
    # Projection confidence
    if projection_method == 'Linear Regression':
        insights.append("📊 Linear projection assumes current trend continues - suitable for stable trends")
    elif projection_method == 'Polynomial Regression':
        insights.append("📈 Polynomial projection captures curve patterns - may be more accurate for complex trends")
    else:
        insights.append("📊 Moving average trend projection based on recent momentum")
    
    for insight in insights:
        st.write(insight)
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: This analysis is for educational purposes only and should not be considered as financial advice. 
    Stock market investments carry risk, and past performance does not guarantee future results. 
    Always consult with a qualified financial advisor before making investment decisions.
    """)

else:
    st.error(f"Unable to fetch data for {selected_stock}. Please check the symbol and try again.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit • Data from Yahoo Finance</div>",
    unsafe_allow_html=True
)