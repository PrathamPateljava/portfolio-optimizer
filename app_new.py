"""
Portfolio Optimization Web Application
Built with Streamlit and Modern Portfolio Theory
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import optimization functions
from portfolio_functions import (
    portfolio_stats,
    calculate_mvp,
    optimize_max_sharpe,
    generate_efficient_frontier,
    calculate_portfolio_performance
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=False)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def download_data(tickers, start_date, end_date):
    """Download stock data with caching"""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    return data

def calculate_log_returns(data):
    """Calculate log returns from price data"""
    returns = np.log(data / data.shift(1)).dropna()
    return returns

# =============================================================================
# SIDEBAR - USER INPUTS
# =============================================================================

st.sidebar.header("⚙️ Configuration")

# Stock selection
st.sidebar.subheader("📈 Select Stocks")

preset = st.sidebar.selectbox(
    "Choose a preset portfolio:",
    ["Custom", "Tech Giants", "Blue Chips", "Dividend Aristocrats", "FAANG"]
)

# Define presets
PRESETS = {
    "Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'ORCL'],
    "Blue Chips": ['AAPL', 'MSFT', 'JPM', 'JNJ', 'V', 'PG', 'WMT', 'KO', 'DIS', 'BA'],
    "Dividend Aristocrats": ['JNJ', 'PG', 'KO', 'PEP', 'MMM', 'CAT', 'XOM', 'CVX', 'MCD', 'WMT'],
    "FAANG": ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
    "Custom": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'NVDA']
}

default_tickers = PRESETS.get(preset, PRESETS["Custom"])

tickers_input = st.sidebar.text_area(
    "Enter stock tickers (one per line):",
    value='\n'.join(default_tickers),
    height=200
)

tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

# Date range
st.sidebar.subheader("📅 Time Period")

col1, col2 = st.sidebar.columns(2)
end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)  # 4 years default

start = col1.date_input("Start Date", start_date, max_value=end_date)
end = col2.date_input("End Date", end_date)

# Risk-free rate
st.sidebar.subheader("💰 Risk Parameters")
rfr = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
    help="Typically the 10-year Treasury yield"
) / 100

# Run button
st.sidebar.markdown("---")
run_optimization = st.sidebar.button(
    "🚀 Run Optimization", 
    type="primary",
    use_container_width=True
)

# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.markdown('<p class="main-header">📊 Portfolio Optimization Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Built with Modern Portfolio Theory (Markowitz 1952)</p>', unsafe_allow_html=True)

if run_optimization:
    if len(tickers) < 2:
        st.error("⚠️ Please enter at least 2 stock tickers.")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Download data
            status_text.info("📥 Downloading stock data from Yahoo Finance...")
            progress_bar.progress(20)
            
            data = download_data(tickers, start, end)
            
            if data.empty:
                st.error("❌ No data found. Please check your tickers and date range.")
                st.stop()
            
            # Handle missing data
            if data.isnull().any().any():
                st.warning("⚠️ Some stocks have missing data. They will be excluded.")
                data = data.dropna(axis=1)
                tickers = data.columns.tolist()
            
            # Calculate returns
            returns = calculate_log_returns(data)
            
            status_text.info("⚙️ Running portfolio optimization...")
            progress_bar.progress(50)
            
            # Calculate portfolios
            equal_weights = np.array([1/len(returns.columns)] * len(returns.columns))
            mvp_weights = calculate_mvp(returns)
            max_sharpe_weights = optimize_max_sharpe(returns, rfr)
            
            # Get statistics
            equal_ret, equal_vol, equal_sharpe = portfolio_stats(equal_weights, returns, rfr)
            mvp_ret, mvp_vol, mvp_sharpe = portfolio_stats(mvp_weights, returns, rfr)
            max_ret, max_vol, max_sharpe = portfolio_stats(max_sharpe_weights, returns, rfr)
            
            status_text.info("📊 Generating efficient frontier...")
            progress_bar.progress(75)
            
            # Generate frontier
            frontier_returns, frontier_vols = generate_efficient_frontier(returns)
            
            progress_bar.progress(100)
            status_text.success(f"✅ Optimization complete! Analyzed {len(returns.columns)} stocks over {len(returns)} trading days")
            
            # Clear progress
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # =================================================================
            # DISPLAY RESULTS
            # =================================================================
            
            st.header("📊 Optimization Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Max Sharpe Ratio",
                    f"{max_sharpe:.3f}",
                    delta=f"+{((max_sharpe/equal_sharpe - 1) * 100):.1f}% vs Equal"
                )
            
            with col2:
                st.metric(
                    "📈 Max Sharpe Return",
                    f"{max_ret*100:.2f}%",
                    delta=f"{((max_ret - equal_ret) * 100):.2f}%"
                )
            
            with col3:
                st.metric(
                    "🛡️ Min Variance Vol",
                    f"{mvp_vol*100:.2f}%",
                    delta=f"{((mvp_vol - equal_vol) * 100):.2f}%",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "📊 # of Stocks",
                    len(returns.columns),
                    delta=f"{len(returns)} days"
                )
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Portfolios", 
                "📈 Efficient Frontier", 
                "📊 Stocks Analysis", 
                "🔢 Raw Data"
            ])
            
            # TAB 1: OPTIMAL PORTFOLIOS
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Maximum Sharpe Ratio Portfolio")
                    st.markdown(f"**Sharpe:** `{max_sharpe:.3f}` | **Return:** `{max_ret*100:.2f}%` | **Risk:** `{max_vol*100:.2f}%`")
                    
                    # Filter weights > 1%
                    weights_df = pd.DataFrame({
                        'Stock': returns.columns,
                        'Weight (%)': max_sharpe_weights * 100
                    })
                    weights_df = weights_df[weights_df['Weight (%)'] > 1].sort_values('Weight (%)', ascending=False)
                    
                    # Pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=weights_df['Stock'],
                        values=weights_df['Weight (%)'],
                        hole=0.4,
                        marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
                    )])
                    fig.update_layout(height=350, showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(
                        weights_df.style.format({'Weight (%)': '{:.2f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    st.subheader("🛡️ Minimum Variance Portfolio")
                    st.markdown(f"**Sharpe:** `{mvp_sharpe:.3f}` | **Return:** `{mvp_ret*100:.2f}%` | **Risk:** `{mvp_vol*100:.2f}%`")
                    
                    weights_df = pd.DataFrame({
                        'Stock': returns.columns,
                        'Weight (%)': mvp_weights * 100
                    })
                    weights_df = weights_df[weights_df['Weight (%)'] > 1].sort_values('Weight (%)', ascending=False)
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=weights_df['Stock'],
                        values=weights_df['Weight (%)'],
                        hole=0.4,
                        marker=dict(colors=['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA'])
                    )])
                    fig.update_layout(height=350, showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(
                        weights_df.style.format({'Weight (%)': '{:.2f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Comparison table
                st.subheader("📊 Strategy Comparison")
                comparison_df = pd.DataFrame({
                    'Strategy': ['Equal Weight', 'Minimum Variance', 'Maximum Sharpe'],
                    'Return': [f"{equal_ret*100:.2f}%", f"{mvp_ret*100:.2f}%", f"{max_ret*100:.2f}%"],
                    'Volatility': [f"{equal_vol*100:.2f}%", f"{mvp_vol*100:.2f}%", f"{max_vol*100:.2f}%"],
                    'Sharpe Ratio': [f"{equal_sharpe:.3f}", f"{mvp_sharpe:.3f}", f"{max_sharpe:.3f}"]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # TAB 2: EFFICIENT FRONTIER
            with tab2:
                st.subheader("Efficient Frontier Visualization")
                
                fig = go.Figure()
                
                # Frontier
                fig.add_trace(go.Scatter(
                    x=[v*100 for v in frontier_vols],
                    y=[r*100 for r in frontier_returns],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # Individual stocks
                for ticker in returns.columns:
                    annual_ret = returns[ticker].mean() * 252
                    annual_vol = returns[ticker].std() * np.sqrt(252)
                    fig.add_trace(go.Scatter(
                        x=[annual_vol*100],
                        y=[annual_ret*100],
                        mode='markers+text',
                        name=ticker,
                        text=[ticker],
                        textposition="top center",
                        marker=dict(size=10)
                    ))
                
                # Special portfolios
                fig.add_trace(go.Scatter(
                    x=[equal_vol*100],
                    y=[equal_ret*100],
                    mode='markers',
                    name=f'Equal Weight (Sharpe={equal_sharpe:.2f})',
                    marker=dict(size=15, symbol='diamond', color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=[mvp_vol*100],
                    y=[mvp_ret*100],
                    mode='markers',
                    name=f'Min Variance (Sharpe={mvp_sharpe:.2f})',
                    marker=dict(size=20, symbol='star', color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=[max_vol*100],
                    y=[max_ret*100],
                    mode='markers',
                    name=f'Max Sharpe (Sharpe={max_sharpe:.2f})',
                    marker=dict(size=20, symbol='star', color='red')
                ))
                
                # Capital Allocation Line
                cal_x = [0, max(frontier_vols)*1.1*100]
                cal_y = [rfr*100, rfr*100 + (max_ret - rfr) / max_vol * (max(frontier_vols)*1.1)*100]
                fig.add_trace(go.Scatter(
                    x=cal_x,
                    y=cal_y,
                    mode='lines',
                    name='Capital Allocation Line',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title="Annual Volatility (%)",
                    yaxis_title="Annual Return (%)",
                    hovermode='closest',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # TAB 3: STOCKS ANALYSIS
            with tab3:
                st.subheader("Individual Stock Performance")
                
                stocks_data = []
                for ticker in returns.columns:
                    annual_ret = returns[ticker].mean() * 252
                    annual_vol = returns[ticker].std() * np.sqrt(252)
                    sharpe = (annual_ret - rfr) / annual_vol
                    stocks_data.append({
                        'Ticker': ticker,
                        'Annual Return': f"{annual_ret*100:.2f}%",
                        'Annual Volatility': f"{annual_vol*100:.2f}%",
                        'Sharpe Ratio': f"{sharpe:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(stocks_data), use_container_width=True, hide_index=True)
                
                # Correlation heatmap
                st.subheader("📊 Correlation Matrix")
                corr_matrix = returns.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 9}
                ))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # TAB 4: RAW DATA
            with tab4:
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sharpe_df = pd.DataFrame({
                        'Stock': returns.columns,
                        'Weight': max_sharpe_weights
                    }).sort_values('Weight', ascending=False)
                    
                    csv = sharpe_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Max Sharpe Weights",
                        csv,
                        "max_sharpe_weights.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    mvp_df = pd.DataFrame({
                        'Stock': returns.columns,
                        'Weight': mvp_weights
                    }).sort_values('Weight', ascending=False)
                    
                    csv = mvp_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Min Variance Weights",
                        csv,
                        "min_variance_weights.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                st.subheader("📊 Historical Price Data (Last 100 Days)")
                st.dataframe(data.tail(100), use_container_width=True)
                
                st.subheader("📈 Returns Data (Last 100 Days)")
                st.dataframe(returns.tail(100), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

else:
    # Welcome screen
    st.info("👈 **Configure your portfolio in the sidebar and click 'Run Optimization'**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This Tool Does
        
        Uses **Modern Portfolio Theory** to find optimal asset allocations.
        
        ### 📊 Features
        
        - **Maximum Sharpe Ratio Portfolio**: Best risk-adjusted returns
        - **Minimum Variance Portfolio**: Lowest possible risk  
        - **Efficient Frontier**: All optimal portfolio combinations
        - **Correlation Analysis**: How stocks move together
        
        ### 🚀 Quick Start
        
        1. Select stocks (preset or custom)
        2. Set time period (3-5 years recommended)
        3. Adjust risk-free rate
        4. Click "Run Optimization"
        5. Explore results across tabs
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Tips
        
        **Stock Selection:**
        - Use 5-10 stocks minimum
        - Mix sectors for diversification
        
        **Time Period:**
        - 3-5 years recommended
        - Consider market cycles
        
        **Risk-Free Rate:**
        - Use 10-yr Treasury (~4%)
        
        ### ⚠️ Disclaimer
        
        Educational purposes only.
        Past performance ≠ future results.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
