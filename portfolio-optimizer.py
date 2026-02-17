import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
# Download data
tickers = ['TSLA', 'JNJ']
data = yf.download(tickers, start='2024-10-01', end='2025-09-30')['Close']

# Verify what we downloaded
print("=" * 60)
print("DATA DOWNLOADED")
print("=" * 60)
print(f"Columns: {data.columns.tolist()}")
print(f"\nPrice changes over the period:")
for ticker in data.columns:
    print(f"{ticker}: Start=${data[ticker].iloc[0]:.2f}, End=${data[ticker].iloc[-1]:.2f}")
    total_return = (data[ticker].iloc[-1] / data[ticker].iloc[0]) - 1
    print(f"        Total return: {total_return:.1%}")

# Calculate log returns
returns = np.log(data / data.shift(1)).dropna()


# =============================================================================
# PORTFOLIO FUNCTIONS
# =============================================================================

def portfolio_stats(weights, returns_df, rfr=0.03):
    """Calculate portfolio statistics"""
    annual_returns = returns_df.mean() * 252
    annual_cov = returns_df.cov() * 252

    portfolio_return = np.sum(annual_returns * weights)
    portfolio_variance = np.dot(weights.T, np.dot(annual_cov, weights))
    portfolio_vol = np.sqrt(portfolio_variance)
    sharpe = (portfolio_return - rfr) / portfolio_vol

    return portfolio_return, portfolio_vol, sharpe


def calculate_mvp(returns_df):
    """Calculate minimum variance portfolio weights"""
    annual_vol = returns_df.std() * np.sqrt(252)
    corr = returns_df.corr().iloc[0, 1]

    vol_A = annual_vol.iloc[0]  # First column
    vol_B = annual_vol.iloc[1]  # Second column

    # MVP formula we derived
    numerator = vol_B ** 2 - vol_A * vol_B * corr
    denominator = vol_A ** 2 + vol_B ** 2 - 2 * vol_A * vol_B * corr

    w_A = numerator / denominator
    w_B = 1 - w_A

    return np.array([w_A, w_B])

def optimize_max_sharpe(returns_df):
    nassets = len(returns_df.columns)
    print(f"\n{nassets} assets")
    def neg_sharpe(weights):
        ret, vol, sharpe = portfolio_stats(weights, returns_df)
        return -sharpe
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0,1),(0,1)]
    initial_weights = [0.5,0.5]

    result = minimize(
        neg_sharpe,
        initial_weights,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
    )

    return result.x
# =============================================================================
# TESTING
# =============================================================================

print("\n" + "=" * 60)
print("INDIVIDUAL STOCK STATISTICS")
print("=" * 60)

for ticker in returns.columns:
    annual_ret = returns[ticker].mean() * 252
    annual_vol = returns[ticker].std() * np.sqrt(252)
    sharpe = (annual_ret - 0.03) / annual_vol
    print(f"\n{ticker}:")
    print(f"  Annual Return:     {annual_ret:>7.2%}")
    print(f"  Annual Volatility: {annual_vol:>7.2%}")
    print(f"  Sharpe Ratio:      {sharpe:>7.3f}")

print("\n" + "=" * 60)
print("PORTFOLIO TESTS")
print("=" * 60)

# Test 1: Equal weights
equal_weights = np.array([0.5, 0.5])
ret, vol, sharpe = portfolio_stats(equal_weights, returns)
print(f"\n50/50 Portfolio:")
print(f"  Weights: {returns.columns[0]}=50%, {returns.columns[1]}=50%")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

# Test 2: Minimum variance
mvp_weights = calculate_mvp(returns)
ret, vol, sharpe = portfolio_stats(mvp_weights, returns)
print(f"\nMinimum Variance Portfolio:")
print(f"  Weights: {returns.columns[0]}={mvp_weights[0]:.1%}, {returns.columns[1]}={mvp_weights[1]:.1%}")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

# Test 3: 100% first stock
weights_first = np.zeros(len(returns.columns))
weights_first[0] = 1.0
ret, vol, sharpe = portfolio_stats(weights_first, returns)
print(f"\n100% {returns.columns[0]}:")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

# Test 4: 100% second stock
weights_second = np.zeros(len(returns.columns))
weights_second[1] = 1.0
ret, vol, sharpe = portfolio_stats(weights_second, returns)
print(f"\n100% {returns.columns[1]}:")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

# 4. Maximum Sharpe ratio portfolio
max_sharpe_weights = optimize_max_sharpe(returns)
ret, vol, sharpe = portfolio_stats(max_sharpe_weights, returns)
weights_str = ', '.join([f'{tick}={w:.1%}' for tick, w in zip(returns.columns, max_sharpe_weights)])

print('Portfolio', 'Max Sharpe')
print('Weights', weights_str)
print('Return', ret)
print('Volatility', vol)
print('Sharpe', sharpe)