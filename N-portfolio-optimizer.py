import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Download data - NOW WORKS FOR ANY NUMBER OF TICKERS!
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM',
           'JNJ', 'V', 'PG', 'MA', 'NVDA']
data = yf.download(tickers, start='2020-10-01', end='2026-02-15')['Close']

# Verify what we downloaded
print("=" * 60)
print("DATA DOWNLOADED")
print("=" * 60)
print(f"Number of stocks: {len(data.columns)}")
print(f"Tickers: {data.columns.tolist()}")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

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
    """Calculate minimum variance portfolio weights (works for N assets)"""
    n_assets = len(returns_df.columns)

    if n_assets == 2:
        # Use analytical formula for 2 assets
        annual_vol = returns_df.std() * np.sqrt(252)
        corr = returns_df.corr().iloc[0, 1]

        vol_A = annual_vol.iloc[0]
        vol_B = annual_vol.iloc[1]

        numerator = vol_B ** 2 - vol_A * vol_B * corr
        denominator = vol_A ** 2 + vol_B ** 2 - 2 * vol_A * vol_B * corr

        w_A = numerator / denominator
        w_B = 1 - w_A

        return np.array([w_A, w_B])
    else:
        # Use optimization for N assets
        def portfolio_variance(weights):
            annual_cov = returns_df.cov() * 252
            return np.dot(weights.T, np.dot(annual_cov, weights))

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x


def optimize_max_sharpe(returns_df, rfr=0.03):
    """Optimize for maximum Sharpe ratio (works for N assets)"""
    n_assets = len(returns_df.columns)

    def neg_sharpe(weights):
        ret, vol, sharpe = portfolio_stats(weights, returns_df, rfr)
        return -sharpe

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1 / n_assets] * n_assets)

    result = minimize(
        neg_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("INDIVIDUAL STOCK STATISTICS")
print("=" * 60)

for ticker in returns.columns:
    annual_ret = returns[ticker].mean() * 252
    annual_vol = returns[ticker].std() * np.sqrt(252)
    sharpe = (annual_ret - 0.03) / annual_vol
    print(f"{ticker:6s}: Return={annual_ret:>7.2%}, Vol={annual_vol:>7.2%}, Sharpe={sharpe:>6.3f}")

print("\n" + "=" * 60)
print("PORTFOLIO OPTIMIZATION RESULTS")
print("=" * 60)

# Equal weights baseline
equal_weights = np.array([1 / len(returns.columns)] * len(returns.columns))
ret, vol, sharpe = portfolio_stats(equal_weights, returns)
print(f"\nEqual Weight Portfolio (1/{len(returns.columns)} each):")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

# Minimum variance portfolio
print("\n" + "-" * 60)
print("MINIMUM VARIANCE PORTFOLIO")
print("-" * 60)
mvp_weights = calculate_mvp(returns)
ret, vol, sharpe = portfolio_stats(mvp_weights, returns)

print(f"\nOptimal Weights (showing > 1%):")
for tick, w in zip(returns.columns, mvp_weights):
    if w > 0.01:
        print(f"  {tick}: {w:>6.1%}")

print(f"\nPortfolio Metrics:")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

# Maximum Sharpe ratio portfolio
print("\n" + "-" * 60)
print("MAXIMUM SHARPE RATIO PORTFOLIO")
print("-" * 60)
max_sharpe_weights = optimize_max_sharpe(returns)
ret, vol, sharpe = portfolio_stats(max_sharpe_weights, returns)

print(f"\nOptimal Weights (showing > 1%):")
for tick, w in zip(returns.columns, max_sharpe_weights):
    if w > 0.01:
        print(f"  {tick}: {w:>6.1%}")

print(f"\nPortfolio Metrics:")
print(f"  Return:     {ret:>7.2%}")
print(f"  Volatility: {vol:>7.2%}")
print(f"  Sharpe:     {sharpe:>7.3f}")

print("\n" + "=" * 60)
print(f"✓ Optimization complete for {len(returns.columns)} assets!")
print("=" * 60)

def generate_efficient_frontier(returns_df, n_points=50):
    """Generate efficient frontier"""
    n_assets = len(returns_df.columns)

    # Get range of possible returns
    individual_returns = [returns_df[col].mean() * 252 for col in returns_df.columns]
    min_ret = min(individual_returns)
    max_ret = max(individual_returns)

    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = []
    frontier_returns = []

    for target_ret in target_returns:
        def portfolio_vol(weights):
            return portfolio_stats(weights, returns_df)[1]

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_stats(w, returns_df)[0] - target_ret}
        ]

        bounds = tuple((0, 1) for _ in range(n_assets))
        initial = np.array([1 / n_assets] * n_assets)

        result = minimize(portfolio_vol, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'disp': False, 'ftol': 1e-9})

        if result.success:
            ret, vol, _ = portfolio_stats(result.x, returns_df)
            frontier_returns.append(ret)
            frontier_vols.append(vol)

    return frontier_returns, frontier_vols

# =============================================================================
# VISUALIZATION
# =============================================================================
# Generate frontier
print("\nGenerating efficient frontier...")
frontier_returns, frontier_vols = generate_efficient_frontier(returns)

# Get portfolio stats for plotting
equal_ret, equal_vol, _ = portfolio_stats(equal_weights, returns)
mvp_ret, mvp_vol, mvp_sharpe = portfolio_stats(mvp_weights, returns)
max_ret, max_vol, max_sharpe = portfolio_stats(max_sharpe_weights, returns)

# Create the plot
plt.figure(figsize=(14, 9))

# Plot efficient frontier
plt.plot(frontier_vols, frontier_returns, 'b-', linewidth=3, label='Efficient Frontier', zorder=2)

# Plot individual stocks
for ticker in returns.columns:
    annual_ret = returns[ticker].mean() * 252
    annual_vol = returns[ticker].std() * np.sqrt(252)
    plt.scatter(annual_vol, annual_ret, s=150, alpha=0.6, edgecolors='black', linewidths=1.5)
    plt.annotate(ticker, (annual_vol, annual_ret),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot special portfolios
plt.scatter(equal_vol, equal_ret, s=400, marker='D', color='blue',
            edgecolors='black', linewidths=2,
            label=f'Equal Weight (Sharpe={0.793:.2f})', zorder=5)

plt.scatter(mvp_vol, mvp_ret, s=500, marker='*', color='green',
            edgecolors='black', linewidths=2,
            label=f'Min Variance (Sharpe={mvp_sharpe:.2f})', zorder=5)

plt.scatter(max_vol, max_ret, s=500, marker='*', color='red',
            edgecolors='black', linewidths=2,
            label=f'Max Sharpe (Sharpe={max_sharpe:.2f})', zorder=5)

# Capital Allocation Line (from risk-free rate through max Sharpe)
rf_rate = 0.03
cal_x = np.linspace(0, max(frontier_vols) * 1.1, 100)
cal_y = rf_rate + (max_ret - rf_rate) / max_vol * cal_x
plt.plot(cal_x, cal_y, 'r--', linewidth=2, alpha=0.7,
         label='Capital Allocation Line', zorder=1)

plt.xlabel('Annual Volatility (Risk)', fontsize=12, fontweight='bold')
plt.ylabel('Annual Return', fontsize=12, fontweight='bold')
plt.title('Efficient Frontier - Portfolio Optimization (Oct 2024 - Feb 2026)',
          fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, max(frontier_vols) * 1.15)
plt.ylim(min(frontier_returns) - 0.05, max(frontier_returns) + 0.05)

# Format as percentages
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
print("✓ Efficient frontier saved to efficient_frontier.png")
plt.show()