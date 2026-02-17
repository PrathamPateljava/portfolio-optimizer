"""
Portfolio Optimization Functions
Contains all core functions for portfolio optimization using Modern Portfolio Theory
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_stats(weights, returns_df, rfr=0.03):
    """
    Calculate portfolio statistics
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights for each asset
    returns_df : pd.DataFrame
        DataFrame of daily log returns
    rfr : float
        Risk-free rate (annual)
    
    Returns:
    --------
    tuple : (portfolio_return, portfolio_volatility, sharpe_ratio)
    """
    annual_returns = returns_df.mean() * 252
    annual_cov = returns_df.cov() * 252
    
    portfolio_return = np.sum(annual_returns * weights)
    portfolio_variance = np.dot(weights.T, np.dot(annual_cov, weights))
    portfolio_vol = np.sqrt(portfolio_variance)
    sharpe = (portfolio_return - rfr) / portfolio_vol
    
    return portfolio_return, portfolio_vol, sharpe


def calculate_mvp(returns_df):
    """
    Calculate minimum variance portfolio weights
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of daily log returns
    
    Returns:
    --------
    np.array : Optimal weights for minimum variance portfolio
    """
    n_assets = len(returns_df.columns)
    
    def portfolio_variance(weights):
        annual_cov = returns_df.cov() * 252
        return np.dot(weights.T, np.dot(annual_cov, weights))
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x


def optimize_max_sharpe(returns_df, rfr=0.03):
    """
    Optimize for maximum Sharpe ratio
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of daily log returns
    rfr : float
        Risk-free rate (annual)
    
    Returns:
    --------
    np.array : Optimal weights for maximum Sharpe ratio portfolio
    """
    n_assets = len(returns_df.columns)
    
    def neg_sharpe(weights):
        ret, vol, sharpe = portfolio_stats(weights, returns_df, rfr)
        return -sharpe
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        neg_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x


def generate_efficient_frontier(returns_df, n_points=30):
    """
    Generate efficient frontier
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of daily log returns
    n_points : int
        Number of points to generate on the frontier
    
    Returns:
    --------
    tuple : (frontier_returns, frontier_volatilities)
        Lists of returns and volatilities for portfolios on the efficient frontier
    """
    n_assets = len(returns_df.columns)
    
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
        initial = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_vol, 
            initial, 
            method='SLSQP',
            bounds=bounds, 
            constraints=constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if result.success:
            ret, vol, _ = portfolio_stats(result.x, returns_df)
            frontier_returns.append(ret)
            frontier_vols.append(vol)
    
    return frontier_returns, frontier_vols


def calculate_portfolio_performance(weights, returns_df, rfr=0.03):
    """
    Calculate comprehensive portfolio performance metrics
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    returns_df : pd.DataFrame
        DataFrame of daily log returns
    rfr : float
        Risk-free rate (annual)
    
    Returns:
    --------
    dict : Dictionary containing all performance metrics
    """
    ret, vol, sharpe = portfolio_stats(weights, returns_df, rfr)
    
    # Calculate additional metrics
    daily_returns = returns_df @ weights
    
    # Downside deviation (for Sortino ratio)
    negative_returns = daily_returns[daily_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino = (ret - rfr) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'return': ret,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'downside_deviation': downside_deviation
    }
