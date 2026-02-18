# Portfolio Optimizer

Modern Portfolio Theory implementation with real-time optimization using MVP and Maximum sharpe ratio optimizaiton.

## Demo
🔗 [Live App](https://prathams-portfolio-optimizer.streamlit.app)

## Features
- Maximum Sharpe Ratio optimization
- Minimum Variance Portfolio calculation
- Efficient Frontier visualization
- Real-time market data (Yahoo Finance)

## Tech Stack
Python(3.10 preferred) | NumPy/SciPy | Streamlit | Plotly

## Results
- Achieved 2.28 Sharpe ratio (188% improvement over equal weighting)
- Optimizes 10-asset portfolios in <5 seconds
- Reduced portfolio volatility by 30%

## Installation
\```bash
pip install -r requirements.txt
streamlit run app.py
\```

## Technical Implementation
- **Optimization**: SLSQP algorithm for constrained optimization
- **Data**: yfinance API for real-time market data
- **Frontend**: Streamlit for interactive visualization
- **Deployment**: Streamlit Cloud