import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_factor_returns(returns_df, window=60):
    """
    Compute daily factor returns for:
        - Market: equal‑weighted mean of all ETFs (or SPY if available)
        - Value: IWD - IWF (if available) else zero
    Returns (market_ret, value_ret) as pandas Series.
    """
    # Market
    if 'SPY' in returns_df.columns:
        market_ret = returns_df['SPY']
    else:
        market_ret = returns_df.mean(axis=1)
    # Value
    if 'IWD' in returns_df.columns and 'IWF' in returns_df.columns:
        value_ret = returns_df['IWD'] - returns_df['IWF']
    else:
        value_ret = pd.Series(0, index=returns_df.index)
    return market_ret, value_ret

def compute_factor_exposures(returns_df, window=60):
    """
    For each ETF, compute rolling exposures to market and value factors.
    Also compute ETF‑specific characteristics: momentum (12‑1m), low vol (negative vol rank).
    Returns a DataFrame with columns: ETF_market_beta, ETF_value_beta, ETF_momentum, ETF_low_vol.
    """
    market_ret, value_ret = compute_factor_returns(returns_df, window)
    etfs = returns_df.columns
    n = len(returns_df)
    exposures = pd.DataFrame(index=returns_df.index)
    for etf in etfs:
        ret = returns_df[etf]
        # Rolling betas
        market_beta = np.zeros(n)
        value_beta = np.zeros(n)
        for i in range(window, n):
            X = np.column_stack([market_ret.iloc[i-window:i], value_ret.iloc[i-window:i]])
            y = ret.iloc[i-window:i].values
            valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            if valid.sum() < 10:
                continue
            X_clean = X[valid]
            y_clean = y[valid]
            try:
                lr = LinearRegression()
                lr.fit(X_clean, y_clean)
                market_beta[i] = lr.coef_[0]
                value_beta[i] = lr.coef_[1]
            except:
                pass
        # Momentum: 12‑month (252d) minus 1‑month (21d) cumulative return
        mom = ret.rolling(252).apply(lambda x: (1+x).prod() - 1, raw=False) - ret.rolling(21).apply(lambda x: (1+x).prod() - 1, raw=False)
        # Low volatility: negative of rolling 60d volatility (annualised)
        vol = ret.rolling(60).std() * np.sqrt(252)
        low_vol = -vol
        exposures[f"{etf}_market_beta"] = market_beta
        exposures[f"{etf}_value_beta"] = value_beta
        exposures[f"{etf}_momentum"] = mom
        exposures[f"{etf}_low_vol"] = low_vol
    return exposures
