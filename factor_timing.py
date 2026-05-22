import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_timing_model(macro_df, factor_returns, horizon=21, model_type='logistic'):
    """
    macro_df: DataFrame of macro variables (daily, aligned with returns)
    factor_returns: Series of daily factor returns
    horizon: days ahead to define target (positive if cumulative return > 0)
    Returns trained classifier and scaler, or (None, None) if cannot train.
    """
    # Shift returns forward to create target
    target = (factor_returns.rolling(horizon).sum().shift(-horizon) > 0).astype(int).dropna()
    # Align macro and target
    common = macro_df.index.intersection(target.index)
    if len(common) < 50:
        return None, None
    X = macro_df.loc[common].values
    y = target.loc[common].values
    # Check if y contains both classes
    if len(np.unique(y)) < 2:
        return None, None
    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if model_type == 'logistic':
        model = LogisticRegression()
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def predict_timing(model, scaler, macro_today):
    """Return probability that factor will be positive, or 0.5 if model missing."""
    if model is None:
        return 0.5
    X = scaler.transform(macro_today.reshape(1, -1))
    prob = model.predict_proba(X)[0, 1]
    return prob
