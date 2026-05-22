import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from factor_definitions import compute_factor_exposures, compute_factor_returns
from factor_timing import train_timing_model, predict_timing

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Factor Timing TS) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 100:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 100:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            # Use last `win` days
            returns_win = returns.iloc[-win:]
            macro_win = macro.iloc[-win:]

            # Compute factor returns (market, value) for the window
            market_ret, value_ret = compute_factor_returns(returns_win, window=60)
            # Compute factor exposures for each ETF
            exposures = compute_factor_exposures(returns_win, window=60)
            # The exposures DataFrame has many columns; we need to align with ETFs.
            # For each ETF, we will have a vector of factor exposures (market_beta, value_beta, momentum, low_vol).
            # Then we need timing probabilities for each factor (market, value, momentum, low_vol) separately.
            # Train timing models for each factor using macro data
            timing_probs = {}
            # Market factor timing: use market_ret as the factor return
            model_mkt, scaler_mkt = train_timing_model(macro_win, market_ret, horizon=config.FORECAST_HORIZON, model_type=config.TIMING_MODEL)
            timing_probs['market'] = predict_timing(model_mkt, scaler_mkt, macro_win.iloc[-1].values)
            # Value factor timing
            model_val, scaler_val = train_timing_model(macro_win, value_ret, horizon=config.FORECAST_HORIZON, model_type=config.TIMING_MODEL)
            timing_probs['value'] = predict_timing(model_val, scaler_val, macro_win.iloc[-1].values)
            # For momentum and low volatility, we need factor returns: we can define momentum factor return as the cross‑sectional average of ETF momentum?
            # For simplicity, we'll use the average of ETF momentum as factor return.
            # We'll compute momentum factor return as the mean of ETF momentum (calculated from exposures? Actually exposures are rolling; we need a time series of factor returns.
            # Better: compute ETF momentum as a time series (daily) and average across ETFs.
            # Let's compute momentum factor return as equal‑weighted ETF momentum (daily).
            # First, compute daily momentum for each ETF: 12‑month minus 1‑month return (rolling).
            momentum_factor = pd.Series(0, index=returns_win.index)
            lowvol_factor = pd.Series(0, index=returns_win.index)
            for etf in tickers:
                ret = returns_win[etf]
                mom = ret.rolling(252).apply(lambda x: (1+x).prod() - 1, raw=False) - ret.rolling(21).apply(lambda x: (1+x).prod() - 1, raw=False)
                momentum_factor += mom.fillna(0)
            momentum_factor = momentum_factor / len(tickers)
            # Low volatility factor: negative of cross‑sectional volatility rank? Not straightforward.
            # For timing, we'll use the average ETF low volatility characteristic (which is already ETF‑specific).
            # But we need a factor return for training. We'll use the equal‑weighted return of ETFs with low volatility (bottom 20%) minus top 20%.
            # For simplicity, we'll skip training for low_vol and use a constant 0.5 probability.
            timing_probs['momentum'] = 0.5
            timing_probs['low_vol'] = 0.5
            # If we have enough data, we can train:
            model_mom, scaler_mom = train_timing_model(macro_win, momentum_factor, horizon=config.FORECAST_HORIZON, model_type=config.TIMING_MODEL)
            if model_mom is not None:
                timing_probs['momentum'] = predict_timing(model_mom, scaler_mom, macro_win.iloc[-1].values)
            # Now compute per‑ETF composite score
            # For each ETF, we need its current factor exposures (the last row of exposures)
            last_exposures = exposures.iloc[-1].to_dict()
            scores = {}
            for etf in tickers:
                mkt_beta = last_exposures.get(f"{etf}_market_beta", 0.0)
                val_beta = last_exposures.get(f"{etf}_value_beta", 0.0)
                mom_char = last_exposures.get(f"{etf}_momentum", 0.0)
                lowvol_char = last_exposures.get(f"{etf}_low_vol", 0.0)   # negative volatility
                # Composite score = sum of factor exposure * timing probability
                score = (mkt_beta * timing_probs['market'] +
                         val_beta * timing_probs['value'] +
                         mom_char * timing_probs['momentum'] +
                         lowvol_char * timing_probs['low_vol'])
                scores[etf] = score
            window_results[win] = scores
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "timing_score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by factor timing score: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/factor_timing_ts_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Factor Timing TS Engine complete ===")

if __name__ == "__main__":
    main()
