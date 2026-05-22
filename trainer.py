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
            returns_win = returns.iloc[-win:]
            macro_win = macro.iloc[-win:]

            # Compute factor returns (market, value)
            market_ret, value_ret = compute_factor_returns(returns_win, window=60)
            # Compute factor exposures for each ETF
            exposures = compute_factor_exposures(returns_win, window=60)
            if exposures.empty:
                continue

            # Factor timing probabilities
            timing_probs = {}

            # Market factor
            if market_ret.std() > 0 and len(market_ret.dropna()) > 50:
                model_mkt, scaler_mkt = train_timing_model(macro_win, market_ret, horizon=config.FORECAST_HORIZON, model_type=config.TIMING_MODEL)
                timing_probs['market'] = predict_timing(model_mkt, scaler_mkt, macro_win.iloc[-1].values)
            else:
                timing_probs['market'] = 0.5

            # Value factor
            if value_ret.std() > 0 and len(value_ret.dropna()) > 50:
                model_val, scaler_val = train_timing_model(macro_win, value_ret, horizon=config.FORECAST_HORIZON, model_type=config.TIMING_MODEL)
                timing_probs['value'] = predict_timing(model_val, scaler_val, macro_win.iloc[-1].values)
            else:
                timing_probs['value'] = 0.5

            # Momentum and low volatility: use default 0.5 (can be enhanced later)
            timing_probs['momentum'] = 0.5
            timing_probs['low_vol'] = 0.5

            # Compute per‑ETF composite score
            last_exposures = exposures.iloc[-1].to_dict()
            scores = {}
            for etf in tickers:
                mkt_beta = last_exposures.get(f"{etf}_market_beta", 0.0)
                val_beta = last_exposures.get(f"{etf}_value_beta", 0.0)
                mom_char = last_exposures.get(f"{etf}_momentum", 0.0)
                lowvol_char = last_exposures.get(f"{etf}_low_vol", 0.0)
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
