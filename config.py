import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-factor-timing-ts-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling windows for training (days)
WINDOWS = [63, 252, 504, 1008, 2016, 4032]

# Macro columns for regime prediction
MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Factors to time (list of factor names)
FACTORS = ["value", "momentum", "low_vol", "quality"]   # you can add more
# Lookback for factor performance to create target (monthly horizon)
FORECAST_HORIZON = 21   # days (≈1 month)

# Factor timing model hyperparameters
TIMING_MODEL = "logistic"   # or "random_forest"
TIMING_WINDOW = 252         # days for training the timing classifier

TOP_N = 3
