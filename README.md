# Factor Timing Engine

Implements time‑series factor timing (Asness et al., 2017). For each factor (market beta, value, momentum, low volatility), a macro‑driven classifier predicts whether the factor will outperform over the next month. The final ETF score is the sum of factor exposures multiplied by the timing probabilities. Higher score → overweight signal.

- **Factors:** market beta, value, momentum, low volatility
- **Timing model:** logistic regression (or random forest) using macro variables (VIX, DXY, yields, credit spreads)
- **Exposures:** rolling OLS betas or ETF characteristics
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
