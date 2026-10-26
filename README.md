# Sigmoid Round 2 — Bike Sharing: More Bikes or Better Allocation?

Human-led, AI-assisted case study by **Aditya Anand** (NIT Warangal), Sigmoid Data Science Trainee, Round 2.
Data: UCI Bike Sharing Dataset (Capital Bikeshare, Washington D.C., 2011-2012; Fanaee-T & Gama, 2013).

**Question.** Riders can't find bikes while bikes sit idle elsewhere. Should the company buy more bikes, or allocate
existing ones better?

**Answer in one paragraph.** Demand is strongly concentrated in time: on working days the 07-09 and 17-19 commute
windows carry 51.7% of daily rentals vs 26.0% on non-working days (H1, p ≈ 4e-102). Under stated assumptions,
scenario estimates of the bikes needed at the 2012 95th-percentile hour (~491) are far below the case's fictional
2,000-bike fleet; that fleet would only be too small if bikes had to be positioned ~8x more than are ever riding at once.
**Recommendation (Aditya's decision):** investigate forecast-guided redistribution and fleet-health improvements
*before* buying bikes, run as an **operational pilot to validate**. Redistribution is **not proven** to be the root
cause: that needs station-level availability and unmet-demand data, which this dataset does not contain.

## The four deliverables

| Deliverable | Where |
|---|---|
| A. Data quality & feature engineering | `analysis.ipynb` §A, `outputs/data_quality_checks.csv`, charts 01-03 |
| B. Hypotheses & testing (H1 commuter peaks, H3 summer heat) | `analysis.ipynb` §B, charts 04-05, `outputs/h3_threshold_sensitivity_EXPLORATORY.csv` |
| C. Demand model & fleet estimate | `analysis.ipynb` §C, `models/`, `outputs/model_metrics_test_2012H2.csv`, `outputs/fleet_*.csv` |
| D. Natural-language prototype | `app.py` + `nl_assistant.py` (Gemini function calling + deterministic fallback) |
| Report | `Final_Report.pdf` |
| Decisions & AI use | `DECISION_LOG.md`, `AI_USAGE.md` |

## Folder structure
```
data/            hour.csv, day.csv (UCI, unmodified)
model_utils.py   all analysis logic: quality checks, features, tests, models, fleet maths
build_notebook.py  builds + executes analysis.ipynb (reproducibility)
analysis.ipynb   executed notebook with all outputs
nl_assistant.py  6 validated analysis tools, Gemini loop, deterministic router, grounding check
app.py           Streamlit UI (entrypoint)
models/          demand_model.joblib (HistGradientBoosting, refit on all data), seasonal_baseline.joblib
outputs/         charts (PNG) + numeric results (CSV/JSON)
tests/           test_prototype.py (66 automated checks)
```

## Run locally
```bash
pip install -r requirements.txt          # Python 3.13 tested; sklearn pinned for the saved models
streamlit run app.py                     # works without a key (deterministic mode)
```
Enable Gemini (key is read only from the environment, never stored):
```bash
export GEMINI_API_KEY="your-key"         # Windows PowerShell:  $env:GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-3.8-flash"   # optional; default tries gemini-3.8-flash then gemini-3.5-flash-lite
streamlit run app.py
```
Reproduce everything: `python build_notebook.py` (re-executes the notebook, rewrites outputs/ and models/; ~1-2 min).
Tests: `python tests/test_prototype.py`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a **private** GitHub repo (no secrets are in it; `.gitignore` excludes `.env` and `.streamlit/secrets.toml`).
2. share.streamlit.io → Create app → repo, branch `main`, main file `app.py` → Advanced settings: Python 3.13,
   Secrets: `GEMINI_API_KEY = "..."` → Deploy.
3. In the app sidebar click **Run Gemini diagnostic**. A green message naming the model and functions called means live
   Gemini tool calling works. Every answer also shows "Answered by Gemini (model)" or the fallback reason.

## Key methodology
- **Chronological split:** train 2011-01-01 to 2012-06-30, test 2012-07-01 to 2012-12-31.
- **No leakage:** `casual` and `registered` (sum to the target), `instant`, and lag features are excluded.
- **Weather inputs** use recorded weather (= a perfect weather forecast assumption).
- **Hypothesis tests** use one row per day (hours in a day are not independent); incomplete days excluded.
- **Matched estimands:** rank tests reported with Hodges-Lehmann shifts; mean differences with permutation tests.
- **Fleet scenario:** `fleet = demand/hr × trip hours × spatial buffer ÷ (1 − maintenance share)`.
  Trip length, buffer and maintenance share are assumptions; the 2,000-bike fleet is fictional.

## Results
| Model (test Jul-Dec 2012) | MAE | RMSE | R² | Peak-hour MAE | Peak bias |
|---|---|---|---|---|---|
| Seasonal baseline | 56.2 | 92.6 | 0.82 | 87.9 | +46.9 |
| **HistGradientBoosting (primary)** | **44.5** | **71.0** | **0.90** | **82.8** | **−51.0** |
| RandomForest | 49.9 | 78.9 | 0.87 | 87.8 | −49.9 |

## Limitations (read before using any number)
- The data is **completed rentals**, citywide and hourly. It contains no station stock, trip origins/destinations,
  trip durations, failed rentals, fleet size or rebalancing records. Unmet demand is invisible.
- HistGradientBoosting **under-predicts commute-peak hours by ~51 rentals/hr** on the test period; forecast-driven fleet
  estimates therefore lean low. Observed percentiles are shown alongside as a reference.
- H3 (heat) evidence is **modest** at the pre-specified 32 °C threshold; stronger results at 34-36 °C are post-hoc and exploratory.
- 2011-2012 D.C. demand is paired with a fictional company's fleet; fleet numbers are **scenarios, not requirements**.
- The app's model is refit on all data, so predictions for 2011-2012 dates are in-sample; post-2012 dates are extrapolations.

## Data that would validate the redistribution pilot
Station dock-status snapshots, trip-level start/end stations and durations, failed/abandoned rental logs, bike health
status, and driver rebalancing logs. A/B pilot: forecast-guided rebalancing in some zones vs the current schedule,
measuring empty-station minutes at commute peaks and completed rentals.

## Citation
Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge.
*Progress in Artificial Intelligence*, 1-15. doi:10.1007/s13748-013-0040-3
