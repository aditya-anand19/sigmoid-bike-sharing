# Decision Log

Each entry records the decision, alternatives, rationale, what the AI contributed, and the **actual** human approval.
Times are IST on 26 Oct 2026.

## Round 1 reasoning carried forward (human, pre-existing)
Aditya's Round 1 answers (submitted before this AI-assisted round) argued:
- the problem is more likely **distribution** than absolute fleet shortage, because bikes sit idle;
- possible causes: bikes piling up at drop locations, commuter imbalance, poorly targeted driver redistribution, unhealthy/offline bikes, tech failure at peak, weather/season;
- completed rentals do not reveal riders who failed to find a bike;
- with a halved budget: fix idle/unhealthy bikes and redistribute by time/location first, buy only if still short;
- 2012 > 2011 does not prove people like cycling more (supply/adoption could explain it).
The office/residential commute movement remains a **hypothesis**: this dataset has no station or trip data.

---

## Checkpoint 1 — analytical plan (approved 19:14, "approve all")

| # | Decision | Alternatives considered | Rationale | AI contribution | Human approval |
|---|---|---|---|---|---|
| 1 | Merge stages: Stage 2+3 together, then 4, then 5 | Five separate checkpoints | ~2 h to deadline | Proposed timing | Approved |
| 2 | Hypotheses: H1 commuter peaks (primary) + H3 summer heat; weather (H2) described only | H1 only; H1+H2 | H1 is the required commuter test; H3 tests the Round 1 "monsoon" guess against evidence | Proposed; pre-check showed July 2011 was hottest & driest | Approved |
| 3 | Missing hours: keep + flag days < 20 h, no imputation; exclude flagged days from tests | Impute; drop | Missing rows are mostly 0-6 am; Sandy days are real events | Proposed | Approved |
| 4 | Split: train ≤ 2012-06-30, test 2012-07-01..12-31 | Train 2011/test 2012; rolling CV | Both years in training; test covers summer, heat, winter | Proposed | Approved |
| 5 | No lag features; recorded weather as input (perfect-forecast assumption) | Lagged nowcasting model | Planning happens days ahead | Proposed | Approved |
| 6 | Fleet defaults: p95 demand, 20-min trips, 10% maintenance, 2x spatial buffer; adjustable | Other values | Business assumptions, not measurements | Proposed | Approved |
| 7 | Prototype: deterministic router first, LLM tool-calling if a key exists | Deterministic only | Aditya has a Google AI Studio (Gemini) key → build Gemini mode | Proposed | Approved; Gemini chosen by Aditya |
| 8 | Exclude Round 1 receipt / roll number from package | — | Privacy | Proposed | Approved |

## Checkpoint 2 — results of Stage 2+3 (approved with refinements, 19:30)

| # | Decision | Alternatives considered | Rationale | AI contribution | Human decision |
|---|---|---|---|---|---|
| 1 | H3: keep the pre-specified 32 °C result as primary; report 34-36 °C as **exploratory (post-hoc)** | Switch to 35 °C | Changing a threshold after seeing results to gain significance is threshold-shopping | Flagged that 35 °C was chosen after viewing the scatter; recommended (a) | **Aditya: keep 32 °C; 35 °C exploratory only** |
| 2 | Accept and document the peak-hour under-prediction (~-51 rentals/hr on 2012-H2); no further tuning | Add trend feature / peak weighting | Strict deadline; fleet sizing uses observed percentiles as reference | Diagnosed bias (trees cannot extrapolate 2012 growth) | **Aditya: accept and document** |
| 3 | HistGradientBoosting = primary hourly model; seasonal baseline kept for comparison | Baseline; RandomForest | Best MAE 44.5 / RMSE 71.0 / R² 0.90 on held-out Jul-Dec 2012 | Trained and evaluated all three | **Aditya: HGB primary** |
| 4 | Business recommendation: **prioritise investigating forecast-guided redistribution and fleet-health improvements before buying bikes**, framed as an **operational pilot to validate**. Redistribution is not proven as root cause without station-level availability and unmet-demand data. Suggested rebalancing times are **trial scheduling ideas**, not optimal schedules. | Stronger "don't buy" claim; softer framing | Consistent with Round 1; limited by aggregate data | Drafted option (a) | **Aditya's decision, with his own refinement on pilot framing** |

### Verification requested by Aditya at Checkpoint 2 (19:30) — findings and corrections

1. **"8.8% vs 221 bikes" inconsistency — he was right that these differ; they are different statistics.**
   8.8% = mean rentals of the busiest hour-of-day (working days 17:00, 525/hr) × 20 min ÷ 2,000.
   221 bikes = 2012 **95th-percentile** hour (662/hr) × 20 min = **11.0%** of 2,000. The single busiest hour (977) = 16.3%.
   *Correction:* the Stage 3 summary wrongly said "at most 8.8% are riding at any hour". The notebook now prints all three with labels.

2. **H3: p = 0.003 with a 95% CI including zero — different estimands and sidedness.**
   Mann-Whitney (one-sided) tests a *rank/location* shift; the CI was a *two-sided 95%* bootstrap of the *mean* difference.
   A one-sided 5% test pairs with a two-sided 90% interval, and the mean is pulled by a few low-demand cooler days.
   *Correction:* `compare_groups()` now reports matched pairs:
   rank-based — MW one-sided p = 0.0034 ↔ Hodges-Lehmann shift −5.0 pts, 95% CI [−8.4, −1.3];
   mean-based — permutation one-sided p = 0.032 ↔ mean diff −3.7 pts, 95% CI [−8.0, +0.7], 90% CI [−7.2, −0.1].
   Interpretation: modest evidence; median-based effect clearer than mean-based.

3. **Fleet calculator accepts ML predictions.** Added `fleet_from_forecast()` and `observed_demand_reference()`;
   the notebook compares forecast-driven and observed-reference scenarios for Jul-Dec 2012; the app's
   `fleet_scenario(demand_basis="forecast_peak", date=...)` sizes from the model's highest predicted hour.
   All outputs are labelled scenario estimates, not verified requirements.
   Re-running the notebook reproduced the model metrics exactly (seeds fixed).

## Stage 4 — prototype (instructed 19:30)
| Decision | Made by | Note |
|---|---|---|
| Gemini via `google-genai` function calling; deterministic fallback | Aditya (instruction) | Key from `GEMINI_API_KEY` only |
| Default model `gemini-3.8-flash`, then `gemini-3.5-flash-lite`; override with `GEMINI_MODEL` | AI, per Google's stable model list checked 26 Oct 2026 | Configurable in sidebar |
| Manual function-calling loop (automatic calling disabled) so every call is validated and logged | AI design | Tool whitelist + argument validation |
| Automatic grounding check: numbers in the answer must appear in tool output | AI design | Shown in UI |
| Live Gemini call **not verified** (Google API unreachable from the build environment) | — | Aditya to test locally |

## Stage 5 — final packaging (instructed 19:54)
| Decision | Made by | Note |
|---|---|---|
| Secure submission (report + ZIP) before any deployment work | Aditya | Deployment optional, must not risk submission |
| Deploy via private GitHub repo + Streamlit Community Cloud; key set by Aditya in Streamlit Secrets, never shared in chat or committed | Aditya | `.gitignore` excludes `.env`, `.streamlit/secrets.toml` |
| Unsupported questions refused explicitly (no adjacent answers) | AI, per Aditya's "fail gracefully" requirement | Tested |
| No changes to approved methodology or results in Stage 5 | — | Report numbers read from `outputs/` |
