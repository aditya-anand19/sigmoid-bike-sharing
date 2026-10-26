"""Builds analysis.ipynb from the cells below and executes it in place (python build_notebook.py)."""
import nbformat as nbf
from nbclient import NotebookClient

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Sigmoid Round 2 — Bike-sharing: more bikes, or better allocation?
Human-led, AI-assisted analysis. All logic is in `model_utils.py`; this notebook runs it and records outputs.

**What this data can and cannot show.** `hour.csv`/`day.csv` are *citywide completed rentals* (Capital Bikeshare, D.C., 2011-12).
They contain no station stock, origin/destination, failed requests, fleet size or trip duration.
So we keep five quantities apart: (1) observed completed rentals, (2) predicted completed rentals,
(3) unobserved/suppressed demand (not measurable here), (4) scenario-based fleet estimates, (5) actual fleet sufficiency (not measurable here).""")

code("""import json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
import model_utils as mu
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .3, 'font.size': 10})
C_WORK, C_NON, C_ACC = '#2a6f97', '#e07a1f', '#7a7a7a'
mu.OUT_DIR.mkdir(exist_ok=True); mu.MODEL_DIR.mkdir(exist_ok=True)
RESULTS = {}
h_raw, d = mu.load_hour(), mu.load_day()
h = mu.add_features(h_raw)
h.shape, d.shape""")

md("## A. Data quality checks")
code("""qr = mu.quality_report(h_raw, d)
qr.to_csv(mu.OUT_DIR/'data_quality_checks.csv', index=False)
qr""")
code("""hours_per_day = h_raw.groupby('dteday').size()
inc = mu.incomplete_days(h_raw)
print('Incomplete days (<20 recorded hours):', [str(x.date()) for x in inc])
print('\\nRecorded hours per day (distribution):'); print(hours_per_day.value_counts().sort_index())
# Which hours go missing? Mostly low-demand night hours -> consistent with 'no rentals logged' rather than data loss
full = pd.MultiIndex.from_product([d.dteday, range(24)], names=['dteday','hr'])
missing = full.difference(pd.MultiIndex.from_frame(h_raw[['dteday','hr']]))
miss_hr = pd.Series(missing.get_level_values('hr')).value_counts().sort_index()
print('\\nMissing rows by hour of day:'); print(miss_hr.to_string())
RESULTS['missing_rows'] = int(len(missing)); RESULTS['incomplete_days'] = [str(x.date()) for x in inc]""")
md("""**Decision (approved):** do not impute missing hours. Most are 0-6 am hours, which most likely had zero logged rentals.
Days with <20 recorded hours (incl. 2012-10-29/30, Hurricane Sandy) are flagged and excluded from hypothesis tests;
they stay in the model data because they are genuine low-demand events.
`hum==0` (22 rows) and `windspeed==0` (2,180 rows) look like missing values; tree models are robust to this, so we leave them and note it.""")

md("## A. Feature engineering and exploratory analysis")
code("""print(h[['temp_c','atemp_c','hum_pct','wind_kmh']].describe().round(1))
# Hourly profile by day type and user type
prof = h.groupby(['day_type','hr'])[['cnt','registered','casual']].mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, dt, col in [(ax[0],'Working day',C_WORK),(ax[1],'Non-working day',C_NON)]:
    p = prof.loc[dt]
    a.plot(p.index, p.cnt, color=col, lw=2.5, label='Total')
    a.plot(p.index, p.registered, color=col, lw=1.2, ls='--', label='Registered')
    a.plot(p.index, p.casual, color=C_ACC, lw=1.2, ls=':', label='Casual')
    for x in mu.PEAK_HOURS: a.axvspan(x-.5, x+.5, color='#f2e8cf', zorder=0)
    a.set_title(f'{dt}s — mean rentals per hour'); a.set_xlabel('Hour of day'); a.set_xticks(range(0,24,2))
ax[0].set_ylabel('Mean completed rentals / hour'); ax[0].legend()
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'01_hourly_profile_by_daytype.png'); plt.show()
print((prof['cnt'].unstack(0).loc[[8,13,17,18]]).round(0))""")
code("""# Heatmap hour x weekday
pv = h.pivot_table(index='weekday', columns='hr', values='cnt', aggfunc='mean')
fig, ax = plt.subplots(figsize=(12, 3.2))
im = ax.imshow(pv, aspect='auto', cmap='Blues')
ax.set_yticks(range(7)); ax.set_yticklabels(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
ax.set_xticks(range(24)); ax.set_xlabel('Hour of day'); ax.grid(False)
plt.colorbar(im, label='Mean rentals / hour'); ax.set_title('Demand concentrates at 08:00 and 17-18:00 on weekdays')
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'02_heatmap_weekday_hour.png'); plt.show()""")
code("""# Monthly trend with temperature, and weather effect
m = h.groupby(['year','mnth']).agg(cnt=('cnt','mean'), temp=('temp_c','mean')).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for y, col in [(2011, C_WORK), (2012, C_NON)]:
    s = m[m.year==y]; ax[0].plot(s.mnth, s.cnt, marker='o', color=col, label=f'{y} rentals')
ax0b = ax[0].twinx(); s = m.groupby('mnth').temp.mean()
ax0b.plot(s.index, s.values, color=C_ACC, ls='--', label='Mean temp (°C, both yrs)'); ax0b.set_ylabel('°C'); ax0b.grid(False)
ax[0].set_title('Monthly demand vs temperature'); ax[0].set_xlabel('Month'); ax[0].set_ylabel('Mean rentals / hour')
ax[0].legend(loc='upper left'); ax0b.legend(loc='lower center')
w = h[h.weathersit<=3].groupby('weather_label').cnt.mean().reindex(list(mu.WEATHER_LABELS.values())[:3])
ax[1].bar(w.index, w.values, color=[C_WORK, '#6c9ab8', '#a9c4d6'])
ax[1].set_title('Mean rentals / hour by weather (raw, not controlled)'); ax[1].set_ylabel('Mean rentals / hour')
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'03_monthly_and_weather.png'); plt.show()
print(w.round(1)); RESULTS['yearly_totals'] = h.groupby('year').cnt.sum().to_dict()""")
code("""# How peaky is demand? (relevant to fleet sizing)
pct = h.groupby('year').cnt.quantile([.5,.9,.95,.99]).unstack().round(0)
pct['max'] = h.groupby('year').cnt.max(); print(pct)
RESULTS['hourly_percentiles_by_year'] = pct.to_dict(orient='index')""")

md("""## B. Hypothesis H1 — commuter-hour demand
**H0:** the share of a day's rentals that fall in the commute windows (07-09 & 17-19) is the same on working and non-working days.
**H1:** the share is higher on working days.

*Design:* unit = **day** (hours within a day are correlated, so hourly rows would overstate significance). Incomplete days excluded.
One-sided Mann-Whitney U (no normality assumption), rank-biserial effect size, 95% bootstrap CI of the mean difference, repeated within each season as a robustness check.

*What it can and cannot tell us:* it establishes **when** demand concentrates (a rebalancing deadline). It **cannot** show residential→office movement — that needs origin/destination trip data.""")
code("""h1 = mu.test_h1(h)
for k in ['n_working_days','n_nonworking_days','mean_peak_share_working','mean_peak_share_nonworking',
          'U','p_value','rank_biserial','hodges_lehmann_shift','hodges_lehmann_ci95','mean_diff_ci95',
          'permutation_p_one_sided','registered_share_at_peak_working','registered_share_at_peak_nonworking']:
    print(f'{k:40s} {h1[k]}')
h1['by_season'].round(4)""")
code("""t = h1['daily_table']
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(t[t.workingday==1].peak_share, bins=30, alpha=.75, color=C_WORK, label='Working days')
ax.hist(t[t.workingday==0].peak_share, bins=30, alpha=.75, color=C_NON, label='Non-working days')
ax.set_xlabel('Share of daily rentals in 07-09 & 17-19 windows'); ax.set_ylabel('Number of days'); ax.legend()
ax.set_title('H1: commute windows carry a much larger share of working-day demand')
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'04_h1_peak_share.png'); plt.show()
RESULTS['H1'] = {k: h1[k] for k in h1 if k not in ('by_season','daily_table')}
RESULTS['H1']['by_season'] = h1['by_season'].to_dict(orient='records')""")

md("""## B. Hypothesis H3 — is the July dip heat, not rain?
In Round 1 the July dip was tentatively attributed to a monsoon. D.C. does not have a monsoon, so we test the alternative: heat.

**H0:** on dry summer days (Jun-Aug), demand on hot days (daily max ≥ 32 °C) is the same as on other summer days.
**H1:** hot days have lower demand.

*Design:* unit = day; wet days (weathersit ≥ 3) and incomplete days excluded; daily rentals divided by the mean of the same (year × day-type) group so 2011→2012 growth and the commute effect are removed. One-sided Mann-Whitney + Spearman correlation.
*Limitation:* temperature and humidity move together, so this tests heat-and-humidity together, not heat alone; small sample.

**Estimands (corrected after review).** An earlier version paired a one-sided Mann-Whitney p-value (a *rank* comparison) with a two-sided 95% bootstrap CI of the *mean* difference.
Those answer different questions, and a one-sided 5% test corresponds to a two-sided 90% interval. We now report matched pairs:
- rank-based: Mann-Whitney one-sided p ↔ Hodges-Lehmann location shift + bootstrap CI;
- mean-based: one-sided permutation p on the mean difference ↔ bootstrap CI of the mean difference (95% and 90%).""")
code("""h3 = mu.test_h3(h, d)
for k in ['n_hot_days','n_other_days','median_relative_demand_hot','median_relative_demand_other',
          'p_value','rank_biserial','hodges_lehmann_shift','hodges_lehmann_ci95',
          'mean_diff','permutation_p_one_sided','mean_diff_ci95','mean_diff_ci90',
          'spearman_rho_temp_vs_demand','spearman_p','spearman_rho_temp_vs_humidity']:
    print(f'{k:35s} {h3[k]}')
h3['monthly_context'].round(3)""")
md("""**Exploratory sensitivity (post-hoc, NOT confirmatory).** The pre-specified threshold is 32 °C (approved decision: keep it).
The thresholds below were examined *after* seeing the scatter plot, so their smaller p-values must not be read as stronger evidence.""")
code("""sens = mu.h3_sensitivity(h, d); sens.to_csv(mu.OUT_DIR/'h3_threshold_sensitivity_EXPLORATORY.csv', index=False)
RESULTS['H3_sensitivity_exploratory'] = sens.to_dict(orient='records'); sens.round(4)""")
code("""t3 = h3['daily_table']
fig, ax = plt.subplots(figsize=(7, 3.8))
for wd, col, lab in [(1, C_WORK, 'Working day'), (0, C_NON, 'Non-working day')]:
    s = t3[t3.workingday==wd]; ax.scatter(s.max_temp_c, s.relative_demand, color=col, alpha=.7, label=lab)
ax.axvline(h3['hot_threshold_c'], color=C_ACC, ls='--'); ax.axhline(1, color=C_ACC, lw=.8)
ax.set_xlabel('Daily max temperature (°C)'); ax.set_ylabel('Demand relative to group mean')
ax.set_title('H3: summer demand vs heat (dry days, Jun-Aug)'); ax.legend()
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'05_h3_heat.png'); plt.show()
RESULTS['H3'] = {k: h3[k] for k in h3 if k not in ('monthly_context','daily_table')}
RESULTS['H3']['monthly_context'] = h3['monthly_context'].to_dict(orient='records')""")

md("""## C. Demand model — chronological evaluation
Train: 2011-01-01 → 2012-06-30. Test: 2012-07-01 → 2012-12-31 (summer peak, July heat, winter, Sandy).
Features: calendar + recorded weather (= assumes a perfect weather forecast; acceptable for planning scenarios).
Excluded: `casual`, `registered` (sum to target), `instant`, lag features.
No scaling/encoding is fitted on the data, so no preprocessing leakage. Target modelled as log1p(cnt).""")
code("""train, test = mu.chrono_split(h)
print('train', train.dteday.min().date(), '->', train.dteday.max().date(), len(train))
print('test ', test.dteday.min().date(), '->', test.dteday.max().date(), len(test))
base = mu.SeasonalBaseline().fit(train)
preds = {'Seasonal baseline': base.predict(test)}
models = {}
for name, mdl in mu.make_models().items():
    models[name] = mu.fit_log_model(mdl, train)
    preds[name] = mu.predict_log_model(models[name], test)
metrics = pd.DataFrame({n: mu.evaluate(test.cnt, p, test.is_peak) for n, p in preds.items()}).T.round(2)
metrics.to_csv(mu.OUT_DIR/'model_metrics_test_2012H2.csv'); print('Baseline growth factor', round(base.growth_, 3))
metrics""")
code("""# Daily-level accuracy (what a planner sees) and a sample fortnight
daily = test.assign(**{n: p for n, p in preds.items()}).groupby('dteday')[['cnt']+list(preds)].sum()
dm = {n: (np.abs(daily[n]-daily.cnt)/daily.cnt).median()*100 for n in preds}
print('Median absolute % error on daily totals:', {k: round(v,1) for k,v in dm.items()})
RESULTS['daily_median_ape_pct'] = dm
best = metrics['MAE'].idxmin(); RESULTS['best_model'] = best; RESULTS['metrics'] = metrics.to_dict(orient='index')
win = test[(test.dteday>='2012-09-10')&(test.dteday<'2012-09-24')]
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(win.dteday + pd.to_timedelta(win.hr, unit='h'), win.cnt, color='black', lw=1.2, label='Actual')
idx = win.index - test.index[0]
ax.plot(win.dteday + pd.to_timedelta(win.hr, unit='h'), preds[best][test.index.get_indexer(win.index)], color=C_WORK, lw=1.2, label=best)
ax.plot(win.dteday + pd.to_timedelta(win.hr, unit='h'), preds['Seasonal baseline'][test.index.get_indexer(win.index)], color=C_NON, lw=1, ls='--', label='Baseline')
ax.set_title('Test period sample: 10-23 Sep 2012'); ax.set_ylabel('Rentals / hour'); ax.legend()
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'06_forecast_sample.png'); plt.show()""")
code("""# Which inputs matter? (permutation importance on the test set, best model)
from sklearn.inspection import permutation_importance
mdl = models[best] if best in models else models['HistGradientBoosting']
pi = permutation_importance(mdl, test[mu.FEATURES], np.log1p(test.cnt), n_repeats=5, random_state=42, scoring='neg_mean_absolute_error')
imp = pd.Series(pi.importances_mean, index=mu.FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(6, 3.8)); ax.barh(imp.index, imp.values, color=C_WORK)
ax.set_title('Permutation importance (test, log-MAE increase)'); plt.tight_layout()
plt.savefig(mu.OUT_DIR/'07_feature_importance.png'); plt.show(); print(imp.sort_values(ascending=False).round(3))""")
code("""# Refit the chosen ML model on ALL data for the prototype (evaluation above used the held-out split)
final_name = best if best in models else 'HistGradientBoosting'
final = mu.fit_log_model(mu.make_models()[final_name], h)
joblib.dump({'model': final, 'features': mu.FEATURES, 'name': final_name,
             'trained_on': '2011-01-01..2012-12-31', 'note': 'log1p target; evaluate via notebook split'},
            mu.MODEL_DIR/'demand_model.joblib')
joblib.dump(base, mu.MODEL_DIR/'seasonal_baseline.joblib')
print('saved', final_name)""")

md("""## C. Scenario-based fleet estimate
`fleet = demand/hr × trip duration (h) × spatial buffer ÷ (1 − maintenance share)` — Little's law plus two operational allowances.
Every input except demand is an **assumption** (no trip-duration, fleet or station data in this dataset). Defaults (approved): p95 demand, 20-min trips, 10 % maintenance, 2× spatial buffer.
The 2,000-bike fleet is the *fictional* Round 1 figure, used only as a reference line.""")
code("""ref_period = h[h.year==2012]
demand_levels = {
  'Observed 2012 p90': ref_period.cnt.quantile(.90),
  'Observed 2012 p95 (default)': ref_period.cnt.quantile(.95),
  'Observed 2012 p99': ref_period.cnt.quantile(.99),
  'Observed max hour': h.cnt.max(),
  'Predicted p95, test period': np.percentile(preds[best], 95),
}
rows = []
for lab, dem in demand_levels.items():
    r = mu.fleet_required(dem); r['demand_scenario'] = lab; rows.append(r)
fleet = pd.DataFrame(rows).set_index('demand_scenario').round(1); fleet.to_csv(mu.OUT_DIR/'fleet_scenarios.csv'); print(fleet)
grid = mu.fleet_grid(demand_levels['Observed 2012 p95 (default)']); grid.to_csv(mu.OUT_DIR/'fleet_sensitivity_grid.csv')
print('\\nSensitivity (p95 demand, 10% maintenance): rows = trip minutes, cols = spatial buffer'); print(grid)
RESULTS['fleet_scenarios'] = fleet.to_dict(orient='index'); RESULTS['fleet_grid'] = grid.to_dict()""")
code("""# Utilisation view: with the fictional 2,000-bike fleet, what share is riding at once?
FICTIONAL_FLEET = 2000
util = h.groupby(['day_type','hr']).cnt.mean().unstack(0) * 20/60 / FICTIONAL_FLEET * 100
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(util.index, util['Working day'], color=C_WORK, lw=2, label='Working day')
ax.plot(util.index, util['Non-working day'], color=C_NON, lw=2, label='Non-working day')
ax.set_ylabel('% of 2,000 bikes in motion'); ax.set_xlabel('Hour of day'); ax.set_xticks(range(0,24,2)); ax.legend()
ax.set_title('Scenario (20-min trips, fictional 2,000 fleet): AVERAGE hour profile')
plt.tight_layout(); plt.savefig(mu.OUT_DIR/'08_utilisation_scenario.png'); plt.show()
# Three DIFFERENT statistics -- do not mix them up (corrected after review):
ref = mu.observed_demand_reference(h)
pct_in_motion = {
  'Mean of busiest hour-of-day (working days, 17:00, both years)': util.max().max(),
  'Observed 2012 p95 hour': ref['p95']*20/60/FICTIONAL_FLEET*100,
  'Observed maximum hour (2012)': ref['max']*20/60/FICTIONAL_FLEET*100,
}
for k, v in pct_in_motion.items(): print(f'{k:62s} {v:5.1f}% of 2,000 bikes in motion')
RESULTS['pct_in_motion_2000_20min'] = pct_in_motion""")
md("""**Clarification.** 8.8% is the *average* 17:00 working-day hour (mean 525 rentals). 11.0% is the 2012 *95th-percentile* hour (662 rentals = 221 bikes in motion). 16.3% is the single busiest hour (977).
They are different statistics; none of them says whether the *right stations* had bikes.""")

md("""### Fleet scenario driven by the ML forecast
The calculator accepts **predicted** hourly rentals from the trained model as input. Observed percentiles stay alongside as a historical reference.
Because the model under-predicts peak hours on the test period, forecast-driven estimates lean low.""")
code("""# Forecast-driven scenario on the held-out test window, using the model trained only on data up to 2012-06-30
pred_hgb = preds['HistGradientBoosting']
fc_rows = {
  'ML forecast: max predicted hour (2012-H2)': mu.fleet_from_forecast(pred_hgb),
  'ML forecast: p95 of predicted hours (2012-H2)': mu.fleet_from_forecast(pred_hgb, percentile=95),
  'Observed reference: 2012-H2 p95 (actual)': mu.fleet_required(np.percentile(test.cnt, 95)),
  'Observed reference: 2012-H2 max (actual)': mu.fleet_required(test.cnt.max()),
}
fc = pd.DataFrame(fc_rows).T[['hourly_demand','bikes_in_motion','bikes_positioned','fleet_required']].astype(float).round(1)
fc.to_csv(mu.OUT_DIR/'fleet_forecast_vs_observed.csv'); RESULTS['fleet_forecast_vs_observed'] = fc.to_dict(orient='index'); fc""")

md("""## Conclusions (human decision, Checkpoint 2)
- **H1 supported:** commute windows carry about twice the share of daily rentals on working days. This tells us *when* bikes must be in place, not *where*.
- **H3:** at the pre-specified 32 °C threshold the evidence is modest (see matched estimands above); higher thresholds look stronger but are exploratory.
  Rain does not explain July: July 2011 had the fewest rain/snow hours of the summer.
- **Model:** HistGradientBoosting is the primary hourly model (best MAE/RMSE/R²) but under-predicts peaks by ~51 rentals/hr on 2012-H2.
- **Fleet:** under the stated assumptions, scenario estimates are far below the fictional 2,000 bikes. These are **scenario estimates, not verified fleet requirements**.
- **Recommendation (Aditya):** investigate forecast-guided redistribution and fleet-health improvements *before* buying bikes, run as an **operational pilot to validate**.
  Redistribution is not proven to be the root cause without station-level availability and unmet-demand data; suggested timings (finish rebalancing before ~07:00 and ~16:30 on working days) are *trial scheduling ideas*.""")
code("""def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, pd.Timestamp): return str(o.date())
    return o
json.dump(_clean(RESULTS), open(mu.OUT_DIR/'results.json','w'), indent=2); print('results.json written')""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbf.write(nb, "analysis.ipynb")
print("analysis.ipynb executed and saved")
