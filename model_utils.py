"""
model_utils.py — shared logic for the Sigmoid bike-sharing case study.

Everything the notebook (analysis.ipynb) and the prototype (app.py) need lives here,
so both use exactly the same data cleaning, features, tests, model and fleet maths.

Sections
  1. Loading + data-quality checks
  2. Feature engineering
  3. Hypothesis tests (H1 commuter peaks, H3 summer heat)
  4. Forecasting (baseline vs ML, chronological split)
  5. Scenario-based fleet estimation (Little's law + assumptions)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

# Commute windows chosen from Exhibit A / hourly profile: 07:00-09:59 and 17:00-19:59
PEAK_HOURS = [7, 8, 9, 17, 18, 19]

# Model inputs. casual/registered are EXCLUDED (they sum to the target -> leakage).
# instant/dteday excluded (row id / raw date). No lag features (planning use-case).
FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"

# Chronological split (approved at Checkpoint 1)
TRAIN_END = "2012-06-30"
TEST_START = "2012-07-01"

WEATHER_LABELS = {1: "Clear / few clouds", 2: "Mist / cloudy",
                  3: "Light rain / snow", 4: "Heavy rain / snow"}


# ---------------------------------------------------------------- 1. loading + quality
def load_hour(path: Path | str = DATA_DIR / "hour.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["dteday"])


def load_day(path: Path | str = DATA_DIR / "day.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["dteday"])


def quality_report(h: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Return a table of data-quality checks with their observed results."""
    hours_per_day = h.groupby("dteday").size()
    daily_from_hourly = h.groupby("dteday")["cnt"].sum()
    checks = [
        ("hour.csv rows", len(h), "17,379 per Readme"),
        ("day.csv rows", len(d), "731 per Readme"),
        ("Null values (hour / day)", f"{h.isna().sum().sum()} / {d.isna().sum().sum()}", "0 expected"),
        ("Duplicate rows (hour)", int(h.duplicated().sum()), "0 expected"),
        ("Duplicate (date, hr) keys", int(h.duplicated(["dteday", "hr"]).sum()), "0 expected"),
        ("cnt == casual + registered (hour)", bool((h.cnt == h.casual + h.registered).all()), "identity"),
        ("Hourly sums reconcile with day.csv", bool((daily_from_hourly.values == d.cnt.values).all()), "identity"),
        ("Expected hours (731 x 24)", 731 * 24, ""),
        ("Missing hour-rows", 731 * 24 - len(h), "hours absent from the log"),
        ("Days with < 24 recorded hours", int((hours_per_day < 24).sum()), ""),
        ("Days with < 20 recorded hours", int((hours_per_day < 20).sum()), "flagged as incomplete"),
        ("weathersit = 4 rows", int((h.weathersit == 4).sum()), "too rare to model separately"),
        ("hum == 0 rows", int((h.hum == 0).sum()), "physically implausible -> likely missing"),
        ("windspeed == 0 rows", int((h.windspeed == 0).sum()), "possibly missing / calm"),
        ("Value ranges in [0,1] (temp, atemp, hum, windspeed)",
         bool(h[["temp", "atemp", "hum", "windspeed"]].apply(lambda s: s.between(0, 1).all()).all()), ""),
        ("hr range", f"{h.hr.min()}-{h.hr.max()}", "0-23"),
        ("cnt min / median / max", f"{h.cnt.min()} / {int(h.cnt.median())} / {h.cnt.max()}", ""),
    ]
    return pd.DataFrame(checks, columns=["check", "result", "note"])


def incomplete_days(h: pd.DataFrame, min_hours: int = 20) -> list[pd.Timestamp]:
    """Days with fewer than `min_hours` recorded hours (e.g. 2012-10-29/30, Hurricane Sandy)."""
    n = h.groupby("dteday").size()
    return list(n[n < min_hours].index)


# ---------------------------------------------------------------- 2. features
def add_features(h: pd.DataFrame) -> pd.DataFrame:
    """Readable units + planning-oriented features. Does not modify the input."""
    df = h.copy()
    # De-normalise using the Readme divisors
    df["temp_c"] = df["temp"] * 41
    df["atemp_c"] = df["atemp"] * 50
    df["hum_pct"] = df["hum"] * 100
    df["wind_kmh"] = df["windspeed"] * 67
    df["year"] = df["yr"].map({0: 2011, 1: 2012})
    df["day_type"] = np.where(df["workingday"] == 1, "Working day", "Non-working day")
    df["is_peak"] = df["hr"].isin(PEAK_HOURS).astype(int)
    df["daypart"] = pd.cut(df["hr"], bins=[-1, 5, 9, 15, 19, 23],
                           labels=["Night (0-5)", "AM peak (6-9)", "Midday (10-15)",
                                   "PM peak (16-19)", "Evening (20-23)"])
    df["weather_label"] = df["weathersit"].map(WEATHER_LABELS)
    bad = set(incomplete_days(h))
    df["incomplete_day"] = df["dteday"].isin(bad).astype(int)
    return df


# ---------------------------------------------------------------- 3. hypothesis tests
def _rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1 - 2 * u / (n1 * n2)


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 42):
    """Two-sided 95% bootstrap CI for the difference in MEANS (a - b)."""
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _hodges_lehmann(a: np.ndarray, b: np.ndarray) -> float:
    """Median of all pairwise differences a_i - b_j: the location shift estimated by Mann-Whitney."""
    return float(np.median(np.subtract.outer(a, b)))


def compare_groups(a: np.ndarray, b: np.ndarray, alternative: str, n_boot: int = 1000,
                   n_perm: int = 10000, seed: int = 42) -> dict:
    """
    Compare two independent groups with MATCHED estimands, so p-values and intervals agree:

      Rank-based:  Mann-Whitney U (one-sided)  <->  Hodges-Lehmann shift + 95% bootstrap CI
      Mean-based:  permutation test on mean difference (one-sided)  <->  95% bootstrap CI on mean diff

    `alternative` is 'greater' (a > b) or 'less' (a < b), as in scipy.
    A one-sided test at 5% corresponds to a two-sided 90% interval, so both 95% and 90% CIs are returned.
    """
    rng = np.random.default_rng(seed)
    u, p_mw = stats.mannwhitneyu(a, b, alternative=alternative)
    hl = _hodges_lehmann(a, b)
    hl_boot = [_hodges_lehmann(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    mean_diff = float(a.mean() - b.mean())
    mean_boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot * 5)]
    pooled, n = np.r_[a, b], len(a)
    perm = np.array([(lambda s: s[:n].mean() - s[n:].mean())(rng.permutation(pooled)) for _ in range(n_perm)])
    p_perm = float(np.mean(perm >= mean_diff) if alternative == "greater" else np.mean(perm <= mean_diff))
    r = 1 - 2 * u / (len(a) * len(b))  # rank-biserial; sign: positive means a < b under scipy's U
    return {
        "n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "mannwhitney_U": float(u), "mannwhitney_p_one_sided": float(p_mw),
        "rank_biserial": float(-r if alternative == "greater" else r),  # positive = effect in hypothesised direction
        "hodges_lehmann_shift": hl,
        "hodges_lehmann_ci95": tuple(float(x) for x in np.percentile(hl_boot, [2.5, 97.5])),
        "mean_diff": mean_diff, "permutation_p_one_sided": p_perm,
        "mean_diff_ci95": tuple(float(x) for x in np.percentile(mean_boot, [2.5, 97.5])),
        "mean_diff_ci90": tuple(float(x) for x in np.percentile(mean_boot, [5, 95])),
    }


def daily_peak_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per day: total rentals, rentals in commute windows, peak share, registered share at peak."""
    g = df.groupby("dteday")
    out = pd.DataFrame({
        "workingday": g["workingday"].first(),
        "season": g["season"].first(),
        "incomplete_day": g["incomplete_day"].first(),
        "total": g["cnt"].sum(),
        "peak": df[df.is_peak == 1].groupby("dteday")["cnt"].sum(),
        "peak_registered": df[df.is_peak == 1].groupby("dteday")["registered"].sum(),
    }).fillna(0)
    out["peak_share"] = out["peak"] / out["total"]
    out["registered_share_at_peak"] = out["peak_registered"] / out["peak"].replace(0, np.nan)
    return out


def test_h1(df: pd.DataFrame) -> dict:
    """
    H1 (commuter peaks).
    H0: the share of daily rentals in the 07-09 & 17-19 windows is the same on working and non-working days.
    H1: it is higher on working days.
    Unit = day (hourly rows within a day are not independent). Incomplete days excluded.
    Test = one-sided Mann-Whitney U; effect size = rank-biserial r; 95% bootstrap CI on mean difference.
    """
    t = daily_peak_table(df)
    t = t[t.incomplete_day == 0]
    w = t.loc[t.workingday == 1, "peak_share"].values
    nw = t.loc[t.workingday == 0, "peak_share"].values
    u, p = stats.mannwhitneyu(w, nw, alternative="greater")
    lo, hi = _bootstrap_diff(w, nw)
    cmp = compare_groups(w, nw, "greater", n_boot=300, n_perm=5000)
    by_season = []
    for s, name in {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}.items():
        ws = t[(t.season == s) & (t.workingday == 1)].peak_share
        ns = t[(t.season == s) & (t.workingday == 0)].peak_share
        us, ps = stats.mannwhitneyu(ws, ns, alternative="greater")
        by_season.append({"season": name, "working_mean": ws.mean(), "nonworking_mean": ns.mean(),
                          "n_working": len(ws), "n_nonworking": len(ns), "p_value": ps,
                          "rank_biserial": -_rank_biserial(us, len(ws), len(ns))})
    return {
        "n_working_days": len(w), "n_nonworking_days": len(nw),
        "mean_peak_share_working": float(w.mean()), "mean_peak_share_nonworking": float(nw.mean()),
        "median_peak_share_working": float(np.median(w)), "median_peak_share_nonworking": float(np.median(nw)),
        "U": float(u), "p_value": float(p),
        # sign flipped so a positive value means "working days higher"
        "rank_biserial": float(-_rank_biserial(u, len(w), len(nw))),
        "mean_diff_ci95": (lo, hi),
        "hodges_lehmann_shift": cmp["hodges_lehmann_shift"], "hodges_lehmann_ci95": cmp["hodges_lehmann_ci95"],
        "permutation_p_one_sided": cmp["permutation_p_one_sided"],
        "registered_share_at_peak_working": float(t.loc[t.workingday == 1, "registered_share_at_peak"].mean()),
        "registered_share_at_peak_nonworking": float(t.loc[t.workingday == 0, "registered_share_at_peak"].mean()),
        "by_season": pd.DataFrame(by_season),
        "daily_table": t,
    }


def test_h3(df: pd.DataFrame, day: pd.DataFrame, hot_c: float = 32.0) -> dict:
    """
    H3 (summer heat, replaces the 'monsoon' explanation for the July dip).
    H0: in Jun-Aug, on dry days, daily demand does not differ between hot and other days.
    H1: hot days (daily max temp >= hot_c) have LOWER demand than other summer days.
    Demand is made comparable across years/day types by dividing by the mean of the same
    (year x workingday) group in Jun-Aug (removes the 2011->2012 growth and commute effect).
    Unit = day. Wet days (day-level weathersit >= 3) and incomplete days excluded.
    Also reports Spearman correlation between daily max temperature and relative demand.
    """
    dmax = df.groupby("dteday")["temp_c"].max().rename("max_temp_c")
    dhum = df.groupby("dteday")["hum_pct"].mean().rename("mean_hum_pct")
    t = day.set_index("dteday").join([dmax, dhum])
    t = t[t.mnth.isin([6, 7, 8]) & (t.weathersit <= 2)]
    t = t[~t.index.isin(incomplete_days(df))]
    t["relative_demand"] = t["cnt"] / t.groupby(["yr", "workingday"])["cnt"].transform("mean")
    hot = t.loc[t.max_temp_c >= hot_c, "relative_demand"].values
    other = t.loc[t.max_temp_c < hot_c, "relative_demand"].values
    cmp = compare_groups(hot, other, "less")
    rho, p_rho = stats.spearmanr(t.max_temp_c, t.relative_demand)
    rho_h, p_rho_h = stats.spearmanr(t.max_temp_c, t.mean_hum_pct)
    # monthly context: is July wetter (monsoon idea) or hotter?
    ctx = (df.groupby(["year", "mnth"])
             .agg(mean_hourly_cnt=("cnt", "mean"), mean_temp_c=("temp_c", "mean"),
                  share_rain_snow_hours=("weathersit", lambda s: (s >= 3).mean()))
             .reset_index())
    ctx = ctx[ctx.mnth.isin([5, 6, 7, 8, 9])]
    return {
        "hot_threshold_c": hot_c, "n_hot_days": len(hot), "n_other_days": len(other),
        "mean_relative_demand_hot": cmp["mean_a"], "mean_relative_demand_other": cmp["mean_b"],
        "median_relative_demand_hot": cmp["median_a"], "median_relative_demand_other": cmp["median_b"],
        # rank-based estimand (what Mann-Whitney tests)
        "U": cmp["mannwhitney_U"], "p_value": cmp["mannwhitney_p_one_sided"], "rank_biserial": cmp["rank_biserial"],
        "hodges_lehmann_shift": cmp["hodges_lehmann_shift"], "hodges_lehmann_ci95": cmp["hodges_lehmann_ci95"],
        # mean-based estimand
        "mean_diff": cmp["mean_diff"], "permutation_p_one_sided": cmp["permutation_p_one_sided"],
        "mean_diff_ci95": cmp["mean_diff_ci95"], "mean_diff_ci90": cmp["mean_diff_ci90"],
        "spearman_rho_temp_vs_demand": float(rho), "spearman_p": float(p_rho),
        "spearman_rho_temp_vs_humidity": float(rho_h), "spearman_p_temp_humidity": float(p_rho_h),
        "monthly_context": ctx, "daily_table": t,
    }


# ---------------------------------------------------------------- 4. forecasting
def chrono_split(df: pd.DataFrame):
    train = df[df.dteday <= TRAIN_END]
    test = df[df.dteday >= TEST_START]
    return train, test


class SeasonalBaseline:
    """
    Transparent baseline: average 2011 rentals for the same (month, hour, workingday) cell,
    scaled by the year-on-year growth factor measured on Jan-Jun (2012 H1 mean / 2011 H1 mean).
    Uses training data only.
    """
    def fit(self, train: pd.DataFrame):
        y11 = train[train.yr == 0]
        self.table_ = y11.groupby(["mnth", "hr", "workingday"])["cnt"].mean()
        h1_11 = y11[y11.mnth <= 6].cnt.mean()
        h1_12 = train[(train.yr == 1) & (train.mnth <= 6)].cnt.mean()
        self.growth_ = h1_12 / h1_11
        self.fallback_ = y11.groupby(["hr", "workingday"])["cnt"].mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = list(zip(X.mnth, X.hr, X.workingday))
        base = np.array([self.table_.get(k, self.fallback_.get((k[1], k[2]))) for k in keys], dtype=float)
        scale = np.where(X.yr.values == 1, self.growth_, 1.0)
        return base * scale


def make_models(seed: int = 42) -> dict:
    """ML candidates. Both are trained on log1p(cnt) and predictions are back-transformed."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
    }


def fit_log_model(model, train: pd.DataFrame):
    model.fit(train[FEATURES], np.log1p(train[TARGET]))
    return model


def predict_log_model(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(model.predict(X[FEATURES])), 0, None)


def evaluate(y_true, y_pred, is_peak) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred, is_peak = np.asarray(y_true), np.asarray(y_pred), np.asarray(is_peak).astype(bool)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Peak-hour MAE": mean_absolute_error(y_true[is_peak], y_pred[is_peak]),
        "Peak-hour bias (pred-actual)": float((y_pred[is_peak] - y_true[is_peak]).mean()),
    }


# ---------------------------------------------------------------- 5. fleet scenarios
def fleet_required(hourly_demand: float, trip_minutes: float = 20, maintenance_share: float = 0.10,
                   spatial_buffer: float = 2.0) -> dict:
    """
    Scenario estimate of fleet size. ASSUMPTION-DRIVEN, not a measurement.

    Step 1  Little's law: bikes in motion = rentals per hour x average trip duration (hours)
    Step 2  Spatial buffer: bikes must already be parked at the right stations before demand
            arrives, so we need more parked bikes than are ever riding at once (factor >= 1).
    Step 3  Maintenance: only (1 - maintenance_share) of the fleet is usable.
    """
    in_motion = hourly_demand * trip_minutes / 60
    positioned = in_motion * spatial_buffer
    fleet = positioned / (1 - maintenance_share)
    return {"hourly_demand": hourly_demand, "trip_minutes": trip_minutes,
            "maintenance_share": maintenance_share, "spatial_buffer": spatial_buffer,
            "bikes_in_motion": in_motion, "bikes_positioned": positioned,
            "fleet_required": int(np.ceil(fleet))}


def h3_sensitivity(df: pd.DataFrame, day: pd.DataFrame, thresholds=(32, 34, 35, 36)) -> pd.DataFrame:
    """EXPLORATORY: H3 at other thresholds. 32 C was pre-specified; the others were chosen after
    seeing the scatter plot and must not be read as confirmatory evidence."""
    rows = []
    for t in thresholds:
        r = test_h3(df, day, hot_c=t)
        rows.append({"threshold_c": t, "status": "pre-specified" if t == 32 else "exploratory (post-hoc)",
                     "n_hot": r["n_hot_days"], "n_other": r["n_other_days"],
                     "HL_shift": r["hodges_lehmann_shift"], "MW_p_one_sided": r["p_value"],
                     "mean_diff": r["mean_diff"], "mean_diff_ci95": r["mean_diff_ci95"],
                     "perm_p_one_sided": r["permutation_p_one_sided"]})
    return pd.DataFrame(rows)


def fleet_from_forecast(hourly_predictions, trip_minutes: float = 20, maintenance_share: float = 0.10,
                        spatial_buffer: float = 2.0, percentile: float | None = None) -> dict:
    """
    Scenario fleet estimate driven by MODEL PREDICTIONS (predicted completed rentals).
    - percentile=None -> size for the single highest predicted hour in the supplied window.
    - percentile=95   -> size for the 95th percentile of predicted hours.
    Predicted completed rentals != true demand (suppressed demand is unobserved), and the
    current model under-predicts peak hours (~-51 rentals/hr on the 2012-H2 test), so this
    estimate leans LOW. It is a scenario, not a verified fleet requirement.
    """
    p = np.asarray(hourly_predictions, dtype=float)
    level = float(p.max() if percentile is None else np.percentile(p, percentile))
    out = fleet_required(level, trip_minutes, maintenance_share, spatial_buffer)
    out["demand_basis"] = "max predicted hour" if percentile is None else f"p{percentile:g} of predicted hours"
    out["n_hours_in_window"] = int(len(p))
    return out


def observed_demand_reference(df: pd.DataFrame, year: int = 2012) -> dict:
    """Historical reference levels of OBSERVED completed rentals per hour."""
    s = df[df["yr"] == (year - 2011)]["cnt"]
    return {"p90": float(s.quantile(.90)), "p95": float(s.quantile(.95)),
            "p99": float(s.quantile(.99)), "max": float(s.max()), "year": year}


def fleet_grid(hourly_demand: float, trips=(15, 20, 30), buffers=(1.5, 2.0, 3.0),
               maintenance_share: float = 0.10) -> pd.DataFrame:
    rows = [fleet_required(hourly_demand, t, maintenance_share, b) for t in trips for b in buffers]
    g = pd.DataFrame(rows)
    return g.pivot(index="trip_minutes", columns="spatial_buffer", values="fleet_required")


# ---------------------------------------------------------------- helpers for the app
def typical_weather(df: pd.DataFrame, mnth: int, hr: int) -> dict:
    """Median normalised weather for a month/hour, used when the user gives no weather."""
    s = df[(df.mnth == mnth) & (df.hr == hr)]
    return {c: float(s[c].median()) for c in ["temp", "atemp", "hum", "windspeed"]}
