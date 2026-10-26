"""
nl_assistant.py — natural-language layer for the operations-manager prototype.

    question -> (Gemini function calling | deterministic keyword router)
             -> validated Python tool (this file) -> real numbers from data / trained model
             -> grounded answer (+ automatic check that numbers in the answer exist in tool output)

Both modes call the SAME whitelisted tools, so the analysis is identical; only the language
understanding differs. Gemini never executes code and never sees the raw dataset — it only
chooses a tool + arguments and explains the returned JSON.

The API key is read from the GEMINI_API_KEY environment variable only. Nothing is stored.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

import model_utils as mu

# Supported stable models per Google's model list (checked 26 Oct 2026). Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

DATA_TYPE_NOTE = {
    "observed": "OBSERVED completed rentals (2011-12 history). Unserved demand is not in the data.",
    "predicted": "PREDICTED completed rentals from the trained model (not true demand; model under-predicts peak hours).",
    "scenario": "SCENARIO ESTIMATE built on stated assumptions. Not a verified fleet requirement.",
}


class ToolError(ValueError):
    """Raised when tool arguments fail validation; the message is shown to the user / Gemini."""


def _r(x, nd=1):
    return float(round(float(x), nd))


# =============================================================== analytical tools
class Analyst:
    """Holds data + model once; exposes whitelisted, validated analysis functions."""

    def __init__(self):
        self.h = mu.add_features(mu.load_hour())
        self.d = mu.load_day()
        bundle = joblib.load(mu.MODEL_DIR / "demand_model.joblib")
        self.model, self.model_name = bundle["model"], bundle["name"]
        res_path = mu.OUT_DIR / "results.json"
        self.results = json.loads(res_path.read_text()) if res_path.exists() else {}
        self.holidays = set(self.h.loc[self.h.holiday == 1, "dteday"].dt.date)
        # season label as used in the dataset, looked up by (month, day)
        s = self.h.drop_duplicates("dteday")
        self.season_lookup = {(r.dteday.month, r.dteday.day): r.season for r in s.itertuples()}
        # simple linear map temp -> feels-like temp (both normalised), fitted on the data
        self.atemp_coef = np.polyfit(self.h.temp, self.h.atemp, 1)

    # ---------------------------------------------------------- 1. hourly profile / peaks
    def hourly_profile(self, day_type: str = "both", year: int | None = None) -> dict:
        day_type = (day_type or "both").lower().replace("-", "_").replace(" ", "_")
        if day_type not in {"working", "non_working", "both"}:
            raise ToolError("day_type must be 'working', 'non_working' or 'both'.")
        df = self._year_filter(self.h, year)
        out = {"data_type": "observed", "note": DATA_TYPE_NOTE["observed"], "year": year or "2011-2012",
               "unit": "mean completed rentals per hour", "profiles": {}}
        for lab, wd in [("working", 1), ("non_working", 0)]:
            if day_type not in (lab, "both"):
                continue
            p = df[df.workingday == wd].groupby("hr").cnt.mean()
            out["profiles"][lab] = {
                "hourly_mean": {int(k): _r(v) for k, v in p.items()},
                "top_3_hours": [{"hour": int(k), "mean_rentals": _r(v)} for k, v in p.nlargest(3).items()],
                "quietest_hour": int(p.idxmin()),
            }
        return out

    # ---------------------------------------------------------- 2. working vs non-working
    def compare_day_types(self, year: int | None = None) -> dict:
        df = self._year_filter(self.h, year)
        daily = df.groupby(["dteday", "workingday"]).agg(total=("cnt", "sum")).reset_index()
        peak = df[df.is_peak == 1].groupby("dteday").cnt.sum()
        daily["peak_share"] = daily.dteday.map(peak).fillna(0) / daily.total
        g = daily.groupby("workingday")
        prof = df.groupby(["workingday", "hr"]).cnt.mean()
        reg = df[df.is_peak == 1].groupby("workingday")[["registered", "cnt"]].sum()
        out = {"data_type": "observed", "note": DATA_TYPE_NOTE["observed"], "year": year or "2011-2012",
               "commute_windows": "07:00-09:59 and 17:00-19:59"}
        for lab, wd in [("working_days", 1), ("non_working_days", 0)]:
            out[lab] = {
                "n_days": int(g.size()[wd]),
                "mean_daily_rentals": _r(g.total.mean()[wd], 0),
                "mean_share_of_rentals_in_commute_windows_pct": _r(g.peak_share.mean()[wd] * 100),
                "busiest_hour": int(prof[wd].idxmax()), "busiest_hour_mean_rentals": _r(prof[wd].max()),
                "registered_user_share_in_commute_windows_pct": _r(reg.loc[wd, "registered"] / reg.loc[wd, "cnt"] * 100),
            }
        h1 = self.results.get("H1")
        if h1 and year is None:
            out["hypothesis_test_H1"] = {
                "result": "supported",
                "test": "one-sided Mann-Whitney on daily commute-window share (incomplete days excluded)",
                "p_value": h1["p_value"], "rank_biserial_effect": _r(h1["rank_biserial"], 3),
                "hodges_lehmann_shift_pct_points": _r(h1.get("hodges_lehmann_shift", np.nan) * 100),
                "limitation": "Shows WHEN demand concentrates, not WHERE bikes travel (no station/trip data).",
            }
        return out

    # ---------------------------------------------------------- 3. weather
    def weather_effect(self) -> dict:
        df = self.h[self.h.weathersit <= 3].copy()
        raw = df.groupby("weathersit").cnt.mean()
        # controlled comparison: each hour vs the mean CLEAR-weather rentals of the same year, month, hour, day type
        keys = ["yr", "mnth", "hr", "workingday"]
        clear = df[df.weathersit == 1].groupby(keys).cnt.mean().rename("clear_mean")
        df = df.join(clear, on=keys)
        df = df[df.clear_mean > 0]
        ratio = (df.cnt / df.clear_mean).groupby(df.weathersit).mean()
        return {
            "data_type": "observed", "note": DATA_TYPE_NOTE["observed"],
            "raw_mean_rentals_per_hour": {mu.WEATHER_LABELS[int(k)]: _r(v) for k, v in raw.items()},
            "controlled_pct_of_clear_weather_demand": {mu.WEATHER_LABELS[int(k)]: _r(v * 100) for k, v in ratio.items()},
            "hours_observed": {mu.WEATHER_LABELS[int(k)]: int(v) for k, v in df.groupby("weathersit").size().items()},
            "method": "Controlled = rentals divided by the clear-weather mean for the same year, month, hour and day type.",
            "caveat": "Heavy rain/snow (category 4) has only 3 hours and is excluded.",
        }

    # ---------------------------------------------------------- 4. seasonal / monthly
    def seasonal_demand(self, year: int | None = None) -> dict:
        df = self._year_filter(self.h, year)
        m = df.groupby(["year", "mnth"]).agg(mean_rentals_per_hour=("cnt", "mean"), mean_temp_c=("temp_c", "mean"),
                                             rain_snow_hours_pct=("weathersit", lambda s: (s >= 3).mean() * 100))
        monthly = [{"year": int(y), "month": int(mo), **{k: _r(v) for k, v in row.items()}}
                   for (y, mo), row in m.iterrows()]
        out = {"data_type": "observed", "note": DATA_TYPE_NOTE["observed"], "monthly": monthly,
               "total_rentals_by_year": {str(k): int(v) for k, v in df.groupby("year").cnt.sum().items()},
               "growth_caveat": "2012 > 2011 may reflect more bikes/stations or adoption, not only preference."}
        h3 = self.results.get("H3")
        if h3:
            out["hypothesis_test_H3_summer_heat"] = {
                "pre_specified_threshold_c": h3["hot_threshold_c"],
                "rank_test": f"Mann-Whitney one-sided p={h3['p_value']:.4f}; median shift "
                             f"{h3.get('hodges_lehmann_shift', np.nan) * 100:.1f} pct points",
                "mean_test": f"permutation one-sided p={h3.get('permutation_p_one_sided', np.nan):.3f}; "
                             f"mean diff {h3.get('mean_diff', np.nan) * 100:.1f} pct points (95% CI includes 0)",
                "spearman_rho_temp_vs_demand": _r(h3["spearman_rho_temp_vs_demand"], 2),
                "interpretation": "Modest evidence that very hot summer days have lower demand; rain does not explain "
                                  "the July dip (July 2011 had the fewest rain/snow hours). Higher thresholds (34-36 C) "
                                  "look stronger but were chosen after seeing the data, so they are exploratory only.",
            }
        return out

    # ---------------------------------------------------------- 5. model prediction
    def predict_demand(self, date: str, hour: int | None = None, weather: int = 1, temp_c: float | None = None,
                       humidity_pct: float | None = None, windspeed_kmh: float | None = None) -> dict:
        d = self._parse_date(date)
        weather = int(weather or 1)
        if weather not in (1, 2, 3, 4):
            raise ToolError("weather must be 1 (clear), 2 (mist/cloudy), 3 (light rain/snow) or 4 (heavy).")
        if hour is not None and not (0 <= int(hour) <= 23):
            raise ToolError("hour must be between 0 and 23.")
        if temp_c is not None and not (-10 <= temp_c <= 45):
            raise ToolError("temp_c must be between -10 and 45.")
        X, caveats = self._features_for_day(d, weather, temp_c, humidity_pct, windspeed_kmh)
        preds = mu.predict_log_model(self.model, X)
        hourly = {int(hh): _r(p) for hh, p in zip(X.hr, preds)}
        bias = self.results.get("metrics", {}).get(self.model_name, {}).get("Peak-hour bias (pred-actual)")
        if bias is not None:
            caveats.append(f"On the Jul-Dec 2012 test the model under-predicted commute-peak hours by ~{abs(bias):.0f} "
                           f"rentals/hour on average.")
        caveats.append("Uses the weather you give (or the typical weather for that month and hour) as if it were "
                       "a perfect forecast.")
        out = {"data_type": "predicted", "note": DATA_TYPE_NOTE["predicted"], "model": self.model_name,
               "date": d.isoformat(), "weekday": d.strftime("%A"),
               "working_day": bool(X.workingday.iloc[0]), "weather": mu.WEATHER_LABELS[weather],
               "caveats": caveats}
        if hour is not None:
            out["hour"] = int(hour)
            out["predicted_rentals"] = hourly[int(hour)]
        else:
            out["predicted_hourly"] = hourly
            out["predicted_daily_total"] = _r(sum(hourly.values()), 0)
            out["predicted_peak_hour"] = max(hourly, key=hourly.get)
            out["predicted_peak_rentals"] = max(hourly.values())
        actual = self.h[self.h.dteday == pd.Timestamp(d)]
        if len(actual):
            out["actual_observed_for_comparison"] = (
                int(actual.loc[actual.hr == int(hour), "cnt"].sum()) if hour is not None else int(actual.cnt.sum()))
            out["caveats"].append("This date is inside the training data (model refit on all data), so the comparison "
                                  "is in-sample and optimistic.")
        return out

    # ---------------------------------------------------------- 6. fleet scenario
    def fleet_scenario(self, demand_basis: str = "observed_p95", date: str | None = None,
                       custom_hourly_demand: float | None = None, trip_minutes: float = 20,
                       maintenance_share: float = 0.10, spatial_buffer: float = 2.0,
                       reference_fleet: int = 2000) -> dict:
        valid = {"observed_p90", "observed_p95", "observed_p99", "observed_max", "forecast_peak", "custom"}
        if demand_basis not in valid:
            raise ToolError(f"demand_basis must be one of {sorted(valid)}.")
        if not (5 <= trip_minutes <= 120):
            raise ToolError("trip_minutes must be between 5 and 120.")
        if not (0 <= maintenance_share < 0.9):
            raise ToolError("maintenance_share must be between 0 and 0.9 (e.g. 0.10 for 10%).")
        if not (1 <= spatial_buffer <= 10):
            raise ToolError("spatial_buffer must be between 1 and 10.")
        ref = mu.observed_demand_reference(self.h, 2012)
        extra = {}
        if demand_basis.startswith("observed_"):
            level = ref[demand_basis.split("_")[1]]
            basis_text = f"observed 2012 {demand_basis.split('_')[1]} of hourly rentals"
        elif demand_basis == "forecast_peak":
            if not date:
                raise ToolError("demand_basis 'forecast_peak' needs a date (YYYY-MM-DD).")
            pred = self.predict_demand(date)
            fc = mu.fleet_from_forecast(list(pred["predicted_hourly"].values()), trip_minutes, maintenance_share,
                                        spatial_buffer)
            level = fc["hourly_demand"]
            basis_text = f"ML forecast: highest predicted hour on {pred['date']} ({pred['weekday']})"
            extra = {"forecast_peak_hour": pred["predicted_peak_hour"]}
        else:
            if custom_hourly_demand is None or not (1 <= custom_hourly_demand <= 10000):
                raise ToolError("custom requires custom_hourly_demand between 1 and 10000.")
            level, basis_text = float(custom_hourly_demand), "custom hourly demand supplied by user"
        r = mu.fleet_required(level, trip_minutes, maintenance_share, spatial_buffer)
        breakeven = reference_fleet * (1 - maintenance_share) / r["bikes_in_motion"]
        return {
            "data_type": "scenario", "note": DATA_TYPE_NOTE["scenario"],
            "demand_basis": basis_text, **extra,
            "hourly_demand_used": _r(level),
            "assumptions": {"trip_minutes": trip_minutes, "maintenance_share": maintenance_share,
                            "spatial_buffer": spatial_buffer},
            "formula": "fleet = demand/hr x trip_hours x spatial_buffer / (1 - maintenance_share)",
            "bikes_in_motion": _r(r["bikes_in_motion"]),
            "bikes_positioned": _r(r["bikes_positioned"]),
            "scenario_fleet_estimate": r["fleet_required"],
            "reference_fleet_fictional": reference_fleet,
            "pct_of_reference_fleet_in_motion": _r(r["bikes_in_motion"] / reference_fleet * 100),
            "spatial_buffer_at_which_reference_fleet_becomes_insufficient": _r(breakeven, 1),
            "observed_reference_2012": {k: _r(v) for k, v in ref.items() if k != "year"},
            "caveats": [
                "Trip duration, maintenance share and spatial buffer are assumptions (no trip, fleet or station data).",
                "Demand = completed rentals; riders who found no bike are not counted, so true demand may be higher.",
                "The 2,000-bike fleet is a fictional case figure, not Capital Bikeshare's real fleet.",
                "Says nothing about whether the RIGHT stations had bikes; that needs station-level availability data.",
            ],
        }

    # ---------------------------------------------------------- helpers
    def _year_filter(self, df, year):
        if year in (None, "", 0):
            return df
        if int(year) not in (2011, 2012):
            raise ToolError("year must be 2011 or 2012 (the data covers only these years).")
        return df[df.year == int(year)]

    @staticmethod
    def _parse_date(date: str) -> dt.date:
        try:
            return dt.date.fromisoformat(str(date)[:10])
        except ValueError:
            raise ToolError("date must be in YYYY-MM-DD format.")

    def _features_for_day(self, d: dt.date, weather: int, temp_c, hum_pct, wind_kmh):
        caveats = []
        if d.year in (2011, 2012):
            yr = d.year - 2011
            holiday = int(d in self.holidays)
        else:
            yr = 1
            holiday = 0
            caveats.append(f"{d.year} is outside 2011-2012: the model treats it at 2012 demand levels and does not "
                           f"know that year's holidays (extrapolation).")
        weekday = (d.weekday() + 1) % 7  # dataset: 0 = Sunday
        workingday = int(weekday not in (0, 6) and not holiday)
        season = self.season_lookup.get((d.month, d.day), self.season_lookup.get((d.month, 28)))
        rows = []
        for hr in range(24):
            w = mu.typical_weather(self.h, d.month, hr)
            if temp_c is not None:
                w["temp"] = temp_c / 41
                w["atemp"] = float(np.polyval(self.atemp_coef, w["temp"]))
            if hum_pct is not None:
                w["hum"] = hum_pct / 100
            if wind_kmh is not None:
                w["windspeed"] = wind_kmh / 67
            rows.append({"season": season, "yr": yr, "mnth": d.month, "hr": hr, "holiday": holiday,
                         "weekday": weekday, "workingday": workingday, "weathersit": weather, **w})
        if temp_c is None:
            caveats.append("No temperature given: used the median historical temperature for that month and hour.")
        return pd.DataFrame(rows)[mu.FEATURES], caveats


# =============================================================== tool registry (whitelist)
TOOL_SCHEMAS = {
    "hourly_profile": {
        "description": "Average OBSERVED rentals for each hour of the day, with the top commute/peak hours. "
                       "Use for questions about peak hours, rush hour, busiest or quietest times.",
        "parameters": {"type": "object", "properties": {
            "day_type": {"type": "string", "enum": ["working", "non_working", "both"]},
            "year": {"type": "integer", "description": "2011 or 2012; omit for both years"}}},
    },
    "compare_day_types": {
        "description": "Compare working days vs weekends/holidays: daily rentals, share of rentals in commute "
                       "windows, busiest hour, registered-user share, and the H1 hypothesis test result.",
        "parameters": {"type": "object", "properties": {"year": {"type": "integer"}}},
    },
    "weather_effect": {
        "description": "How weather (clear, mist/cloudy, light rain/snow) changes OBSERVED rentals, raw and "
                       "controlled for year, month, hour and day type.",
        "parameters": {"type": "object", "properties": {}},
    },
    "seasonal_demand": {
        "description": "Monthly/seasonal OBSERVED demand with temperature and rain share by month, year totals, "
                       "and the H3 summer-heat hypothesis result (July dip).",
        "parameters": {"type": "object", "properties": {"year": {"type": "integer"}}},
    },
    "predict_demand": {
        "description": "PREDICT completed rentals with the trained model for a date (and optionally one hour), "
                       "under given weather. Without an hour it returns all 24 hours and the daily total.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "hour": {"type": "integer", "description": "0-23; omit for the whole day"},
            "weather": {"type": "integer", "description": "1 clear, 2 mist/cloudy, 3 light rain/snow, 4 heavy"},
            "temp_c": {"type": "number"}, "humidity_pct": {"type": "number"}, "windspeed_kmh": {"type": "number"}},
            "required": ["date"]},
    },
    "fleet_scenario": {
        "description": "SCENARIO estimate of bikes needed (Little's law + spatial buffer + maintenance). "
                       "demand_basis: observed_p90/p95/p99/max (history), forecast_peak (needs date; uses the ML "
                       "model's highest predicted hour), or custom (needs custom_hourly_demand).",
        "parameters": {"type": "object", "properties": {
            "demand_basis": {"type": "string", "enum": ["observed_p90", "observed_p95", "observed_p99",
                                                         "observed_max", "forecast_peak", "custom"]},
            "date": {"type": "string", "description": "YYYY-MM-DD, for forecast_peak"},
            "custom_hourly_demand": {"type": "number"},
            "trip_minutes": {"type": "number", "description": "average trip length, default 20"},
            "maintenance_share": {"type": "number", "description": "fraction unusable, default 0.10"},
            "spatial_buffer": {"type": "number", "description": "parked-vs-riding factor, default 2.0"},
            "reference_fleet": {"type": "integer", "description": "fleet to compare against, default 2000 (fictional)"}}},
    },
}

_INT_ARGS = {"year", "hour", "weather", "reference_fleet"}
_FLOAT_ARGS = {"temp_c", "humidity_pct", "windspeed_kmh", "custom_hourly_demand", "trip_minutes",
               "maintenance_share", "spatial_buffer"}


def call_tool(analyst: Analyst, name: str, args: dict | None) -> dict:
    """Validate the tool name and argument names/types, then run it. Never executes arbitrary code."""
    if name not in TOOL_SCHEMAS:
        raise ToolError(f"Unknown tool '{name}'.")
    allowed = set(TOOL_SCHEMAS[name]["parameters"].get("properties", {}))
    clean = {}
    for k, v in (args or {}).items():
        if k not in allowed:
            raise ToolError(f"Unexpected argument '{k}' for {name}.")
        if v is None:
            continue
        try:
            clean[k] = int(v) if k in _INT_ARGS else float(v) if k in _FLOAT_ARGS else v
        except (TypeError, ValueError):
            raise ToolError(f"Argument '{k}' has an invalid value: {v!r}.")
    return getattr(analyst, name)(**clean)


# =============================================================== grounding check
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def _numbers_in(obj) -> list[float]:
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in _numbers_in(k) + _numbers_in(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _numbers_in(v)]
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(m.replace(",", "")) for m in _NUM.findall(obj) if m.replace(",", "").replace("-", "").replace(".", "")]
    return []


def grounding_check(answer: str, tool_results: list[dict]) -> dict:
    """
    Every number >= 10 in the answer (ignoring years/dates) must match a number in the tool results,
    allowing rounding (+-1% or +-1) and percent<->fraction conversion. Returns unmatched numbers.
    """
    source = _numbers_in(tool_results)
    source += [x * 100 for x in source if abs(x) <= 1] + [x / 100 for x in source]
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", answer)
    found = [float(m.replace(",", "")) for m in _NUM.findall(text) if m.replace(",", "").strip("-.")]
    unmatched = []
    for x in found:
        if abs(x) < 10 or 2000 <= x <= 2035 and float(x).is_integer() and x != 2000:
            continue
        if not any(abs(x - s) <= max(1.0, 0.01 * abs(s)) for s in source):
            unmatched.append(x)
    return {"numbers_checked": len(found), "unmatched": unmatched, "grounded": not unmatched}


# =============================================================== Gemini mode
SYSTEM_PROMPT = """You are an analytics assistant for the operations manager of a city bike-sharing company.
You answer ONLY by calling the provided tools, which run real analysis on 2011-2012 hourly rental data
and a trained demand model. Rules:
1. Call a tool for every factual or numerical claim. Never invent or estimate numbers yourself.
   Every number in your answer must come from a tool result.
2. Always say which kind of number it is: OBSERVED completed rentals, PREDICTED completed rentals,
   or SCENARIO estimate. Never present a scenario fleet estimate as a verified requirement.
3. Completed rentals do not include riders who could not find a bike. The data has no station-level,
   trip origin/destination, fleet-size or rebalancing information, so do not claim where bikes go or that
   redistribution is proven to be the cause. Redistribution ideas are pilots to validate.
4. Relay the most important caveat from the tool result in one short sentence.
5. If the question needs data the tools do not have, say so plainly and suggest what data would answer it.
6. Be concise: 3-6 sentences or a short list, in plain business language. Use today's assumptions only
   if the user gives them; otherwise use tool defaults and state them.
Dates: the data covers 2011-01-01 to 2012-12-31. If the user gives a weekday or month without a year,
ask nothing — pick a matching date in 2012 and state which date you used."""


class GeminiUnavailable(RuntimeError):
    pass


def _gemini_declarations():
    from google.genai import types
    return [types.FunctionDeclaration(name=n, description=s["description"], parameters_json_schema=s["parameters"])
            for n, s in TOOL_SCHEMAS.items()]


def ask_gemini(analyst: Analyst, question: str, api_key: str | None = None, model: str | None = None,
               history: list[tuple[str, str]] | None = None, client=None, max_rounds: int = 4) -> dict:
    """
    Gemini function-calling loop (manual, so every call is validated and logged):
      user question -> model returns function_call(s) -> call_tool -> function_response -> ... -> final text.
    `client` can be injected for testing. Raises GeminiUnavailable if no key / SDK / network.
    """
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as e:
        raise GeminiUnavailable("google-genai is not installed") from e
    if client is None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiUnavailable("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)
    models = [model] if model else ([os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL")
                                    else DEFAULT_GEMINI_MODELS)
    tools = [types.Tool(function_declarations=_gemini_declarations())]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, tools=tools, temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    cfg_final = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, tools=tools, temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")))

    contents = []
    for role, text in (history or [])[-6:]:  # short conversational memory
        contents.append(types.Content(role="user" if role == "user" else "model",
                                      parts=[types.Part.from_text(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    last_err = None
    for model_name in models:
        try:
            calls_log, convo = [], list(contents)
            for round_ in range(max_rounds + 1):
                resp = client.models.generate_content(model=model_name, contents=convo,
                                                      config=cfg if round_ < max_rounds else cfg_final)
                fcalls = resp.function_calls or []
                if not fcalls:
                    answer = (resp.text or "").strip()
                    results = [c["result"] for c in calls_log if "result" in c]
                    return {"mode": "gemini", "model": model_name, "answer": answer, "tool_calls": calls_log,
                            "grounding": grounding_check(answer, results) if results else
                            {"grounded": False, "unmatched": [], "numbers_checked": 0,
                             "note": "No tool was called; treat any figures with caution."}}
                convo.append(resp.candidates[0].content)
                parts = []
                for fc in fcalls:
                    entry = {"tool": fc.name, "args": dict(fc.args or {})}
                    try:
                        entry["result"] = call_tool(analyst, fc.name, entry["args"])
                        payload = {"result": entry["result"]}
                    except ToolError as e:
                        entry["error"] = str(e)
                        payload = {"error": str(e)}
                    calls_log.append(entry)
                    parts.append(types.Part.from_function_response(name=fc.name, response=_jsonable(payload)))
                convo.append(types.Content(role="user", parts=parts))
        except errors.ClientError as e:  # e.g. model not found / not permitted -> try next model
            last_err = e
            if getattr(e, "code", None) in (400, 401, 403) and "API key" in str(e):
                raise GeminiUnavailable(f"Gemini rejected the API key: {e}") from e
            continue
        except errors.APIError as e:
            last_err = e
            continue
        except Exception as e:  # network errors etc.
            raise GeminiUnavailable(f"{type(e).__name__}: {e}") from e
    raise GeminiUnavailable(f"All Gemini models failed: {last_err}")


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


# =============================================================== deterministic fallback
_MONTHS = {m.lower(): i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                               "August", "September", "October", "November", "December"], 1)}
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _extract(q: str) -> dict:
    ql = q.lower()
    out = {}
    m = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", ql)
    if m:
        out["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    else:
        day = next((d for d in _DAYS if d in ql), None)
        mon = next((v for k, v in _MONTHS.items() if re.search(rf"\b{k}\b", ql)), None)
        if day or mon or "tomorrow" in ql or "next" in ql:
            base = dt.date(2012, mon or 7, 1)
            if day:  # first matching weekday in that month of 2012
                while base.weekday() != _DAYS.index(day):
                    base += dt.timedelta(days=1)
            out["date"] = base.isoformat()
            out["date_assumed"] = True
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", ql) or re.search(r"\b(?:at|hour)\s*(\d{1,2})(?::00)?\b", ql) \
        or re.search(r"\b(\d{1,2}):00\b", ql)
    if m:
        hr = int(m.group(1))
        if len(m.groups()) > 1 and m.group(2) == "pm" and hr < 12:
            hr += 12
        if len(m.groups()) > 1 and m.group(2) == "am" and hr == 12:
            hr = 0
        if 0 <= hr <= 23:
            out["hour"] = hr
    if re.search(r"heavy rain|storm|blizzard", ql):
        out["weather"] = 4
    elif re.search(r"rain|snow|wet|drizzle", ql):
        out["weather"] = 3
    elif re.search(r"mist|cloud|fog|overcast", ql):
        out["weather"] = 2
    m = re.search(r"(-?\d{1,2}(?:\.\d)?)\s*(?:°|deg|degrees)?\s*c\b", ql)
    if m:
        out["temp_c"] = float(m.group(1))
    m = re.search(r"\b(2011|2012)\b", ql)
    if m and "date" not in out:
        out["year"] = int(m.group(1))
    m = re.search(r"(\d{1,3})\s*(?:-|\s)?min", ql)
    if m:
        out["trip_minutes"] = float(m.group(1))
    m = re.search(r"buffer (?:of )?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*x\s*buffer", ql)
    if m:
        out["spatial_buffer"] = float(m.group(1) or m.group(2))
    m = re.search(r"(\d{1,2})\s*%\s*(?:of bikes )?(?:in )?(?:maintenance|broken|repair)", ql)
    if m:
        out["maintenance_share"] = float(m.group(1)) / 100
    m = re.search(r"\bp(90|95|99)\b|(90|95|99)(?:th)? percentile", ql)
    if m:
        out["percentile"] = m.group(1) or m.group(2)
    return out


UNSUPPORTED_PATTERN = re.compile(
    r"station|where (do|are|should)|which (area|location|neighbou?rhood|dock)|origin|destination|route|"
    r"trip (length|duration)|how long (are|do) trips|empty dock|run(s)? out|stock ?out|failed rental|"
    r"unmet demand|lost demand|driver|rebalanc|redistribut|real fleet|actual fleet|how many bikes do we (have|own)")

UNSUPPORTED_TEXT = (
    "**This dataset cannot answer that.** It contains only citywide hourly totals of *completed* rentals for "
    "2011-2012. It has no station-level stock, trip origins/destinations, trip durations, failed rental attempts, "
    "fleet size or driver/rebalancing records, so I won't guess.\n\nWhat would answer it: station-level dock "
    "status snapshots (bikes/docks available every few minutes), trip-level records with start/end station and "
    "duration, app logs of failed or abandoned rentals, and rebalancing logs. What I *can* tell you from this data: "
    "**when** demand peaks (ask about commute peaks), predicted demand for a date/hour, and scenario fleet sizes "
    "under stated assumptions.")


def route_deterministic(q: str) -> tuple[str, dict]:
    """Keyword intent router. Order matters: most specific intents first."""
    ql, e = q.lower(), _extract(q)
    if UNSUPPORTED_PATTERN.search(ql):
        return "__unsupported__", {}
    if re.search(r"fleet|how many bikes|bikes (do we )?need|bikes required|buy (more )?bikes|enough bikes|capacity", ql):
        args = {k: e[k] for k in ("trip_minutes", "spatial_buffer", "maintenance_share") if k in e}
        if "date" in e and re.search(r"forecast|predict|on |for ", ql):
            args.update(demand_basis="forecast_peak", date=e["date"])
        elif re.search(r"max|worst|busiest ever|record", ql):
            args["demand_basis"] = "observed_max"
        else:
            args["demand_basis"] = f"observed_p{e.get('percentile', '95')}"
        return "fleet_scenario", args
    if "date" in e and re.search(r"predict|forecast|expect|estimate|how many (rentals|rides|trips)|will|demand on|demand at", ql) \
            or re.search(r"predict|forecast", ql):
        if "date" not in e:
            e["date"], e["date_assumed"] = "2012-07-02", True
        return "predict_demand", {k: e[k] for k in ("date", "hour", "weather", "temp_c") if k in e}
    if re.search(r"weather|rain|snow|mist|cloud|storm", ql):
        return "weather_effect", {}
    if re.search(r"season|month|summer|winter|spring|autumn|fall\b|july|growth|year over year|trend|heat|hot", ql):
        return "seasonal_demand", {k: e[k] for k in ("year",) if k in e}
    yr = {k: e[k] for k in ("year",) if k in e}
    if re.search(r"compare|\bvs\.?\b|versus|differ|difference", ql):
        return "compare_day_types", yr
    if re.search(r"peak|rush|commut|busiest|busy|quiet|hour|time of day|when", ql):
        dtp = "non_working" if re.search(r"weekend|holiday|non-working", ql) else \
              "working" if re.search(r"work|commut|office", ql) else "both"
        return "hourly_profile", {"day_type": dtp, **yr}
    if re.search(r"weekend|working day|workday|weekday|holiday|non-working|nonworking", ql):
        return "compare_day_types", yr
    return "", {}


def _fmt_hours(items):
    return ", ".join(f"{d['hour']:02d}:00 (~{d['mean_rentals']:.0f}/hr)" for d in items)


def template_answer(tool: str, res: dict, assumed_date: bool = False) -> str:
    """Plain-language answer built ONLY from the tool result."""
    if tool == "hourly_profile":
        parts = []
        for lab, p in res["profiles"].items():
            parts.append(f"**{lab.replace('_', '-').capitalize()} days** — busiest hours: {_fmt_hours(p['top_3_hours'])}; "
                         f"quietest hour {p['quietest_hour']:02d}:00.")
        return ("Observed completed rentals, " + str(res["year"]) + ":\n\n- " + "\n- ".join(parts) +
                "\n\n_These are averages of completed rentals; they show when demand concentrates, not where._")
    if tool == "compare_day_types":
        w, n = res["working_days"], res["non_working_days"]
        s = (f"Observed ({res['year']}): working days average **{w['mean_daily_rentals']:.0f}** rentals/day vs "
             f"**{n['mean_daily_rentals']:.0f}** on non-working days. Commute windows ({res['commute_windows']}) carry "
             f"**{w['mean_share_of_rentals_in_commute_windows_pct']:.1f}%** of working-day rentals vs "
             f"**{n['mean_share_of_rentals_in_commute_windows_pct']:.1f}%** on non-working days. Busiest hour: "
             f"{w['busiest_hour']:02d}:00 (~{w['busiest_hour_mean_rentals']:.0f}/hr) on working days vs "
             f"{n['busiest_hour']:02d}:00 (~{n['busiest_hour_mean_rentals']:.0f}/hr) otherwise. Registered users make up "
             f"{w['registered_user_share_in_commute_windows_pct']:.1f}% of working-day commute-window rentals.")
        if "hypothesis_test_H1" in res:
            t = res["hypothesis_test_H1"]
            s += (f"\n\nH1 test: {t['result']} (one-sided Mann-Whitney p = {t['p_value']:.1e}, rank-biserial "
                  f"{t['rank_biserial_effect']}). _{t['limitation']}_")
        return s
    if tool == "weather_effect":
        c = res["controlled_pct_of_clear_weather_demand"]
        return ("Observed: compared with clear weather at the same year, month, hour and day type, demand is about "
                + "; ".join(f"**{v:.1f}%** in {k.lower()}" for k, v in c.items()) +
                ". Raw averages per hour: " + "; ".join(f"{k}: {v:.1f}" for k, v in res["raw_mean_rentals_per_hour"].items())
                + f". _{res['caveat']}_")
    if tool == "seasonal_demand":
        m = pd.DataFrame(res["monthly"])
        top = m.loc[m.mean_rentals_per_hour.idxmax()]
        low = m.loc[m.mean_rentals_per_hour.idxmin()]
        s = (f"Observed: highest month {int(top.month)}/{int(top.year)} (~{top.mean_rentals_per_hour:.0f} rentals/hr, "
             f"{top.mean_temp_c:.1f} °C); lowest {int(low.month)}/{int(low.year)} (~{low.mean_rentals_per_hour:.0f}/hr, "
             f"{low.mean_temp_c:.1f} °C). Totals: " +
             ", ".join(f"{k}: {v:,}" for k, v in res["total_rentals_by_year"].items()) + f". _{res['growth_caveat']}_")
        if "hypothesis_test_H3_summer_heat" in res:
            t = res["hypothesis_test_H3_summer_heat"]
            s += f"\n\nSummer heat (H3, {t['pre_specified_threshold_c']:.0f} °C pre-specified): {t['interpretation']}"
        return s
    if tool == "predict_demand":
        lead = "Predicted completed rentals" + (" (date assumed: " + res["date"] + ")" if assumed_date else "")
        if "hour" in res:
            s = (f"{lead} for {res['weekday']} {res['date']} at {res['hour']:02d}:00, {res['weather'].lower()}: "
                 f"**~{res['predicted_rentals']:.0f} rentals** ({'working' if res['working_day'] else 'non-working'} day, "
                 f"model: {res['model']}).")
        else:
            s = (f"{lead} for {res['weekday']} {res['date']}, {res['weather'].lower()}: **~{res['predicted_daily_total']:.0f}"
                 f" rentals** for the day; peak at {res['predicted_peak_hour']:02d}:00 (~{res['predicted_peak_rentals']:.0f}/hr).")
        cav = res["caveats"][:2]
        if "actual_observed_for_comparison" in res:
            s += f" Observed that {'hour' if 'hour' in res else 'day'}: {res['actual_observed_for_comparison']}."
            cav = [res["caveats"][-1]] + cav[:1]  # in-sample warning first
        return s + "\n\n_" + " ".join(cav) + "_"
    if tool == "fleet_scenario":
        a = res["assumptions"]
        return (f"**Scenario estimate (not a verified requirement):** basis = {res['demand_basis']} "
                f"(**{res['hourly_demand_used']:.0f} rentals/hr**). With {a['trip_minutes']:.0f}-min trips about "
                f"**{res['bikes_in_motion']:.0f} bikes are in motion**; with a {a['spatial_buffer']}x spatial buffer and "
                f"{a['maintenance_share'] * 100:.0f}% in maintenance the scenario needs **~{res['scenario_fleet_estimate']} bikes**. "
                f"That is {res['pct_of_reference_fleet_in_motion']:.1f}% of the fictional {res['reference_fleet_fictional']:,}-bike "
                f"fleet in motion; that fleet would only be too small if the spatial buffer had to exceed "
                f"~{res['spatial_buffer_at_which_reference_fleet_becomes_insufficient']:.1f}x.\n\n_{res['caveats'][3]}_")
    return ""


HELP_TEXT = ("I couldn't match that to an analysis. Try asking about **peak/commute hours**, **working days vs weekends**, "
             "**weather**, **seasons/months**, a **prediction** (e.g. *'predict demand on 2012-07-10 at 8am in rain'*) "
             "or **fleet size** (e.g. *'how many bikes do we need with 30 min trips?'*).")


def ask_deterministic(analyst: Analyst, question: str) -> dict:
    tool, args = route_deterministic(question)
    if tool == "__unsupported__":
        return {"mode": "deterministic", "answer": UNSUPPORTED_TEXT, "tool_calls": []}
    if not tool:
        return {"mode": "deterministic", "answer": HELP_TEXT, "tool_calls": []}
    assumed = bool(_extract(question).get("date_assumed"))
    try:
        res = call_tool(analyst, tool, args)
    except ToolError as e:
        return {"mode": "deterministic", "answer": f"Input problem: {e}", "tool_calls": [{"tool": tool, "args": args,
                                                                                       "error": str(e)}]}
    return {"mode": "deterministic", "answer": template_answer(tool, res, assumed),
            "tool_calls": [{"tool": tool, "args": args, "result": res}]}


def ask(analyst: Analyst, question: str, prefer_gemini: bool = True, model: str | None = None,
        history=None) -> dict:
    """Entry point: Gemini if available and preferred, otherwise (or on failure) the deterministic router."""
    if prefer_gemini:
        try:
            return ask_gemini(analyst, question, model=model, history=history)
        except GeminiUnavailable as e:
            out = ask_deterministic(analyst, question)
            out["fallback_reason"] = str(e)
            return out
    return ask_deterministic(analyst, question)
