"""
Tests for the NL prototype. Run from the project folder:  python tests/test_prototype.py
(no pytest needed). Live Gemini is NOT called here; the Gemini loop is tested with a fake client.
"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import model_utils as mu  # noqa: E402
import nl_assistant as nla  # noqa: E402

A = nla.Analyst()
PASSED = []


def check(name, cond, detail=""):
    assert cond, f"FAIL: {name} {detail}"
    PASSED.append(name)
    print(f"PASS  {name}")


# ---------------------------------------------------------------- tools return real numbers
h = A.h
r = A.hourly_profile("working")
check("hourly_profile matches pandas", abs(r["profiles"]["working"]["hourly_mean"][17]
                                         - h[(h.workingday == 1) & (h.hr == 17)].cnt.mean()) < 0.1)
check("working-day top hour is 17:00", r["profiles"]["working"]["top_3_hours"][0]["hour"] == 17)

c = A.compare_day_types()
check("compare_day_types commute share ~51.7% vs ~26%",
      50 < c["working_days"]["mean_share_of_rentals_in_commute_windows_pct"] < 53
      and 24 < c["non_working_days"]["mean_share_of_rentals_in_commute_windows_pct"] < 28)
check("H1 summary present", c["hypothesis_test_H1"]["result"] == "supported")

w = A.weather_effect()
pct = w["controlled_pct_of_clear_weather_demand"]
check("rain < mist < clear (controlled)", pct["Light rain / snow"] < pct["Mist / cloudy"] < 101)

s = A.seasonal_demand()
check("year totals match data", s["total_rentals_by_year"] == {"2011": 1243103, "2012": 2049576})

# ---------------------------------------------------------------- prediction reproduces the saved model
p = A.predict_demand("2012-07-10", hour=8, weather=1)
X, _ = A._features_for_day(pd.Timestamp("2012-07-10").date(), 1, None, None, None)
direct = float(np.expm1(A.model.predict(X[mu.FEATURES]))[8])
check("predict_demand == direct model call", abs(p["predicted_rentals"] - round(direct, 1)) < 0.11)
check("2012-07-10 recognised as working Tuesday", p["working_day"] and p["weekday"] == "Tuesday")
rainy = A.predict_demand("2012-07-10", hour=8, weather=3)["predicted_rentals"]
check("rain lowers prediction", rainy < p["predicted_rentals"])
sat = A.predict_demand("2012-07-14", hour=8)["predicted_rentals"]
check("Saturday 8am < Tuesday 8am", sat < p["predicted_rentals"])
hol = A.predict_demand("2012-07-04", hour=8)
check("4 July 2012 flagged as non-working (holiday)", hol["working_day"] is False)
fut = A.predict_demand("2026-11-02")
check("out-of-range year warns extrapolation", any("extrapolation" in x for x in fut["caveats"]))

# ---------------------------------------------------------------- fleet calculator
f = A.fleet_scenario()
check("fleet default = notebook value (491)", f["scenario_fleet_estimate"] == 491, f["scenario_fleet_estimate"])
check("fleet labelled scenario", f["data_type"] == "scenario" and "not a verified" in f["note"].lower())
check("p95 in motion = 11.0% of 2,000", abs(f["pct_of_reference_fleet_in_motion"] - 11.0) < 0.1)
ff = A.fleet_scenario("forecast_peak", date="2012-09-14")
check("fleet accepts ML forecast as input", "ML forecast" in ff["demand_basis"] and ff["hourly_demand_used"] > 0)
check("observed reference kept with forecast", ff["observed_reference_2012"]["p95"] > 600)
manual = mu.fleet_required(ff["hourly_demand_used"], 20, 0.10, 2.0)["fleet_required"]
check("forecast fleet consistent with formula", abs(manual - ff["scenario_fleet_estimate"]) <= 1)

# ---------------------------------------------------------------- validation rejects bad input
for bad in [("predict_demand", {"date": "10/07/2012"}), ("predict_demand", {"date": "2012-07-10", "hour": 25}),
            ("fleet_scenario", {"trip_minutes": 500}), ("fleet_scenario", {"demand_basis": "forecast_peak"}),
            ("hourly_profile", {"year": 2015}), ("drop_table", {}), ("weather_effect", {"sql": "x"})]:
    try:
        nla.call_tool(A, *bad)
        check(f"rejects {bad}", False)
    except nla.ToolError:
        check(f"rejects {bad[0]} {bad[1]}", True)

# ---------------------------------------------------------------- deterministic router
cases = {
    "When are the commute peaks on working days?": "hourly_profile",
    "what time is rush hour": "hourly_profile",
    "How do weekends compare with working days?": "compare_day_types",
    "How much does rain reduce demand?": "weather_effect",
    "Why is there a dip in July?": "seasonal_demand",
    "Predict demand on 2012-07-10 at 8am if it rains": "predict_demand",
    "how many rentals should we expect next Monday in August at 5pm": "predict_demand",
    "How many bikes do we need with 30 min trips and a 3x buffer?": "fleet_scenario",
    "Should we buy more bikes?": "fleet_scenario",
    "Using the forecast for 2012-09-14, how many bikes would we need?": "fleet_scenario",
}
for q, tool in cases.items():
    got, args = nla.route_deterministic(q)
    check(f"route '{q[:45]}' -> {tool}", got == tool, f"got {got} {args}")
_, a = nla.route_deterministic("Predict demand on 2012-07-10 at 8am if it rains")
check("extracts date/hour/weather", a == {"date": "2012-07-10", "hour": 8, "weather": 3}, a)
_, a = nla.route_deterministic("How many bikes do we need with 30 min trips and a 3x buffer?")
check("extracts trip/buffer", a.get("trip_minutes") == 30 and a.get("spatial_buffer") == 3, a)
_, a = nla.route_deterministic("Using the forecast for 2012-09-14, how many bikes would we need?")
check("forecast fleet routed with date", a.get("demand_basis") == "forecast_peak" and a.get("date") == "2012-09-14", a)

for q in cases:
    out = nla.ask_deterministic(A, q)
    res = [c["result"] for c in out["tool_calls"] if "result" in c]
    g = nla.grounding_check(out["answer"], res)
    check(f"template answer grounded: '{q[:40]}'", g["grounded"], g)
for q in ["Which stations run out of bikes at 8am?", "Where should drivers move bikes in the morning?",
          "How long are trips on average?"]:
    check(f"unsupported refused honestly: '{q[:35]}'", "cannot answer" in nla.ask_deterministic(A, q)["answer"])
check("unknown question -> help text", "couldn't match" in nla.ask_deterministic(A, "tell me a joke")["answer"])

# ---------------------------------------------------------------- grounding check catches invented numbers
g = nla.grounding_check("Demand will be 1234 rentals at 8am.", [A.fleet_scenario()])
check("grounding flags invented number", not g["grounded"] and 1234.0 in g["unmatched"])

# ---------------------------------------------------------------- Gemini loop with a fake client
from google.genai import types  # noqa: E402


class FakeModels:
    """Turn 1: request predict_demand. Turn 2: answer using the returned number."""
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        last = contents[-1]
        fr = [pt.function_response for pt in last.parts if getattr(pt, "function_response", None)]
        if not fr:
            fc = types.FunctionCall(name="predict_demand", args={"date": "2012-07-10", "hour": 8, "weather": 3})
            content = types.Content(role="model", parts=[types.Part(function_call=fc)])
            return SimpleNamespace(function_calls=[fc], candidates=[SimpleNamespace(content=content)], text=None)
        val = fr[0].response["result"]["predicted_rentals"]
        return SimpleNamespace(function_calls=None, candidates=[],
                               text=f"PREDICTED completed rentals: about {val} at 08:00 on 2012-07-10 in light rain.")


fake = SimpleNamespace(models=FakeModels())
out = nla.ask_gemini(A, "How busy is 8am on 10 July 2012 if it rains?", client=fake, model="fake-model")
check("gemini loop executes tool and returns text", out["mode"] == "gemini" and out["tool_calls"][0]["tool"] == "predict_demand")
check("gemini answer grounded in tool result", out["grounding"]["grounded"], out["grounding"])
check("tool declarations accepted by SDK types", len(nla._gemini_declarations()) == len(nla.TOOL_SCHEMAS))
cfg = fake.models.calls[0][2]
check("automatic function calling disabled (manual validation)", cfg.automatic_function_calling.disable is True)


class BadArgsModels(FakeModels):
    def generate_content(self, model, contents, config):
        last = contents[-1]
        fr = [pt.function_response for pt in last.parts if getattr(pt, "function_response", None)]
        if not fr:
            fc = types.FunctionCall(name="predict_demand", args={"date": "tomorrow"})
            content = types.Content(role="model", parts=[types.Part(function_call=fc)])
            return SimpleNamespace(function_calls=[fc], candidates=[SimpleNamespace(content=content)], text=None)
        return SimpleNamespace(function_calls=None, candidates=[], text=f"Error: {fr[0].response['error']}")


out = nla.ask_gemini(A, "x", client=SimpleNamespace(models=BadArgsModels()), model="fake")
check("invalid Gemini args returned as tool error, not crash", "error" in out["tool_calls"][0])

# ---------------------------------------------------------------- fallback when no key
os.environ.pop("GEMINI_API_KEY", None)
out = nla.ask(A, "How much does rain reduce demand?", prefer_gemini=True)
check("no key -> deterministic fallback with reason", out["mode"] == "deterministic" and "GEMINI_API_KEY" in out["fallback_reason"])
os.environ["GEMINI_API_KEY"] = "invalid-test-key"
out = nla.ask(A, "How much does rain reduce demand?", prefer_gemini=True)
check("unreachable/invalid Gemini -> fallback", out["mode"] == "deterministic" and out.get("fallback_reason"))
os.environ.pop("GEMINI_API_KEY", None)

# ---------------------------------------------------------------- Streamlit app runs headless
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
check("app loads without exceptions", not at.exception, at.exception)
for q in ["When are the commute peaks on working days?", "Predict demand on 2012-07-10 at 8am if it rains",
          "How many bikes do we need with 30 min trips and a 3x buffer?"]:
    at.chat_input[0].set_value(q).run()
    check(f"app answers: '{q[:40]}'", not at.exception and len(at.session_state.messages) >= 2, at.exception)
last = at.session_state.messages[-1]["out"]
check("app answer is from fleet tool", last["tool_calls"][0]["tool"] == "fleet_scenario")
at.button(key="ex3").click().run()
check("example button works", not at.exception and at.session_state.messages[-1]["out"]["tool_calls"][0]["tool"] == "seasonal_demand")

print(f"\nAll {len(PASSED)} checks passed.")
