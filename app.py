"""
app.py — Operations-manager prototype: ask demand questions in plain English.

Run:   streamlit run app.py
Gemini (optional):  set GEMINI_API_KEY in the environment before launching.
Without a key (or if Gemini fails) the app uses the deterministic keyword router.
"""
import json
import os

import pandas as pd
import streamlit as st

import nl_assistant as nla

st.set_page_config(page_title="Bike Demand Assistant", page_icon="🚲", layout="wide")

# Streamlit Community Cloud: secrets set in the app's "Secrets" settings live in st.secrets.
# Copy them into environment variables so nl_assistant reads GEMINI_API_KEY the same way everywhere.
# Locally, with no secrets file, this silently does nothing and the shell environment is used.
for _name in ("GEMINI_API_KEY", "GEMINI_MODEL"):
    try:
        if _name in st.secrets and not os.environ.get(_name):
            os.environ[_name] = str(st.secrets[_name])
    except Exception:
        pass


@st.cache_resource
def get_analyst():
    return nla.Analyst()


analyst = get_analyst()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    key_present = bool(os.environ.get("GEMINI_API_KEY"))
    mode = st.radio("Language understanding",
                    ["Gemini (falls back automatically)", "Deterministic only"],
                    index=0 if key_present else 1)
    model = st.text_input("Gemini model", value=os.environ.get("GEMINI_MODEL", nla.DEFAULT_GEMINI_MODELS[0]))
    st.caption(f"GEMINI_API_KEY detected: {'yes' if key_present else 'no'} (the key is never displayed or stored).")
    st.divider()
    st.markdown("**What the numbers mean**")
    for k, v in nla.DATA_TYPE_NOTE.items():
        st.markdown(f"- **{k.capitalize()}** — {v}")
    st.caption(f"Forecast model: {analyst.model_name}. Data: Capital Bikeshare (Washington D.C.), 2011-2012, "
               "citywide hourly totals; no station-level data.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
    st.divider()
    if st.button("Run Gemini diagnostic"):
        # Calls Gemini directly (no fallback) so any key/model error is shown verbatim.
        try:
            d = nla.ask_gemini(analyst, "Compare working days and weekends.", model=model or None)
            ok = d["tool_calls"] and all("result" in c for c in d["tool_calls"])
            st.success(f"Gemini OK — model `{d['model']}`, functions called: "
                       f"{[c['tool'] for c in d['tool_calls']]}, grounded: {d['grounding']['grounded']}"
                       if ok else f"Gemini replied (model `{d['model']}`) but called no valid function.")
        except nla.GeminiUnavailable as e:
            st.error(f"Gemini unavailable: {e}")

# ------------------------------------------------------------------ main
st.title("🚲 Bike Demand Assistant")
st.write("Ask about commute peaks, working days vs weekends, weather, seasons, hourly predictions, "
         "or scenario fleet size. Every answer comes from a validated Python function run on the real data or "
         "the trained model; expand **How this was answered** to see the function and raw result.")

EXAMPLES = [
    "When are the commute peaks on working days?",
    "How do weekends compare with working days?",
    "How much does rain reduce demand?",
    "Why is there a dip in July?",
    "Predict demand on 2012-07-10 at 8am if it rains",
    "How many bikes do we need with 30 min trips and a 3x buffer?",
    "Using the forecast for 2012-09-14, how many bikes would we need?",
]
cols = st.columns(4)
clicked = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, key=f"ex{i}", use_container_width=True):
        clicked = ex

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_chart(call: dict):
    """Draw a small chart when the tool result contains a time profile."""
    r = call.get("result") or {}
    if call.get("tool") == "hourly_profile":
        df = pd.DataFrame({k.replace("_", "-"): v["hourly_mean"] for k, v in r["profiles"].items()})
        df.index.name = "hour"
        st.line_chart(df, x_label="Hour of day", y_label="Mean observed rentals / hour")
    elif call.get("tool") == "predict_demand" and "predicted_hourly" in r:
        df = pd.Series(r["predicted_hourly"], name="predicted rentals").to_frame()
        df.index = df.index.astype(int)
        st.bar_chart(df, x_label="Hour of day", y_label="Predicted rentals / hour")
    elif call.get("tool") == "seasonal_demand":
        df = pd.DataFrame(r["monthly"]).pivot(index="month", columns="year", values="mean_rentals_per_hour")
        df.columns = df.columns.astype(str)
        st.line_chart(df, x_label="Month", y_label="Mean observed rentals / hour")
    elif call.get("tool") == "weather_effect":
        s = pd.Series(r["controlled_pct_of_clear_weather_demand"], name="% of clear-weather demand")
        st.bar_chart(s, y_label="% of clear-weather demand (controlled)")


def render_reply(out: dict):
    st.markdown(out["answer"] or "_(empty answer)_")
    badge = (f"Answered by Gemini ({out.get('model')})" if out["mode"] == "gemini"
             else "Answered by deterministic router")
    if out.get("fallback_reason"):
        badge += f" — Gemini unavailable: {out['fallback_reason']}"
    st.caption(badge)
    g = out.get("grounding")
    if g is not None:
        if g.get("grounded"):
            st.caption(f"✅ Grounding check: all {g['numbers_checked']} numbers matched the tool output.")
        else:
            st.warning("Grounding check: some figures in the answer were not found in the tool output "
                       f"{g.get('unmatched')}. Trust the raw result below. {g.get('note', '')}")
    for call in out.get("tool_calls", []):
        if call.get("result"):
            render_chart(call)
    with st.expander("How this was answered (function, arguments, raw result)"):
        if not out.get("tool_calls"):
            st.write("No analysis function was called.")
        for call in out.get("tool_calls", []):
            st.markdown(f"**Function:** `{call['tool']}`  \n**Arguments:** `{json.dumps(call.get('args', {}))}`")
            if "error" in call:
                st.error(call["error"])
            else:
                st.json(call["result"], expanded=False)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_reply(m["out"])

question = st.chat_input("e.g. How busy will 8am be next Monday in July if it's cloudy?") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    history = [(m["role"], m["content"] if m["role"] == "user" else m["out"]["answer"])
               for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            out = nla.ask(analyst, question, prefer_gemini=mode.startswith("Gemini"),
                          model=model or None, history=history)
        render_reply(out)
    st.session_state.messages.append({"role": "assistant", "out": out})
