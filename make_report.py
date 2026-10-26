"""Builds Final_Report.pdf from outputs/ (python make_report.py). All numbers are read from executed results."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
R = json.loads((OUT / "results.json").read_text())
M = pd.read_csv(OUT / "model_metrics_test_2012H2.csv", index_col=0)
FC = pd.read_csv(OUT / "fleet_forecast_vs_observed.csv", index_col=0)
GRID = pd.read_csv(OUT / "fleet_sensitivity_grid.csv", index_col=0)
SENS = pd.read_csv(OUT / "h3_threshold_sensitivity_EXPLORATORY.csv")
H1, H3 = R["H1"], R["H3"]

from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable

# Typography modelled on a plain Word-style document: Liberation Sans (Arial metrics), black text,
# rules between numbered sections, plain black-grid tables. Falls back to Helvetica if the font is missing.
_FD = Path("/usr/share/fonts/truetype/liberation")
try:
    for name, f in [("LS", "Regular"), ("LS-B", "Bold"), ("LS-I", "Italic"), ("LS-BI", "BoldItalic")]:
        pdfmetrics.registerFont(TTFont(name, str(_FD / f"LiberationSans-{f}.ttf")))
    addMapping("LS", 0, 0, "LS"); addMapping("LS", 1, 0, "LS-B")
    addMapping("LS", 0, 1, "LS-I"); addMapping("LS", 1, 1, "LS-BI")
    FONT, FONT_B, FONT_I = "LS", "LS-B", "LS-I"
except Exception:
    FONT, FONT_B, FONT_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"

ss = getSampleStyleSheet()
INK = colors.black
B = ParagraphStyle("B", parent=ss["BodyText"], fontName=FONT, fontSize=9.5, leading=14, textColor=INK,
                   alignment=TA_LEFT, spaceAfter=6)
H = ParagraphStyle("H", parent=B, fontName=FONT_B, fontSize=11, leading=14, spaceBefore=2, spaceAfter=8,
                   keepWithNext=1)
H2 = ParagraphStyle("H2", parent=B, fontName=FONT_B, fontSize=9.5, leading=13, spaceBefore=4, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontName=FONT_I, fontSize=8.5, leading=11.5, textColor=colors.HexColor("#333333"))
BOX = ParagraphStyle("BOX", parent=B)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=24, bulletIndent=12, spaceAfter=4)
META = ParagraphStyle("META", parent=B, spaceAfter=3)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.5, leading=11, spaceAfter=0)
CELL_B = ParagraphStyle("CELL_B", parent=CELL, fontName=FONT_B)
CELL_R = ParagraphStyle("CELL_R", parent=CELL, alignment=2)
CELL_BR = ParagraphStyle("CELL_BR", parent=CELL_B, alignment=2)


def p(t, s=B):
    return Paragraph(t, s)


def rule():
    hr = HRFlowable(width="100%", thickness=0.8, color=INK, spaceBefore=8, spaceAfter=12)
    hr.keepWithNext = True  # never leave a rule or heading stranded at the bottom of a page
    return hr


def sec(title):
    """Numbered section: horizontal rule above, bold heading."""
    return [rule(), p(title, H)]


def bullets(items):
    return [Paragraph(i, BUL, bulletText="\u2022") for i in items]


def _is_num(v):
    try:
        float(str(v).replace(",", "").replace("+", "").replace("%", ""))
        return True
    except ValueError:
        return False


def table(rows, widths=None, header=True):
    body = rows[1:] if header else rows
    avail = A4[0] - 4.6 * cm  # page width minus margins
    if widths and sum(widths) > avail:
        widths = [w * avail / sum(widths) for w in widths]
    numeric = [bool(body) and all(_is_num(r[c]) for r in body) for c in range(len(rows[0]))]
    cells = []
    for i, r in enumerate(rows):  # wrap every cell so long text breaks inside its column
        head = header and i == 0
        cells.append([Paragraph(str(v), (CELL_BR if numeric[c] else CELL_B) if head else
                                (CELL_R if numeric[c] else CELL)) for c, v in enumerate(r)])
    t = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return KeepTogether([t])


def img(name, w=16):
    from reportlab.lib.utils import ImageReader
    path = str(OUT / name)
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=w * cm, height=w * cm * ih / iw)


def fig(name, caption, w=16):
    return KeepTogether([img(name, w), p(caption, SM)])


hl = lambda x: f"{x * 100:+.1f}"
story = []

# ------------------------------------------------------------------ title + executive summary
TITLE = ParagraphStyle("T", parent=H, fontSize=17, leading=21, spaceAfter=14)
story += [p("Sigmoid Case Study Round 2 – Bike-Sharing Demand: More Bikes or Better Allocation?", TITLE),
          p("Final Report: Analysis, Demand Model and Natural-Language Prototype", ParagraphStyle(
              "ST", parent=H2, spaceAfter=8)),
          p("<b>Candidate:</b> Aditya Anand, NIT Warangal", META),
          p("<b>Role:</b> Data Science Trainee (2026-27), Sigmoid", META),
          p("<b>Data:</b> UCI Bike Sharing Dataset – Capital Bikeshare, Washington D.C., 2011-2012 (hourly and daily)", META),
          p("<b>Approach:</b> Human-led, AI-assisted; code, notebook, prototype and decision log accompany this report", META),
          p('<b>Live prototype:</b> <link href="https://sigmoid-bike-sharing-4sbnqpjtvqpqxdpayyzvjz.streamlit.app/" color="blue"><u>https://sigmoid-bike-sharing-4sbnqpjtvqpqxdpayyzvjz.streamlit.app/</u></link>', META),
          p("<b>Date:</b> 26 October 2026", META)]

story += [*sec("Executive Summary"), p(
    "<b>Question.</b> Riders cannot find bikes, yet bikes sit idle. Should the company buy more bikes, or allocate "
    "existing ones better?", B), p(
    f"<b>What the data shows.</b> Demand is sharply concentrated in time. On working days the 07-09 and 17-19 commute "
    f"windows carry <b>{H1['mean_peak_share_working'] * 100:.1f}%</b> of daily rentals vs "
    f"<b>{H1['mean_peak_share_nonworking'] * 100:.1f}%</b> on non-working days (H1 supported, p ≈ "
    f"{H1['p_value']:.0e}, effect r = {H1['rank_biserial']:.2f}). A gradient-boosting model forecasts hourly "
    f"rentals on a held-out half-year with R<super>2</super> = {M.loc['HistGradientBoosting', 'R2']:.2f}, but under-predicts commute "
    f"peaks by about {abs(M.loc['HistGradientBoosting', 'Peak-hour bias (pred-actual)']):.0f} rentals/hour.", B), p(
    f"<b>Fleet scenarios.</b> At the 2012 95th-percentile hour (662 rentals), 20-minute trips put ~221 bikes in motion; "
    f"with a 2x positioning buffer and 10% in maintenance the scenario needs <b>~491 bikes</b> (sensitivity range "
    f"{int(GRID.values.min())}-{int(GRID.values.max())}). The case's fictional 2,000-bike fleet would only be too small "
    f"if bikes had to be positioned ~8x more than are ever riding at once. This is a scenario, not a verified "
    f"requirement.", B), p(
    "<b>Recommendation (candidate's decision).</b> Prioritise investigating <b>forecast-guided redistribution and "
    "fleet-health improvements before buying bikes</b>, as an <b>operational pilot to validate</b>. Redistribution is "
    "not proven to be the root cause: the dataset has no station-level availability or unmet-demand data. Suggested "
    "timings (rebalancing finished before ~07:00 and ~16:30 on working days) are trial scheduling ideas, not an "
    "optimised schedule.", BOX)]

# ------------------------------------------------------------------ 1 problem + continuity
story += [*sec("1. Problem framing and continuity with Round 1"), p(
    "In Round 1 (submitted before this AI-assisted round) I argued the problem is more likely <b>distribution</b> than "
    "fleet shortage, because bikes sit idle while riders complain. I listed causes: bikes piling up at drop locations, "
    "commuter imbalance between residential and office areas, poorly targeted driver redistribution, unhealthy or "
    "offline bikes, tech failures at peak, and weather/season. I also noted that completed rentals do not show riders "
    "who failed to get a bike. Round 2 tests what the provided data can test and is explicit about what it cannot.", B),
    p("To keep claims honest, the analysis separates five quantities:", ParagraphStyle("KN", parent=B, keepWithNext=1)),
    table([["Quantity", "Available here?", "How used"],
           ["1. Observed completed rentals", "Yes", "EDA, hypothesis tests, model training"],
           ["2. Predicted completed rentals", "Yes (model)", "Forecasts, forecast-driven scenarios"],
           ["3. Unobserved / suppressed demand", "No", "Stated as a limitation; true demand may be higher"],
           ["4. Scenario fleet estimates", "Yes (assumptions)", "Little's law + buffers, all inputs adjustable"],
           ["5. Actual fleet sufficiency / station availability", "No", "Requires station-level data (Section 8)"]],
          [6 * cm, 3.3 * cm, 7.7 * cm])]

# ------------------------------------------------------------------ 2 data quality
story += [*sec("2. Data quality checks and feature engineering"), table(
    [["Check", "Result"],
     ["Rows (hour.csv / day.csv)", "17,379 / 731 — match the Readme"],
     ["Nulls, duplicate rows, duplicate (date, hour) keys", "0 / 0 / 0"],
     ["cnt = casual + registered; hourly sums = day.csv", "Holds for every row / every day"],
     ["Missing hour-rows (of 17,544 expected)", f"165, mostly 00-06 h (likely zero-rental hours)"],
     ["Days with < 20 recorded hours", "8, incl. 2012-10-29/30 (Hurricane Sandy) and 2011-08-27/28"],
     ["Suspicious values", "hum = 0 in 22 rows; windspeed = 0 in 2,180 rows (likely missing)"],
     ["weathersit = 4 (heavy rain/snow)", "3 rows only — too rare to analyse separately"]],
    [8 * cm, 9 * cm]), Spacer(1, 4), p(
    "<b>Decisions.</b> No imputation of counts. Incomplete days are flagged and excluded from hypothesis tests but kept "
    "for modelling (they are real low-demand events). Zero humidity/wind values are left as-is: tree models are robust "
    "to them, and they are documented.", B), p(
    "<b>Features.</b> De-normalised weather (°C, %, km/h) for readability; commute-window flag (07-09, 17-19); "
    "day-part; day type; incomplete-day flag. <b>Excluded from the model:</b> <i>casual</i> and <i>registered</i> "
    "(they sum to the target — leakage), <i>instant</i> (row id), and lag features (the use-case is planning days "
    "ahead, when recent counts are unknown).", B),
    fig("01_hourly_profile_by_daytype.png", "Figure 1. Mean completed rentals per hour; shaded = commute windows. "
        "Working days show sharp 08:00 and 17-18:00 peaks driven by registered users; non-working days a midday plateau.")]

# ------------------------------------------------------------------ 3 H1
story += [*sec("3. Hypothesis H1 — commuter-hour concentration"), p(
    "<b>H0:</b> the share of a day's rentals in the commute windows is the same on working and non-working days. "
    "<b>H1:</b> it is higher on working days.", B), p(
    "<b>Design.</b> Unit of analysis = one day (hours within a day are correlated; treating 17,379 hours as independent "
    "would overstate significance). Incomplete days excluded. One-sided Mann-Whitney U (no normality assumption) with "
    "matched effect sizes; repeated within each season.", B),
    table([["Statistic", "Working days", "Non-working days"],
           ["Days", H1["n_working_days"], H1["n_nonworking_days"]],
           ["Mean share in commute windows", f"{H1['mean_peak_share_working'] * 100:.1f}%",
            f"{H1['mean_peak_share_nonworking'] * 100:.1f}%"],
           ["Registered-user share in commute windows", f"{H1['registered_share_at_peak_working'] * 100:.1f}%",
            f"{H1['registered_share_at_peak_nonworking'] * 100:.1f}%"]], [8 * cm, 4 * cm, 4 * cm]), Spacer(1, 4),
    p(f"<b>Result.</b> Mann-Whitney one-sided p ≈ {H1['p_value']:.1e}; rank-biserial r = {H1['rank_biserial']:.3f}; "
      f"Hodges-Lehmann shift {H1['hodges_lehmann_shift'] * 100:.1f} percentage points (95% bootstrap CI "
      f"{H1['hodges_lehmann_ci95'][0] * 100:.1f} to {H1['hodges_lehmann_ci95'][1] * 100:.1f}). Significant in all four "
      f"seasons. <b>H1 is supported.</b>", B),
    p("<b>What it does and does not show.</b> It establishes <i>when</i> bikes must already be in place — a deadline "
      "for rebalancing. It does <b>not</b> show residential-to-office movement: that remains a hypothesis requiring "
      "trip origin/destination data.", B),
    fig("04_h1_peak_share.png", "Figure 2. Distribution of daily commute-window share by day type.", 12)]

# ------------------------------------------------------------------ 4 H3
s32 = SENS[SENS.threshold_c == 32].iloc[0]
story += [*sec("4. Hypothesis H3 — is the July dip heat, not rain?"), p(
    "In Round 1 I tentatively attributed a July dip to a monsoon. Washington D.C. has no monsoon, so the data was "
    "checked: July 2011 was the hottest month (31.1 °C mean) and had the <b>fewest</b> rain/snow hours of the summer "
    "(2.0% vs 4.7% in June and 6.6% in August). Rain does not explain the dip; heat is the candidate.", B), p(
    "<b>H0:</b> on dry Jun-Aug days, demand on hot days (daily max ≥ 32 °C, pre-specified) equals demand on other "
    "days. <b>H1:</b> it is lower. Daily rentals are divided by the mean of the same year × day type, removing growth "
    "and the commute effect. Wet and incomplete days excluded.", B),
    table([["Estimand (matched test ↔ interval)", "Estimate", "Interval", "One-sided p"],
           ["Rank: H-L shift ↔ Mann-Whitney", f"{hl(H3['hodges_lehmann_shift'])} pts",
            f"95% CI {hl(H3['hodges_lehmann_ci95'][0])} to {hl(H3['hodges_lehmann_ci95'][1])}", f"{H3['p_value']:.4f}"],
           ["Mean diff ↔ permutation", f"{hl(H3['mean_diff'])} pts",
            f"95% CI {hl(H3['mean_diff_ci95'][0])} to {hl(H3['mean_diff_ci95'][1])}; "
            f"90% CI {hl(H3['mean_diff_ci90'][0])} to {hl(H3['mean_diff_ci90'][1])}",
            f"{H3['permutation_p_one_sided']:.3f}"],
           ["Spearman: max temp vs demand", f"rho = {H3['spearman_rho_temp_vs_demand']:.2f}", "—",
            f"{H3['spearman_p']:.1e} (two-sided)"]], [6 * cm, 2.6 * cm, 5.6 * cm, 2.8 * cm]), Spacer(1, 4),
    p("<b>Result: modest evidence</b> that hot summer days have lower demand. The rank-based effect is clear; the "
      "mean-based effect is borderline because a few cooler days had unusually low demand. Temperature and humidity "
      "co-vary, so this tests heat and humidity together. <b>Exploratory only:</b> at 34-36 °C the effect looks "
      "larger (Hodges-Lehmann −7.8 to −11.5 pts), but those thresholds were chosen after viewing the data and are not "
      "confirmatory. <i>Correction log:</i> an earlier draft paired the rank test with a two-sided mean CI; the "
      "estimands were aligned after review.", B),
    fig("05_h3_heat.png", "Figure 3. Relative summer demand vs daily max temperature (dry days). Dashed line: "
        "pre-specified 32 °C threshold.", 12)]

# ------------------------------------------------------------------ 5 model
mrows = [["Model", "MAE", "RMSE", "R2", "Peak-hour MAE", "Peak bias"]]
for n, r in M.iterrows():
    mrows.append([n, f"{r.MAE:.1f}", f"{r.RMSE:.1f}", f"{r.R2:.2f}", f"{r['Peak-hour MAE']:.1f}",
                  f"{r['Peak-hour bias (pred-actual)']:+.1f}"])
story += [*sec("5. Demand forecasting model"), p(
    "<b>Setup.</b> Train 2011-01-01 to 2012-06-30; test 2012-07-01 to 2012-12-31 (covers summer peak, July heat, "
    "winter and Sandy). Chronological split, no shuffling. Inputs: calendar fields and recorded weather. Using recorded "
    "weather assumes a perfect weather forecast — acceptable for planning scenarios, stated as a limitation. Target "
    "modelled as log(1 + rentals). No fitted preprocessing, so no preprocessing leakage.", B), p(
    f"<b>Baseline.</b> Mean 2011 rentals for the same month × hour × day type, scaled by the Jan-Jun growth factor "
    f"(×1.74). <b>ML.</b> HistGradientBoosting (primary) and RandomForest (robustness).", B),
    table(mrows, [4.6 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 2.8 * cm, 2.2 * cm]), Spacer(1, 4), p(
    "<b>Selection.</b> HistGradientBoosting is the primary model (best MAE, RMSE, R<super>2</super>). <b>Known limitation:</b> it "
    "under-predicts commute peaks by ~51 rentals/hour, because tree models cannot extrapolate the 2012 growth beyond "
    "levels seen in training; the baseline over-predicts them. On daily totals the baseline is marginally better "
    "(median absolute error 11.9% vs 12.8%). The decision was to document this rather than tune late. Permutation "
    "importance: hour of day dominates, then working day and temperature.", B),
    fig("06_forecast_sample.png", "Figure 4. Test-period sample (10-23 Sep 2012): actual vs model vs baseline. "
        "Peaks are under-predicted by the model.")]

# ------------------------------------------------------------------ 6 fleet
story += [*sec("6. Scenario-based fleet estimation"), p(
    "<b>Method.</b> Little's law gives the bikes in motion at once: rentals per hour × average trip length (hours). "
    "Bikes must also already be parked at the right stations before riders arrive (spatial buffer ≥ 1), and some bikes "
    "are in maintenance. <b>Fleet = demand/hr × trip hours × spatial buffer ÷ (1 − maintenance share).</b> Approved "
    "defaults: p95 demand, 20-minute trips, 10% maintenance, 2x buffer. Every factor except demand is an "
    "<b>assumption</b>: the data has no trip-length, fleet or station information.", B)]
grows = [["Trip length \\ buffer"] + [f"{float(c):g}x" for c in GRID.columns]]
for idx, r in GRID.iterrows():
    grows.append([f"{idx} min"] + [f"{int(v)}" for v in r.values])
frows = [["Demand basis", "Rentals/hr", "In motion", "Scenario fleet"]]
for idx, r in FC.iterrows():
    frows.append([idx, f"{r.hourly_demand:.0f}", f"{r.bikes_in_motion:.0f}", f"{int(r.fleet_required)}"])
story += [table(frows, [8.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm]), Spacer(1, 4),
          p("Sensitivity at 2012 p95 demand (662/hr), 10% maintenance — scenario fleet size:", ParagraphStyle("KN2", parent=B, keepWithNext=1)),
          table(grows, [4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]), Spacer(1, 4), p(
    "<b>Reading the numbers.</b> With 20-minute trips, the share of the fictional 2,000 bikes riding at once is 8.8% "
    "for the <i>average</i> busiest hour (17:00 on working days), 11.0% at the 2012 95th-percentile hour, and 16.3% at "
    "the single busiest hour. (A draft wrongly called 8.8% the maximum; corrected after review.) Forecast-driven "
    "estimates run below observed-percentile ones because of the model's peak bias, so observed percentiles are kept "
    "as the reference. <b>None of these is a verified fleet requirement:</b> they combine 2011-12 D.C. demand with a "
    "fictional fleet and assumed trip lengths, and say nothing about whether the right stations had bikes.", B),
    fig("08_utilisation_scenario.png", "Figure 5. Scenario: % of a 2,000-bike fleet in motion by hour (average "
        "profile, 20-minute trips).", 12)]

# ------------------------------------------------------------------ 7 prototype
story += [*sec("7. Natural-language prototype for the operations manager"), p(
    "A Streamlit app (<i>app.py</i>) lets a manager ask questions in plain English. <b>Architecture:</b> question → "
    "Gemini function calling (google-genai SDK) → one of six whitelisted, argument-validated Python functions → real "
    "numbers from the data or the trained model → explanation. Automatic function calling is disabled so every call is "
    "validated and logged; Gemini never executes code or sees the raw data. A <b>grounding check</b> flags any number "
    "in the answer that does not appear in the function output. If no key is set or Gemini fails, a <b>deterministic "
    "keyword router</b> answers with templates built from the same functions, and the UI states why.", B),
    table([["Function", "Questions it answers", "Number type"],
           ["hourly_profile", "Commute peaks, busiest/quietest hours", "Observed"],
           ["compare_day_types", "Working vs non-working days, H1 result", "Observed"],
           ["weather_effect", "Weather impact, raw and controlled", "Observed"],
           ["seasonal_demand", "Monthly/seasonal demand, July dip, H3, growth", "Observed"],
           ["predict_demand", "Rentals for a date/hour under given weather", "Predicted"],
           ["fleet_scenario", "Bikes needed from observed percentiles, ML forecast or custom demand", "Scenario"]],
          [3.6 * cm, 10 * cm, 3.4 * cm]), Spacer(1, 4), p(
    "Questions the data cannot answer (stations, routes, trip lengths, failed rentals, rebalancing) receive an explicit "
    "refusal that names the data needed, rather than an adjacent but misleading number. Each answer shows which function "
    "ran, its arguments and the raw result. <b>Testing:</b> 66 automated checks (function outputs vs direct "
    "pandas/model calls, input validation, routing, grounding, the Gemini loop with a mocked client, fallback, and a "
    "headless Streamlit run). <b>Deployment:</b> live on Streamlit Community Cloud at <link href='https://sigmoid-bike-sharing-4sbnqpjtvqpqxdpayyzvjz.streamlit.app/' color='blue'><u>https://sigmoid-bike-sharing-4sbnqpjtvqpqxdpayyzvjz.streamlit.app/</u></link>, built from a private GitHub repository. The Gemini key is held only in Streamlit Secrets, never in code or the repository; without it the app falls back to deterministic mode and says so on every answer.", B)]

# ------------------------------------------------------------------ 8 recommendation
story += [*sec("8. Recommendation and validation pilot"), p(
    "<b>Decision.</b> Do not make bike purchases the first move. Investigate forecast-guided redistribution and fleet "
    "health first, framed as a pilot whose success criteria are set in advance. Buy bikes only if the pilot shows "
    "stations still run empty at peaks after rebalancing and repairs.", BOX)]
story += bullets([
    "<b>Fleet health first (cheapest):</b> measure how many of the fleet are rentable at 07:00 and 16:30; repair or "
    "replace unhealthy bikes before buying new ones. This addresses the halved-budget scenario from Round 1.",
    "<b>Trial schedule:</b> finish morning rebalancing before ~07:00 and add an afternoon round finishing before "
    "~16:30 on working days; shift driver hours seasonally (summer and autumn peaks are highest). Trial ideas, not "
    "an optimised schedule.",
    "<b>Pilot design:</b> forecast-guided rebalancing in some zones vs the current schedule in comparable zones for "
    "4-6 weeks. Primary metric: empty-station minutes during commute windows; secondary: completed rentals, "
    "complaints, cost per rebalanced bike.",
    "<b>Data needed to validate:</b> station dock-status snapshots, trip-level start/end stations and durations, "
    "failed/abandoned rental logs, bike-health status, driver rebalancing logs.",
    "<b>When the answer changes:</b> if, after rebalancing and repairs, stations still empty at peaks and demand keeps "
    "growing (2012 rentals were 65% above 2011), targeted purchases for the stations that still run out are justified."])

story += [*sec("9. Limitations and confidence")]
story += bullets([
    "<b>Most confident:</b> demand timing (H1) and the weather effect — large, consistent effects.",
    "<b>Moderately confident:</b> the forecast model overall (R<super>2</super> 0.90 on unseen months), with known peak under-prediction.",
    "<b>Least confident:</b> fleet sizes (assumption-driven, fictional fleet) and the heat effect (modest, co-varies "
    "with humidity). Why 2012 exceeded 2011 cannot be separated into supply growth vs adoption with this data.",
    "Completed rentals hide unmet demand, so all demand figures are lower bounds on true demand."])

story += [*sec("10. How this work was done"), p(
    "Human-led and AI-assisted. I set the brief and constraints, approved the plan, chose the hypotheses, kept the "
    "pre-specified H3 threshold rather than a post-hoc one, accepted the documented model limitation, and made the "
    "business recommendation and its pilot framing. Claude (Anthropic) wrote and executed the code and drafted analyses "
    "and options at each checkpoint. My review caught two errors in the AI's summary (the utilisation statistic and the "
    "H3 estimand mismatch), both corrected. Details: DECISION_LOG.md and AI_USAGE.md.", B), Spacer(1, 6), p(
    "Data citation: Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background "
    "knowledge. Progress in Artificial Intelligence, 1-15. doi:10.1007/s13748-013-0040-3", SM)]


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(1.8 * cm, 1.1 * cm, "Sigmoid Round 2 — Bike-sharing case study · Aditya Anand")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


SimpleDocTemplate(str(ROOT / "Final_Report.pdf"), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                  topMargin=2.2 * cm, bottomMargin=2.2 * cm, title="More Bikes or Better Allocation?",
                  author="Aditya Anand").build(story)
print("Final_Report.pdf written")
