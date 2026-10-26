// Builds Final_Report.docx (Word version of Final_Report.pdf). Run: node make_word.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType,
  WidthType, BorderStyle, LevelFormat, ExternalHyperlink,
} = require("docx");

const OUT = path.join(__dirname, "outputs");
const URL = "https://sigmoid-bike-sharing-4sbnqpjtvqpqxdpayyzvjz.streamlit.app/";
const FONT = "Arial";
const W = 9638; // A4 text width in DXA with 2.0 cm margins

// ---- inline markup: **bold**, _italic_ -------------------------------------------------
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|(?<![\w])_[^_]+_(?![\w]))/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t), spacing: { after: 120, line: 288 }, ...o });
const META = (label, value) => new Paragraph({ spacing: { after: 60 },
  children: [new TextRun({ text: label + ": ", bold: true }), ...runs(value)] });
const RULE = () => new Paragraph({ spacing: { before: 120, after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 1 } }, children: [] });
const SEC = (t) => [RULE(), new Paragraph({ keepNext: true, spacing: { after: 160 },
  children: [new TextRun({ text: t, bold: true, size: 22 })] })];
const BUL = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 276 },
  children: runs(t) });
const CAP = (t) => new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: t, italics: true, size: 17 })] });

function FIG(file, widthPx = 600) {
  const buf = fs.readFileSync(path.join(OUT, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG IHDR
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf,
      transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })] });
}

const isNum = (v) => /^[-+]?[\d,]*\.?\d+%?$/.test(String(v).trim());
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
function TBL(rows, fractions) {
  const widths = fractions.map((f) => Math.round(W * f / fractions.reduce((a, b) => a + b, 0)));
  widths[widths.length - 1] = W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const numeric = rows[0].map((_, c) => rows.slice(1).every((r) => isNum(r[c])));
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, cantSplit: true,
      children: r.map((v, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        borders: { top: border, bottom: border, left: border, right: border },
        margins: { top: 70, bottom: 70, left: 100, right: 100 },
        children: [new Paragraph({ alignment: numeric[c] ? AlignmentType.RIGHT : AlignmentType.LEFT,
          children: runs(String(v), { size: 17, bold: i === 0 || undefined }) })],
      })) })),
  });
}
const GAP = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---- content (numbers identical to Final_Report.pdf / outputs/) -----------------------
const c = [];
c.push(new Paragraph({ spacing: { after: 240 },
  children: [new TextRun({ text: "Sigmoid Case Study Round 2 – Bike-Sharing Demand: More Bikes or Better Allocation?",
    bold: true, size: 34 })] }));
c.push(new Paragraph({ spacing: { after: 160 },
  children: [new TextRun({ text: "Final Report: Analysis, Demand Model and Natural-Language Prototype", bold: true })] }));
c.push(META("Candidate", "Aditya Anand, NIT Warangal"));
c.push(META("Role", "Data Science Trainee (2026-27), Sigmoid"));
c.push(META("Data", "UCI Bike Sharing Dataset – Capital Bikeshare, Washington D.C., 2011-2012 (hourly and daily)"));
c.push(META("Approach", "Human-led, AI-assisted; code, notebook, prototype and decision log accompany this report"));
c.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Live prototype: ", bold: true }),
  new ExternalHyperlink({ link: URL, children: [new TextRun({ text: URL, style: "Hyperlink" })] })] }));
c.push(META("Date", "26 October 2026"));

c.push(...SEC("Executive Summary"));
c.push(P("**Question.** Riders cannot find bikes, yet bikes sit idle. Should the company buy more bikes, or allocate existing ones better?"));
c.push(P("**What the data shows.** Demand is sharply concentrated in time. On working days the 07-09 and 17-19 commute windows carry **51.7%** of daily rentals vs **26.0%** on non-working days (H1 supported, p ≈ 4e-102, effect r = 0.99). A gradient-boosting model forecasts hourly rentals on a held-out half-year with R² = 0.90, but under-predicts commute peaks by about 51 rentals/hour."));
c.push(P("**Fleet scenarios.** At the 2012 95th-percentile hour (662 rentals), 20-minute trips put ~221 bikes in motion; with a 2x positioning buffer and 10% in maintenance the scenario needs **~491 bikes** (sensitivity range 276-1104). The case's fictional 2,000-bike fleet would only be too small if bikes had to be positioned ~8x more than are ever riding at once. This is a scenario, not a verified requirement."));
c.push(P("**Recommendation (candidate's decision).** Prioritise investigating **forecast-guided redistribution and fleet-health improvements before buying bikes**, as an **operational pilot to validate**. Redistribution is not proven to be the root cause: the dataset has no station-level availability or unmet-demand data. Suggested timings (rebalancing finished before ~07:00 and ~16:30 on working days) are trial scheduling ideas, not an optimised schedule."));

c.push(...SEC("1. Problem framing and continuity with Round 1"));
c.push(P("In Round 1 (submitted before this AI-assisted round) I argued the problem is more likely **distribution** than fleet shortage, because bikes sit idle while riders complain. I listed causes: bikes piling up at drop locations, commuter imbalance between residential and office areas, poorly targeted driver redistribution, unhealthy or offline bikes, tech failures at peak, and weather/season. I also noted that completed rentals do not show riders who failed to get a bike. Round 2 tests what the provided data can test and is explicit about what it cannot."));
c.push(P("To keep claims honest, the analysis separates five quantities:", { keepNext: true }));
c.push(TBL([["Quantity", "Available here?", "How used"],
  ["1. Observed completed rentals", "Yes", "EDA, hypothesis tests, model training"],
  ["2. Predicted completed rentals", "Yes (model)", "Forecasts, forecast-driven scenarios"],
  ["3. Unobserved / suppressed demand", "No", "Stated as a limitation; true demand may be higher"],
  ["4. Scenario fleet estimates", "Yes (assumptions)", "Little's law + buffers, all inputs adjustable"],
  ["5. Actual fleet sufficiency / station availability", "No", "Requires station-level data (Section 8)"]], [6, 3.3, 7.7]));

c.push(...SEC("2. Data quality checks and feature engineering"));
c.push(TBL([["Check", "Result"],
  ["Rows (hour.csv / day.csv)", "17,379 / 731 — match the Readme"],
  ["Nulls, duplicate rows, duplicate (date, hour) keys", "0 / 0 / 0"],
  ["cnt = casual + registered; hourly sums = day.csv", "Holds for every row / every day"],
  ["Missing hour-rows (of 17,544 expected)", "165, mostly 00-06 h (likely zero-rental hours)"],
  ["Days with < 20 recorded hours", "8, incl. 2012-10-29/30 (Hurricane Sandy) and 2011-08-27/28"],
  ["Suspicious values", "hum = 0 in 22 rows; windspeed = 0 in 2,180 rows (likely missing)"],
  ["weathersit = 4 (heavy rain/snow)", "3 rows only — too rare to analyse separately"]], [8, 9]));
c.push(GAP());
c.push(P("**Decisions.** No imputation of counts. Incomplete days are flagged and excluded from hypothesis tests but kept for modelling (they are real low-demand events). Zero humidity/wind values are left as-is: tree models are robust to them, and they are documented."));
c.push(P("**Features.** De-normalised weather (°C, %, km/h) for readability; commute-window flag (07-09, 17-19); day-part; day type; incomplete-day flag. **Excluded from the model:** _casual_ and _registered_ (they sum to the target — leakage), _instant_ (row id), and lag features (the use-case is planning days ahead, when recent counts are unknown)."));
c.push(FIG("01_hourly_profile_by_daytype.png", 620));
c.push(CAP("Figure 1. Mean completed rentals per hour; shaded = commute windows. Working days show sharp 08:00 and 17-18:00 peaks driven by registered users; non-working days a midday plateau."));

c.push(...SEC("3. Hypothesis H1 — commuter-hour concentration"));
c.push(P("**H0:** the share of a day's rentals in the commute windows is the same on working and non-working days. **H1:** it is higher on working days."));
c.push(P("**Design.** Unit of analysis = one day (hours within a day are correlated; treating 17,379 hours as independent would overstate significance). Incomplete days excluded. One-sided Mann-Whitney U (no normality assumption) with matched effect sizes; repeated within each season.", { keepNext: true }));
c.push(TBL([["Statistic", "Working days", "Non-working days"],
  ["Days", "494", "229"], ["Mean share in commute windows", "51.7%", "26.0%"],
  ["Registered-user share in commute windows", "91.8%", "74.1%"]], [8, 4, 4]));
c.push(GAP());
c.push(P("**Result.** Mann-Whitney one-sided p ≈ 4.1e-102; rank-biserial r = 0.989; Hodges-Lehmann shift 26.2 percentage points (95% bootstrap CI 25.7 to 26.8). Significant in all four seasons. **H1 is supported.**"));
c.push(P("**What it does and does not show.** It establishes _when_ bikes must already be in place — a deadline for rebalancing. It does **not** show residential-to-office movement: that remains a hypothesis requiring trip origin/destination data."));
c.push(FIG("04_h1_peak_share.png", 440));
c.push(CAP("Figure 2. Distribution of daily commute-window share by day type."));

c.push(...SEC("4. Hypothesis H3 — is the July dip heat, not rain?"));
c.push(P("In Round 1 I tentatively attributed a July dip to a monsoon. Washington D.C. has no monsoon, so the data was checked: July 2011 was the hottest month (31.1 °C mean) and had the **fewest** rain/snow hours of the summer (2.0% vs 4.7% in June and 6.6% in August). Rain does not explain the dip; heat is the candidate."));
c.push(P("**H0:** on dry Jun-Aug days, demand on hot days (daily max ≥ 32 °C, pre-specified) equals demand on other days. **H1:** it is lower. Daily rentals are divided by the mean of the same year × day type, removing growth and the commute effect. Wet and incomplete days excluded.", { keepNext: true }));
c.push(TBL([["Estimand (matched test ↔ interval)", "Estimate", "Interval", "One-sided p"],
  ["Rank: H-L shift ↔ Mann-Whitney", "-5.0 pts", "95% CI -8.4 to -1.3", "0.0034"],
  ["Mean diff ↔ permutation", "-3.7 pts", "95% CI -8.0 to +0.7; 90% CI -7.2 to -0.1", "0.032"],
  ["Spearman: max temp vs demand", "rho = -0.35", "—", "1.6e-06 (two-sided)"]], [6, 2.6, 5.6, 2.8]));
c.push(GAP());
c.push(P("**Result: modest evidence** that hot summer days have lower demand. The rank-based effect is clear; the mean-based effect is borderline because a few cooler days had unusually low demand. Temperature and humidity co-vary, so this tests heat and humidity together. **Exploratory only:** at 34-36 °C the effect looks larger (Hodges-Lehmann -7.8 to -11.5 pts), but those thresholds were chosen after viewing the data and are not confirmatory. _Correction log:_ an earlier draft paired the rank test with a two-sided mean CI; the estimands were aligned after review."));
c.push(FIG("05_h3_heat.png", 440));
c.push(CAP("Figure 3. Relative summer demand vs daily max temperature (dry days). Dashed line: pre-specified 32 °C threshold."));

c.push(...SEC("5. Demand forecasting model"));
c.push(P("**Setup.** Train 2011-01-01 to 2012-06-30; test 2012-07-01 to 2012-12-31 (covers summer peak, July heat, winter and Sandy). Chronological split, no shuffling. Inputs: calendar fields and recorded weather. Using recorded weather assumes a perfect weather forecast — acceptable for planning scenarios, stated as a limitation. Target modelled as log(1 + rentals). No fitted preprocessing, so no preprocessing leakage."));
c.push(P("**Baseline.** Mean 2011 rentals for the same month × hour × day type, scaled by the Jan-Jun growth factor (×1.74). **ML.** HistGradientBoosting (primary) and RandomForest (robustness).", { keepNext: true }));
c.push(TBL([["Model", "MAE", "RMSE", "R2", "Peak-hour MAE", "Peak bias"],
  ["Seasonal baseline", "56.2", "92.6", "0.82", "87.9", "+46.9"],
  ["HistGradientBoosting", "44.5", "71.0", "0.90", "82.8", "-51.0"],
  ["RandomForest", "49.9", "78.9", "0.87", "87.8", "-49.9"]], [4.6, 1.8, 1.8, 1.5, 2.8, 2.2]));
c.push(GAP());
c.push(P("**Selection.** HistGradientBoosting is the primary model (best MAE, RMSE, R²). **Known limitation:** it under-predicts commute peaks by ~51 rentals/hour, because tree models cannot extrapolate the 2012 growth beyond levels seen in training; the baseline over-predicts them. On daily totals the baseline is marginally better (median absolute error 11.9% vs 12.8%). The decision was to document this rather than tune late. Permutation importance: hour of day dominates, then working day and temperature."));
c.push(FIG("06_forecast_sample.png", 620));
c.push(CAP("Figure 4. Test-period sample (10-23 Sep 2012): actual vs model vs baseline. Peaks are under-predicted by the model."));

c.push(...SEC("6. Scenario-based fleet estimation"));
c.push(P("**Method.** Little's law gives the bikes in motion at once: rentals per hour × average trip length (hours). Bikes must also already be parked at the right stations before riders arrive (spatial buffer ≥ 1), and some bikes are in maintenance. **Fleet = demand/hr × trip hours × spatial buffer ÷ (1 − maintenance share).** Approved defaults: p95 demand, 20-minute trips, 10% maintenance, 2x buffer. Every factor except demand is an **assumption**: the data has no trip-length, fleet or station information.", { keepNext: true }));
c.push(TBL([["Demand basis", "Rentals/hr", "In motion", "Scenario fleet"],
  ["ML forecast: max predicted hour (2012-H2)", "827", "276", "613"],
  ["ML forecast: p95 of predicted hours (2012-H2)", "602", "201", "447"],
  ["Observed reference: 2012-H2 p95 (actual)", "702", "234", "521"],
  ["Observed reference: 2012-H2 max (actual)", "977", "326", "724"]], [8.5, 2.5, 2.5, 3]));
c.push(GAP());
c.push(P("Sensitivity at 2012 p95 demand (662/hr), 10% maintenance — scenario fleet size:", { keepNext: true }));
c.push(TBL([["Trip length \\ buffer", "1.5x", "2x", "3x"],
  ["15 min", "276", "368", "552"], ["20 min", "368", "491", "736"], ["30 min", "552", "736", "1104"]], [4, 2.5, 2.5, 2.5]));
c.push(GAP());
c.push(P("**Reading the numbers.** With 20-minute trips, the share of the fictional 2,000 bikes riding at once is 8.8% for the _average_ busiest hour (17:00 on working days), 11.0% at the 2012 95th-percentile hour, and 16.3% at the single busiest hour. (A draft wrongly called 8.8% the maximum; corrected after review.) Forecast-driven estimates run below observed-percentile ones because of the model's peak bias, so observed percentiles are kept as the reference. **None of these is a verified fleet requirement:** they combine 2011-12 D.C. demand with a fictional fleet and assumed trip lengths, and say nothing about whether the right stations had bikes."));
c.push(FIG("08_utilisation_scenario.png", 460));
c.push(CAP("Figure 5. Scenario: % of a 2,000-bike fleet in motion by hour (average profile, 20-minute trips)."));

c.push(...SEC("7. Natural-language prototype for the operations manager"));
c.push(P("A Streamlit app (_app.py_) lets a manager ask questions in plain English. **Architecture:** question → Gemini function calling (google-genai SDK) → one of six whitelisted, argument-validated Python functions → real numbers from the data or the trained model → explanation. Automatic function calling is disabled so every call is validated and logged; Gemini never executes code or sees the raw data. A **grounding check** flags any number in the answer that does not appear in the function output. If no key is set or Gemini fails, a **deterministic keyword router** answers with templates built from the same functions, and the UI states why.", { keepNext: true }));
c.push(TBL([["Function", "Questions it answers", "Number type"],
  ["hourly_profile", "Commute peaks, busiest/quietest hours", "Observed"],
  ["compare_day_types", "Working vs non-working days, H1 result", "Observed"],
  ["weather_effect", "Weather impact, raw and controlled", "Observed"],
  ["seasonal_demand", "Monthly/seasonal demand, July dip, H3, growth", "Observed"],
  ["predict_demand", "Rentals for a date/hour under given weather", "Predicted"],
  ["fleet_scenario", "Bikes needed from observed percentiles, ML forecast or custom demand", "Scenario"]], [3.6, 10, 3.4]));
c.push(GAP());
c.push(P("Questions the data cannot answer (stations, routes, trip lengths, failed rentals, rebalancing) receive an explicit refusal that names the data needed, rather than an adjacent but misleading number. Each answer shows which function ran, its arguments and the raw result. **Testing:** 66 automated checks (function outputs vs direct pandas/model calls, input validation, routing, grounding, the Gemini loop with a mocked client, fallback, and a headless Streamlit run)."));
c.push(new Paragraph({ spacing: { after: 120, line: 288 }, children: [new TextRun({ text: "Deployment: ", bold: true }),
  new TextRun("live on Streamlit Community Cloud at "),
  new ExternalHyperlink({ link: URL, children: [new TextRun({ text: URL, style: "Hyperlink" })] }),
  new TextRun(", built from a private GitHub repository. The Gemini key is held only in Streamlit Secrets, never in code or the repository; without it the app falls back to deterministic mode and says so on every answer.")] }));

c.push(...SEC("8. Recommendation and validation pilot"));
c.push(P("**Decision.** Do not make bike purchases the first move. Investigate forecast-guided redistribution and fleet health first, framed as a pilot whose success criteria are set in advance. Buy bikes only if the pilot shows stations still run empty at peaks after rebalancing and repairs."));
[
  "**Fleet health first (cheapest):** measure how many of the fleet are rentable at 07:00 and 16:30; repair or replace unhealthy bikes before buying new ones. This addresses the halved-budget scenario from Round 1.",
  "**Trial schedule:** finish morning rebalancing before ~07:00 and add an afternoon round finishing before ~16:30 on working days; shift driver hours seasonally (summer and autumn peaks are highest). Trial ideas, not an optimised schedule.",
  "**Pilot design:** forecast-guided rebalancing in some zones vs the current schedule in comparable zones for 4-6 weeks. Primary metric: empty-station minutes during commute windows; secondary: completed rentals, complaints, cost per rebalanced bike.",
  "**Data needed to validate:** station dock-status snapshots, trip-level start/end stations and durations, failed/abandoned rental logs, bike-health status, driver rebalancing logs.",
  "**When the answer changes:** if, after rebalancing and repairs, stations still empty at peaks and demand keeps growing (2012 rentals were 65% above 2011), targeted purchases for the stations that still run out are justified.",
].forEach((t) => c.push(BUL(t)));

c.push(...SEC("9. Limitations and confidence"));
[
  "**Most confident:** demand timing (H1) and the weather effect — large, consistent effects.",
  "**Moderately confident:** the forecast model overall (R² 0.90 on unseen months), with known peak under-prediction.",
  "**Least confident:** fleet sizes (assumption-driven, fictional fleet) and the heat effect (modest, co-varies with humidity). Why 2012 exceeded 2011 cannot be separated into supply growth vs adoption with this data.",
  "Completed rentals hide unmet demand, so all demand figures are lower bounds on true demand.",
].forEach((t) => c.push(BUL(t)));

c.push(...SEC("10. How this work was done"));
c.push(P("Human-led and AI-assisted. I set the brief and constraints, approved the plan, chose the hypotheses, kept the pre-specified H3 threshold rather than a post-hoc one, accepted the documented model limitation, and made the business recommendation and its pilot framing. Claude (Anthropic) wrote and executed the code and drafted analyses and options at each checkpoint. My review caught two errors in the AI's summary (the utilisation statistic and the H3 estimand mismatch), both corrected. Details: DECISION_LOG.md and AI_USAGE.md."));
c.push(CAP("Data citation: Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 1-15. doi:10.1007/s13748-013-0040-3"));

const doc = new Document({
  creator: "Aditya Anand", title: "More Bikes or Better Allocation?",
  styles: { default: { document: { run: { font: FONT, size: 19, color: "000000" } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 640, hanging: 280 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 },
    margin: { top: 1247, bottom: 1247, left: 1134, right: 1134 } } }, children: c }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(path.join(__dirname, "Final_Report.docx"), b); console.log("Final_Report.docx written"); });
