// Builds Final_Presentation.pptx. Run: node make_slides.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const OUT = path.join(__dirname, "outputs");
const URL = "https://sigmoid-bike-sharing-4sbnqpjtvqpqxdpayyzvjz.streamlit.app/";

const DARK = "1B2A35", INK = "1B2A35", MUTED = "5B6770", BLUE = "2A6F97", ORANGE = "E07A1F", TINT = "EEF3F6", WHITE = "FFFFFF";
const F = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Aditya Anand";
pres.title = "More Bikes or Better Allocation?";

const T = (s, text, o) => s.addText(text, { fontFace: F, color: INK, isTextBox: true, margin: 0, valign: "top", ...o });
function content(title, sub) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  T(s, title, { x: 0.5, y: 0.35, w: 9, h: 0.6, fontSize: 26, bold: true });
  if (sub) T(s, sub, { x: 0.5, y: 0.95, w: 9, h: 0.35, fontSize: 13, color: MUTED });
  s.addText(`Sigmoid Round 2 · Aditya Anand`, { x: 0.5, y: 5.25, w: 5, h: 0.25, fontFace: F, fontSize: 9, color: MUTED, margin: 0, isTextBox: true });
  return s;
}
const bullets = (items, size = 13) => items.map((t, i) => ({
  text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size } }));
function stat(s, x, y, w, big, label, color = BLUE) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.55, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
  T(s, big, { x: x + 0.2, y: y + 0.18, w: w - 0.4, h: 0.7, fontSize: 34, bold: true, color });
  T(s, label, { x: x + 0.2, y: y + 0.9, w: w - 0.4, h: 0.6, fontSize: 11.5, color: INK });
}
const img = (s, file, x, y, w, h) => s.addImage({ path: path.join(OUT, file), x, y, w, h, sizing: { type: "contain", w, h } });

// 1 Title
let s = pres.addSlide();
s.background = { color: DARK };
T(s, "More Bikes or Better Allocation?", { x: 0.6, y: 1.5, w: 8.8, h: 0.9, fontSize: 38, bold: true, color: WHITE });
T(s, "Bike-sharing demand analysis, forecast model and natural-language prototype", { x: 0.6, y: 2.45, w: 8.8, h: 0.5, fontSize: 16, color: "C9D6DE" });
T(s, "Sigmoid Data Science Trainee · Case Study Round 2\nAditya Anand · NIT Warangal · 26 October 2026", { x: 0.6, y: 3.6, w: 8.8, h: 0.7, fontSize: 12, color: "C9D6DE" });
s.addNotes("Riders can't find bikes while bikes sit idle. The CEO asks: buy more bikes, or allocate better? This is a human-led, AI-assisted analysis of the 2011-12 Capital Bikeshare data.");

// 2 Answer up front
s = content("Answer first: timing, not fleet size", "What the data supports, and what it cannot prove");
stat(s, 0.5, 1.5, 2.85, "51.7%", "of working-day rentals fall in the 07-09 and 17-19 commute windows (vs 26.0% on non-working days)");
stat(s, 3.575, 1.5, 2.85, "R² 0.90", "hourly forecast on unseen Jul-Dec 2012; under-predicts peaks by ~51/hr", ORANGE);
stat(s, 6.65, 1.5, 2.85, "~491", "bikes needed in the default scenario, vs a fictional fleet of 2,000 (scenario, not a requirement)");
T(s, [{ text: "Recommendation: ", options: { bold: true } },
  { text: "pilot forecast-guided redistribution and fleet-health fixes before buying bikes. Redistribution is not proven as the root cause: there is no station-level or unmet-demand data." }],
  { x: 0.5, y: 3.4, w: 9, h: 0.9, fontSize: 14 });
s.addNotes("Lead with the conclusion. All three numbers come from the executed notebook. The fleet number is a scenario built on assumptions.");

// 3 Data quality
s = content("Data quality: clean, with documented gaps", "17,379 hourly rows · 731 days · 2011-2012");
s.addText(bullets([
  "No nulls or duplicates; cnt = casual + registered on every row; hourly sums match day.csv",
  "165 missing hours, mostly 00-06 h; 8 incomplete days incl. Hurricane Sandy, flagged and excluded from tests",
  "hum = 0 (22 rows) and windspeed = 0 (2,180 rows) look like missing values; documented",
  "Leakage guard: casual and registered never used as model inputs",
], 12.5), { x: 0.5, y: 1.45, w: 4.1, h: 3.6, fontFace: F, color: INK, valign: "top", margin: 0, isTextBox: true });
img(s, "01_hourly_profile_by_daytype.png", 4.8, 1.45, 4.7, 2.2);
T(s, "Working days: sharp 08:00 and 17-18:00 peaks. Non-working days: a midday plateau.", { x: 4.8, y: 3.75, w: 4.7, h: 0.5, fontSize: 11, color: MUTED, italic: true });

// 4 H1
s = content("H1: commute windows dominate working days", "One row per day · one-sided Mann-Whitney · incomplete days excluded");
img(s, "04_h1_peak_share.png", 0.5, 1.45, 4.6, 2.9);
s.addText(bullets([
  "51.7% vs 26.0% of daily rentals in commute windows (494 vs 229 days)",
  "p ≈ 4e-102; rank-biserial r = 0.99; shift 26.2 pts (95% CI 25.7-26.8)",
  "Holds in all four seasons; 92% of peak riders are registered users",
  "Shows WHEN bikes must be in place, not WHERE they travel: that needs trip data",
], 12.5), { x: 5.35, y: 1.5, w: 4.15, h: 3.5, fontFace: F, color: INK, valign: "top", margin: 0, isTextBox: true });

// 5 H3
s = content("H3: the July dip looks like heat, not rain", "Round 1 guessed 'monsoon'; the data says otherwise");
img(s, "05_h3_heat.png", 0.5, 1.45, 4.6, 2.9);
s.addText(bullets([
  "July 2011: hottest month (31.1 °C) and fewest rain hours (2.0%)",
  "Pre-specified 32 °C test: median shift −5.0 pts (95% CI −8.4 to −1.3), p = 0.003",
  "Mean-based test borderline (95% CI includes 0): modest evidence",
  "34-36 °C look stronger but were chosen after seeing data: exploratory only",
], 12.5), { x: 5.35, y: 1.5, w: 4.15, h: 3.5, fontFace: F, color: INK, valign: "top", margin: 0, isTextBox: true });

// 6 Model (native chart)
s = content("Forecast model: best overall, misses peaks", "Train to 30 Jun 2012 · test Jul-Dec 2012 · no leakage features, no lags");
s.addChart(pres.charts.BAR, [
  { name: "MAE", labels: ["Seasonal baseline", "HistGradientBoosting", "RandomForest"], values: [56.2, 44.5, 49.9] },
  { name: "Peak-hour MAE", labels: ["Seasonal baseline", "HistGradientBoosting", "RandomForest"], values: [87.9, 82.8, 87.8] },
], { x: 0.5, y: 1.4, w: 5.0, h: 3.6, barDir: "col", chartColors: [BLUE, ORANGE], showValue: true, dataLabelPosition: "outEnd",
  dataLabelFontSize: 10, dataLabelFontFace: "Arial", catAxisLabelFontSize: 10, catAxisLabelFontFace: "Arial", valAxisLabelFontSize: 9,
  valAxisLabelFontFace: "Arial", catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, valGridLine: { color: "E2E8EC", size: 0.5 },
  catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "Arial",
  showTitle: true, title: "Error on unseen months (rentals/hour, lower is better)", titleFontSize: 11, titleFontFace: "Arial", titleColor: INK });
s.addText(bullets([
  "Primary model: HistGradientBoosting (MAE 44.5, R² 0.90)",
  "Known limitation: under-predicts commute peaks by ~51/hr; trees can't extrapolate 2012 growth",
  "Documented rather than tuned late; fleet sizing keeps observed percentiles as reference",
], 12.5), { x: 5.8, y: 1.5, w: 3.7, h: 3.5, fontFace: F, color: INK, valign: "top", margin: 0, isTextBox: true });

// 7 Fleet
s = content("Fleet scenarios: few bikes ride at once", "fleet = demand/hr × trip hours × spatial buffer ÷ (1 − maintenance share)");
const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: TINT } } });
s.addTable([
  [hdr("Trip length \\ buffer"), hdr("1.5x"), hdr("2x"), hdr("3x")],
  ["15 min", "276", "368", "552"], ["20 min", "368", { text: "491", options: { bold: true } }, "736"], ["30 min", "552", "736", "1,104"],
], { x: 0.5, y: 1.5, w: 4.3, colW: [1.6, 0.9, 0.9, 0.9], fontFace: F, fontSize: 12, color: INK, border: { type: "solid", pt: 0.75, color: "B8C4CC" },
  align: "center", valign: "middle", rowH: 0.42 });
T(s, "Scenario fleet at 2012 p95 demand (662/hr), 10% maintenance", { x: 0.5, y: 3.35, w: 4.3, h: 0.4, fontSize: 10.5, color: MUTED, italic: true });
s.addText(bullets([
  "Share of a 2,000-bike fleet riding at once (20-min trips): 8.8% average busiest hour, 11.0% p95 hour, 16.3% single busiest hour",
  "2,000 bikes only fall short if positioning needs a buffer of ~8x",
  "Assumed trip length, buffer and fictional fleet: scenarios, not verified requirements",
], 12.5), { x: 5.1, y: 1.5, w: 4.4, h: 3.5, fontFace: F, color: INK, valign: "top", margin: 0, isTextBox: true });

// 8 Prototype
s = content("Prototype: ask demand questions in plain English", "Streamlit app · Gemini function calling · deterministic fallback");
const steps = ["Manager's question", "Gemini picks a validated function", "Python runs on real data / model", "Grounded answer + raw result"];
steps.forEach((t, i) => {
  const x = 0.5 + i * 2.33;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 2.0, h: 1.0, fill: { color: i === 2 ? BLUE : TINT }, line: { color: i === 2 ? BLUE : TINT }, rectRadius: 0.08 });
  T(s, t, { x: x + 0.12, y: 1.6, w: 1.76, h: 0.9, fontSize: 12.5, bold: true, align: "center", valign: "middle", color: i === 2 ? WHITE : INK });
  if (i < 3) s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + 2.07, y: 1.92, w: 0.2, h: 0.26, rotate: 90, fill: { color: MUTED }, line: { color: MUTED } });
});
s.addText(bullets([
  "6 whitelisted functions: peaks, day types, weather, seasons, predictions, fleet scenarios",
  "Grounding check flags any number not in the function output; unsupported questions (stations, routes) are refused honestly",
  "No key or API failure → deterministic router, stated on every answer · 66 automated tests",
], 12.5), { x: 0.5, y: 2.85, w: 9, h: 1.7, fontFace: F, color: INK, valign: "top", margin: 0, isTextBox: true });
s.addText([{ text: "Live app: ", options: { bold: true } }, { text: URL, options: { hyperlink: { url: URL }, color: BLUE } }],
  { x: 0.5, y: 4.65, w: 9, h: 0.35, fontFace: F, fontSize: 11.5, color: INK, margin: 0, isTextBox: true });

// 9 Recommendation (dark)
s = pres.addSlide();
s.background = { color: DARK };
T(s, "Recommendation: pilot before purchase", { x: 0.6, y: 0.45, w: 8.8, h: 0.6, fontSize: 28, bold: true, color: WHITE });
const recs = [
  ["Fix fleet health first", "Count rentable bikes at 07:00 and 16:30; repair before buying (fits the halved budget)"],
  ["Trial a forecast-guided schedule", "Rebalance before ~07:00 and ~16:30 on working days; shift driver hours seasonally (trial ideas)"],
  ["Measure it properly", "4-6 week zone pilot vs current schedule; metric: empty-station minutes at commute peaks"],
  ["Buy only if still short", "If stations still run empty after rebalancing and repairs, buy for those stations"],
];
recs.forEach(([h, b], i) => {
  const y = 1.35 + i * 0.9;
  s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.05, w: 0.45, h: 0.45, fill: { color: ORANGE }, line: { color: ORANGE } });
  T(s, String(i + 1), { x: 0.6, y: y + 0.05, w: 0.45, h: 0.45, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
  T(s, h, { x: 1.3, y, w: 8.1, h: 0.35, fontSize: 15, bold: true, color: WHITE });
  T(s, b, { x: 1.3, y: y + 0.36, w: 8.1, h: 0.45, fontSize: 12, color: "C9D6DE" });
});
T(s, "Data needed to validate: station dock status, trip start/end and duration, failed rentals, bike health, rebalancing logs", { x: 0.6, y: 5.0, w: 8.8, h: 0.35, fontSize: 10.5, italic: true, color: "C9D6DE" });

pres.writeFile({ fileName: path.join(__dirname, "Final_Presentation.pptx") }).then(() => console.log("Final_Presentation.pptx written"));
