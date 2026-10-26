# AI Usage

This case study was **human-led and AI-assisted**. Aditya Anand directed the work; Claude (Anthropic, Opus 5.5)
was used as a collaborator through a chat workspace that could run Python. This file records who did what.


## What Aditya did independently
- **Round 1 (before any AI use in this round):** framed the problem as likely distribution rather than fleet
  shortage, listed causes (idle/unhealthy bikes, commuter imbalance, poorly targeted redistribution, weather/season),
  and noted that completed rentals hide unmet demand. This reasoning is the starting point of Round 2.
- Wrote the project brief: human-in-the-loop workflow, scientific constraints, leakage rules, deliverables, privacy.
- **Approved all methodological choices at Checkpoint 1** (hypotheses, missing-data handling, chronological split,
  no lag features, fleet assumptions, Gemini for the prototype).
- **At Checkpoint 2:** kept the pre-specified 32 °C H3 threshold and refused to promote the post-hoc 35 °C result;
  accepted the model's peak bias instead of late tuning; chose HistGradientBoosting; set the business
  recommendation and reframed redistribution as a **pilot to validate**, with timings as trial ideas.
- **Caught three analytical issues in the AI's Stage 3 summary** and asked for verification:
  (1) the 8.8% vs 221-bike inconsistency, (2) the p-value / confidence-interval mismatch in H3,
  (3) the need for the fleet calculator to accept ML predictions. All three were confirmed and corrected
  (details in `DECISION_LOG.md`).

## What the AI did
- Read the source files and profiled the data; proposed the analysis plan, hypotheses and options at each checkpoint.
- Wrote and executed the code: `model_utils.py`, `build_notebook.py` → `analysis.ipynb`, `nl_assistant.py`, `app.py`,
  `tests/test_prototype.py`, `make_report.py` → `Final_Report.pdf`, README and deployment files.
- Ran data-quality checks, hypothesis tests, model training/evaluation and fleet scenarios; produced charts and CSV/JSON results.
- Pointed out its own risky choices for human decision (e.g. that the 35 °C threshold was chosen after seeing data).
- Made two errors that Aditya's review caught (items 1-2 above) and corrected them.
- Checked Google's current Gemini model list (26 Oct 2026) to choose a supported model.

## Verification
- All reported numbers come from executed code; the notebook is saved with outputs.
- Re-executing the notebook reproduced the model metrics exactly (fixed random seeds).
- `tests/test_prototype.py`: 66 automated checks (tool outputs vs direct pandas/model calls, input validation,
  router intents, grounding of answers, Gemini loop with a mocked client, fallback behaviour, headless Streamlit run).
- **Not verified by the AI:** live Gemini API calls (Google's API was unreachable from the build environment).
  Aditya to test locally with his own key.

## Stage 5 (AI, at Aditya's instruction)
- Audit: no secrets, no private correspondence or roll number in the package, no leakage features, 0 notebook errors,
  relative paths work from any directory.
- Hardening for deployment: Streamlit secrets bridged to `GEMINI_API_KEY`; questions the data cannot answer
  (stations, routes, trip lengths, failed rentals, rebalancing) now get an explicit refusal instead of an adjacent number;
  a "Run Gemini diagnostic" button reports the model and functions called, or the exact error.
- Wrote README and the PDF report from executed outputs.

## Limits of AI involvement
- No external datasets were added.
- The AI did not make the business recommendation; it drafted options and Aditya decided.
