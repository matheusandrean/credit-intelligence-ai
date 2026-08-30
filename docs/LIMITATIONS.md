# Known Limitations

Stated explicitly, per this project's principle that limitations should be
disclosed rather than discovered.

## Data

- **100% synthetic.** No real customer, financial, or personal data exists
  anywhere in this project. Statistical relationships are designed to be
  plausible, not empirically derived from any real portfolio.
- **No macroeconomic variables.** Stress scenarios shock income/expenses/
  utilization directly rather than through a macro model (e.g. unemployment
  rate → income shock).
- **No concentration modeling.** Geographic, industry, or single-obligor
  concentration limits are not modeled or monitored.

## Modeling

- **Reject inference is not implemented.** The model trains only on
  originated accounts; there is no declined-applicant population. This is
  the classic reject-inference problem in credit scoring and is disclosed,
  not solved, in this project (see [`GOVERNANCE.md`](../GOVERNANCE.md)).
- **No independent validation.** No model, calibration, or threshold in
  this repository has undergone independent model validation, legal
  review, or regulatory approval.
- **Illustrative score/band only.** The 300-900 score and A-E bands are a
  standard scorecard transformation of PD for demonstration — not a real
  commercial pricing or approval policy.
- **SHAP explains the raw model, not the calibrated PD directly.**
  Calibration is monotonic, so direction/ranking of drivers is preserved,
  but exact SHAP magnitudes are computed pre-calibration.

## Vintage / Roll-Rate Analytics

- The monthly delinquency-bucket panel is simulated via a decile-
  conditioned Markov chain seeded by observable risk drivers — it is
  plausible and internally consistent, but it is not literally derived
  from `target_default_90d`, so small numerical differences between
  "modeled PD" and "panel-implied bad rate" are expected and not a bug.
- Panel history is capped at 18 months per account for tractability;
  longer-tenured accounts' full history is not represented in the panel
  (though it is represented in `account_tenure_months`).

## Generative AI / Agent

- **Demo Mode is not a language model.** It routes questions to tools via
  keyword matching and renders real tool output in a template — it cannot
  answer open-ended questions outside that routing table's coverage.
- **No cross-question memory.** Each agent call starts a fresh provider
  conversation; the dashboard's chat history is for display only.
- **English + Portuguese keyword coverage in Demo Mode is not exhaustive.**
  A real LLM provider (Anthropic/OpenAI/Ollama) handles arbitrary phrasing;
  Demo Mode's intent router covers the demo question set plus common
  variants (see `tests/unit/test_demo_provider.py`).
- **RAG uses TF-IDF, not neural embeddings**, for full offline operation —
  see [`DEMO_MODE.md`](DEMO_MODE.md). This is adequate for a 5-document
  policy knowledge base but would not scale semantically to a much larger,
  more nuanced corpus.

## Screenshots

This repository was built and verified in a non-interactive, headless
development environment without browser/GUI screenshot capture available.
The Streamlit app and FastAPI docs were verified to run and respond
correctly (see the Testing section of `README.md`), but static screenshot
images are not included — run `make app` locally to see the live UI.
