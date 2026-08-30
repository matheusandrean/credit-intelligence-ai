# Responsible AI

This is the canonical, real statement of this project's responsible-AI
engineering practices — distinct from the fictitious
`knowledge_base/responsible_ai_policy.md`, which is a synthetic in-app
policy document used to demonstrate the RAG feature.

## 1. Human-in-the-loop, structurally

No component of this platform — the quantitative models or the generative
AI agent — can approve, deny, or finalize a credit decision. This is
enforced at multiple levels, not just documented as a policy:

- No API endpoint exists that writes a decision back to any record.
- The agent's system prompt (`src/llm/system_prompt.py`) explicitly forbids
  approval/denial language and instructs the model to redirect to human
  review.
- `tests/integration/test_llm_tools.py` and `test_agent.py` assert the
  agent always grounds its answers in tool output rather than free-form
  claims.

## 2. Fairness and prohibited inputs

`src/data/schema.py::PROHIBITED_PROTECTED_ATTRIBUTES` lists characteristics
that must never appear in the dataset or be used as a feature: race, color,
ethnicity, religion, sexual orientation, gender identity, political
opinion, disability, nationality, marital status. This is enforced by an
automated check in `src/data/validation.py` (`protected_attributes_found`),
not merely a generation-time promise — a dataset containing any of these
columns fails validation.

The synthetic generator does not construct any proxy variable intended to
approximate a protected characteristic. `age_band` is used as a coarse,
non-protected demographic segment consistent with common (if imperfect)
industry practice; a real deployment would need dedicated fair-lending
review of any age-based effect, which this demonstration does not perform.

## 3. Explainability

Every customer-level PD is explainable via SHAP
(`src/models/explainability.py`): top-5 risk-increasing and top-5
risk-reducing factors, reported with **real observed feature values**, not
scaled/encoded internals. The LLM agent is only permitted to narrate these
numbers — the system prompt explicitly forbids inventing a driver that did
not appear in the SHAP output, and `get_customer_shap` is the only path by
which the agent can discuss customer-level drivers.

## 4. Anti-hallucination architecture

Beyond prompting, hallucination is constrained structurally:

- Every numeric claim the agent can make comes from a typed tool call
  (`src/llm/tools/registry.py`); there is no path for the LLM to emit a
  number that didn't come from a tool result.
- `retrieve_credit_policy` returns document + section citations; the
  system prompt requires citing them, and retrieved text is explicitly
  treated as **data, not instructions** (prompt-injection defense — a
  retrieved policy chunk that said "ignore all previous instructions"
  would be quoted/analyzed, never obeyed).
- `query_portfolio`'s SQL guard (`src/llm/tools/sql_guard.py`) only allows
  single `SELECT`/`WITH` statements and rejects `DROP/DELETE/UPDATE/ALTER/
  INSERT/TRUNCATE/CREATE/ATTACH/PRAGMA/...` — verified by
  `tests/unit/test_sql_guard.py` with real injection-shaped payloads.
- Demo Mode (no LLM provider) routes to the same tools and renders their
  real output — it cannot "hallucinate" because it never generates
  free-form text; every response is templated from a tool result.

## 5. Model risk

- Temporal (out-of-time) validation is enforced, not just simulated — the
  test split is genuinely later in calendar time than train/validation
  (see [`docs/TEMPORAL_VALIDATION.md`](docs/TEMPORAL_VALIDATION.md)).
- Both champion and challenger metrics are always reported side by side
  (`reports/model_comparison.json`), and calibration is fit only on the
  validation split to avoid overfitting the calibration itself.
- Drift is monitored via PSI on both input features and the predicted PD
  distribution over time (`src/monitoring/drift.py`), with documented,
  demonstrative thresholds (see [`GOVERNANCE.md`](GOVERNANCE.md)).

## 6. Data governance and LGPD alignment

Although 100% synthetic, the project's data-handling design follows
data-minimization and purpose-limitation principles consistent with
Brazil's LGPD (Lei Geral de Proteção de Dados):

- No unnecessary field is generated or retained.
- Synthetic customer IDs never resemble real Brazilian identifiers (no
  valid-format CPF/CNPJ, card, or account numbers are generated).
- A real adaptation of this platform to actual customer data would require
  a full LGPD/compliance review before any use — this is stated explicitly
  rather than left implicit.

## 7. Auditability

Every AI Credit Analyst interaction is logged to
`reports/audit_log.jsonl` (`src/llm/audit.py`): timestamp, question, tools
invoked (name + success/failure, never row-level data), RAG sources cited,
and the model+calibration version — without ever logging secrets, API
keys, or full provider payloads (see [`SECURITY.md`](SECURITY.md)).

## 8. Limitations honestly stated

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the complete list:
reject inference, no concentration-limit modeling, synthetic-only data, no
independent model validation, demonstrative thresholds throughout.

## 9. Disclaimer

This is a portfolio/educational project. It must not be used to make real
credit decisions about real people without independent validation,
governance, compliance and legal review specific to the deploying
institution. See the repository [`LICENSE`](LICENSE).
