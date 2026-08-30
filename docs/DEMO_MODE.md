# Demo Mode

Credit Intelligence AI is designed so a recruiter or reviewer **without any
paid API key** can still exercise the full platform.

## What works with zero configuration

Everything except free-form LLM conversation:

- Synthetic data generation, validation, feature engineering.
- Both models (Logistic Regression, LightGBM), calibration, SHAP.
- Every analytics module: portfolio KPIs, segmentation, vintage/MOB,
  roll-rate, stress testing, what-if, drift monitoring.
- The full Streamlit dashboard (all 12 pages).
- The full FastAPI surface (`/health` through `/ai/chat`).
- RAG retrieval over the synthetic policy knowledge base (TF-IDF-based,
  no network call or model download required).

## The AI Credit Analyst in Demo Mode

Set `LLM_PROVIDER=demo` (the `.env.example` default). `DemoProvider`
(`src/llm/providers/demo_provider.py`) is a **deterministic, rule-based**
router — not a language model:

1. It matches the question against keyword patterns (English and
   Portuguese, accented and unaccented) to pick the same tool a real LLM
   would call — e.g. "compare risk bands A and D" → `compare_segments`.
2. It executes that tool for real, against the real scored portfolio.
3. It renders the tool's actual JSON output into a labeled response,
   prefixed with a clear `[Demo Mode: ...]` disclaimer.

This means every number a Demo Mode response shows is genuine — it simply
does not perform open-ended reasoning or handle arbitrary phrasing the way
a real LLM would.

## Enabling a real LLM provider

Set one of:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1   # or another tool-calling-capable local model
```

Ollama requires a locally running [Ollama](https://ollama.com) server with
a tool-calling-capable model pulled (`ollama pull llama3.1`) — no API key,
fully local and free.

If `LLM_PROVIDER` is set to a paid provider but the corresponding API key
is missing, `src/llm/providers/factory.py` logs a warning and falls back
to Demo Mode rather than crashing the app.

## Why TF-IDF instead of a neural embedding model for RAG

Downloading a sentence-transformer model requires network access, which
this platform should not assume is available (or desirable) in Demo Mode.
A `TfidfVectorizer` fit on the 5-document policy knowledge base gives
deterministic, reproducible, dependency-light vectors that are entirely
sufficient at this corpus size, while still exercising a real ChromaDB
vector-store workflow end to end (`src/rag/embeddings.py`).
