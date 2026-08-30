"""Credit Intelligence AI - FastAPI application.

Exposes the platform's analytics, model, monitoring, simulation and AI
Credit Analyst capabilities as a REST API with automatic OpenAPI docs at
`/docs`. This is a decision-support API: no endpoint approves, denies, or
finalizes any credit decision (see RESPONSIBLE_AI.md).
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routers import chat, customer, model, monitoring, portfolio, simulation
from src.utils.config import get_settings
from src.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Credit Intelligence AI API",
    description=(
        "Decision-support API for a synthetic-data Credit Intelligence platform: "
        "credit risk model outputs, explainability, portfolio analytics, stress "
        "testing / what-if simulation, drift monitoring and an AI Credit Analyst. "
        "This API never approves or denies credit - see RESPONSIBLE_AI.md."
    ),
    version="0.1.0",
)

app.include_router(portfolio.router)
app.include_router(customer.router)
app.include_router(model.router)
app.include_router(monitoring.router)
app.include_router(simulation.router)
app.include_router(chat.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}
