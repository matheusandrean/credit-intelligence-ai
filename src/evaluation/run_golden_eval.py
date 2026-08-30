"""Golden dataset evaluation for the Credit Intelligence Agent.

Runs every question in `evaluation/credit_questions.json` through the
agent and checks:

- **Tool correctness**: did the agent call the expected tool for this
  question type? (A model that answers "the average PD is 8%" without
  calling `portfolio_summary` first would fail this - the project's core
  anti-hallucination requirement, made testable.)
- **Groundedness / evidence**: for policy questions, does the final answer
  actually contain the expected retrieved evidence snippet?

Runs against Demo Mode by default (deterministic, no API key required) so
this evaluation is always runnable in CI; pass a different provider to
evaluate a real LLM's tool-selection behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.agents.credit_intelligence_agent import CreditIntelligenceAgent
from src.llm.providers.base import BaseLLMProvider
from src.llm.providers.demo_provider import DemoProvider
from src.llm.system_prompt import SYSTEM_PROMPT
from src.llm.tools.registry import list_tool_specs
from src.utils.config import PROJECT_ROOT
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "credit_questions.json"


@dataclass
class EvalResult:
    question_id: str
    question: str
    expected_tool: str
    actual_tools: list[str]
    tool_correct: bool
    evidence_found: bool | None


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def run_golden_eval(provider: BaseLLMProvider | None = None) -> list[EvalResult]:
    questions = load_questions()
    results: list[EvalResult] = []

    for q in questions:
        # A fresh provider/agent per question keeps each evaluation isolated.
        agent_provider = provider or DemoProvider(SYSTEM_PROMPT, list_tool_specs())
        agent = CreditIntelligenceAgent(provider=agent_provider)
        response = agent.ask(q["question"])

        tool_correct = q["expected_tool"] in response.tools_called

        evidence_found = None
        expected_evidence = q.get("expected_evidence_contains")
        if expected_evidence:
            evidence_found = expected_evidence.lower() in response.answer.lower()

        results.append(
            EvalResult(
                question_id=q["id"],
                question=q["question"],
                expected_tool=q["expected_tool"],
                actual_tools=response.tools_called,
                tool_correct=tool_correct,
                evidence_found=evidence_found,
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict:
    n = len(results)
    tool_accuracy = sum(r.tool_correct for r in results) / n if n else 0.0
    evidence_checks = [r for r in results if r.evidence_found is not None]
    evidence_accuracy = (
        sum(r.evidence_found for r in evidence_checks) / len(evidence_checks)
        if evidence_checks
        else None
    )
    return {
        "n_questions": n,
        "tool_selection_accuracy": tool_accuracy,
        "evidence_grounding_accuracy": evidence_accuracy,
        "results": [
            {
                "id": r.question_id,
                "question": r.question,
                "expected_tool": r.expected_tool,
                "actual_tools": r.actual_tools,
                "tool_correct": r.tool_correct,
                "evidence_found": r.evidence_found,
            }
            for r in results
        ],
    }


def main() -> None:
    configure_logging("INFO")
    results = run_golden_eval()
    summary = summarize(results)
    output_path = PROJECT_ROOT / "reports" / "golden_eval_report.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "golden_eval_complete",
        tool_selection_accuracy=summary["tool_selection_accuracy"],
        evidence_grounding_accuracy=summary["evidence_grounding_accuracy"],
    )


if __name__ == "__main__":
    main()
