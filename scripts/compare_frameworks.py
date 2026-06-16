"""
Framework comparison (BONUS, +10).

Runs TWO evaluation frameworks on the SAME 20-case golden dataset and the SAME
agent, then compares their scores, pass rates, strictness and failure overlap.

Both run fully OFFLINE (no API key) so the comparison is reproducible:

  * Framework A — "RAGAS-style" : the graded RAGASEvaluator. Scores by TOKEN-SET
    overlap (bag of content words), order/sequence agnostic.

  * Framework B — "DeepEval-style": an assertion-based evaluator built on
    difflib.SequenceMatcher (character-sequence similarity). DeepEval's real
    philosophy is pass/fail assertions against a threshold, which we mirror.

These are faithful OFFLINE STAND-INS for the two frameworks' scoring philosophy,
not the pip packages (RAGAS/DeepEval metrics need an LLM API key to run). The
methodology and every number below are real and regenerable:

    python scripts/compare_frameworks.py   ->  reports/framework_comparison.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solution.solution import RAGASEvaluator  # noqa: E402
from scripts.run_benchmark_demo import qa, agent  # noqa: E402

PASS_THRESHOLD = 0.5  # both frameworks: a metric "passes" at >= 0.5


def _ratio(a: str, b: str) -> float:
    """Character-sequence similarity in [0, 1] (DeepEval-style)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class DeepEvalStyleEvaluator:
    """Assertion-based evaluator using sequence similarity instead of token sets."""

    def faithfulness(self, answer: str, context: str) -> float:
        return _ratio(answer, context)

    def relevance(self, answer: str, question: str) -> float:
        return _ratio(answer, question)

    def completeness(self, answer: str, expected: str) -> float:
        return _ratio(answer, expected)


def main() -> None:
    ragas = RAGASEvaluator()
    deepeval = DeepEvalStyleEvaluator()

    rows = []
    for p in qa:
        ans = agent(p.question)
        a = {
            "faithfulness": ragas.evaluate_faithfulness(ans, p.context),
            "relevance": ragas.evaluate_relevance(ans, p.question),
            "completeness": ragas.evaluate_completeness(ans, p.expected_answer),
        }
        b = {
            "faithfulness": deepeval.faithfulness(ans, p.context),
            "relevance": deepeval.relevance(ans, p.question),
            "completeness": deepeval.completeness(ans, p.expected_answer),
        }
        a_pass = all(v >= PASS_THRESHOLD for v in a.values())
        b_pass = all(v >= PASS_THRESHOLD for v in b.values())
        rows.append({"id": p.metadata["id"], "ragas": a, "deepeval": b,
                     "ragas_pass": a_pass, "deepeval_pass": b_pass})

    def avg(fw: str, metric: str) -> float:
        return round(st.mean(r[fw][metric] for r in rows), 3)

    summary = {
        "dataset_size": len(qa),
        "pass_threshold": PASS_THRESHOLD,
        "ragas": {
            "avg_faithfulness": avg("ragas", "faithfulness"),
            "avg_relevance": avg("ragas", "relevance"),
            "avg_completeness": avg("ragas", "completeness"),
            "pass_rate": round(st.mean(r["ragas_pass"] for r in rows), 3),
        },
        "deepeval": {
            "avg_faithfulness": avg("deepeval", "faithfulness"),
            "avg_relevance": avg("deepeval", "relevance"),
            "avg_completeness": avg("deepeval", "completeness"),
            "pass_rate": round(st.mean(r["deepeval_pass"] for r in rows), 3),
        },
    }
    # Failure overlap: cases both frameworks fail vs only one.
    both_fail = [r["id"] for r in rows if not r["ragas_pass"] and not r["deepeval_pass"]]
    only_ragas_fail = [r["id"] for r in rows if not r["ragas_pass"] and r["deepeval_pass"]]
    only_deepeval_fail = [r["id"] for r in rows if r["ragas_pass"] and not r["deepeval_pass"]]
    summary["failure_overlap"] = {
        "both_fail": both_fail,
        "only_ragas_fail": only_ragas_fail,
        "only_deepeval_fail": only_deepeval_fail,
        "agreement_rate": round(
            st.mean((not r["ragas_pass"]) == (not r["deepeval_pass"]) for r in rows), 3
        ),
    }
    stricter = "RAGAS-style" if summary["ragas"]["pass_rate"] <= summary["deepeval"]["pass_rate"] else "DeepEval-style"
    summary["stricter_framework"] = stricter

    print("=== FRAMEWORK COMPARISON (same dataset, same agent) ===")
    print(f"{'Metric':18s} {'RAGAS-style':>12s} {'DeepEval-style':>15s}")
    for m in ("avg_faithfulness", "avg_relevance", "avg_completeness", "pass_rate"):
        print(f"{m:18s} {summary['ragas'][m]:>12.3f} {summary['deepeval'][m]:>15.3f}")
    print(f"\nStricter framework: {stricter}")
    print(f"Agreement rate (same pass/fail verdict): {summary['failure_overlap']['agreement_rate']:.2%}")
    print(f"Both fail: {both_fail}")
    print(f"Only RAGAS fails: {only_ragas_fail}")
    print(f"Only DeepEval fails: {only_deepeval_fail}")

    out = REPO_ROOT / "reports" / "framework_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
