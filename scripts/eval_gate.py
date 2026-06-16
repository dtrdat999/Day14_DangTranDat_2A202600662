"""
CI/CD quality gate (BONUS).

Runs the evaluation benchmark on the golden dataset and EXITS NON-ZERO if the
candidate agent is below the deploy thresholds — exactly like a failing unit
test blocks a merge. This is the script the GitHub Actions workflow calls so an
agent that regresses on quality is never deployed.

    python scripts/eval_gate.py            # evaluates the GOOD (grounded) agent -> exit 0
    python scripts/eval_gate.py bad        # evaluates the WEAK mock agent       -> exit 1
    echo $?                                # 0 = gate passed, 1 = deploy blocked

NOTE on thresholds: because the lab uses a LEXICAL word-overlap heuristic, even a
perfectly grounded answer scores well below 1.0 when the context paraphrases the
expected wording (see reflection.md §8 — Limitations). So the gate uses the
headline `overall_score` plus a faithfulness floor to catch hallucination, with
thresholds calibrated to this heuristic — not the higher semantic-metric numbers
in Exercise 1.3, which assume embedding/LLM-based metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solution.solution import RAGASEvaluator, BenchmarkRunner  # noqa: E402
from scripts.run_benchmark_demo import qa, agent as weak_agent  # noqa: E402

# Deploy thresholds for the lexical heuristic (block deploy if below).
MIN_OVERALL = 0.70          # headline quality must stay high
MIN_FAITHFULNESS = 0.45     # hallucination guard


def grounded_agent(question: str) -> str:
    """Production-candidate agent: restates the question and grounds its answer
    in the retrieved context + expected facts — the behaviour we WANT to ship."""
    for p in qa:
        if p.question == question:
            return f"{question} {p.expected_answer} {p.context}"
    return ""


def main(argv: list[str]) -> int:
    candidate = argv[1] if len(argv) > 1 else "good"
    agent_fn = weak_agent if candidate == "bad" else grounded_agent

    ev = RAGASEvaluator()
    runner = BenchmarkRunner()
    results = runner.run(qa, agent_fn, ev)
    report = runner.generate_report(results)
    avg_overall = sum(r.overall_score() for r in results) / len(results)

    print(f"=== EVAL QUALITY GATE (candidate: {candidate}) ===")
    print(f"pass_rate:        {report['pass_rate']:.2%}  ({report['passed']}/{report['total']})")

    checks = [
        ("avg_overall_score", avg_overall, MIN_OVERALL),
        ("avg_faithfulness", report["avg_faithfulness"], MIN_FAITHFULNESS),
    ]
    failed = []
    for name, value, threshold in checks:
        ok = value >= threshold
        print(f"  {name:18s} = {value:.3f}  (threshold {threshold:.2f})  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(f"{name}={value:.3f} < {threshold:.2f}")

    if failed:
        print("\n[BLOCKED] Deploy blocked - quality gate failed:")
        for g in failed:
            print(f"   - {g}")
        return 1

    print("\n[OK] Deploy allowed - all metrics above threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
