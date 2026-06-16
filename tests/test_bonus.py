"""Bonus tests — custom metric (token-level F1) on RAGASEvaluator.

Loaded the same way as the main suite (solution/solution.py preferred).
"""
import importlib.util
import sys
import unittest
from pathlib import Path

DAY_DIR = Path(__file__).parent.parent
SOLUTION_DIR = DAY_DIR / "solution"


def _load(path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


if (SOLUTION_DIR / "solution.py").exists():
    _m = _load(SOLUTION_DIR / "solution.py", f"{DAY_DIR.name}.bonus_solution")
else:
    _m = _load(DAY_DIR / "template.py", f"{DAY_DIR.name}.bonus_template")

RAGASEvaluator = getattr(_m, "RAGASEvaluator")


class TestCustomF1Metric(unittest.TestCase):
    def setUp(self):
        self.ev = RAGASEvaluator()

    def test_identical_is_one(self):
        score = self.ev.evaluate_f1("Python is a programming language",
                                    "Python is a programming language")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_disjoint_is_zero(self):
        score = self.ev.evaluate_f1("Jupiter Neptune Saturn", "Python programming language")
        self.assertEqual(score, 0.0)

    def test_both_empty_is_one(self):
        self.assertEqual(self.ev.evaluate_f1("", ""), 1.0)

    def test_one_empty_is_zero(self):
        self.assertEqual(self.ev.evaluate_f1("something", ""), 0.0)

    def test_in_range(self):
        score = self.ev.evaluate_f1("Python is a great language", "Python is a programming language")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_penalises_bloated_answer(self):
        """Bloated-but-correct answer should score LOWER on F1 than on one-sided completeness."""
        expected = "Paris is the capital"
        bloated = ("Paris is the capital and also here is a long irrelevant tangent about "
                   "bananas weather sports history and many other unrelated topics entirely")
        completeness = self.ev.evaluate_completeness(bloated, expected)
        f1 = self.ev.evaluate_f1(bloated, expected)
        self.assertGreater(completeness, f1)


if __name__ == "__main__":
    unittest.main()
