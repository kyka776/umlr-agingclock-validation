from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_characterizer():
    path = Path(__file__).parents[1] / "upstream-tests" / "characterize_upstream.py"
    spec = importlib.util.spec_from_file_location("characterize_upstream", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.characterize


def test_characterization_distinguishes_formula_boundary() -> None:
    characterize = _load_characterizer()
    strict = characterize(
        "young <- which(y < mean.y)\nold <- which(y > mean.y)\n"
        "opt_sol <- opt$message$primal_solution"
    )
    paper = characterize("young <- which(y <= mean.y)\nold <- which(y > mean.y)")
    assert strict["code_lower_region_uses_strict_less_than"] is True
    assert strict["paper_lower_region_uses_less_than_or_equal"] is False
    assert strict["solver_status_checked"] is False
    assert paper["paper_lower_region_uses_less_than_or_equal"] is True
