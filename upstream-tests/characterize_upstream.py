"""Characterize a user-fetched official UMLR file without redistributing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED_SHA256 = "8050563de787434452f6f8a8105ef1163227a7ddecb7a389f8f9acbf7fb5073b"


def characterize(source_text: str) -> dict[str, bool | str]:
    compact = re.sub(r"\s+", "", source_text)
    return {
        "paper_lower_region_uses_less_than_or_equal": "which(y<=mean.y)" in compact,
        "code_lower_region_uses_strict_less_than": "which(y<mean.y)" in compact,
        "upper_region_uses_strict_greater_than": "which(y>mean.y)" in compact,
        "explicit_constant_outcome_guard": bool(
            re.search(
                r"(constant|zero\\s*variance|length\\(young\\)\\s*==\\s*0)",
                source_text,
                re.I,
            )
        ),
        "solver_status_checked": bool(
            re.search(r"opt\\$(status|code)|solution\\s*status|solver_status", source_text, re.I)
        ),
        "constraint_residual_checked": bool(
            re.search(r"(constraint.*residual|Aeq.*betahat.*beq)", source_text, re.I)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    content = arguments.source.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            f"unexpected upstream source: expected {EXPECTED_SHA256}, got {actual_hash}"
        )
    report: dict[str, object] = {
        "upstream_commit": "3cf11abcecc5c3c99b9748d64c04f570d51243c3",
        "file": "Lasso_UP.R",
        "sha256": actual_hash,
        **characterize(content.decode("utf-8")),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
